"""Micro-benchmarks etendus pour l'adaptation RTX 4060 Ti (sm_89).

Sections :
0) bande passante DRAM effective (copie)
1) head GEMM bf16 vs fp8 _scaled_mm (6144x768x50304)
2) MLP : eager vs kernel fusionne (lrs_fwd/lrs_bwd), bf16 vs fp8, sweep de configs
3) CE : kernel seul, chemin autograd complet fwd+bwd, pic VRAM en fonction de T
4) transposes fp8 du backward CE + grad_w fp8 vs bf16
5) Muon/Polar Express : XXT, XTX, ba_plus_cAA
6) attention : FA2 varlen vs SDPA (flash/cudnn) vs FlexAttention (sliding window)
7) validation numerique contre references fp32

Usage : venv/bin/python bench_kernels.py
"""
import torch
import triton
import torch.nn.functional as F

import triton_kernels
from triton_kernels import (ce_fwd_bwd, transpose_copy, XXT, XTX, ba_plus_cAA,
                            FusedSoftcappedCrossEntropy, FusedLinearReLUSquareFunction)

props = torch.cuda.get_device_properties(0)
print(f"{props.name}: {props.multi_processor_count} SMs, {props.total_memory / 1e9:.1f} GB")
print(f"torch {torch.__version__}, triton {triton.__version__}")


def bench(fn, n_warmup=10, n_rep=50):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_rep):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_rep


def rel_err(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


T, D, V, H = 6144, 768, 50304, 3072

# ---- 0) Bande passante DRAM ----
a = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device="cuda")
b = torch.empty_like(a)
ms = bench(lambda: b.copy_(a))
print(f"\n[mem] copy 512MB: {ms:.3f} ms -> {2 * a.numel() / (ms / 1e3) / 1e9:.0f} GB/s effectifs")
del a, b
torch.cuda.empty_cache()

# ---- 1) Head GEMM : bf16 vs fp8 ----
x = torch.randn(T, D, device="cuda", dtype=torch.bfloat16)
w = torch.randn(D, V, device="cuda", dtype=torch.bfloat16)
flops = 2 * T * D * V

ms = bench(lambda: x @ w)
print(f"\n[head] bf16 mm        : {ms:.3f} ms ({flops / ms / 1e9:.0f} TFLOPS)")

x_s, w_s = 100 / 448, 1.6 / 448
x_f8 = (x / x_s).to(torch.float8_e4m3fn)
w_f8 = (w / w_s).to(torch.float8_e4m3fn)
w_f8_col = w_f8.T.contiguous().T
scale_a = torch.tensor(x_s, device="cuda")
scale_b = torch.tensor(w_s, device="cuda")

ms = bench(lambda: torch._scaled_mm(x_f8, w_f8_col, out_dtype=torch.bfloat16,
                                    scale_a=scale_a, scale_b=scale_b, use_fast_accum=True))
print(f"[head] fp8 _scaled_mm : {ms:.3f} ms ({flops / ms / 1e9:.0f} TFLOPS)")


def fp8_head_with_quant():
    xf = (x / x_s).to(torch.float8_e4m3fn)
    wf = (w / w_s).to(torch.float8_e4m3fn)
    wfc = wf.T.contiguous().T
    return torch._scaled_mm(xf, wfc, out_dtype=torch.bfloat16,
                            scale_a=scale_a, scale_b=scale_b, use_fast_accum=True)


ms = bench(fp8_head_with_quant)
print(f"[head] fp8 + quant    : {ms:.3f} ms (inclut les casts et la transpose)")

# ---- 2) MLP : eager vs kernel fusionne ----
w1 = torch.randn(H, D, device="cuda", dtype=torch.bfloat16)
w2 = torch.randn(H, D, device="cuda", dtype=torch.bfloat16)
flops_up = 2 * T * D * H

ms = bench(lambda: x @ w1.T)
print(f"\n[mlp] up bf16   : {ms:.3f} ms ({flops_up / ms / 1e9:.0f} TFLOPS)")

w1_f8 = (w1 / w_s).to(torch.float8_e4m3fn)
ms = bench(lambda: torch._scaled_mm(x_f8, w1_f8.T, out_dtype=torch.bfloat16,
                                    scale_a=scale_a, scale_b=scale_b, use_fast_accum=True))
print(f"[mlp] up fp8    : {ms:.3f} ms ({flops_up / ms / 1e9:.0f} TFLOPS)")


@torch.compile
def eager_mlp(x, w1, w2):
    pre = x @ w1.T
    post = torch.relu(pre) ** 2
    return post @ w2


ms = bench(lambda: eager_mlp(x, w1, w2))
print(f"[mlp] eager relu2 fwd : {ms:.3f} ms (2 GEMMs + pointwise)")

# -- 2b) kernel fusionne : fwd / bwd / step complet, bf16 et fp8 --
g_out = torch.randn(T, D, device="cuda", dtype=torch.bfloat16)
pre, post = torch.ops.nanogpt.lrs_fwd(x, w1)

ms = bench(lambda: torch.ops.nanogpt.lrs_fwd(x, w1))
print(f"\n[mlp] lrs_fwd bf16      : {ms:.3f} ms")
ms = bench(lambda: torch.ops.nanogpt.lrs_bwd(g_out, w2, pre))
print(f"[mlp] lrs_bwd bf16      : {ms:.3f} ms")

dq = torch.tensor(x_s * w_s, device="cuda", dtype=torch.float32)
ms = bench(lambda: torch.ops.nanogpt.lrs_fwd(x, w1, a_f8=x_f8, b_f8=w1_f8, dequant_scale_ptr=dq))
print(f"[mlp] lrs_fwd fp8       : {ms:.3f} ms (poids pre-quantifies)")


def fused_mlp_with_quant():
    xf = (x / x_s).to(torch.float8_e4m3fn)
    wf = (w1 / w_s).to(torch.float8_e4m3fn)
    return torch.ops.nanogpt.lrs_fwd(x, w1, a_f8=xf, b_f8=wf, dequant_scale_ptr=dq)


ms = bench(fused_mlp_with_quant)
print(f"[mlp] lrs_fwd fp8+quant : {ms:.3f} ms (casts inclus)")

# step complet fwd+bwd via l'autograd Function (chemin de production)
x3d = x.view(1, T, D).requires_grad_(True)
w1g = w1.clone().requires_grad_(True)
w2g = w2.clone().requires_grad_(True)
g3d = g_out.view(1, T, D)


def fused_step():
    out = FusedLinearReLUSquareFunction.apply(x3d, w1g, w2g)
    out.backward(g3d)


def eager_step():
    out = eager_mlp(x3d, w1g, w2g)
    out.backward(g3d)


ms = bench(fused_step)
print(f"[mlp] step fusionne fwd+bwd : {ms:.3f} ms (autograd Function)")
ms = bench(eager_step)
print(f"[mlp] step eager    fwd+bwd : {ms:.3f} ms (compile)")

# -- 2c) sweep de configs LRS (recompile a chaque combo, warmup inclus) --
print("\n[mlp] sweep configs lrs (BM, BN, st_fwd, st_bwd, warps)")
for bm_, bn_, sf, sb, wp in [(64, 128, 4, 3, 8), (64, 128, 3, 3, 8),
                             (128, 128, 3, 3, 8), (128, 64, 3, 3, 8)]:
    triton_kernels.LRS_BM, triton_kernels.LRS_BN = bm_, bn_
    triton_kernels.LRS_STAGES_FWD = sf
    triton_kernels.LRS_STAGES_BWD = sb
    triton_kernels.LRS_WARPS = wp
    try:
        msf = bench(lambda: torch.ops.nanogpt.lrs_fwd(x, w1))
        msb = bench(lambda: torch.ops.nanogpt.lrs_bwd(g_out, w2, pre))
        print(f"  {bm_:>3}x{bn_:<3} st{sf}/{sb} w{wp} : fwd {msf:.3f} ms | bwd {msb:.3f} ms")
    except Exception as e:
        print(f"  {bm_:>3}x{bn_:<3} st{sf}/{sb} w{wp} : ECHEC ({type(e).__name__})")
# restaure la config recommandee
triton_kernels.LRS_BM, triton_kernels.LRS_BN = 64, 128
triton_kernels.LRS_STAGES_FWD, triton_kernels.LRS_STAGES_BWD, triton_kernels.LRS_WARPS = 4, 3, 8

# ---- 3) Kernel CE ----
torch.manual_seed(0)
logits = torch.randn(T, V, device="cuda", dtype=torch.bfloat16) * 5
targets = torch.randint(0, 50257, (T + 4,), device="cuda", dtype=torch.int64)
losses = torch.empty(T, dtype=torch.float32, device="cuda")
grad_input = torch.empty(T, V, dtype=torch.float8_e5m2, device="cuda")

mtp1 = torch.tensor([1.0], device="cuda")
ms = bench(lambda: ce_fwd_bwd(logits, targets, mtp1, losses, grad_input, T, 1, 23.0, 5.0, 7.5, 1.0, 1.0))
bytes_ce = T * V * (2 + 1)
print(f"\n[ce] fwd_bwd mtp=1 : {ms:.3f} ms ({bytes_ce / (ms / 1e3) / 1e9:.0f} GB/s)")

mtp3 = torch.tensor([1.0, 0.5, 0.25], device="cuda")
ms = bench(lambda: ce_fwd_bwd(logits, targets, mtp3, losses, grad_input, T, 3, 23.0, 5.0, 7.5, 1.0, 1.0))
print(f"[ce] fwd_bwd mtp=3 : {ms:.3f} ms")

# -- 3b) chemin autograd complet (scaled_mm + kernel + transposes + grad_w) --
# NB : on passe les 11 arguments explicitement pour matcher le backward (11 retours)
lm_head_w = (torch.randn(V, D, device="cuda", dtype=torch.bfloat16) * 0.02).requires_grad_(True)
xh = x.clone().requires_grad_(True)
ones = torch.ones(T, device="cuda")


def ce_full_step():
    ls = FusedSoftcappedCrossEntropy.apply(xh, targets, mtp3, lm_head_w,
                                           x_s, w_s, 1.0, 1.0, 23.0, 5.0, 7.5)
    ls.backward(ones)


ms = bench(ce_full_step)
print(f"[ce] step complet fwd+bwd : {ms:.3f} ms (head fp8 + CE + grads)")

# -- 3c) pic VRAM du step CE en fonction de T --
print("\n[ce] pic VRAM fwd+bwd (save_for_backward allege ?)")
for t in (6144, 12288, 24576):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    xt = torch.randn(t, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    tgt = torch.randint(0, V, (t + 4,), device="cuda", dtype=torch.int64)
    onest = torch.ones(t, device="cuda")
    ls = FusedSoftcappedCrossEntropy.apply(xt, tgt, mtp3, lm_head_w,
                                           x_s, w_s, 1.0, 1.0, 23.0, 5.0, 7.5)
    ls.backward(onest)
    peak = torch.cuda.max_memory_allocated() / 2**20
    print(f"  T={t:>6} : pic {peak:,.0f} MiB")
    del xt, tgt, onest, ls
del lm_head_w, xh, ones
torch.cuda.empty_cache()

# ---- 4) Transposes fp8 du backward CE ----
gi_T = torch.empty(V, T, dtype=torch.float8_e5m2, device="cuda")
ms = bench(lambda: transpose_copy(grad_input, gi_T))
print(f"\n[bwd] transpose grad_input fp8 ({T}x{V}): {ms:.3f} ms ({2 * T * V / (ms / 1e3) / 1e9:.0f} GB/s)")

x_f8_T = torch.empty(D, T, dtype=torch.float8_e4m3fn, device="cuda")
ms = bench(lambda: torch._scaled_mm(x_f8_T, gi_T.T, out_dtype=torch.float32,
                                    scale_a=scale_a, scale_b=scale_b, use_fast_accum=False))
print(f"[bwd] grad_w fp8 mm  : {ms:.3f} ms ({2 * D * T * V / ms / 1e9:.0f} TFLOPS)")

grad_bf16 = torch.randn(T, V, device="cuda", dtype=torch.bfloat16)
ms = bench(lambda: grad_bf16.T @ x)
print(f"[bwd] grad_w bf16 mm : {ms:.3f} ms ({2 * V * T * D / ms / 1e9:.0f} TFLOPS, sans transpose prealable)")
del grad_bf16
torch.cuda.empty_cache()

# ---- 5) Muon / Polar Express : XXT, XTX, ba_plus_cAA ----
g_sq = torch.randn(768, 768, device="cuda", dtype=torch.bfloat16)
out_sq = torch.empty(768, 768, device="cuda", dtype=torch.bfloat16)
g_tall = torch.randn(H, D, device="cuda", dtype=torch.bfloat16)
out_768 = torch.empty(768, 768, device="cuda", dtype=torch.bfloat16)

ms = bench(lambda: XXT(g_sq, out_sq))
print(f"\n[muon] XXT  (768x768)        : {ms:.3f} ms")
ms = bench(lambda: XTX(g_tall, out_768))
print(f"[muon] XTX  (3072x768 -> 768^2): {ms:.3f} ms")
ms = bench(lambda: ba_plus_cAA(g_sq, 1.0, 0.5, out_sq))
print(f"[muon] ba_plus_cAA (768x768)   : {ms:.3f} ms")

# ---- 6) Attention : FA2 varlen vs SDPA vs FlexAttention ----
print("\n[attn] duel des backends (adaptez S/NH/DH/WINDOW a votre config)")
S, NH, DH, WINDOW = 1024, 12, 64, 512
B = max(1, T // S)
q = torch.randn(B, NH, S, DH, device="cuda", dtype=torch.bfloat16)
k_ = torch.randn_like(q)
v_ = torch.randn_like(q)


def attn_tflops(ms, win):
    eff = min(win, S)
    return 4 * B * NH * S * eff * DH / ms / 1e9


# 6a) FA2 varlen (chemin de production)
try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
    qv = q.transpose(1, 2).reshape(B * S, NH, DH).contiguous()
    kv = k_.transpose(1, 2).reshape(B * S, NH, DH).contiguous()
    vv = v_.transpose(1, 2).reshape(B * S, NH, DH).contiguous()
    cu = torch.arange(0, (B + 1) * S, S, device="cuda", dtype=torch.int32)
    ms = bench(lambda: flash_attn_varlen_func(qv, kv, vv, cu, cu, S, S,
                                              causal=True, window_size=(WINDOW, 0)))
    print(f"  FA2 varlen causal w={WINDOW} : {ms:.3f} ms ({attn_tflops(ms, WINDOW):.0f} TFLOPS)")
except Exception as e:
    print(f"  FA2 varlen indisponible : {type(e).__name__}: {str(e)[:80]}")

# 6b) SDPA flash / cudnn (padde, causal plein -> plus de FLOPs que la fenetre)
try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
    for name, be in [("flash", SDPBackend.FLASH_ATTENTION), ("cudnn", SDPBackend.CUDNN_ATTENTION)]:
        try:
            with sdpa_kernel([be]):
                ms = bench(lambda: F.scaled_dot_product_attention(q, k_, v_, is_causal=True))
            print(f"  SDPA {name:<6} causal plein : {ms:.3f} ms ({attn_tflops(ms, S):.0f} TFLOPS)")
        except Exception as e:
            print(f"  SDPA {name:<6} indisponible : {type(e).__name__}: {str(e)[:80]}")
except ImportError:
    print("  sdpa_kernel indisponible sur cette version de torch")

# 6c) FlexAttention avec sliding window + causal (remplace potentiellement le varlen)
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask

    def sliding_causal(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & (q_idx - kv_idx < WINDOW)

    bm = create_block_mask(sliding_causal, B=None, H=None, Q_LEN=S, KV_LEN=S, device="cuda")
    flex_c = torch.compile(flex_attention)
    ms = bench(lambda: flex_c(q, k_, v_, block_mask=bm))
    print(f"  FlexAttention w={WINDOW}      : {ms:.3f} ms ({attn_tflops(ms, WINDOW):.0f} TFLOPS)")
except Exception as e:
    print(f"  FlexAttention indisponible : {type(e).__name__}: {str(e)[:80]}")

# ---- 7) Validation numerique (references fp32) ----
print("\n[check] validation numerique (err relative L2, bf16 ~1e-2, fp8 grad ~quelques %)")
torch.manual_seed(0)
xs = torch.randn(2048, D, device="cuda", dtype=torch.bfloat16)
w1s = torch.randn(H, D, device="cuda", dtype=torch.bfloat16)
w2s = torch.randn(H, D, device="cuda", dtype=torch.bfloat16)

pre_t, post_t = torch.ops.nanogpt.lrs_fwd(xs, w1s)
ref_pre = xs.float() @ w1s.float().T
ref_post = torch.relu(ref_pre) ** 2
print(f"  lrs_fwd  : pre {rel_err(pre_t, ref_pre):.2e} | post {rel_err(post_t, ref_post):.2e}")

g_s = torch.randn(2048, D, device="cuda", dtype=torch.bfloat16)
dpre_t = torch.ops.nanogpt.lrs_bwd(g_s, w2s, pre_t)
ref_dpre = 2 * (g_s.float() @ w2s.float().T) * torch.relu(ref_pre)
print(f"  lrs_bwd  : dpre {rel_err(dpre_t, ref_dpre):.2e}")

n = 4096
lg = torch.randn(n, V, device="cuda", dtype=torch.bfloat16) * 5
tg = torch.randint(0, V, (n + 4,), device="cuda", dtype=torch.int64)
ls = torch.empty(n, dtype=torch.float32, device="cuda")
gi = torch.empty(n, V, dtype=torch.float8_e5m2, device="cuda")
ce_fwd_bwd(lg, tg, mtp1, ls, gi, n, 1, 23.0, 5.0, 7.5, 1.0, 1.0)

sig = torch.sigmoid((lg.float() + 5.0) / 7.5)
z = 23.0 * sig
ref_loss = torch.logsumexp(z, dim=-1) - z.gather(1, tg[:n, None]).squeeze(1)
print(f"  ce loss  : {rel_err(ls, ref_loss):.2e}")

p = torch.softmax(z, dim=-1)
onehot = torch.zeros_like(p).scatter_(1, tg[:n, None], 1.0)
ref_grad = (23.0 / 7.5) * (p - onehot) * sig * (1 - sig)
print(f"  ce grad  : {rel_err(gi, ref_grad):.2e} (fp8 e5m2 : quelques % normal)")