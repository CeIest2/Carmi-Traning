# -*- coding: utf-8 -*-
"""
perf_model.py — Estimation analytique + mesurée des performances du modèle.

1. Charge les définitions de train_gpt.py SANS lancer l'entraînement
   (coupe du source juste avant la section "int main").
2. Compte exact des paramètres par groupe.
3. Estimation analytique (roofline) : FLOPs/token, FLOPs/step, temps de step
   pour chaque stage du schedule, tokens/s en inférence.
4. Mesure réelle : forward+backward sur batch synthétique (eager par défaut,
   --compile pour torch.compile ~7 min de chauffe) -> MFU réel de la carte.
5. Inférence : estimation bande passante + mesure optionnelle du decode.

Usage:
    python perf_model.py              # analyse + mesure eager fwd/bwd
    python perf_model.py --compile    # idem mais avec torch.compile (long)
    python perf_model.py --decode     # ajoute une mesure de decode batch=1
    python perf_model.py --no-measure # analyse analytique uniquement
"""
import os
import sys
import math

# --- Environnement distribué minimal (train_gpt.py l'exige au top-level) ---
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29777")

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_gpt.py")

with open(SRC) as f:
    src = f.read()

# Coupe avant "int main" : tout ce qui précède est des définitions + setup dist
cut = src.index("# int main")
ns = {"__name__": "perf_model_bootstrap", "__file__": SRC}
exec(compile(src[:cut], SRC, "exec"), ns)

import torch

GPT = ns["GPT"]
ForwardScheduleConfig = ns["ForwardScheduleConfig"]
get_bigram_hash = ns["get_bigram_hash"]
grad_scale = ns["grad_scale"]
grad_accum_steps = ns["grad_accum_steps"]
args = ns["args"]
BOS_ID = ns.get("BOS_ID", 50257)

# -----------------------------------------------------------------------------
# Specs GPU (pics théoriques denses)
# -----------------------------------------------------------------------------
GPU_SPECS = {
    "4060 Ti":  dict(bf16=88.0,  fp8=176.0, bw=288e9),    # 4352 cœurs @2.54 GHz, GDDR6 128-bit
    "4090":     dict(bf16=165.0, fp8=330.0, bw=1008e9),
    "A100":     dict(bf16=312.0, fp8=None,  bw=1555e9 if "80" else 1555e9),
    "H100":     dict(bf16=989.0, fp8=1979.0, bw=3350e9),
}

def gpu_specs():
    name = torch.cuda.get_device_name(0)
    for key, spec in GPU_SPECS.items():
        if key in name:
            return name, spec
    print(f"!! GPU '{name}' inconnu : valeurs par défaut RTX 4060 Ti. "
          "Éditer GPU_SPECS dans perf_model.py sinon.")
    return name, GPU_SPECS["4060 Ti"]

# -----------------------------------------------------------------------------
# 1. Instanciation et comptage des paramètres
# -----------------------------------------------------------------------------
MODEL_CFG = dict(vocab_size=50257, num_layers=11, num_heads=6, head_dim=128,
                 model_dim=768, max_seq_len=4096)

def build_model():
    model = GPT(**MODEL_CFG).cuda()
    for m in model.modules():
        if isinstance(m, (torch.nn.Embedding, torch.nn.Linear)):
            m.weight.data = m.weight.data.bfloat16()
    for name in ["attn_gate_bank", "ve_gate_bank", "qk_bank", "vo_bank",
                 "mlp_bank", "mudd_w1", "mudd_w2", "mudd_b2"]:
        getattr(model, name).data = getattr(model, name).data.bfloat16()
    return model

def param_report(model):
    groups = {}
    for name, p in model.named_parameters():
        key = name.split(".")[0]
        groups.setdefault(key, 0)
        groups[key] += p.numel()
    total = sum(groups.values())
    print("\n=== Paramètres par groupe ===")
    for k, v in sorted(groups.items(), key=lambda kv: -kv[1]):
        print(f"  {k:20s} {v/1e6:9.2f} M  ({v*2/1e6:8.1f} Mo en bf16)")
    print(f"  {'TOTAL':20s} {total/1e6:9.2f} M  ({total*2/1e6:8.1f} Mo en bf16)")

    # mlp_bank contient une 12e couche de padding inutilisée (boucle = 11 couches)
    mlp_used = model.mlp_bank[:MODEL_CFG["num_layers"]].numel()
    body = (model.qk_bank.numel() + model.vo_bank.numel() + mlp_used
            + model.mudd_w1.numel() + model.mudd_w2.numel())
    embeds = (model.embed.weight.numel() + model.value_embeds.numel()
              + model.bigram_embed.weight.numel())
    lm_head = model.lm_head.weight.numel()
    return dict(total=total, body=body, embeds=embeds, lm_head=lm_head,
                vocab=model.vocab_size, dim=MODEL_CFG["model_dim"])

# -----------------------------------------------------------------------------
# 2. FLOPs analytiques
# -----------------------------------------------------------------------------
# Pattern des fenêtres d'attention : [s,s,s,l,s,s,None,s,s,s,l] (couche 6 sans attn)
BM_PATTERN = ["s", "s", "s", "l", "s", "s", None, "s", "s", "s", "l"]
BLOCK = 128

def flops_per_token(counts, stage, seq_len):
    """FLOPs forward par token. Training ≈ 3x (fwd + bwd~2x fwd)."""
    d = counts["dim"]
    matmul = 2 * counts["body"]                      # poids matriciels du corps
    lm_head = 2 * counts["lm_head"]                  # 1 seul appel (MTP géré dans le kernel CE)
    ws, wl = stage["ws_short"] * BLOCK, stage["ws_long"] * BLOCK
    attn = 0
    for b in BM_PATTERN:
        if b is None:
            continue
        w = ws if b == "s" else wl
        w_eff = min(w, seq_len) / 2                  # moyenne causale dans la fenêtre
        attn += 4 * d * w_eff                        # QK^T + AV : 2 matmuls × 2·w·d
    return matmul + lm_head + attn, matmul, lm_head, attn

STAGES = [
    dict(name="stage 1 (steps 0-459)",   tokens=8 * 2048 * 8,  ws_short=1, ws_long=3,  seq=896),
    dict(name="stage 2 (steps 460-919)", tokens=16 * 2048 * 8, ws_short=3, ws_long=7,  seq=2048),
    dict(name="stage 3 (steps 920-1380)",tokens=24 * 2048 * 8, ws_short=5, ws_long=11, seq=2048),
]

def analytic_report(counts, spec):
    peak = spec["bf16"] * 1e12
    print("\n=== Analyse roofline (training) ===")
    print(f"Corps (matmuls 6N)   : {counts['body']/1e6:.1f} M params -> {6*counts['body']/1e6:.0f} MFLOP/token")
    print(f"lm_head              : {counts['lm_head']/1e6:.1f} M params -> {6*counts['lm_head']/1e6:.0f} MFLOP/token")
    print(f"Embeddings (lookup)  : {counts['embeds']/1e6:.1f} M params -> ~0 FLOP (gather, pas de matmul)")
    total_tokens = sum(s["tokens"] for s in STAGES)  # 1/3 du schedule chacun
    for s in STAGES:
        fpt, matmul, lm, attn = flops_per_token(counts, s, s["seq"])
        train_fpt = 3 * fpt
        flop_step = train_fpt * s["tokens"]
        print(f"\n  {s['name']}: {s['tokens']:,} tokens/step, seq<={s['seq']}")
        print(f"    FLOPs/token (fwd) : {fpt/1e6:.0f} M  dont attn {attn/1e6:.0f} M ({100*attn/fpt:.1f} %)")
        print(f"    FLOPs/step (train): {flop_step/1e12:.0f} TFLOP")
        for mfu in (0.20, 0.30, 0.40, 0.50):
            print(f"      MFU {mfu:.0%} -> {flop_step / (peak * mfu):6.2f} s/step")
    # temps total approximatif : chaque stage ~ 1/3 de 1380 steps + 10 ext
    print("\n  Durée totale estimée (1390 steps, 460 steps/stage en moyenne) :")
    for mfu in (0.20, 0.30, 0.40, 0.50):
        t = 0
        for s in STAGES:
            fpt, *_ = flops_per_token(counts, s, s["seq"])
            t += 464 * (3 * fpt * s["tokens"]) / (peak * mfu)
        print(f"      MFU {mfu:.0%} -> {t/3600:.2f} h")
    print("  (hors optimizer NorMuon/polar-express, data loading, evals : ajouter ~10-20 %)")

def inference_report(counts, spec, ctx=2048):
    bw = spec["bw"]
    d = counts["dim"]
    # Decode : on relit les poids matriciels + lm_head. Les tables d'embedding
    # ne sont PAS relues (lookup d'une seule ligne par token).
    weights_bytes = 2 * (counts["body"] + counts["lm_head"])
    # KV cache lu à chaque token généré (fenêtre moyenne, causal)
    ws, wl = 5 * BLOCK, 11 * BLOCK
    kv_bytes = 0
    for b in BM_PATTERN:
        if b is None:
            continue
        w = ws if b == "s" else wl
        w_eff = min(w, ctx) / 2
        kv_bytes += 2 * d * 2 * w_eff   # K et V, bf16
    per_tok = weights_bytes + kv_bytes
    print("\n=== Inférence (decode, batch=1, memory-bound) ===")
    print(f"  Poids relus/token      : {weights_bytes/1e6:.0f} Mo")
    print(f"  KV cache relu/token    : {kv_bytes/1e6:.2f} Mo (ctx={ctx})")
    print(f"  Total                  : {per_tok/1e6:.0f} Mo/token")
    print(f"  Max théorique          : {bw/per_tok:,.0f} tok/s")
    for eff in (0.4, 0.5, 0.6, 0.7):
        print(f"    efficacité BW {eff:.0%} -> {eff*bw/per_tok:6.0f} tok/s")
    # Prefill : compute-bound, 2N par token
    peak = spec["bf16"] * 1e12
    flop_tok = 2 * (counts["body"] + counts["lm_head"])
    print(f"\n  Prefill (compute-bound) : {flop_tok/1e6:.0f} MFLOP/token")
    for mfu in (0.3, 0.5):
        print(f"    MFU {mfu:.0%} -> {mfu*peak/flop_tok:7,.0f} tok/s")
    print("  (le batching multiplie le débit de decode jusqu'à saturation du compute)")

# -----------------------------------------------------------------------------
# 3. Mesure réelle forward+backward
# -----------------------------------------------------------------------------
def make_batch(num_tokens, doc_len, ws_short, ws_long, seq):
    """Batch synthétique calqué sur DistributedDataGenerator._produce_batch."""
    assert num_tokens % doc_len == 0
    n_docs = num_tokens // doc_len
    g = torch.Generator().manual_seed(1234)
    inputs = torch.randint(0, 50257, (num_tokens,), generator=g, dtype=torch.int32)
    inputs[::doc_len] = BOS_ID
    targets = torch.randint(0, 50257, (num_tokens,), generator=g, dtype=torch.int64)
    max_docs = 128
    cum = torch.full((max_docs,), num_tokens, dtype=torch.int32)
    cum[0] = 0
    for i in range(1, n_docs + 1):
        cum[i] = i * doc_len
    bigram = get_bigram_hash(inputs.clone())
    cfg = ForwardScheduleConfig(
        mtp_weights=torch.tensor([1.0], device="cuda"),
        ws_short=ws_short * BLOCK, ws_long=ws_long * BLOCK,
        train_max_seq_len=seq,
    )
    return (inputs.cuda(), targets.cuda(), cum.cuda(), bigram.cuda(), cfg)

def measure_train_step(model, counts, spec, use_compile):
    stage = STAGES[2]  # stage 3 : le plus lourd, micro-batch 6144 tokens
    micro_tokens = stage["tokens"] // grad_accum_steps
    print(f"\n=== Mesure fwd+bwd ({'compile' if use_compile else 'eager'}) ===")
    print(f"  micro-batch : {micro_tokens} tokens (= batch {stage['tokens']:,} / accum {grad_accum_steps})")
    batch = make_batch(micro_tokens, min(2048, micro_tokens), stage["ws_short"], stage["ws_long"], stage["seq"])

    if use_compile:
        print("  torch.compile en cours (plusieurs minutes la 1re fois)...")
        model = torch.compile(model, dynamic=False, fullgraph=True)
    model.train()
    model.quantize_mlp_fp8()

    n_warm, n_iter = 3, 10
    for _ in range(n_warm):
        loss = model(*batch).sum() * grad_scale
        loss.backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        loss = model(*batch).sum() * grad_scale
        loss.backward()
        model.zero_grad(set_to_none=True)
    end.record()
    torch.cuda.synchronize()

    ms = start.elapsed_time(end) / n_iter
    tok_s = micro_tokens / (ms / 1e3)
    fpt, *_ = flops_per_token(counts, stage, stage["seq"])
    tflops = 3 * fpt * tok_s / 1e12
    mfu = tflops / spec["bf16"]
    step_s = ms * grad_accum_steps / 1e3
    print(f"  temps/micro-step : {ms:8.1f} ms")
    print(f"  débit            : {tok_s:,.0f} tok/s")
    print(f"  FLOPs effectifs  : {tflops:.1f} TFLOP/s  ->  MFU = {mfu:.1%} (pic bf16 {spec['bf16']:.0f} TFLOPS)")
    print(f"  => step complet ({grad_accum_steps} micro-steps, hors optimizer) : ~{step_s:.2f} s")
    print(f"  => run complet (1390 steps) : ~{step_s*1390/3600:.1f} h  (+ optimizer/evals)")
    print(f"  mémoire pic : {torch.cuda.max_memory_allocated()/2**20:.0f} Mio")
    return model

def measure_decode(model, counts, spec, n=100):
    print("\n=== Mesure decode batch=1 (eval, no_grad) ===")
    model.eval()
    cfg = ForwardScheduleConfig(
        mtp_weights=torch.tensor([1.0], device="cuda"),
        ws_short=5 * BLOCK, ws_long=11 * BLOCK, train_max_seq_len=2048,
    )
    inputs = torch.tensor([BOS_ID], dtype=torch.int32, device="cuda")
    targets = torch.zeros(1, dtype=torch.int64, device="cuda")
    cum = torch.ones(128, dtype=torch.int32, device="cuda")
    cum[0] = 0
    bigram = get_bigram_hash(inputs.cpu()).cuda()
    try:
        with torch.no_grad():
            for _ in range(10):
                model(inputs, targets, cum, bigram, cfg)
            torch.cuda.synchronize()
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            t0.record()
            for _ in range(n):
                model(inputs, targets, cum, bigram, cfg)
            t1.record()
            torch.cuda.synchronize()
        ms = t0.elapsed_time(t1) / n
        print(f"  {ms:.2f} ms/token -> {1e3/ms:.0f} tok/s (contexte quasi vide, eager)")
        print("  (sans KV cache : chaque forward recalcule tout, mais à T=1 c'est le régime memory-bound)")
    except Exception as e:
        print(f"  mesure impossible ({type(e).__name__}: {e}) — se fier à l'estimation analytique.")

# -----------------------------------------------------------------------------
def main():
    use_compile = "--compile" in sys.argv
    do_decode = "--decode" in sys.argv
    no_measure = "--no-measure" in sys.argv

    name, spec = gpu_specs()
    print(f"GPU : {name}  |  pic dense bf16 {spec['bf16']:.0f} TFLOPS, "
          f"fp8 {spec['fp8'] and f'{spec['fp8']:.0f} TFLOPS'}, BW {spec['bw']/1e9:.0f} Go/s")
    print(f"grad_accum_steps = {grad_accum_steps} (GRAD_ACCUM_STEPS)")

    model = build_model()
    counts = param_report(model)
    analytic_report(counts, spec)
    inference_report(counts, spec)

    if not no_measure:
        model = measure_train_step(model, counts, spec, use_compile)
        if do_decode:
            measure_decode(model, counts, spec)

if __name__ == "__main__":
    main()
