import math
import os
from dataclasses import dataclass

from dist_setup import device, grad_accum_steps, grad_scale, world_size
from config import args

# torch._inductor.config.coordinate_descent_tuning = True # we have banned this flag for new records because it causes compilation to take 30min
import torch
import torch._dynamo as dynamo
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

import fp8  # enregistre les custom ops nanogpt::mm_t (utilisés via torch.ops.nanogpt.mm_t)

# --- PATCH (RTX 4060 Ti / Ada Lovelace) ---
# Le repo original fetch un kernel Flash-Attention 3 précompilé (Hopper/SM90 only, via
# `kernels-community/flash-attn3`). FA3 utilise des instructions (TMA, warp specialization,
# wgmma) qui n'existent PAS sur Ada Lovelace (SM89) : ce n'est pas juste un binaire manquant,
# c'est une limitation matérielle. On utilise donc Flash-Attention 2 (package `flash-attn`),
# qui supporte officiellement Ampere/Ada/Hopper et a une API quasi identique
# (mêmes arguments cu_seqlens_q/k, max_seqlen_q/k, causal, softmax_scale, window_size).
from flash_attn.flash_attn_interface import flash_attn_varlen_func

from triton_kernels import FusedLinearReLUSquareFunction, FusedSoftcappedCrossEntropy
# Fused triton kernel: relu(x @ W1.T)^2 @ W2.T
# https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
# PATCH (RTX 4060 Ti / Ada Lovelace) : FusedLinearReLUSquareFunction s'appuie sur des
# TensorDescriptor Triton (TMA), une fonctionnalité matérielle exclusive à Hopper/Blackwell.
# Sous torch.compile(fullgraph=True) ça casse au traçage (et même patché pour tracer,
# l'exécution réelle nécessiterait du vrai matériel TMA absent sur Ada). On bascule donc
# sur l'équivalent mathématique en PyTorch pur : matmul/relu/carré/matmul, entièrement
# différentiable via l'autograd standard (donc backward identique, pas besoin de le
# réécrire à la main) et 100% traçable par Dynamo en fullgraph. torch.compile/Inductor
# fusionne de toute façon les opérations élément-par-élément en un kernel Triton généré
# automatiquement, donc la perte de perf reste limitée.
def ReLUSqrdMLP(x, W1, W2, W1_f8=None, dequant_scale=None, x_f8=None):
    x_flat = x.view(-1, x.shape[-1])
    pre = x_flat @ W1.T
    post = torch.relu(pre)
    post = post * post
    out = post @ W2
    return out.view(x.shape)

dynamo.config.recompile_limit = 64

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the model

def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))


class CastedLinearT(nn.Module):
    """
    Linear layer with transposed weight storage (in_features, out_features) which
    addresses the slow kernel that was used for gradient accumulation. @chrisjmccormick
    """
    def __init__(self, in_features: int, out_features: int, use_fp8=False, x_s=1.0, w_s=1.0, grad_s=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_fp8 = use_fp8
        self.x_s = x_s
        self.w_s = w_s
        self.grad_s = grad_s

        self.weight = nn.Parameter(torch.empty(in_features, out_features, dtype=torch.bfloat16))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            nn.init.zeros_(self.weight) # @Grad62304977 and others

    def forward(self, x: Tensor):
        if self.use_fp8 and self.training:
            _x = x.flatten(0, -2)
            out = torch.ops.nanogpt.mm_t(_x, self.weight, x_s=self.x_s, w_s=self.w_s, grad_s=self.grad_s)[0]
            return out.reshape(*x.shape[:-1], -1)
        else:
            return x @ self.weight.type_as(x)

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the model

class Yarn(nn.Module):
    def __init__(self, head_dim, max_seq_len, paired=False):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.paired = paired
        self.reset()

    def rotary(self, x_BTHD):
        assert self.factor1.size(0) >= x_BTHD.size(-3)
        factor1, factor2 = (
            self.factor1[None, : x_BTHD.size(-3), None, :],
            self.factor2[None, : x_BTHD.size(-3), None, :],
        )
        x_flip = x_BTHD.view(*x_BTHD.shape[:-1], x_BTHD.shape[-1] // 2, 2).flip(-1).view(x_BTHD.shape)
        return factor1 * x_BTHD + factor2 * x_flip

    def reset(self):
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=self.head_dim//4, dtype=torch.float32, device=device)
        angular_freq = angular_freq.repeat_interleave(2)
        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(self.head_dim//2)])
        t = torch.arange(2*self.max_seq_len, dtype=torch.float32, device=device)
        if not self.paired:
            theta = torch.outer(t, angular_freq)
            self.factor1 = nn.Buffer(
                theta.cos().to(torch.bfloat16), persistent=False
            )
            self.factor2 = nn.Buffer(
                theta.sin().to(torch.bfloat16), persistent=False
            )
        else:
            t_even = 2 * t
            t_odd = t_even + 1
            theta1 = torch.outer(t_even, angular_freq)
            theta2 = torch.outer(t_odd, angular_freq)
            self.factor1 = nn.Buffer(
                torch.cat((theta1.cos(), theta2.cos()), dim=-1).to(torch.bfloat16),
                persistent=False
            )
            self.factor2 = nn.Buffer(
                torch.cat((theta1.sin(), theta2.sin()), dim=-1).to(torch.bfloat16),
                persistent=False
            )
        self.factor2[..., 1::2] *= -1
        self.angular_freq = angular_freq
        # start with 0.1, inspired by 0.12 from @leloykun and learnable scalars used by @brendanh0gan https://x.com/hi_tysam/status/1879693583898591283
        self.attn_scale = 0.1

    def apply(self, old_window: int, new_window: int, alpha: int=1, beta: int=32):
        rotations = old_window * self.angular_freq / (2 * torch.pi)
        scaling_factor = old_window / new_window
        interpolation_weight = torch.clamp((rotations - alpha) / (beta - alpha), 0, 1)
        self.angular_freq *= scaling_factor + interpolation_weight * (1 - scaling_factor)
        t = torch.arange(2*self.max_seq_len, dtype=torch.float32, device=self.angular_freq.device)
        if not self.paired:
            theta = torch.outer(t, self.angular_freq)
            self.factor1.copy_(theta.cos())
            self.factor2.copy_(theta.sin())
        else:
            t_even = 2 * t
            t_odd = t_even + 1
            theta1 = torch.outer(t_even, self.angular_freq)
            theta2 = torch.outer(t_odd, self.angular_freq)
            self.factor1.copy_(torch.cat((theta1.cos(), theta2.cos()), dim=-1))
            self.factor2.copy_(torch.cat((theta1.sin(), theta2.sin()), dim=-1))
        self.factor2[..., 1::2] *= -1
        self.attn_scale *= 0.2 * math.log(new_window / old_window) + 1

@dataclass(slots=True)
class AttnArgs:
    sa_lambdas: torch.Tensor
    seqlens: torch.Tensor
    bm_size: int
    yarn: Yarn
    key_offset: bool
    attn_gate_w: torch.Tensor
    aux_v: torch.Tensor | None
    xsa_alpha: torch.Tensor | None
    train_max_seq_len: torch.Tensor

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim: int, num_heads: int, paired: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dim = dim
        self.hdim = num_heads * head_dim
        self.paired = paired
        assert self.hdim == self.dim, "num_heads * head_dim must equal model_dim"
        # Weights are stored in parameter banks and passed via forward()

    def forward(self, x: Tensor, attn_args: AttnArgs, qkvo_w: Tensor):
        B, T = x.size(0), x.size(1) # batch size, sequence length
        assert B == 1, "varlen sequences requires B == 1"
        assert T % 16 == 0
        # unpack attention args
        aux_v, attn_gate_w = attn_args.aux_v, attn_args.attn_gate_w
        sa_lambdas, key_offset = attn_args.sa_lambdas, attn_args.key_offset
        seqlens, bm_size = attn_args.seqlens, attn_args.bm_size
        train_max_seq_len, yarn = attn_args.train_max_seq_len, attn_args.yarn

        q, k, v = F.linear(x, sa_lambdas[0] * qkvo_w[:self.dim * 3].type_as(x)).view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
        max_len = train_max_seq_len if self.training else (args.val_batch_size // (grad_accum_steps * world_size))

        q, k = norm(q), norm(k) # QK norm @Grad62304977

        if not self.paired:
            q, k = yarn.rotary(q), yarn.rotary(k)

            if key_offset:
                # shift keys forward for the stationary head dims. Enables 1-layer induction.
                k[:, 1:, :, self.head_dim // 2:] = k[:, :-1, :, self.head_dim // 2:]

            if aux_v is not None:
                v = v + aux_v.view_as(v)

        else:
            # Paired heads: adjacent heads' queries attend to each other's keys.
            # Two copies of the input stream are interleaved to achieve this, which:
            # - doubles the length of each sequence
            # - halves the effective window size
            q = q.view(B, T, self.num_heads // 2, self.head_dim * 2)
            k = k.view(B, T, self.num_heads // 2, self.head_dim * 2)
            v = v.reshape(B, T * 2, self.num_heads // 2, self.head_dim)

            q, k = yarn.rotary(q), yarn.rotary(k)

            q = q.view(B, T * 2, self.num_heads // 2, self.head_dim)
            k = k.view(B, T * 2, self.num_heads // 2, self.head_dim)

            if aux_v is not None:
                v = v + aux_v.view_as(v)

            seqlens = 2 * seqlens
            max_len = 2 * max_len

        # use flash_attn over flex_attn @varunneal. flash_attn_varlen suggested by @YouJiacheng
        # PATCH: flash-attn 2 (Ada-compatible) instead of flash-attn 3 (Hopper-only)
        y = flash_attn_varlen_func(q[0], k[0], v[0], cu_seqlens_q=seqlens, cu_seqlens_k=seqlens,
                                    max_seqlen_q=max_len, max_seqlen_k=max_len,
                                    causal=True, softmax_scale=yarn.attn_scale, window_size=(bm_size, 0))
        y = y.view(B, T, self.num_heads, self.head_dim)
        # Gated XSA (arXiv:2603.09078) with learnable strength: subtract per-head fraction tanh(α)
        # of y aligned with v̂. Non-paired only (v shape doesn't line up for paired layers).
        if attn_args.xsa_alpha is not None and not self.paired:
            vn = F.normalize(v, dim=-1, eps=1e-4)
            proj = (y * vn).sum(-1, keepdim=True)
            alpha = torch.tanh(attn_args.xsa_alpha).type_as(y).view(1, 1, self.num_heads, 1)
            y = y - alpha * proj * vn
        y = y * torch.sigmoid(F.linear(x[..., :12], attn_gate_w)).view(B, T, self.num_heads, 1)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim) # re-assemble all head outputs side by side
        y = F.linear(y, sa_lambdas[1] * qkvo_w[self.dim * 3:].type_as(y))  # sa_lambdas[1] pre-multiplied to O @shenberg
        return y


# -----------------------------------------------------------------------------
# The main model

def next_multiple_of_n(v: float | int, *, n: int):
    return math.ceil(v / n) * n

@dataclass(slots=True)
class ForwardScheduleConfig:
    mtp_weights: torch.Tensor
    ws_short: int
    ws_long: int
    train_max_seq_len: int

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, head_dim: int, model_dim: int, max_seq_len: int):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        # there are only 50257 unique GPT-2 tokens; extend to nearest multiple of 128 for efficiency.
        # suggested by @Grad62304977, originates from Karpathy's experiments.
        self.vocab_size = next_multiple_of_n(vocab_size, n=128)

        # Transposed weight storage for faster gradient accumulation
        use_fp8 = not os.environ.get("DISABLE_FP8", False)
        self.lm_head = CastedLinearT(model_dim, self.vocab_size, use_fp8=use_fp8, x_s=100/448, w_s=1.6/448, grad_s=grad_scale * 0.75/448)
        nn.init.normal_(self.lm_head.weight, mean=0, std=0.005)

        self.embed = nn.Embedding(self.vocab_size, model_dim)
        with torch.no_grad():
            # tie embed and lm_head at init
            self.embed.weight.copy_(self.lm_head.weight.T)

        self.init_attn(model_dim, head_dim, num_heads, num_layers, max_seq_len)
        self.init_mlp(model_dim)
        self.init_misc(model_dim, num_layers)
        self.init_mudd(num_layers, model_dim)

        # Auto-label parameters
        for name, param in self.named_parameters():
            param.label = name.replace('.weight', '')

    def init_attn(self, model_dim, head_dim, num_heads, num_layers, max_seq_len):
        # Cache layers for skip / backout snapshots taken at end of loop iter.
        self.cache_layers = [3, 7]

        # Attention modules (no learned params -- weights come from qk_bank/vo_bank)
        self.paired_head_layers = [0, 2, 5, 9]
        self.attn = CausalSelfAttention(model_dim, head_dim, num_heads, paired=False)
        self.attn_paired = CausalSelfAttention(model_dim, head_dim, num_heads, paired=True)
        self.yarn = Yarn(head_dim, max_seq_len)
        self.yarn_paired_head = Yarn(head_dim, max_seq_len, paired=True)

        # token value embeddings by @KoszarskyB - inspired by @Grad62304977's value residual implementation following https://arxiv.org/abs/2410.17897
        # value embedding code simplification inspired by @ragulpr https://github.com/KellerJordan/modded-nanogpt/pull/78
        # spherical gaussian init by @photomz
        self.value_embeds = nn.Parameter(0.01 * torch.randn(5 * self.vocab_size, model_dim, dtype=torch.bfloat16))

        # parameter banks for attention and value embedding gate weights
        self.attn_gate_bank = nn.Parameter(torch.zeros(10, num_heads, 12)) # 10 layers
        self.ve_gate_bank = nn.Parameter(torch.zeros(5, num_heads, 12)) # 5 unique gates
        self.gate_filler_nones = [None] * (num_layers - 6)

        # Parameter banks for sharded optimization, by @chrisjmccormick
        # Attention is skipped in layer 6 by @YouJiacheng
        num_attn_layers = num_layers - 1
        hdim = num_heads * head_dim

        # QK bank: per-head-pair Muon groups for Q, K weights
        # Each pair of adjacent heads gets its own independent polar express orthogonalization
        self._num_attn_layers = num_attn_layers
        num_qk_groups = num_attn_layers * 2 * (num_heads // 2)  # 10 * 2 * 3 = 60
        self._num_qk_groups = num_qk_groups
        num_qk_padded = next_multiple_of_n(num_qk_groups, n=world_size)  # 64
        self.qk_bank = nn.Parameter(torch.empty(num_qk_padded, head_dim * 2, model_dim))
        self.qk_bank.reshape = (num_qk_padded, head_dim * 2, model_dim)

        # VO bank: per-layer Muon groups for V and O weights
        num_vo_real = num_attn_layers * 2  # 20
        num_vo_padded = next_multiple_of_n(num_vo_real, n=world_size)  # 24
        self.vo_bank = nn.Parameter(torch.empty(num_vo_padded, hdim, hdim))
        self.vo_bank.reshape = (num_vo_padded, hdim, hdim)

        # improved init scale by @YouJiacheng and @srashedll
        std = 0.5 * model_dim ** -0.5
        bound = (3 ** 0.5) * std
        with torch.no_grad():
            self.qk_bank[:num_qk_groups].uniform_(-bound, bound)
            self.qk_bank[num_qk_groups:].zero_()
            self.vo_bank[:num_vo_real].uniform_(-bound, bound)
            self.vo_bank[num_vo_real:].zero_()

    def init_mlp(self, model_dim):        
        # MLP bank: stores c_fc and c_proj for all MLP layers
        # We add 1 padding layer (index 11) to get 12*2=24 matrices for even distribution across 8 GPUs
        mlp_hdim = 4 * model_dim
        self.mlp_bank = nn.Parameter(torch.empty(12, 2, mlp_hdim, model_dim))  # (12, 2, 3072, 768)
        self.mlp_bank.reshape = (24, mlp_hdim, model_dim)  # Shape for sharding: (24, 3072, 768)

        # improved init scale by @YouJiacheng and @srashedll
        std = 0.5 * model_dim ** -0.5
        bound = (3 ** 0.5) * std
        with torch.no_grad():
            self.mlp_bank[:, 0, :, :].uniform_(-bound, bound)  # c_fc
            self.mlp_bank[:, 1, :, :].zero_()  # c_proj - zero init suggested by @Grad62304977

    def init_misc(self, model_dim, num_layers):
        self.smear_gate = nn.Linear(12, 1, bias=False)
        nn.init.zeros_(self.smear_gate.weight)

        self.skip_gate = nn.Linear(12, 1, bias=False)
        nn.init.zeros_(self.skip_gate.weight)

        self.bigram_embed = nn.Embedding(args.bigram_vocab_size, args.bigram_dim)
        nn.init.zeros_(self.bigram_embed.weight)
        bigram_sign_table = torch.randn(args.bigram_sign_table_rows, args.bigram_dim).sign().to(torch.bfloat16)
        self.register_buffer('bigram_sign_table', bigram_sign_table)

        self.post_lambdas = nn.Parameter(torch.ones(num_layers, 2))

        # Per-layer injection coefficients for x0 and bigram
        self.x0_lambdas = nn.Parameter(torch.zeros(num_layers))
        self.bigram_lambdas = nn.Parameter(0.05 * torch.ones(num_layers))

        # Per-sublayer residual scaling: [num_layers, 2] where [:,0]=attn, [:,1]=mlp
        # sqrt(1.1) per sublayer so cumulative per-layer scaling is 1.1
        self.resid_lambdas = nn.Parameter(torch.full((num_layers, 2), 1.1**0.5))

        # Per-(layer, head) learnable XSA gate; zero-init -> tanh(0)=0 disables XSA at step 0
        self.xsa_alphas = nn.Parameter(torch.zeros(num_layers, self.num_heads))

        pad = (-num_layers * 2 - 2) % dist.get_world_size()
        self.scalars = nn.Parameter(
            torch.cat(
                [
                    *[torch.tensor([0.5, 1.0]) for _ in range(num_layers)],  # SA lambdas
                    torch.zeros(1), # smear_lambda
                    -1.5 * torch.ones(1),  # skip_lambda -> σ(-1.5) ≈ 0.18
                    torch.ones(pad),
                ]
            )
        )

    def init_mudd(self, num_layers: int, model_dim: int):
        """
        Multiway Dynamic Dense Connections @lishengping. https://arxiv.org/abs/2502.12170
        Expressive and efficient mechanism for data dependent skip connections.
        Given current activation x, return n skip coefficients computed via ~mlp(x).
        Trimmed for speedrun: invoked at start of last layer and post-loop only.

        Start of last layer produces 14 coefficients:
          mu[0..2]  = v_mudd source coefs  (cache[0], cache[7], x)   -> added into V
          mu[3..5]  = residual source coefs (cache[0], cache[7], x)  -> residual recombination
          mu[6..7]  = per-pair ve_gate (2 channels, tiled to num_heads)
          mu[8..9]  = resid_attn / post_attn lambdas (dynamic)
          mu[10..11]= x0 / bigram injection lambdas (dynamic)
          mu[12..13]= resid_mlp / post_mlp lambdas (dynamic)

        Post-loop produces 5 residual coefs over
          {cache[0], cache[7], cache[9], ve_bank0, cache[3]}.
        """
        num_mudd_layers = 2
        self._mudd_scale = 0.1
        mudd_dim = 64
        max_num_coef = 14

        self.mudd_w1 = nn.Parameter(torch.empty(num_mudd_layers, mudd_dim, model_dim))
        for j in range(num_mudd_layers):
            nn.init.kaiming_uniform_(self.mudd_w1.data[j], a=math.sqrt(5))

        self.mudd_w2 = nn.Parameter(torch.zeros(num_mudd_layers, max_num_coef, mudd_dim))

        # Bias init in pre-scaled domain (effective = bias * _mudd_scale).
        bs_init = torch.zeros(num_mudd_layers, max_num_coef)
        # Per-pair ve_gate baseline (matches max of `2*sigmoid` used at other layers):
        bs_init[0, 6]  = 2.0 / self._mudd_scale       # ve_gate lane 0
        bs_init[0, 7]  = 2.0 / self._mudd_scale       # ve_gate lane 1
        # Layer-0 layer-10 dynamic lambdas (effective values match per-layer defaults):
        bs_init[0, 8]  = 1.1**0.5 / self._mudd_scale  # resid_attn[10]
        bs_init[0, 9]  = 1.0 / self._mudd_scale       # post_attn[10]
        bs_init[0, 10] = 0.0                          # x0_lambda[10] (init 0)
        bs_init[0, 11] = 0.05 / self._mudd_scale      # bigram_lambda[10]
        bs_init[0, 12] = 1.1**0.5 / self._mudd_scale  # resid_mlp[10]
        bs_init[0, 13] = 1.0 / self._mudd_scale       # post_mlp[10]
        # Layer-1 (post-loop): -0.5 backout absorbed into residual h7 coef.
        bs_init[1, 1]  = -0.5 / self._mudd_scale      # post-loop residual h7 coef
        self.mudd_b2 = nn.Parameter(bs_init)

    def forward_mudd(self, x, id, num_coef):
        """Returns `num_coef` per-token MUDD coefficients from block `id` (0 or 1)."""
        x = F.gelu(F.linear(x, self.mudd_w1[id]))
        x = (F.linear(x, self.mudd_w2[id, :num_coef]) + self.mudd_b2[id, :num_coef]) * self._mudd_scale
        return x.split(1, dim=-1)

    def quantize_mlp_fp8(self):
        """Refresh the FP8 copy of the MLP up-projection weights after optimizer steps."""
        E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max
        with torch.no_grad():
            if not hasattr(self, "_mlp_up_proj_f8"):
                self._mlp_up_proj_f8 = torch.zeros_like(self.mlp_bank[:, 0], dtype=torch.float8_e4m3fn)
                self._mlp_up_proj_scales = torch.ones(12, dtype=torch.float32, device=self.mlp_bank.device)
                self._mlp_dequant_scale_buf = torch.ones(1, dtype=torch.float32, device=self.mlp_bank.device)
            flat = self.mlp_bank[:, 0].view(12, -1)
            scales = flat.abs().amax(dim=1).clamp(min=1e-12) / E4M3_MAX
            self._mlp_up_proj_scales[:] = scales.float()
            self._mlp_up_proj_f8[:] = (self.mlp_bank[:, 0] / scales.view(12, 1, 1)).to(torch.float8_e4m3fn)

    def forward(self, input_seq: Tensor, target_seq: Tensor, seqlens: Tensor, bigram_input_seq: Tensor, schedule_cfg: ForwardScheduleConfig):
        assert input_seq.ndim == 1

        # ---- Schedule and layer topology ----
        mtp_weights, train_max_seq_len = schedule_cfg.mtp_weights, schedule_cfg.train_max_seq_len
        ws_short, ws_long = schedule_cfg.ws_short, schedule_cfg.ws_long
        # set block masks and key shift
        bm_sizes = [ws_short, ws_short, ws_short, ws_long, ws_short, ws_short, None, ws_short, ws_short, ws_short, ws_long]
        assert len(bm_sizes) == self.num_layers
        key_offset = [b==ws_long for b in bm_sizes] # apply partial key offset to long windows

        use_mlp_fp8 = self.training and not os.environ.get("DISABLE_FP8", False)
        if use_mlp_fp8:
            mlp_up_proj_f8 = self._mlp_up_proj_f8.unbind(0)
            mlp_up_proj_scales = [self._mlp_up_proj_scales[i:i+1] for i in range(12)]

        # ---- Unbind parameters (avoid select_backward kernels) ----
        sa_lambdas = self.scalars[: 2 * self.num_layers].view(-1, 2)
        smear_lambda = self.scalars[2 * self.num_layers]
        skip_lambda = self.scalars[2 * self.num_layers + 1]
        resid_lambdas_attn = self.resid_lambdas[:, 0].bfloat16().unbind(0)
        resid_lambdas_mlp  = self.resid_lambdas[:, 1].bfloat16().unbind(0)
        post_lambdas_attn = self.post_lambdas[:, 0].bfloat16().unbind(0)
        post_lambdas_mlp  = self.post_lambdas[:, 1].bfloat16().unbind(0)
        x0_lambdas = self.x0_lambdas.bfloat16().unbind(0)
        bigram_lambdas = self.bigram_lambdas.bfloat16().unbind(0)
        ag = self.attn_gate_bank.unbind(0)
        veg = self.ve_gate_bank.unbind(0)
        attn_gates = [*ag[:6], None, *ag[6:]]
        ve_gates = [None, veg[0], veg[1], *self.gate_filler_nones, veg[2], veg[3], veg[4]]
        # XSA on non-paired attn layers only; paired {0,2,5,9} and MLP-only layer 6 skipped
        xsa_alpha_per_layer = self.xsa_alphas.unbind(0)
        xsa_alphas = [xsa_alpha_per_layer[j] if j in {1, 3, 4, 7, 8, 10} else None for j in range(self.num_layers)]
        assert len(attn_gates) == self.num_layers
        assert len(ve_gates) == self.num_layers
        qk_all = self.qk_bank[:self._num_qk_groups].view(self._num_attn_layers, -1, self.qk_bank.shape[-1])
        vo_flat = self.vo_bank[:self._num_attn_layers * 2].view(self._num_attn_layers, 2, *self.vo_bank.shape[1:]).flatten(1, 2)
        attn_weights = torch.cat([qk_all, vo_flat], dim=1).unbind(0)
        mlp_all = self.mlp_bank.flatten(0, 1).unbind(0)  # 24 tensors of [mlp_hdim, dim]
        mlp_fcs = mlp_all[0::2]    # even indices: c_fc
        mlp_projs = mlp_all[1::2]  # odd indices: c_proj

        # ---- Embeddings and input preparation ----
        x = self.embed(input_seq) # embed is synced from lm_head during tied phase by optimizer
        
        # Use sign-trick to better compress multiple bigrams into a shared bigram embedding row
        # (details in https://github.com/KellerJordan/modded-nanogpt/pull/299 by @trianxy)
        sign_idx = torch.zeros_like(input_seq)
        sign_idx[1:] = (input_seq[:-1] ^ input_seq[1:]) % self.bigram_sign_table.shape[0]  # (8192,)
        bigram_signs = self.bigram_sign_table[sign_idx]                                    # (seq, bigram_dim)
        x0_bigram = (self.bigram_embed(bigram_input_seq) * bigram_signs)[None]             # (1, seq, bigram_dim)

        # Value embeddings - always computed (not precomputed)
        ve = self.value_embeds.view(5, self.vocab_size, -1)[:, input_seq]
        # Shifted .01 ... 234 structure on token value embeddings by @photomz
        ve = [None, ve[0], ve[1], *self.gate_filler_nones, ve[2], ve[3], ve[4]]
        assert len(ve) == self.num_layers

        # smear token embed forward 1 position @classiclarryd
        smear_gate_out = smear_lambda * torch.sigmoid(self.smear_gate(x[1:, :self.smear_gate.weight.size(-1)]))
        x = torch.cat([x[:1], x[1:] + smear_gate_out * x[:-1]])
        x = x0 = norm(x[None])

        # Initialize residual stream with pre-layer-0 bigram injection
        x[..., :args.bigram_dim] = x[..., :args.bigram_dim] + x0_bigram * bigram_lambdas[0]

        # Precompute x0/bigram injection (added to attention output each layer)
        # Layer 0: bigram already injected above, so only x0 component
        x0_inject = tuple(x0 * x0_lambdas[i] for i in range(self.num_layers))
        bg_inject = (None,) + tuple(x0_bigram * bigram_lambdas[i] for i in range(1, self.num_layers))
        skip_gate_out = torch.sigmoid(skip_lambda) * 2 * torch.sigmoid(self.skip_gate(x0[..., :self.skip_gate.weight.size(-1)]))

        # cache[k] is the layer-k snapshot used downstream by MUDD.
        # cache[0] = residual stream after bigram injection (input to layer 0).
        cache = {0: x}
        for i in range(self.num_layers):
            is_paired = i in self.paired_head_layers
            yarn = self.yarn_paired_head if is_paired else self.yarn
            attn = self.attn_paired if is_paired else self.attn
            c_fc = mlp_fcs[i]
            c_proj = mlp_projs[i]
            if use_mlp_fp8:
                up_proj_f8, up_proj_scale = mlp_up_proj_f8[i], mlp_up_proj_scales[i]
            mu = None

            # process attn. skip on layer 6 @YouJiacheng
            if i == 6:
                x = x + skip_gate_out * cache[3]
            else:
                qkvo_w = attn_weights[i - (i > 6)]
                attn_in_normed = norm(cache.get(7, x))
                B, T = attn_in_normed.size(0), attn_in_normed.size(1)

                if i == self.num_layers - 1:
                    cache[9] = x
                    mu = self.forward_mudd(x, id=0, num_coef=14)
                    v_mudd = (mu[0] * cache[0] + mu[1] * cache[7] + mu[2] * x).view(B, T, self.num_heads, self.head_dim)
                    x = (1 + mu[5]) * x + mu[3] * cache[0] + mu[4] * cache[7]
                    ve_gate = torch.cat([mu[6], mu[7]], dim=-1).repeat_interleave(
                        self.num_heads // 2, dim=-1
                    ).unsqueeze(-1)
                    ve_view = ve[i].view(B, T, self.num_heads, self.head_dim)
                    aux_v = (ve_gate * ve_view + v_mudd).view(B, T, -1)
                elif ve[i] is not None:
                    # gate pattern g(x[:6] + ve[:6]) by @photomz
                    gate_in = torch.cat([attn_in_normed[..., :6], ve[i][None, ..., :6]], dim=-1)
                    ve_gate_out = 2 * torch.sigmoid(F.linear(gate_in, ve_gates[i])).view(B, T, self.num_heads, 1)
                    ve_view = ve[i].view(B, T, self.num_heads, self.head_dim)
                    aux_v = (ve_gate_out * ve_view).view(B, T, -1)
                else:
                    aux_v = None

                attn_args = AttnArgs(
                    sa_lambdas=sa_lambdas[i],
                    seqlens=seqlens,
                    bm_size=bm_sizes[i],
                    yarn=yarn,
                    key_offset=key_offset[i],
                    attn_gate_w=attn_gates[i],
                    aux_v=aux_v,
                    xsa_alpha=xsa_alphas[i],
                    train_max_seq_len=train_max_seq_len,
                )
                attn_out = attn(attn_in_normed, attn_args, qkvo_w)

                if mu is not None:
                    x = mu[8] * x + mu[9] * attn_out + mu[10] * cache[0] 
                    x[..., :args.bigram_dim] = x[..., :args.bigram_dim] + mu[11] * x0_bigram
                else:
                    x = resid_lambdas_attn[i] * x + post_lambdas_attn[i] * attn_out + x0_inject[i]
                    if bg_inject[i] is not None:
                        x[..., :args.bigram_dim] = x[..., :args.bigram_dim] + bg_inject[i]

            # process mlp
            normed = norm(x)
            if use_mlp_fp8:
                amax = normed.detach().abs().max().clamp(min=1e-12)
                x_f8 = (normed.detach() * (448.0 / amax)).to(torch.float8_e4m3fn)
                self._mlp_dequant_scale_buf.copy_(up_proj_scale).mul_(amax).div_(448.0)
                mlp_args = (c_fc, c_proj, up_proj_f8, self._mlp_dequant_scale_buf, x_f8)
            else:
                mlp_args = (c_fc, c_proj)

            if mu is not None:
                x = mu[12] * x + mu[13] * ReLUSqrdMLP(normed, *mlp_args)
            else:
                x = resid_lambdas_mlp[i] * x + post_lambdas_mlp[i] * ReLUSqrdMLP(normed, *mlp_args)

            if i in self.cache_layers:
                cache[i] = x

        # Post-loop MUDD: 5 residual coefs over {cache[0], cache[7], cache[9], ve_bank0, cache[3]}.
        mu = self.forward_mudd(x, id=1, num_coef=5)
        ve_bank0 = ve[1][None].to(dtype=x.dtype)  # (1, T, D), same VE as layer-1 attn
        x = x + mu[0] * cache[0] + mu[1] * cache[7] + mu[2] * cache[9] + mu[3] * ve_bank0 + mu[4] * cache[3]

        x = norm(x)
        # @Grad62304977 added tanh softcapping following Gemma 2 paper, @KoszarskyB reduced it from 30 to 15
        # @YouJiacheng shifted it by +15 (2*sigmoid(2*x)=tanh(x)+1). @classiclarryd updated to 23*sigmoid((logits+5)/7.5)
        if self.training:
            loss_per_token = FusedSoftcappedCrossEntropy.apply(x.view(-1, x.size(-1)), target_seq, mtp_weights, self.lm_head.weight, self.lm_head.x_s, self.lm_head.w_s, self.lm_head.grad_s, grad_scale)
        else:
            logits = self.lm_head(x)
            logits = 23 * torch.sigmoid((logits + 5) / 7.5)
            logits_for_loss = logits.float()
            loss_per_token = F.cross_entropy(logits_for_loss.view(-1, logits_for_loss.size(-1)), target_seq, reduction="none")
        return loss_per_token
