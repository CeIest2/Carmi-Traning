"""
modded_nanogpt.py — implémentation SGLang de l'architecture modded-nanogpt.

Enregistré via SGLANG_EXTERNAL_MODEL_PACKAGE=sglang_ext (EntryClass en bas).
La référence de comportement est v1/generate.py (ModdedNanoGPT) ; ce fichier
doit produire des logits identiques en greedy.

Points d'architecture gérés ici :
  - 11 couches, pas d'attention à la couche 6 (skip gate vers snapshot couche 3) ;
  - têtes appariées (couches 0/2/5/9) : séquence doublée en interne — incompatibles
    avec le KV paginé du moteur, elles utilisent un cache KV dense par requête
    (fenêtre glissante 768 tokens doublés -> 768 lignes conservées) et SDPA ;
  - couches normales (1/3/4/7/8/10) : RadixAttention avec fenêtre glissante par
    couche (768/2560) et softmax scale YaRN custom ;
  - état par requête (indexé par req_pool_indices) pour les 3 dépendances au
    token précédent : bigram (id précédent), smear gate (embedding précédent),
    key offset couches 3/10 (moitié haute de la clé brute précédente) ;
  - value embeddings gatés (1/2/8/9), MUDD (couche 10 + post-loop), XSA,
    QK-norm, RoPE half-truncated (YaRN), softcap 23*sigmoid((logits+5)/7.5),
    vocab 50257 paddé à 50304.

Hypothèses (v1 du port) : radix cache désactivé, CUDA graphs désactivés, TP=1,
bf16, pas de mode MIXED (chunked prefill désactivé côté lanceur).
"""

import torch
import torch.nn.functional as F
from torch import nn
from types import SimpleNamespace

from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.radix_attention import RadixAttention

MUDD_DIM = 64  # dimension interne du MLP MUDD (shape des poids exportés)
PAIRED_CACHE_ROWS = 768  # fenêtre glissante des couches appariées, en tokens doublés


class SoftcapLogitsProcessor(LogitsProcessor):
    """LogitsProcessor sglang + softcap custom 23*sigmoid((logits+5)/7.5).

    Hérite de tout le pipeline (pruning des derniers tokens, input logprobs
    pour la validation, troncature au vocab non paddé) ; on ajoute juste le
    softcap après le matmul lm_head.
    """

    def _compute_lm_head(self, hidden_states, lm_head, embedding_bias=None):
        logits = super()._compute_lm_head(hidden_states, lm_head, embedding_bias)
        sc = self.config.softcap
        return sc["scale"] * torch.sigmoid((logits + sc["shift"]) / sc["div"])


# -----------------------------------------------------------------------------
# Helpers mathématiques (répliques de generate.py)
# -----------------------------------------------------------------------------
def rms_norm(x: torch.Tensor, dim: int) -> torch.Tensor:
    return F.rms_norm(x, (dim,))


def rot(x: torch.Tensor, f1: torch.Tensor, f2: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """rot(x) = f1[pos]*x + f2[pos]*flip_paires(x). x: (T, ..., D) ; pos: (T,)."""
    a = f1[pos].view(x.shape[0], *([1] * (x.ndim - 2)), x.shape[-1])
    b = f2[pos].view(x.shape[0], *([1] * (x.ndim - 2)), x.shape[-1])
    xf = x.view(*x.shape[:-1], x.shape[-1] // 2, 2).flip(-1).view(x.shape)
    return a * x + b * xf


def bigram_hash(ids, prev_ids, first_mask, bigram_vocab_size, sign_rows):
    """Index bigram / signe par token. first_mask: True au 1er token de séquence."""
    ids32 = ids.to(torch.int32)
    prev32 = prev_ids.to(torch.int32)
    mod = bigram_vocab_size - 1
    bi = torch.bitwise_xor(36313 * ids32, 27191 * prev32) % mod
    si = torch.bitwise_xor(prev32, ids32) % sign_rows
    bi = torch.where(first_mask, torch.full_like(bi, mod), bi)
    si = torch.where(first_mask, torch.zeros_like(si), si)
    return bi.long(), si.long()


class _Weight(nn.Module):
    """Conteneur donnant un paramètre nommé '<prefix>.weight'."""

    def __init__(self, *shape):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(*shape))


class _Block(nn.Module):
    """Poids d'une couche, noms alignés sur l'export safetensors."""

    def __init__(self, cfg, layer_id: int):
        super().__init__()
        Dm, H = cfg.model_dim, cfg.num_heads
        if layer_id in cfg.attn_layers:
            self.qkvo_w = nn.Parameter(torch.empty(4 * Dm, Dm))
            self.attn_gate = nn.Parameter(torch.empty(H, 12))
        if layer_id in cfg.ve_banks:
            self.ve_gate = nn.Parameter(torch.empty(H, 12))
        self.mlp_fc = nn.Parameter(torch.empty(4 * Dm, Dm))
        self.mlp_proj = nn.Parameter(torch.empty(4 * Dm, Dm))


class ModdedNanoGPTForCausalLM(nn.Module):
    def __init__(self, config, quant_config=None, prefix: str = ""):
        super().__init__()
        cfg = self.config = config
        Dm, H, D, L = cfg.model_dim, cfg.num_heads, cfg.head_dim, cfg.num_layers
        V = cfg.padded_vocab or 50304
        self.padded_vocab = V

        # --- poids (noms identiques à l'export pour load_weights direct) ---
        self.embed = nn.Embedding(V, Dm)
        self.lm_head = _Weight(Dm, V)  # stockage transposé (768, 50304)
        self.value_embeds = nn.Parameter(torch.empty(5, V, Dm))
        self.bigram_embed = nn.Embedding(cfg.bigram_vocab_size, cfg.bigram_dim)
        self.register_buffer("bigram_sign_table", torch.empty(cfg.sign_rows, cfg.bigram_dim))
        self.smear_gate = nn.Linear(12, 1, bias=False)
        self.skip_gate = nn.Linear(12, 1, bias=False)
        self.sa_lambdas = nn.Parameter(torch.empty(L, 2))
        self.smear_lambda = nn.Parameter(torch.empty(1))
        self.skip_lambda = nn.Parameter(torch.empty(1))
        self.post_lambdas = nn.Parameter(torch.empty(L, 2))
        self.resid_lambdas = nn.Parameter(torch.empty(L, 2))
        self.x0_lambdas = nn.Parameter(torch.empty(L))
        self.bigram_lambdas = nn.Parameter(torch.empty(L))
        self.xsa_alphas = nn.Parameter(torch.empty(L, H))
        self.mudd_w1 = nn.Parameter(torch.empty(2, MUDD_DIM, Dm))
        self.mudd_w2 = nn.Parameter(torch.empty(2, 14, MUDD_DIM))
        self.mudd_b2 = nn.Parameter(torch.empty(2, 14))
        self.blocks = nn.ModuleList([_Block(cfg, i) for i in range(L)])
        self.register_buffer("yarn_angular_freq", torch.empty(D))
        self.register_buffer("yarn_attn_scale", torch.empty(1))

        # pipeline logits du moteur (pruning, input logprobs, slice vocab) + softcap
        self.logits_processor = SoftcapLogitsProcessor(config)

        # --- attention gérée par le moteur (couches non appariées) ---
        self.attn = nn.ModuleDict()
        for j, i in enumerate(cfg.non_paired_attn_layers):
            self.attn[str(i)] = RadixAttention(
                num_heads=H,
                head_dim=D,
                scaling=cfg.yarn_attn_scale,
                num_kv_heads=H,
                layer_id=j,
                sliding_window_size=cfg.window_of(i),
            )

        self._state = None  # buffers par requête, alloués au premier forward

    # ------------------------------------------------------------------
    # Chargement des poids (convention sglang DefaultModelLoader)
    # ------------------------------------------------------------------
    def load_weights(self, weights):
        dst = {**dict(self.named_parameters()), **dict(self.named_buffers())}
        seen = set()
        for name, w in weights:
            if name not in dst:
                raise KeyError(f"poids inattendu dans l'export: {name}")
            t = dst[name]
            if t.shape != w.shape:
                raise ValueError(f"shape mismatch {name}: {tuple(t.shape)} vs {tuple(w.shape)}")
            with torch.no_grad():
                t.copy_(w.to(dtype=t.dtype))
            seen.add(name)
        missing = {n for n in dst if not n.startswith("attn.")} - seen
        if missing:
            raise KeyError(f"poids manquants dans l'export: {sorted(missing)}")
        # DefaultModelLoader n'appelle pas post_load_weights() : on le fait ici
        self.post_load_weights()

    def post_load_weights(self):
        """Précalculs : poids effectifs (sa_lambdas), tables YaRN, scale."""
        cfg = self.config
        dev = self.embed.weight.device
        Dm = cfg.model_dim

        self.wqkv, self.wo = {}, {}
        for i in cfg.attn_layers:
            W = self.blocks[i].qkvo_w.float()
            sa = self.sa_lambdas[i].float()
            self.wqkv[i] = (sa[0] * W[: 3 * Dm]).to(torch.bfloat16).contiguous()
            self.wo[i] = (sa[1] * W[3 * Dm:]).to(torch.bfloat16).contiguous()

        # Tables RoPE/YaRN (réplique de generate.py:112-126)
        af = self.yarn_angular_freq.float().to(dev)
        t = torch.arange(cfg.max_positions, dtype=torch.float32, device=dev)
        theta = torch.outer(t, af)
        f1, f2 = theta.cos(), theta.sin()
        f2[..., 1::2] *= -1
        th1 = torch.outer(2 * t, af)
        th2 = torch.outer(2 * t + 1, af)
        pf1 = torch.cat([th1.cos(), th2.cos()], dim=-1)
        pf2 = torch.cat([th1.sin(), th2.sin()], dim=-1)
        pf2[..., 1::2] *= -1
        self.f1, self.f2 = f1.bfloat16(), f2.bfloat16()
        self.pf1, self.pf2 = pf1.bfloat16(), pf2.bfloat16()
        self.attn_scale = float(self.yarn_attn_scale.item())

        # Vue (vocab, hidden) du lm_head pour LogitsProcessor : weight.T redonne
        # exactement le stockage (768, 50304) -> même matmul que generate.py.
        self._lm_head_obj = SimpleNamespace(weight=self.lm_head.weight.T, quant_method=None)

    # ------------------------------------------------------------------
    # État par requête (dépendances au token précédent + KV apparié)
    # ------------------------------------------------------------------
    def _ensure_state(self, n_req: int, device):
        st = self._state
        if st is not None and st["prev_tok"].shape[0] >= n_req:
            return st
        cfg = self.config
        H, D, Dm = cfg.num_heads, cfg.head_dim, cfg.model_dim
        new = max(n_req, 32)
        nst = {
            "prev_tok": torch.zeros(new, dtype=torch.long, device=device),
            "prev_embed": torch.zeros(new, Dm, dtype=torch.bfloat16, device=device),
            # key offset (couches 3 et 10) : moitié haute de la clé brute précédente
            "prev_k_up": torch.zeros(new, 2, H, D // 2, dtype=torch.bfloat16, device=device),
            # KV dense des couches appariées (fenêtre 768 tokens doublés)
            "kbuf": {
                i: torch.zeros(new, PAIRED_CACHE_ROWS, H // 2, D, dtype=torch.bfloat16, device=device)
                for i in cfg.paired_layers
            },
            "vbuf": {
                i: torch.zeros(new, PAIRED_CACHE_ROWS, H // 2, D, dtype=torch.bfloat16, device=device)
                for i in cfg.paired_layers
            },
        }
        if st is not None:  # croissance : recopie
            old = st["prev_tok"].shape[0]
            nst["prev_tok"][:old] = st["prev_tok"]
            nst["prev_embed"][:old] = st["prev_embed"]
            nst["prev_k_up"][:old] = st["prev_k_up"]
            for i in nst["kbuf"]:
                nst["kbuf"][i][:old] = st["kbuf"][i]
                nst["vbuf"][i][:old] = st["vbuf"][i]
        self._state = nst
        return nst

    def _mudd(self, x, idx: int, n: int):
        h = F.gelu(F.linear(x, self.mudd_w1[idx]))
        c = (F.linear(h, self.mudd_w2[idx, :n]) + self.mudd_b2[idx, :n]) * self.config.mudd_scale
        return list(c.split(1, dim=-1))

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward(self, input_ids, positions, forward_batch, **kwargs):
        cfg = self.config
        fb = forward_batch
        device = input_ids.device
        H, D, Dm, L, BD = cfg.num_heads, cfg.head_dim, cfg.model_dim, cfg.num_layers, cfg.bigram_dim
        T = input_ids.shape[0]

        # ---- informations de batch (extend ou decode ; pas de mode MIXED) ----
        if fb.forward_mode.is_decode():
            req = fb.req_pool_indices
            n_per = torch.ones_like(req)
            prefix = fb.seq_lens - 1
            start = torch.arange(req.shape[0], device=device)
            fresh = torch.zeros_like(req, dtype=torch.bool)
        else:
            assert fb.forward_mode.is_extend() and not fb.forward_mode.is_mixed(), (
                "mode MIXED non supporté (lancer avec chunked_prefill_size=-1)"
            )
            req = fb.req_pool_indices
            n_per = fb.extend_seq_lens
            prefix = fb.extend_prefix_lens
            start = fb.extend_start_loc
            fresh = prefix == 0

        st = self._ensure_state(int(req.max().item()) + 1, device)
        if fresh.any():
            fr = req[fresh]
            st["prev_embed"][fr] = 0
            st["prev_tok"][fr] = 0
            st["prev_k_up"][fr] = 0

        req_of_tok = torch.repeat_interleave(req, n_per)
        prefix_of_tok = torch.repeat_interleave(prefix, n_per)
        chunk_start = positions == prefix_of_tok
        first_tok = positions == 0
        last_idx = start + n_per - 1  # index plat du dernier token de chaque requête

        # ---- embeddings + smear gate (mélange avec l'embedding précédent) ----
        ids = input_ids
        e = self.embed(ids)  # (T, Dm) brut
        prev_e = torch.empty_like(e)
        prev_e[1:] = e[:-1]
        prev_e[chunk_start] = st["prev_embed"][req_of_tok[chunk_start]]
        smear = self.smear_lambda.to(e.dtype) * torch.sigmoid(
            F.linear(e[:, :12], self.smear_gate.weight)
        )
        x = e + smear * prev_e
        st["prev_embed"][req] = e[last_idx]

        # ---- bigram hash embeddings ----
        prev_ids = torch.empty_like(ids)
        prev_ids[1:] = ids[:-1]
        prev_ids[chunk_start] = st["prev_tok"][req_of_tok[chunk_start]]
        bi, si = bigram_hash(ids, prev_ids, first_tok, cfg.bigram_vocab_size, cfg.sign_rows)
        x0b = self.bigram_embed(bi) * self.bigram_sign_table[si]  # (T, BD)
        st["prev_tok"][req] = ids[last_idx]

        x = rms_norm(x, Dm)
        x0 = x
        x = x.clone()
        x[:, :BD] += x0b * self.bigram_lambdas[0].to(x.dtype)

        skip_gate = torch.sigmoid(self.skip_lambda.to(x.dtype)) * 2 * torch.sigmoid(
            F.linear(x0[:, :12], self.skip_gate.weight)
        )  # (T, 1)

        ve_all = self.value_embeds[:, ids]  # (5, T, Dm)

        resid_att = self.resid_lambdas[:, 0].to(x.dtype)
        resid_mlp = self.resid_lambdas[:, 1].to(x.dtype)
        post_att = self.post_lambdas[:, 0].to(x.dtype)
        post_mlp = self.post_lambdas[:, 1].to(x.dtype)
        x0_l = self.x0_lambdas.to(x.dtype)
        bg_l = self.bigram_lambdas.to(x.dtype)

        # ---- boucle des couches ----
        snap = {0: x}
        for i in range(L):
            mu = None
            if i == 6:
                x = x + skip_gate * snap[3]
            else:
                attn_in = rms_norm(snap[7] if i > 7 else x, Dm)
                paired = i in cfg.paired_layers

                # ---- aux_v : value embeddings gatés ou MUDD (couche 10) ----
                if i == L - 1:
                    snap[9] = x
                    mu = self._mudd(x, 0, 14)
                    v_mudd = mu[0] * snap[0] + mu[1] * snap[7] + mu[2] * x  # (T, Dm)
                    x = (1 + mu[5]) * x + mu[3] * snap[0] + mu[4] * snap[7]
                    veg = torch.cat([mu[6], mu[7]], dim=-1).repeat_interleave(
                        H // 2, dim=-1
                    ).unsqueeze(-1)  # (T, H, 1)
                    aux_v = veg * ve_all[cfg.ve_banks[i]].view(T, H, D) + v_mudd.view(T, H, D)
                elif i in cfg.ve_banks:
                    gate_in = torch.cat([attn_in[:, :6], ve_all[cfg.ve_banks[i]][:, :6]], dim=-1)
                    g = 2 * torch.sigmoid(F.linear(gate_in, self.blocks[i].ve_gate))
                    aux_v = g.view(T, H, 1) * ve_all[cfg.ve_banks[i]].view(T, H, D)
                else:
                    aux_v = None

                qkv = F.linear(attn_in, self.wqkv[i]).view(T, 3 * H, D)
                q, k, v = qkv.chunk(3, dim=1)  # (T, H, D)
                q, k = rms_norm(q, D), rms_norm(k, D)

                if paired:
                    # ---- têtes appariées : séquence doublée (2T), cache dense ----
                    q2 = rot(q.view(T, H // 2, 2 * D), self.pf1, self.pf2, positions)
                    k2 = rot(k.view(T, H // 2, 2 * D), self.pf1, self.pf2, positions)
                    q2 = q2.view(2 * T, H // 2, D)
                    k2 = k2.view(2 * T, H // 2, D)
                    if aux_v is not None:
                        v = v + aux_v
                    v2 = v.reshape(2 * T, H // 2, D)
                    y2 = self._paired_attention(i, q2, k2, v2, req, n_per, prefix, start, st)
                    y = y2.view(T, H, D)
                else:
                    # ---- couches normales : KV paginé du moteur ----
                    q = rot(q, self.f1, self.f2, positions)
                    k = rot(k, self.f1, self.f2, positions)
                    if i in cfg.key_offset_layers:
                        slot = 0 if i == 3 else 1
                        ku = k[..., D // 2:]  # clé brute (pour l'état)
                        prev_up = torch.empty_like(ku)
                        prev_up[1:] = ku[:-1]
                        own = chunk_start & (prefix_of_tok == 0)
                        prev_up[own] = ku[own]  # 1er token : sa propre moitié haute
                        cs = chunk_start & ~own
                        prev_up[cs] = st["prev_k_up"][req_of_tok[cs], slot]
                        st["prev_k_up"][req, slot] = ku[last_idx]
                        k = torch.cat([k[..., : D // 2], prev_up], dim=-1)
                    if aux_v is not None:
                        v = v + aux_v
                    y = self.attn[str(i)](
                        q.reshape(T, -1), k.reshape(T, -1), v.reshape(T, -1), fb
                    ).view(T, H, D)
                    # ---- XSA (couches non appariées) ----
                    vn = F.normalize(v, dim=-1, eps=1e-4)
                    proj = (y * vn).sum(-1, keepdim=True)
                    alpha = torch.tanh(self.xsa_alphas[i]).to(y.dtype).view(1, H, 1)
                    y = y - alpha * proj * vn

                g = torch.sigmoid(F.linear(attn_in[:, :12], self.blocks[i].attn_gate))
                y = y * g.view(T, H, 1)
                attn_out = F.linear(y.reshape(T, Dm), self.wo[i])

                # ---- résiduel attention ----
                if mu is not None:
                    x = mu[8] * x + mu[9] * attn_out + mu[10] * snap[0]
                    x[:, :BD] += mu[11] * x0b
                else:
                    x = resid_att[i] * x + post_att[i] * attn_out + x0_l[i] * x0
                    if i >= 1:
                        x[:, :BD] += bg_l[i] * x0b

            # ---- MLP ReLU² ----
            normed = rms_norm(x, Dm)
            h = F.relu(F.linear(normed, self.blocks[i].mlp_fc))
            mlp_out = (h * h) @ self.blocks[i].mlp_proj  # proj sans transpose
            if mu is not None:
                x = mu[12] * x + mu[13] * mlp_out
            else:
                x = resid_mlp[i] * x + post_mlp[i] * mlp_out

            if i in (3, 7):
                snap[i] = x

        # ---- MUDD post-loop ----
        mu2 = self._mudd(x, 1, 5)
        x = (
            x
            + mu2[0] * snap[0]
            + mu2[1] * snap[7]
            + mu2[2] * snap[9]
            + mu2[3] * ve_all[0]
            + mu2[4] * snap[3]
        )

        x = rms_norm(x, Dm)

        # ---- logits via le pipeline du moteur (softcap dans SoftcapLogitsProcessor) ----
        return self.logits_processor(input_ids, x, self._lm_head_obj, forward_batch)

    # ------------------------------------------------------------------
    # Attention des couches appariées (cache KV dense par requête + SDPA)
    # ------------------------------------------------------------------
    def _paired_attention(self, layer, q2, k2, v2, req, n_per, prefix, start, st):
        """q2/k2/v2 : (2T, H/2, D) en espace doublé. Retour : (2T, H/2, D)."""
        cfg = self.config
        device = q2.device
        win = cfg.window_of(layer)
        y2 = torch.empty_like(q2)
        kb, vb = st["kbuf"][layer], st["vbuf"][layer]
        for r in range(req.shape[0]):
            s = int(start[r])
            n = int(n_per[r])
            p = int(prefix[r])
            rq = int(req[r])
            m = min(2 * p, win)  # lignes conservées des chunks précédents
            qs = q2[2 * s: 2 * (s + n)]
            ks_new = k2[2 * s: 2 * (s + n)]
            vs_new = v2[2 * s: 2 * (s + n)]
            K = torch.cat([kb[rq, :m], ks_new], dim=0)
            V = torch.cat([vb[rq, :m], vs_new], dim=0)
            qpos = 2 * p + torch.arange(2 * n, device=device)
            kpos = torch.cat([2 * p - m + torch.arange(m, device=device), qpos])
            allowed = (kpos[None, :] <= qpos[:, None]) & ((qpos[:, None] - kpos[None, :]) <= win)
            out = F.scaled_dot_product_attention(
                qs.transpose(0, 1).unsqueeze(0),
                K.transpose(0, 1).unsqueeze(0),
                V.transpose(0, 1).unsqueeze(0),
                attn_mask=allowed[None, None],
                scale=self.attn_scale,
            )
            y2[2 * s: 2 * (s + n)] = out[0].transpose(0, 1)
            keep = min(K.shape[0], win)
            kb[rq, :keep] = K[-keep:]
            vb[rq, :keep] = V[-keep:]
        return y2


EntryClass = [ModdedNanoGPTForCausalLM]
