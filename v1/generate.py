#!/usr/bin/env python3
"""
generate.py — inférence pour un modèle exporté par export_model.py.

Réimplémentation eager (bf16) du forward de modded-nanogpt en mode génération :
  - KV cache par couche avec fenêtres glissantes (courtes/longues par couche) ;
  - têtes appariées (couches 0/2/5/9) : séquence doublée en interne, cache 2T ;
  - value embeddings + gates, bigram hash embeddings, smear gate, skip gate,
    MUDD (couche 10 + post-loop), XSA, QK-norm, RoPE half-truncated (YaRN),
    key offset (couches 3/10), softcap 23*sigmoid((logits+5)/7.5) ;
  - flash-attn 2 si disponible, sinon fallback SDPA (masques explicites).

Usage :
  python generate.py --model exports/mon_modele --prompt "Once upon a time" \
      --max-new-tokens 200 --temperature 0.8 --top-p 0.9
  python generate.py --model exports/mon_modele -i        # mode interactif
"""

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
    HAS_FLASH = True
except Exception:
    HAS_FLASH = False


# -----------------------------------------------------------------------------
# Tokenizer GPT-2 (tiktoken en priorité, sinon transformers)
# -----------------------------------------------------------------------------
def load_tokenizer():
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        return ("tiktoken", enc)
    except ImportError:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
        return ("hf", tok)


def tok_encode(kind, tok, text):
    if kind == "tiktoken":
        return tok.encode(text)  # pas de tokens spéciaux injectés
    return tok.encode(text, add_special_tokens=False)


def tok_decode(kind, tok, ids):
    if kind == "tiktoken":
        return tok.decode(ids)
    return tok.decode(ids, skip_special_tokens=True)


# -----------------------------------------------------------------------------
# Attention causale à fenêtre glissante, alignée à droite (préfill ou decode)
# q: (1, Tq, H, D)  k/v: (1, Sk, H, D)
# -----------------------------------------------------------------------------
def attend(q, k, v, window: int, softmax_scale: float):
    Tq, Sk = q.shape[1], k.shape[1]
    if HAS_FLASH:
        cu_q = torch.tensor([0, Tq], dtype=torch.int32, device=q.device)
        cu_k = torch.tensor([0, Sk], dtype=torch.int32, device=q.device)
        y = flash_attn_varlen_func(
            q[0], k[0], v[0], cu_q, cu_k, Tq, Sk,
            causal=True, softmax_scale=softmax_scale, window_size=(window, 0),
        )
        return y.unsqueeze(0)
    # Fallback SDPA : masque booléen (clé j visible par la requête i ssi
    # 0 <= pos_i - pos_j <= window, positions absolues alignées à droite)
    qpos = torch.arange(Sk - Tq, Sk, device=q.device)
    kpos = torch.arange(Sk, device=q.device)
    allowed = (kpos[None, :] <= qpos[:, None]) & ((qpos[:, None] - kpos[None, :]) <= window)
    y = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        attn_mask=allowed[None, None], scale=softmax_scale,
    )
    return y.transpose(1, 2)


# -----------------------------------------------------------------------------
# Le modèle
# -----------------------------------------------------------------------------
class ModdedNanoGPT:
    def __init__(self, model_dir: str, device: str = "cuda", max_ctx: int | None = None):
        with open(os.path.join(model_dir, "config.json")) as f:
            self.cfg = json.load(f)
        cfg = self.cfg
        self.device = device
        sd = load_file(os.path.join(model_dir, "model.safetensors"), device=device)
        self.sd = sd
        self.max_ctx = max_ctx or cfg["max_positions"]

        H, Dm = cfg["num_heads"], cfg["model_dim"]
        # Poids effectifs QKV/O avec sa_lambdas pré-multipliés (fait une fois,
        # en fp32 -> bf16, au lieu de le refaire à chaque forward)
        self.wqkv, self.wo = {}, {}
        for i in cfg["attn_layers"] if "attn_layers" in cfg else self._attn_layers():
            W = sd[f"blocks.{i}.qkvo_w"].float()
            sa = sd["sa_lambdas"][i].float()
            self.wqkv[i] = (sa[0] * W[: 3 * Dm]).bfloat16()
            self.wo[i] = (sa[1] * W[3 * Dm:]).bfloat16()

        # Tables YaRN reconstruites depuis angular_freq (float32) -> bf16,
        # exactement comme Yarn.reset()/apply() côté entraînement
        af = sd["yarn_angular_freq"].float()
        self.attn_scale = float(sd["yarn_attn_scale"].item())
        n_tab = max(cfg["max_positions"], self.max_ctx)
        t = torch.arange(n_tab, dtype=torch.float32, device=device)
        theta = torch.outer(t, af)
        f1, f2 = theta.cos(), theta.sin()
        f2[..., 1::2] *= -1
        self.f1, self.f2 = f1.bfloat16(), f2.bfloat16()
        # Version appariée : positions paires/impaires intercalées (2t, 2t+1)
        th1 = torch.outer(2 * t, af)
        th2 = torch.outer(2 * t + 1, af)
        pf1 = torch.cat([th1.cos(), th2.cos()], dim=-1)
        pf2 = torch.cat([th1.sin(), th2.sin()], dim=-1)
        pf2[..., 1::2] *= -1
        self.pf1, self.pf2 = pf1.bfloat16(), pf2.bfloat16()

        self.ve_banks = {int(k): v for k, v in cfg["ve_banks"].items()}
        self.reset()

    def _attn_layers(self):
        return [i for i, b in enumerate(self.cfg["bm_pattern"]) if b is not None]

    def reset(self):
        L = self.cfg["num_layers"]
        self.kc: list[torch.Tensor | None] = [None] * L
        self.vc: list[torch.Tensor | None] = [None] * L
        self.koff_prev: dict[int, torch.Tensor] = {}  # couches key_offset : k brut du dernier token
        self.pos = 0
        self.prev_token: int | None = None
        self.prev_embed: torch.Tensor | None = None

    def _rot(self, x, f1, f2, pos0):
        """RoPE tel que Yarn.rotary : f1*x + f2*flip_paires(x). x: (1,T,H,D)."""
        T = x.shape[1]
        a = f1[None, pos0: pos0 + T, None, :]
        b = f2[None, pos0: pos0 + T, None, :]
        xf = x.view(*x.shape[:-1], x.shape[-1] // 2, 2).flip(-1).view(x.shape)
        return a * x + b * xf

    def _mudd(self, x, idx: int, n: int):
        sd = self.sd
        h = F.gelu(F.linear(x, sd["mudd_w1"][idx]))
        c = (F.linear(h, sd["mudd_w2"][idx, :n]) + sd["mudd_b2"][idx, :n]) * self.cfg["mudd_scale"]
        return list(c.split(1, dim=-1))

    @torch.no_grad()
    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: (T,) int64 sur device. Remplit le KV cache, renvoie les logits
        (softcappés, float32, (padded_vocab,)) de la DERNIÈRE position."""
        cfg, sd = self.cfg, self.sd
        H, D, L = cfg["num_heads"], cfg["head_dim"], cfg["num_layers"]
        Dm, BD = cfg["model_dim"], cfg["bigram_dim"]
        T = ids.numel()
        pos0 = self.pos
        assert pos0 + T <= self.f1.shape[0], "contexte max dépassé (tables YaRN)"
        ws = {"s": cfg["ws_short"], "l": cfg["ws_long"]}

        # ---- Embedding + smear gate (mélange avec l'embedding du token précédent)
        e = sd["embed.weight"][ids]                                     # (T, Dm) brut
        smear = (sd["smear_lambda"] * torch.sigmoid(F.linear(e[:, :12], sd["smear_gate.weight"]))).type_as(e)
        if pos0 == 0:
            prev = torch.cat([torch.zeros_like(e[:1]), e[:-1]], dim=0)
        else:
            prev = torch.cat([self.prev_embed, e[:-1]], dim=0)
        x = e + smear * prev
        self.prev_embed = e[-1:].clone()
        x = F.rms_norm(x[None], (Dm,))                                  # (1, T, Dm)
        x0 = x

        # ---- Bigram hash embeddings (+ table de signes)
        mod = cfg["bigram_vocab_size"] - 1
        ids32 = ids.to(torch.int32)
        if pos0 == 0:
            prev_ids = torch.cat([torch.zeros_like(ids32[:1]), ids32[:-1]])
        else:
            prev_ids = torch.cat([torch.tensor([self.prev_token], dtype=torch.int32,
                                               device=ids.device), ids32[:-1]])
        bi = torch.bitwise_xor(36313 * ids32, 27191 * prev_ids) % mod
        si = torch.bitwise_xor(prev_ids, ids32) % cfg["sign_rows"]
        if pos0 == 0:
            bi[0] = mod
            si[0] = 0
        self.prev_token = int(ids[-1])
        x0b = (sd["bigram_embed.weight"][bi.long()] * sd["bigram_sign_table"][si.long()])[None]  # (1,T,BD)
        x[..., :BD] = x[..., :BD] + x0b * sd["bigram_lambdas"][0]

        skip_gate = (torch.sigmoid(sd["skip_lambda"])
                     * 2 * torch.sigmoid(F.linear(x0[..., :12], sd["skip_gate.weight"]))).type_as(x)

        ve = {i: sd["value_embeds"][b][ids] for i, b in self.ve_banks.items()}  # (T, Dm)

        resid_att = sd["resid_lambdas"][:, 0].bfloat16()
        resid_mlp = sd["resid_lambdas"][:, 1].bfloat16()
        post_att = sd["post_lambdas"][:, 0].bfloat16()
        post_mlp = sd["post_lambdas"][:, 1].bfloat16()
        x0_l = sd["x0_lambdas"].bfloat16()
        bg_l = sd["bigram_lambdas"].bfloat16()

        snap = {0: x}
        for i in range(L):
            mu = None
            if i == 6:
                # pas d'attention : skip depuis le snapshot de la couche 3
                x = x + skip_gate * snap[3]
            else:
                attn_in = F.rms_norm(snap[7] if i > 7 else x, (Dm,))
                paired = i in cfg["paired_layers"]
                window = ws[cfg["bm_pattern"][i]]

                qkv = F.linear(attn_in, self.wqkv[i]).view(1, T, 3 * H, D)
                q, k, v = qkv.chunk(3, dim=2)                           # (1,T,H,D)
                q = F.rms_norm(q, (D,))
                k = F.rms_norm(k, (D,))

                # ---- aux_v : value embeddings gatés (couches 1/2/8/9) ou MUDD (couche 10)
                if i == L - 1:
                    snap[9] = x
                    mu = self._mudd(x, 0, 14)
                    v_mudd = (mu[0] * snap[0] + mu[1] * snap[7] + mu[2] * x).view(1, T, H, D)
                    x = (1 + mu[5]) * x + mu[3] * snap[0] + mu[4] * snap[7]
                    veg = torch.cat([mu[6], mu[7]], dim=-1).repeat_interleave(
                        H // 2, dim=-1).unsqueeze(-1)                   # (1,T,H,1)
                    aux_v = (veg * ve[i][None].view(1, T, H, D) + v_mudd)
                elif i in ve:
                    gate_in = torch.cat([attn_in[..., :6], ve[i][None, :, :6]], dim=-1)  # (1,T,12)
                    g = 2 * torch.sigmoid(F.linear(gate_in, sd[f"blocks.{i}.ve_gate"]))
                    aux_v = g.view(1, T, H, 1) * ve[i][None].view(1, T, H, D)
                else:
                    aux_v = None

                # ---- rotary / appariement / key offset, puis KV cache
                if paired:
                    # têtes adjacentes appariées : la séquence est doublée (2T)
                    q = self._rot(q.view(1, T, H // 2, 2 * D), self.pf1, self.pf2, pos0)
                    k = self._rot(k.view(1, T, H // 2, 2 * D), self.pf1, self.pf2, pos0)
                    q = q.view(1, 2 * T, H // 2, D)
                    k = k.view(1, 2 * T, H // 2, D)
                    if aux_v is not None:
                        v = v + aux_v
                    v = v.reshape(1, 2 * T, H // 2, D)
                else:
                    q = self._rot(q, self.f1, self.f2, pos0)
                    k = self._rot(k, self.f1, self.f2, pos0)
                    if i in cfg["key_offset_layers"]:
                        # la moitié "stationnaire" des dims de k est décalée d'un token
                        half = D // 2
                        if pos0 == 0:
                            prev_up = k[:, :1, :, half:]
                        else:
                            prev_up = self.koff_prev[i]
                        up = torch.cat([prev_up, k[:, :-1, :, half:]], dim=1)
                        self.koff_prev[i] = k[:, -1:, :, half:].clone()
                        k = torch.cat([k[..., :half], up], dim=-1)
                    if aux_v is not None:
                        v = v + aux_v

                kc = k if self.kc[i] is None else torch.cat([self.kc[i], k], dim=1)
                vc = v if self.vc[i] is None else torch.cat([self.vc[i], v], dim=1)
                cap = self.max_ctx * (2 if paired else 1)
                if kc.shape[1] > cap:
                    kc, vc = kc[:, -cap:], vc[:, -cap:]
                self.kc[i], self.vc[i] = kc, vc

                y = attend(q, kc, vc, window, self.attn_scale)
                if paired:
                    y = y.view(1, T, H, D)

                # ---- XSA (couches non appariées uniquement) puis gate de sortie
                if i in cfg["xsa_layers"] and not paired:
                    vn = F.normalize(v, dim=-1, eps=1e-4)               # v après ajout de aux_v
                    proj = (y * vn).sum(-1, keepdim=True)
                    alpha = torch.tanh(sd["xsa_alphas"][i]).type_as(y).view(1, 1, H, 1)
                    y = y - alpha * proj * vn
                g = torch.sigmoid(F.linear(attn_in[..., :12], sd[f"blocks.{i}.attn_gate"]))
                y = y * g.view(1, T, H, 1)
                attn_out = F.linear(y.reshape(1, T, -1), self.wo[i])

                # ---- résiduel attention
                if mu is not None:
                    x = mu[8] * x + mu[9] * attn_out + mu[10] * snap[0]
                    x[..., :BD] = x[..., :BD] + mu[11] * x0b
                else:
                    x = resid_att[i] * x + post_att[i] * attn_out + x0_l[i] * x0
                    if i >= 1:
                        x[..., :BD] = x[..., :BD] + bg_l[i] * x0b

            # ---- MLP ReLU²
            normed = F.rms_norm(x, (Dm,))
            h = F.relu(F.linear(normed, sd[f"blocks.{i}.mlp_fc"]))
            mlp_out = (h * h) @ sd[f"blocks.{i}.mlp_proj"]              # proj sans transpose (cf. ReLUSqrdMLP)
            if mu is not None:
                x = mu[12] * x + mu[13] * mlp_out
            else:
                x = resid_mlp[i] * x + post_mlp[i] * mlp_out

            if i in (3, 7):
                snap[i] = x

        # ---- MUDD post-loop : recombinaison des snapshots
        mu2 = self._mudd(x, 1, 5)
        ve0 = sd["value_embeds"][0][ids][None]                          # banque 0 (= VE de la couche 1)
        x = x + mu2[0] * snap[0] + mu2[1] * snap[7] + mu2[2] * snap[9] \
              + mu2[3] * ve0 + mu2[4] * snap[3]

        x = F.rms_norm(x, (Dm,))
        logits = x[0, -1] @ sd["lm_head.weight"]                        # (padded_vocab,)
        sc = cfg["softcap"]
        logits = sc["scale"] * torch.sigmoid((logits + sc["shift"]) / sc["div"])

        self.pos = pos0 + T
        return logits.float()


# -----------------------------------------------------------------------------
# Sampling
# -----------------------------------------------------------------------------
def sample_next(logits: torch.Tensor, vocab_size: int, temperature: float,
                top_k: int, top_p: float, rng: torch.Generator) -> int:
    logits = logits.clone()
    logits[vocab_size:] = float("-inf")      # lignes de padding du vocab
    if temperature <= 0:
        return int(torch.argmax(logits))
    logits = logits / temperature
    if top_k > 0:
        thresh = torch.topk(logits, top_k).values[-1]
        logits = torch.where(logits < thresh, torch.full_like(logits, float("-inf")), logits)
    if top_p < 1.0:
        svals, sidx = torch.sort(logits, descending=True)
        probs = torch.softmax(svals, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        remove = cum - probs > top_p
        svals = torch.where(remove, torch.full_like(svals, float("-inf")), svals)
        logits = torch.full_like(logits, float("-inf")).scatter(0, sidx, svals)
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1, generator=rng))


# -----------------------------------------------------------------------------
# Boucle de génération
# -----------------------------------------------------------------------------
@torch.no_grad()
def generate(model: ModdedNanoGPT, kind, tok, prompt: str, args, rng) -> str:
    cfg = model.cfg
    ids = tok_encode(kind, tok, prompt)
    if not args.no_bos:
        ids = [cfg["bos_id"]] + ids
    assert len(ids) + args.max_new_tokens <= model.max_ctx, \
        f"prompt + génération > max_ctx ({model.max_ctx})"

    device = model.device
    model.reset()
    t0 = time.perf_counter()
    logits = model.forward(torch.tensor(ids, dtype=torch.long, device=device))
    prefill_dt = time.perf_counter() - t0

    out_ids = []
    t0 = time.perf_counter()
    for _ in range(args.max_new_tokens):
        nxt = sample_next(logits, cfg["vocab_size"], args.temperature,
                          args.top_k, args.top_p, rng)
        out_ids.append(nxt)
        if nxt == cfg["bos_id"] and args.stop_eos:
            break
        logits = model.forward(torch.tensor([nxt], dtype=torch.long, device=device))
    decode_dt = time.perf_counter() - t0

    text = tok_decode(kind, tok, out_ids)
    if args.verbose:
        n = len(out_ids)
        print(f"\n[prefill: {len(ids)} tok en {prefill_dt*1e3:.0f} ms | "
              f"decode: {n} tok en {decode_dt:.2f} s ({n/max(decode_dt,1e-9):.1f} tok/s)]")
    return text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="exports/mon_modele", help="dossier exporté")
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--prompt-file", default=None, help="lire le prompt depuis un fichier")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--no-bos", action="store_true", help="ne pas préfixer par <|endoftext|>")
    p.add_argument("--stop-eos", action="store_true", help="arrêter si <|endoftext|> est échantillonné")
    p.add_argument("--max-ctx", type=int, default=None, help="plafond de contexte (défaut: config)")
    p.add_argument("--cpu", action="store_true", help="forcer le CPU (debug)")
    p.add_argument("-i", "--interactive", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    if device == "cuda" and not HAS_FLASH:
        print("[info] flash-attn introuvable -> fallback SDPA (plus lent mais exact)")
    model = ModdedNanoGPT(args.model, device=device, max_ctx=args.max_ctx)
    kind, tok = load_tokenizer()
    rng = torch.Generator(device=device)
    rng.manual_seed(args.seed)

    if args.interactive:
        print(f"Modèle chargé ({args.model}). Tape ton prompt, 'q' pour quitter.")
        while True:
            try:
                prompt = input("\n>>> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if prompt.lower() in ("q", "quit", "exit"):
                break
            if not prompt:
                continue
            print(generate(model, kind, tok, prompt, args, rng))
    else:
        prompt = open(args.prompt_file).read() if args.prompt_file else args.prompt
        print(generate(model, kind, tok, prompt, args, rng))


if __name__ == "__main__":
    main()
