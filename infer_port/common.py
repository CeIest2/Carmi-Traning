"""
common.py — helpers partagés entre le port SGLang et le port vLLM.

Tout ce qui est indépendant du moteur : lecture du config.json exporté,
reconstruction des tables RoPE/YaRN, listes de couches dérivées de la
topologie. La référence de comportement est v1/generate.py (ModdedNanoGPT).
"""

import json
import os

import torch


def load_config(model_dir: str) -> dict:
    with open(os.path.join(model_dir, "config.json")) as f:
        return json.load(f)


def attn_layers(cfg: dict) -> list[int]:
    """Couches avec attention (toutes sauf la 6, marquée None dans bm_pattern)."""
    return [i for i, b in enumerate(cfg["bm_pattern"]) if b is not None]


def window_of(cfg: dict, layer: int) -> int:
    """Fenêtre glissante (en tokens) d'une couche d'attention."""
    return cfg["ws_short"] if cfg["bm_pattern"][layer] == "s" else cfg["ws_long"]


def build_rotary_tables(angular_freq: torch.Tensor, max_ctx: int, device, dtype=torch.bfloat16):
    """
    Reconstruit (f1, f2) tels que Yarn.rotary : rot(x) = f1*x + f2*flip_paires(x),
    avec flip_paires = vue (..., D//2, 2) retournée sur le dernier axe.
    angular_freq : (head_dim,) float32 — la moitié haute est nulle (dims stationnaires).
    Retourne aussi les tables appariées (positions 2t / 2t+1 intercalées).
    Réplique exacte de generate.py:112-126.
    """
    af = angular_freq.float().to(device)
    t = torch.arange(max_ctx, dtype=torch.float32, device=device)
    theta = torch.outer(t, af)
    f1, f2 = theta.cos(), theta.sin()
    f2[..., 1::2] *= -1
    th1 = torch.outer(2 * t, af)
    th2 = torch.outer(2 * t + 1, af)
    pf1 = torch.cat([th1.cos(), th2.cos()], dim=-1)
    pf2 = torch.cat([th1.sin(), th2.sin()], dim=-1)
    pf2[..., 1::2] *= -1
    return (f1.to(dtype), f2.to(dtype)), (pf1.to(dtype), pf2.to(dtype))


def apply_rotary(x: torch.Tensor, f1: torch.Tensor, f2: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """
    rot(x) = f1[pos]*x + f2[pos]*flip_paires(x).
    x : (T, H, D) ; pos : (T,) long ; f1/f2 : (max_ctx, D).
    """
    a = f1[pos][:, None, :]
    b = f2[pos][:, None, :]
    xf = x.view(*x.shape[:-1], x.shape[-1] // 2, 2).flip(-1).view(x.shape)
    return a * x + b * xf


def bigram_hash(ids: torch.Tensor, prev_ids: torch.Tensor, first_mask: torch.Tensor,
                bigram_vocab_size: int, sign_rows: int):
    """
    Index bigram et index de signe pour chaque token, à partir de l'id courant
    et de l'id du token précédent. Réplique de generate.py:183-193.
    ids, prev_ids : (T,) int32/int64 ; first_mask : (T,) bool, True pour le
    premier token de chaque séquence (le hash y est neutralisé :
    bi = bigram_vocab_size - 1, si = 0 — cette ligne d'embedding est dédiée).
    """
    ids32 = ids.to(torch.int32)
    prev32 = prev_ids.to(torch.int32)
    mod = bigram_vocab_size - 1
    bi = torch.bitwise_xor(36313 * ids32, 27191 * prev32) % mod
    si = torch.bitwise_xor(prev32, ids32) % sign_rows
    bi = torch.where(first_mask, torch.full_like(bi, mod), bi)
    si = torch.where(first_mask, torch.zeros_like(si), si)
    return bi.long(), si.long()
