#!/usr/bin/env python3
"""
export_model.py — exporte un checkpoint d'entraînement modded-nanogpt vers un
dossier d'inférence (model.safetensors + config.json) utilisable par generate.py.

Pourquoi ce script est nécessaire (au lieu d'un simple torch.save) :
  - le checkpoint contient le state_dict du modèle torch.compile (préfixe "_orig_mod.") ;
  - les poids sont stockés en "banks" optimisées pour l'optimizer (qk_bank, vo_bank,
    mlp_bank avec lignes de padding, scalars paddés pour l'alignement multi-GPU) ;
  - les tables YaRN (factor1/factor2) sont des buffers NON persistants : absents du
    checkpoint. Leur état final dépend de l'historique du schedule (yarn.apply à
    chaque changement de fenêtre) : on le recalcule ici de façon déterministe ;
  - vLLM/sglang ne supportent pas cette architecture : ce dossier sert de format
    d'échange propre pour l'inférence maison.

Usage :
  python export_model.py --ckpt logs/ckpt_latest.pt --out exports/mon_modele
"""

import argparse
import json
import math
import os

import torch

# -----------------------------------------------------------------------------
# Constantes recopiées du fichier d'entraînement (modded-nanogpt, run 4060 Ti)
# -----------------------------------------------------------------------------
VOCAB_SIZE = 50257          # vrai vocab GPT-2 ; les lignes au-delà sont du padding
NUM_LAYERS = 11
NUM_HEADS = 6
HEAD_DIM = 128
MODEL_DIM = 768
BIGRAM_DIM = 192
SIGN_ROWS = 8192
PAIRED_LAYERS = [0, 2, 5, 9]
KEY_OFFSET_LAYERS = [3, 10]     # couches à fenêtre longue (bm == ws_long)
XSA_LAYERS = [1, 3, 4, 7, 8, 10]
VE_BANKS = {1: 0, 2: 1, 8: 2, 9: 3, 10: 4}   # couche -> banque de value embeddings
ATTN_LAYERS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10]  # la couche 6 n'a pas d'attention
BM_PATTERN = ["s", "s", "s", "l", "s", "s", None, "s", "s", "s", "l"]
MUDD_SCALE = 0.1
BOS_ID = 50256

# Fenêtres finales (blocs de 128 tokens) : schedule extension (6, 13) puis
# apply_final_ws_ext() passe la fenêtre longue à 20 blocs SANS nouvel apply YaRN.
WS_SHORT_TOK = 6 * 128
WS_LONG_TOK = 20 * 128
# max_seq_len du modèle à l'init = val_batch_size // (grad_accum * world) = 4096,
# les tables YaRN sont dimensionnées sur 2 * max_seq_len.
MAX_POSITIONS = 8192

# Transitions YaRN appliquées pendant le run (advance_schedule, en tokens) :
# stage0 (1,3) -> stage1 (3,7) -> stage2 (5,11) -> extension (6,13).
# apply(old_ws_long*128, new_ws_long*128) à chaque transition.
DEFAULT_TRANSITIONS = [(3 * 128, 7 * 128), (7 * 128, 11 * 128), (11 * 128, 13 * 128)]


def yarn_base_freqs(head_dim: int) -> torch.Tensor:
    """angular_freq initial tel que Yarn.reset() le construit (float32)."""
    af = (1 / 1024) ** torch.linspace(0, 1, steps=head_dim // 4, dtype=torch.float32)
    af = af.repeat_interleave(2)
    # half-truncate RoPE : la moitié haute des dims ne tourne pas
    return torch.cat([af, af.new_zeros(head_dim // 2)])


def yarn_apply(af: torch.Tensor, old_window: int, new_window: int,
               alpha: int = 1, beta: int = 32) -> tuple[torch.Tensor, float]:
    """Rejoue Yarn.apply() : interpolation des fréquences + facteur sur attn_scale."""
    rotations = old_window * af / (2 * math.pi)
    scaling = old_window / new_window
    interp_w = torch.clamp((rotations - alpha) / (beta - alpha), 0, 1)
    af = af * (scaling + interp_w * (1 - scaling))
    scale_mul = 0.2 * math.log(new_window / old_window) + 1
    return af, scale_mul


def parse_transitions(s: str) -> list[tuple[int, int]]:
    # format "384:896,896:1408,1408:1664"
    return [tuple(int(v) for v in part.split(":")) for part in s.split(",") if part.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="logs/ckpt_latest.pt", help="checkpoint d'entraînement")
    p.add_argument("--out", default="exports/mon_modele", help="dossier de sortie")
    p.add_argument("--transitions", default=None,
                   help="transitions YaRN 'old:new,old:new,...' en tokens "
                        f"(défaut: {DEFAULT_TRANSITIONS})")
    p.add_argument("--ws-long", type=int, default=WS_LONG_TOK,
                   help="fenêtre longue finale en tokens (défaut 2560 = 20 blocs)")
    p.add_argument("--no-tokenizer", action="store_true",
                   help="ne pas écrire les fichiers tokenizer GPT-2 dans l'export")
    args = p.parse_args()

    transitions = parse_transitions(args.transitions) if args.transitions else DEFAULT_TRANSITIONS

    print(f"[1/4] Chargement du checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    raw = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    sd = {k.removeprefix("_orig_mod."): v for k, v in raw.items()}

    # Quelques vérifications de cohérence avec l'architecture attendue
    assert sd["embed.weight"].shape == (50304, MODEL_DIM), sd["embed.weight"].shape
    assert sd["value_embeds"].shape == (5 * 50304, MODEL_DIM)
    assert sd["qk_bank"].shape[1:] == (HEAD_DIM * 2, MODEL_DIM)
    assert "bigram_sign_table" in sd, "buffer bigram_sign_table manquant du state_dict"
    if isinstance(ckpt, dict):
        print(f"      step du checkpoint: {ckpt.get('step', '?')} "
              f"(split_embed={ckpt.get('split_embed', '?')})")

    print("[2/4] Découpage des banks (padding retiré, poids par couche)")
    num_attn = len(ATTN_LAYERS)
    num_qk_real = num_attn * 2 * (NUM_HEADS // 2)   # 60
    num_vo_real = num_attn * 2                       # 20

    out: dict[str, torch.Tensor] = {}
    out["embed.weight"] = sd["embed.weight"].contiguous()
    out["lm_head.weight"] = sd["lm_head.weight"].contiguous()      # (768, 50304), stockage transposé
    out["value_embeds"] = sd["value_embeds"].view(-1, 50304, MODEL_DIM).contiguous()
    out["bigram_embed.weight"] = sd["bigram_embed.weight"].contiguous()
    out["bigram_sign_table"] = sd["bigram_sign_table"].contiguous()
    out["smear_gate.weight"] = sd["smear_gate.weight"].contiguous()
    out["skip_gate.weight"] = sd["skip_gate.weight"].contiguous()

    scalars = sd["scalars"][: 2 * NUM_LAYERS + 2].float()
    out["sa_lambdas"] = scalars[: 2 * NUM_LAYERS].view(NUM_LAYERS, 2).contiguous()
    out["smear_lambda"] = scalars[2 * NUM_LAYERS].reshape(1).contiguous()
    out["skip_lambda"] = scalars[2 * NUM_LAYERS + 1].reshape(1).contiguous()

    for name in ["post_lambdas", "resid_lambdas", "x0_lambdas", "bigram_lambdas",
                 "xsa_alphas", "mudd_w1", "mudd_w2", "mudd_b2"]:
        out[name] = sd[name].contiguous()

    qk_bank = sd["qk_bank"][:num_qk_real]
    vo_bank = sd["vo_bank"][:num_vo_real]
    for j, i in enumerate(ATTN_LAYERS):
        qk = qk_bank[2 * (NUM_HEADS // 2) * j: 2 * (NUM_HEADS // 2) * (j + 1)].reshape(-1, MODEL_DIM)
        vo = vo_bank[2 * j: 2 * (j + 1)].reshape(-1, MODEL_DIM)     # V puis O
        out[f"blocks.{i}.qkvo_w"] = torch.cat([qk, vo], dim=0).contiguous()  # (3072, 768)
        out[f"blocks.{i}.attn_gate"] = sd["attn_gate_bank"][j].contiguous()  # (6, 12)

    for i, b in VE_BANKS.items():
        out[f"blocks.{i}.ve_gate"] = sd["ve_gate_bank"][b].contiguous()      # (6, 12)

    for i in range(NUM_LAYERS):
        out[f"blocks.{i}.mlp_fc"] = sd["mlp_bank"][i, 0].contiguous()        # (3072, 768)
        out[f"blocks.{i}.mlp_proj"] = sd["mlp_bank"][i, 1].contiguous()      # (3072, 768)

    print("[3/4] Recalcul de l'état YaRN final")
    print(f"      transitions rejouées: {transitions}")
    af = yarn_base_freqs(HEAD_DIM)
    attn_scale = 0.1
    for old_w, new_w in transitions:
        af, mul = yarn_apply(af, old_w, new_w)
        attn_scale *= mul
    print(f"      attn_scale final = {attn_scale:.6f}")
    out["yarn_angular_freq"] = af.float()
    out["yarn_attn_scale"] = torch.tensor([attn_scale], dtype=torch.float32)

    print("[4/4] Écriture du dossier d'export")
    os.makedirs(args.out, exist_ok=True)
    config = {
        "architectures": ["ModdedNanoGPTForCausalLM"],
        "model_type": "modded_nanogpt",
        "arch": "modded-nanogpt-speedrun",
        "vocab_size": VOCAB_SIZE,
        "padded_vocab": int(sd["embed.weight"].shape[0]),
        "num_layers": NUM_LAYERS,
        "num_heads": NUM_HEADS,
        "head_dim": HEAD_DIM,
        "model_dim": MODEL_DIM,
        "bigram_vocab_size": int(sd["bigram_embed.weight"].shape[0]),
        "bigram_dim": BIGRAM_DIM,
        "sign_rows": SIGN_ROWS,
        "paired_layers": PAIRED_LAYERS,
        "key_offset_layers": KEY_OFFSET_LAYERS,
        "xsa_layers": XSA_LAYERS,
        "ve_banks": {str(k): v for k, v in VE_BANKS.items()},
        "bm_pattern": BM_PATTERN,
        "ws_short": WS_SHORT_TOK,
        "ws_long": args.ws_long,
        "max_positions": MAX_POSITIONS,
        "mudd_scale": MUDD_SCALE,
        "bos_id": BOS_ID,
        "softcap": {"scale": 23.0, "shift": 5.0, "div": 7.5},
        "yarn_transitions_tokens": transitions,
        "dtype": "bfloat16",
    }
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    from safetensors.torch import save_file
    save_file(out, os.path.join(args.out, "model.safetensors"))

    if not args.no_tokenizer:
        # Tokenizer GPT-2, requis par les moteurs d'inférence (vLLM/SGLang)
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
        tok.save_pretrained(args.out)
        print("      tokenizer GPT-2 écrit (tokenizer.json, vocab.json, merges.txt)")

    n_params = sum(t.numel() for t in out.values())
    size_mb = sum(t.numel() * t.element_size() for t in out.values()) / 1e6
    print(f"OK — {n_params/1e6:.1f} M paramètres, {size_mb:.0f} Mo écrits dans {args.out}/")
    print("Fichiers: config.json, model.safetensors")


if __name__ == "__main__":
    main()
