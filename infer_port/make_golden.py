#!/usr/bin/env python3
"""
make_golden.py — génère l'oracle de validation pour le port SGLang/vLLM.

Pour chaque prompt fixe, exécute ModdedNanoGPT (v1/generate.py, l'implémentation
de référence) en greedy et sauvegarde dans infer_port/golden/<idx>.npz :
  - prompt_ids  : ids du prompt (BOS inclus)
  - gen_ids     : ids générés en greedy (argmax, temperature=0)
  - logits      : (n_steps, padded_vocab) float32 — logits softcappés à chaque
                  étape (prefill = étape 0, puis un par token décodé)

Le critère de validation du port est l'égalité exacte des gen_ids ; les logits
servent au diagnostic (écart max / position de divergence).

Usage :
  python infer_port/make_golden.py            # depuis la racine du repo
"""

import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "v1"))

from generate import ModdedNanoGPT  # noqa: E402

MODEL_DIR = os.path.join(ROOT, "exports", "mon_modele")
OUT_DIR = os.path.join(ROOT, "infer_port", "golden")
MAX_NEW_TOKENS = 32

PROMPTS = [
    "Once upon a time",
    "The capital of France is",
    "def quicksort(arr):",
    "In a shocking finding, scientists discovered a herd of unicorns living in",
    "The recipe calls for 2 cups of flour,",
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)

    model = ModdedNanoGPT(MODEL_DIR, device="cuda")
    bos = model.cfg["bos_id"]

    for idx, prompt in enumerate(PROMPTS):
        ids = [bos] + tok.encode(prompt, add_special_tokens=False)
        model.reset()
        all_logits = []
        gen = []
        logits = model.forward(torch.tensor(ids, dtype=torch.long, device="cuda"))
        all_logits.append(logits.cpu())
        for _ in range(MAX_NEW_TOKENS):
            nxt = int(torch.argmax(logits[: model.cfg["vocab_size"]]))
            gen.append(nxt)
            logits = model.forward(torch.tensor([nxt], dtype=torch.long, device="cuda"))
            all_logits.append(logits.cpu())

        path = os.path.join(OUT_DIR, f"prompt{idx}.npz")
        np.savez(
            path,
            prompt=np.array(ids, dtype=np.int64),
            gen_ids=np.array(gen, dtype=np.int64),
            logits=torch.stack(all_logits).numpy(),
            prompt_text=prompt,
        )
        print(f"[{idx}] {prompt!r} -> {tok.decode(gen[:8])!r}... ({path})")

    print(f"OK — {len(PROMPTS)} golden files dans {OUT_DIR}")


if __name__ == "__main__":
    main()
