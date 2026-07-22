#!/usr/bin/env python3
"""
compare_sglang.py — valide le port SGLang contre le golden de generate.py.

Critère principal : teacher forcing. On donne au moteur la séquence complète
(prompt + génération golden) et on récupère les logprobs d'input : pour chaque
position, le token golden doit être classé au sommet des logits du port, à une
tolérance près (bf16 : deux logits quasi égaux peuvent s'inverser selon
l'ordre de réduction du kernel d'attention — ex. tie exact observé à 18.625).

Critère secondaire (informatif) : génération greedy libre, ids identiques tant
qu'il n'y a pas de quasi-tie.

Usage (depuis la racine du repo) :
  ./venv-infer/bin/python infer_port/compare_sglang.py [--attention-backend fa3]
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INFER_PORT = os.path.join(ROOT, "infer_port")

# Le package externe doit être importable dans le process parent ET dans les
# subprocess spawn du scheduler -> variable d'env + sys.path.
os.environ["PYTHONPATH"] = INFER_PORT + os.pathsep + os.environ.get("PYTHONPATH", "")
sys.path.insert(0, INFER_PORT)
os.environ["SGLANG_EXTERNAL_MODEL_PACKAGE"] = "sglang_ext"
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".hf_cache"))
# ninja (compilateur JIT de flashinfer) vit dans le venv, pas dans le PATH
os.environ["PATH"] = os.path.join(ROOT, "venv-infer", "bin") + os.pathsep + os.environ["PATH"]

import numpy as np  # noqa: E402

# Tolérance en unités de logprob softcappée (logits ~±23, ulp bf16 ≈ 0.125 à 21)
TOL = 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "exports", "mon_modele"))
    ap.add_argument("--attention-backend", default=None)
    ap.add_argument("--max-running-requests", type=int, default=8)
    args = ap.parse_args()

    import sglang as sgl

    kwargs = dict(
        model_path=args.model,
        tokenizer_path=args.model,
        dtype="bfloat16",
        trust_remote_code=True,
        disable_radix_cache=True,
        disable_cuda_graph=True,
        chunked_prefill_size=-1,
        max_running_requests=args.max_running_requests,
        context_length=8192,
        max_total_tokens=65536,
        mem_fraction_static=0.75,
    )
    if args.attention_backend:
        kwargs["attention_backend"] = args.attention_backend

    engine = sgl.Engine(**kwargs)

    golden_dir = os.path.join(INFER_PORT, "golden")
    ok = True
    try:
        for path in sorted(f for f in os.listdir(golden_dir) if f.endswith(".npz")):
            g = np.load(os.path.join(golden_dir, path))
            prompt_ids = g["prompt"].tolist()
            expected = g["gen_ids"].tolist()

            # ---- teacher forcing : logprobs d'input sur la séquence golden ----
            full_ids = prompt_ids + expected
            out = engine.generate(
                input_ids=[full_ids],
                sampling_params={"temperature": 0, "max_new_tokens": 1, "ignore_eos": True},
                return_logprob=True,
                logprob_start_len=0,
                top_logprobs_num=1,
            )[0]
            meta = out["meta_info"]
            tok_lp = meta["input_token_logprobs"]   # [(lp, token_id, text), ...]
            top_lp = meta["input_top_logprobs"]     # [[(lp, token_id, text)], ...]

            worst = 0.0
            fail_at = None
            for s in range(len(expected)):
                pos = len(prompt_ids) + s  # position du token golden dans full_ids
                lp_expected = tok_lp[pos][0]
                lp_top = top_lp[pos][0][0]
                gap = lp_top - lp_expected
                worst = max(worst, gap)
                if gap > TOL and fail_at is None:
                    fail_at = s

            # ---- greedy libre (informatif) ----
            out_g = engine.generate(
                input_ids=[prompt_ids],
                sampling_params={
                    "temperature": 0,
                    "max_new_tokens": len(expected),
                    "ignore_eos": True,
                },
            )[0]
            got = list(out_g["output_ids"])
            diff = next((j for j in range(min(len(got), len(expected)))
                         if got[j] != expected[j]), None)

            if fail_at is None:
                note = "greedy exact" if diff is None else f"greedy diverge au token {diff} (quasi-tie)"
                print(f"[OK]   {path}: teacher forcing OK (pire écart {worst:.3f}) — {note}")
            else:
                ok = False
                print(f"[FAIL] {path}: token golden {expected[fail_at]} classé trop bas "
                      f"au step {fail_at} (écart {worst:.3f} > {TOL})")
    finally:
        engine.shutdown()

    print("VALIDATION OK" if ok else "VALIDATION FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
