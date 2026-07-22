#!/usr/bin/env python3
"""
bench.py — compare le débit du port SGLang avec la référence generate.py.

Mesure le decode (tok/s) et le prefill (ms) :
  - generate.py (ModdedNanoGPT eager, batch 1) — nécessite le venv d'entraînement ;
  - sglang.Engine offline, batch 1 puis batch N (mêmes prompts dupliqués).

Usage :
  ./venv-infer/bin/python infer_port/bench.py [--batch 8] [--max-new-tokens 128]
"""

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INFER_PORT = os.path.join(ROOT, "infer_port")

os.environ["PYTHONPATH"] = INFER_PORT + os.pathsep + os.environ.get("PYTHONPATH", "")
sys.path.insert(0, INFER_PORT)
os.environ["SGLANG_EXTERNAL_MODEL_PACKAGE"] = "sglang_ext"
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".hf_cache"))
os.environ["PATH"] = os.path.join(ROOT, "venv-infer", "bin") + os.pathsep + os.environ["PATH"]

import numpy as np  # noqa: E402


def bench_sglang(args):
    import sglang as sgl

    g = np.load(os.path.join(INFER_PORT, "golden", "prompt3.npz"))  # prompt long
    prompt_ids = g["prompt"].tolist()

    engine = sgl.Engine(
        model_path=args.model,
        tokenizer_path=args.model,
        dtype="bfloat16",
        trust_remote_code=True,
        disable_radix_cache=True,
        disable_cuda_graph=not args.cuda_graph,
        chunked_prefill_size=-1,
        max_running_requests=args.batch,
        context_length=8192,
        max_total_tokens=65536,
        mem_fraction_static=0.75,
    )
    try:
        sp = {"temperature": 0, "max_new_tokens": args.max_new_tokens, "ignore_eos": True}
        # warmup
        engine.generate(input_ids=[prompt_ids], sampling_params={**sp, "max_new_tokens": 8})

        for bs in (1, args.batch):
            t0 = time.perf_counter()
            outs = engine.generate(input_ids=[prompt_ids] * bs, sampling_params=sp)
            dt = time.perf_counter() - t0
            n = sum(len(o["output_ids"]) for o in outs)
            print(f"sglang batch={bs:3d} : {n} tokens en {dt:.2f} s -> {n/dt:.1f} tok/s "
                  f"({n/dt/bs:.1f} tok/s/req)")
    finally:
        engine.shutdown()


def bench_reference(args):
    """Lance generate.py (venv training) en sous-process et récupère le tok/s."""
    import re
    import subprocess

    cmd = [
        os.path.join(ROOT, "venv", "bin", "python"),
        os.path.join(ROOT, "v1", "generate.py"),
        "--model", args.model,
        "--prompt", "In a shocking finding, scientists discovered a herd of unicorns living in",
        "--max-new-tokens", str(args.max_new_tokens),
        "--temperature", "0",
        "-v",
    ]
    env = {**os.environ, "HF_HOME": os.path.join(ROOT, ".hf_cache")}
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.join(ROOT, "v1"), env=env)
    m = re.search(r"decode: (\d+) tok en ([\d.]+) s \(([\d.]+) tok/s\)", out.stdout)
    if m:
        print(f"generate.py batch=1 : {m.group(1)} tokens en {m.group(2)} s -> {m.group(3)} tok/s")
    else:
        print("generate.py: mesure illisible\n", out.stdout[-500:], out.stderr[-500:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "exports", "mon_modele"))
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--cuda-graph", action="store_true", help="activer les CUDA graphs sglang")
    ap.add_argument("--skip-reference", action="store_true")
    args = ap.parse_args()

    if not args.skip_reference:
        bench_reference(args)
    bench_sglang(args)


if __name__ == "__main__":
    main()
