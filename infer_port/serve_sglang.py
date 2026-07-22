#!/usr/bin/env python3
"""
serve_sglang.py — lance un serveur OpenAI-compatible sglang pour modded-nanogpt.

Usage :
  ./venv-infer/bin/python infer_port/serve_sglang.py [--port 30000]
puis :
  curl http://localhost:30000/v1/completions -H 'Content-Type: application/json' \
    -d '{"model": "exports/mon_modele", "prompt": "Once upon a time", "max_tokens": 32}'
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INFER_PORT = os.path.join(ROOT, "infer_port")

os.environ["PYTHONPATH"] = INFER_PORT + os.pathsep + os.environ.get("PYTHONPATH", "")
sys.path.insert(0, INFER_PORT)
os.environ["SGLANG_EXTERNAL_MODEL_PACKAGE"] = "sglang_ext"
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".hf_cache"))
os.environ["PATH"] = os.path.join(ROOT, "venv-infer", "bin") + os.pathsep + os.environ["PATH"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "exports", "mon_modele"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--max-running-requests", type=int, default=32)
    args = ap.parse_args()

    from sglang.srt.entrypoints.http_server import launch_server
    from sglang.srt.server_args import prepare_server_args

    server_args = prepare_server_args([
        "--model-path", args.model,
        "--tokenizer-path", args.model,
        "--host", args.host,
        "--port", str(args.port),
        "--dtype", "bfloat16",
        "--trust-remote-code",
        "--disable-radix-cache",
        "--chunked-prefill-size", "-1",
        "--max-running-requests", str(args.max_running_requests),
        "--context-length", "8192",
        "--max-total-tokens", "65536",
        "--mem-fraction-static", "0.75",
    ])
    launch_server(server_args)


if __name__ == "__main__":
    main()
