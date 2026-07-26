#!/usr/bin/env python3
"""
chat.py — test rapide du modèle SFT (checkpoint .pt).
Usage : python v1/chat.py --checkpoint logs/sft_output/sft_step01000.pt
"""

import os
import sys
import types
import argparse

import torch
import torch.nn.functional as F
import torch.distributed as dist

# --- HACK dist_setup ---
os.environ["DISABLE_FP8"] = "1"
os.environ["HF_HOME"] = "/tmp/huggingface_cache"

_fake_dist_setup = types.ModuleType("dist_setup")
_fake_dist_setup.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_fake_dist_setup.grad_accum_steps = 16
_fake_dist_setup.grad_scale = 1.0 / 16
_fake_dist_setup.master_process = True
_fake_dist_setup.world_size = 1
_fake_dist_setup.rank = 0
sys.modules["dist_setup"] = _fake_dist_setup

if not dist.is_initialized():
    dist.init_process_group(backend="gloo", init_method="tcp://localhost:29502", rank=0, world_size=1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import GPT, ForwardScheduleConfig
from transformers import GPT2Tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimpleTokenizer:
    def __init__(self):
        self._tok = GPT2Tokenizer.from_pretrained("gpt2", cache_dir="/tmp/huggingface_cache")
        self._tok.pad_token = self._tok.eos_token
        self.eos_token = self._tok.eos_token
        self.eos_token_id = self._tok.eos_token_id
        # ID de l'espace (Ġ) dans GPT-2
        self.space_id = self._tok.encode(" ", add_special_tokens=False)[0]

    def encode(self, text, add_special_tokens=False):
        return self._tok.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, tokens, skip_special_tokens=True):
        return self._tok.decode(tokens, skip_special_tokens=skip_special_tokens)


tokenizer = SimpleTokenizer()


def load_model(checkpoint_path: str):
    print(f"[CHAT] Chargement: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    model = GPT(vocab_size=50257, num_layers=11, num_heads=6, head_dim=128, model_dim=768, max_seq_len=4096)

    state_dict = ckpt["model"]
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_state_dict[k[len("_orig_mod."):]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model = model.to(device).eval()

    import torch.nn as nn
    for m in model.modules():
        if isinstance(m, (nn.Embedding, nn.Linear)):
            m.weight.data = m.weight.data.bfloat16()
    for buf_name in ["attn_gate_bank", "ve_gate_bank", "qk_bank", "vo_bank",
                     "mlp_bank", "mudd_w1", "mudd_w2", "mudd_b2"]:
        if hasattr(model, buf_name):
            getattr(model, buf_name).data = getattr(model, buf_name).data.bfloat16()

    print(f"[CHAT] Modèle prêt. {sum(p.numel() for p in model.parameters()):,} params.")
    return model


def _pad_to_multiple_of_16(ids: list[int]) -> tuple[list[int], int]:
    real_len = len(ids)
    pad_len = (16 - real_len % 16) % 16
    if pad_len:
        ids = ids + [tokenizer.eos_token_id] * pad_len
    return ids, real_len


def generate(model, prompt_text: str, max_new_tokens: int = 128,
             temperature: float = 0.8, top_p: float = 0.9,
             repetition_penalty: float = 1.2):
    # Format IDENTIQUE au SFT (sft.py::format_example)
    # Le SFT encode : prompt="User: {msg}\nAssistant:" puis response=" {msg}</s>"
    # Donc le premier token de la réponse est UN ESPACE.
    full_prompt = f"User: {prompt_text}\nAssistant:"

    prompt_ids = tokenizer.encode(full_prompt, add_special_tokens=False)
    prompt_ids, real_prompt_len = _pad_to_multiple_of_16(prompt_ids)
    generated_ids = list(prompt_ids)

    schedule_cfg = ForwardScheduleConfig(
        mtp_weights=torch.tensor([1.0], device=device),
        ws_short=4 * 128,
        ws_long=32 * 128,
        train_max_seq_len=4096,
    )

    logits_list = []

    def hook(module, input, output):
        logits_list.append(output.detach().clone())

    handle = model.lm_head.register_forward_hook(hook)

    print("[CHAT] Génération...", end=" ", flush=True)

    with torch.no_grad():
        for token_idx in range(max_new_tokens):
            padded_ids, real_len = _pad_to_multiple_of_16(generated_ids)

            ids = torch.tensor(padded_ids, dtype=torch.long, device=device)
            seqlens = torch.tensor([0, len(padded_ids)], dtype=torch.int32, device=device)
            bigram = ids.clone()

            _ = model(
                input_seq=ids,
                target_seq=ids,
                seqlens=seqlens,
                bigram_input_seq=bigram,
                schedule_cfg=schedule_cfg,
            )

            logits = logits_list[-1][0, real_len - 1, :].float()
            logits_list.clear()

            # Softcapping
            logits = 23 * torch.sigmoid((logits + 5) / 7.5)

            # --- Forcer le premier token de la réponse à être un espace ---
            # Dans le SFT, response = f" {assistant_msg}" + eos
            # Donc le modèle s'attend à un espace comme premier token de réponse
            if token_idx == 0:
                # On force l'espace en mettant -inf partout ailleurs
                forced = torch.full_like(logits, float('-inf'))
                forced[tokenizer.space_id] = logits[tokenizer.space_id]
                logits = forced

            # --- Penalty de répétition ---
            if repetition_penalty != 1.0:
                for token_id in set(generated_ids[real_prompt_len:]):
                    logits[token_id] /= repetition_penalty

            # Temperature + top-p
            logits = logits / temperature
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False
            indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = int(torch.multinomial(probs, num_samples=1))

            if next_token == tokenizer.eos_token_id:
                break

            generated_ids.append(next_token)
            print(".", end="", flush=True)

    handle.remove()

    response_ids = generated_ids[real_prompt_len:]
    return tokenizer.decode(response_ids, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="logs/sft_output/sft_final.pt")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)  # ↑ plus haute pour éviter mode collapse
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"[CHAT] ERREUR: {args.checkpoint} introuvable")
        return

    model = load_model(args.checkpoint)

    print("\n" + "="*60)
    print("  🤖 Chatbot modded-nanogpt (SFT)")
    print("  Tape 'quit' ou 'exit' pour quitter")
    print("="*60 + "\n")

    while True:
        try:
            user_input = input("👤 Toi: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        response = generate(model, user_input, max_new_tokens=args.max_tokens,
                           temperature=args.temperature, top_p=args.top_p,
                           repetition_penalty=args.repetition_penalty)
        print(f"\n🤖 Bot: {response}\n")

    print("\n[CHAT] Au revoir !")


if __name__ == "__main__":
    main()