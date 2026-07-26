"""
SFT (Supervised Fine-Tuning) pour modded-nanogpt sur RTX 4060 Ti 16GB.
Charge le checkpoint de pretraining et fine-tune sur UltraChat avec masking des instructions.
"""

import os
import sys
import math
import time
import glob

os.environ["HF_HOME"] = "/tmp/huggingface_cache"
os.environ["HF_DATASETS_CACHE"] = "/tmp/huggingface_cache/datasets"
os.makedirs("/tmp/huggingface_cache", exist_ok=True)
os.makedirs("/tmp/huggingface_cache/datasets", exist_ok=True)
os.environ["DISABLE_FP8"] = "1"

device = "cuda"

import types
import torch
torch.cuda.init()
import torch.distributed as dist

_fake_dist_setup = types.ModuleType("dist_setup")
_fake_dist_setup.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_fake_dist_setup.grad_accum_steps = 16
_fake_dist_setup.grad_scale = 1.0 / 16
_fake_dist_setup.master_process = True
_fake_dist_setup.world_size = 1
_fake_dist_setup.rank = 0

sys.modules["dist_setup"] = _fake_dist_setup

if not dist.is_initialized():
    dist.init_process_group(
        backend="gloo",
        init_method="tcp://localhost:29500",
        rank=0,
        world_size=1,
    )

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import GPT, ForwardScheduleConfig, next_multiple_of_n
from datasets import load_dataset
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Configuration SFT
SFT_CONFIG = {
    "checkpoint_path": "logs/pretrain_edu_final.pt",  
    "output_dir": "logs/sft_output",
    "dataset": "HuggingFaceH4/ultrachat_200k",
    "max_seq_len": 1024,
    "batch_size": 1,
    "grad_accum_steps": 16,
    "learning_rate": 1e-5,
    "weight_decay": 0.01,
    "warmup_steps": 200,
    "max_steps": 5000,
    "eval_every": 500,     
    "save_every": 1000,
    "max_grad_norm": 2.0,   
}


# -----------------------------------------------------------------------------
class SimpleTokenizer:
    """Tokenizer minimal compatible GPT-2 vocabulaire."""
    def __init__(self):
        from transformers import GPT2Tokenizer
        os.environ["HF_HOME"] = "/tmp/huggingface_cache"
        os.makedirs("/tmp/huggingface_cache", exist_ok=True)
        self._tok = GPT2Tokenizer.from_pretrained("gpt2")
        self._tok.pad_token = self._tok.eos_token
        self.eos_token = self._tok.eos_token
        self.eos_token_id = self._tok.eos_token_id

    def encode(self, text, add_special_tokens=False, truncation=False, max_length=None):
        return self._tok.encode(
            text,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            max_length=max_length
        )

    def __call__(self, *args, **kwargs):
        return self._tok(*args, **kwargs)

tokenizer = SimpleTokenizer()


# -----------------------------------------------------------------------------
def format_example(example):
    messages = example.get("messages", [])
    user_msg = None
    assistant_msg = None

    for msg in messages:
        if msg.get("role") == "user" and user_msg is None:
            user_msg = msg.get("content", "").strip()
        elif msg.get("role") == "assistant" and assistant_msg is None:
            assistant_msg = msg.get("content", "").strip()
            break

    if user_msg is None:
        user_msg = ""
    if assistant_msg is None:
        assistant_msg = ""

    # Format IDENTIQUE au chat.py de test
    prompt = f"User: {user_msg}\nAssistant:"
    response = f" {assistant_msg}" + tokenizer.eos_token  # espace initial = token Ġ

    return {"prompt": prompt, "response": response}


# -----------------------------------------------------------------------------
def load_model(checkpoint_path: str):
    print(f"[SFT] Chargement du checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    model = GPT(
        vocab_size=50257,
        num_layers=11,
        num_heads=6,
        head_dim=128,
        model_dim=768,
        max_seq_len=4096,
    )

    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_state_dict[k[len("_orig_mod."):]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model = model.to("cuda").train()

    import torch.nn as nn
    for m in model.modules():
        if isinstance(m, (nn.Embedding, nn.Linear)):
            m.weight.data = m.weight.data.bfloat16()

    model.attn_gate_bank.data = model.attn_gate_bank.data.bfloat16()
    model.ve_gate_bank.data = model.ve_gate_bank.data.bfloat16()
    model.qk_bank.data = model.qk_bank.data.bfloat16()
    model.vo_bank.data = model.vo_bank.data.bfloat16()
    model.mlp_bank.data = model.mlp_bank.data.bfloat16()
    model.mudd_w1.data = model.mudd_w1.data.bfloat16()
    model.mudd_w2.data = model.mudd_w2.data.bfloat16()
    model.mudd_b2.data = model.mudd_b2.data.bfloat16()

    # Full fine-tune (tout dégelé)
    for param in model.parameters():
        param.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[SFT] Paramètres totaux: {total_params:,}")
    print(f"[SFT] Paramètres entraînables: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")

    return model


# -----------------------------------------------------------------------------
def get_lr(step: int):
    if step < SFT_CONFIG["warmup_steps"]:
        return SFT_CONFIG["learning_rate"] * (step + 1) / SFT_CONFIG["warmup_steps"]
    progress = (step - SFT_CONFIG["warmup_steps"]) / (SFT_CONFIG["max_steps"] - SFT_CONFIG["warmup_steps"])
    return SFT_CONFIG["learning_rate"] * 0.5 * (1.0 + math.cos(math.pi * progress))


# -----------------------------------------------------------------------------
def train_step(model, optimizer, input_ids, prompt_len, real_len, schedule_cfg):
    target = input_ids.clone()
    seqlens = torch.tensor([0, input_ids.size(0)], dtype=torch.int32, device=device)
    bigram_input = input_ids.clone()

    # Forward
    loss_per_token = model(
        input_seq=input_ids,
        target_seq=target,
        seqlens=seqlens,
        bigram_input_seq=bigram_input,
        schedule_cfg=schedule_cfg,
    )

    # Masquer le prompt (ne pas apprendre à prédire l'instruction)
    mask = torch.ones_like(loss_per_token)
    if prompt_len > 0:
        mask[:prompt_len] = 0.0

    # Masquer le padding (tokens eos ajoutés en fin pour alignement 16)
    if real_len < len(input_ids):
        mask[real_len:] = 0.0

    loss = (loss_per_token * mask).sum() / mask.sum().clamp_min(1.0)

    loss.backward()
    return loss.item()


# -----------------------------------------------------------------------------
def eval_step(model, val_examples, schedule_cfg):
    """Évalue la loss sur un petit batch de validation (sans gradient)."""
    model.eval()
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for example in val_examples:
            prompt_ids = tokenizer.encode(example["prompt"], add_special_tokens=False, truncation=True, max_length=SFT_CONFIG["max_seq_len"])
            response_ids = tokenizer.encode(example["response"], add_special_tokens=False, truncation=True, max_length=SFT_CONFIG["max_seq_len"])

            combined = prompt_ids + response_ids
            pad_len = (16 - len(combined) % 16) % 16
            if pad_len:
                combined = combined + [tokenizer.eos_token_id] * pad_len

            prompt_len = len(prompt_ids)
            real_len = len(prompt_ids) + len(response_ids)
            if prompt_len >= len(combined):
                prompt_len = len(combined) - 1

            input_ids = torch.tensor(combined, dtype=torch.long, device=device)
            target = input_ids.clone()
            seqlens = torch.tensor([0, input_ids.size(0)], dtype=torch.int32, device=device)
            bigram_input = input_ids.clone()

            loss_per_token = model(
                input_seq=input_ids,
                target_seq=target,
                seqlens=seqlens,
                bigram_input_seq=bigram_input,
                schedule_cfg=schedule_cfg,
            )

            mask = torch.ones_like(loss_per_token)
            if prompt_len > 0:
                mask[:prompt_len] = 0.0
            if real_len < len(input_ids):
                mask[real_len:] = 0.0

            loss = (loss_per_token * mask).sum() / mask.sum().clamp_min(1.0)
            total_loss += loss.item()
            count += 1

    model.train()
    return total_loss / count if count > 0 else 0.0


# -----------------------------------------------------------------------------
def main():
    os.makedirs(SFT_CONFIG["output_dir"], exist_ok=True)

    # 1. Modèle
    model = load_model(SFT_CONFIG["checkpoint_path"])

    # 2. Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=SFT_CONFIG["learning_rate"],
        weight_decay=SFT_CONFIG["weight_decay"],
        betas=(0.9, 0.99),
    )

    # 3. Reprise sur checkpoint SFT si disponible
    start_step = 0
    resume_ckpt = os.path.join(SFT_CONFIG["output_dir"], "sft_resume.pt")
    if os.path.exists(resume_ckpt):
        print(f"[SFT] Reprise du checkpoint SFT: {resume_ckpt}")
        ckpt = torch.load(resume_ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        print(f"[SFT] Repris au step {start_step}")

    # 4. Dataset
    print(f"[SFT] Chargement du dataset {SFT_CONFIG['dataset']}...")
    ds = load_dataset(SFT_CONFIG["dataset"], split="train_sft")
    ds = ds.map(format_example, remove_columns=ds.column_names)
    print(f"[SFT] {len(ds)} exemples avant filtrage")

    def filter_by_length(example):
        prompt_ids = tokenizer.encode(example["prompt"], add_special_tokens=False)
        response_ids = tokenizer.encode(example["response"], add_special_tokens=False)
        total = len(prompt_ids) + len(response_ids)
        return total <= SFT_CONFIG["max_seq_len"] and len(response_ids) >= 30

    ds = ds.filter(filter_by_length)
    print(f"[SFT] {len(ds)} exemples après filtrage (prompt+response <= 1024)")

    # Split train/val (5% validation pour surveiller l'overfitting)
    ds = ds.train_test_split(test_size=0.05, seed=42)
    train_ds = ds["train"]
    val_ds = ds["test"]
    print(f"[SFT] {len(train_ds)} train | {len(val_ds)} val")

    # 5. Schedule cfg fixe
    schedule_cfg = ForwardScheduleConfig(
        mtp_weights=torch.tensor([1.0], device=device),
        ws_short=4 * 128,
        ws_long=32 * 128,
        train_max_seq_len=SFT_CONFIG["max_seq_len"],
    )

    # 6. Training loop
    print(f"[SFT] Début du fine-tuning (max {SFT_CONFIG['max_steps']} steps)...")
    model.train()

    # DIAGNOSTIC
    print("\n[SFT] === DIAGNOSTIC : 3 premiers exemples ===")
    for idx in range(3):
        ex = train_ds[idx]
        prompt_ids_diag = tokenizer.encode(ex["prompt"], add_special_tokens=False, truncation=True, max_length=64)
        response_ids_diag = tokenizer.encode(ex["response"], add_special_tokens=False, truncation=True, max_length=64)
        print(f"  Exemple {idx}:")
        print(f"    Prompt: {ex['prompt'][:100]}...")
        print(f"    Response: {ex['response'][:100]}...")
        print(f"    Prompt tokens: {len(prompt_ids_diag)} | Response tokens: {len(response_ids_diag)}")
    print("[SFT] ========================================\n")

    global_step = start_step
    accum_loss = 0.0
    step_start = time.perf_counter()
    best_val_loss = float("inf")
    steps_since_improvement = 0

    for epoch in range(3):
        for i, example in enumerate(train_ds):
            if global_step >= SFT_CONFIG["max_steps"]:
                break

            # Tokenize
            prompt_ids = tokenizer.encode(
                example["prompt"], add_special_tokens=False,
                truncation=True, max_length=SFT_CONFIG["max_seq_len"]
            )
            response_ids = tokenizer.encode(
                example["response"], add_special_tokens=False,
                truncation=True, max_length=SFT_CONFIG["max_seq_len"]
            )

            # Pad à un multiple de 16 supérieur avec eos
            combined = prompt_ids + response_ids
            pad_len = (16 - len(combined) % 16) % 16
            if pad_len:
                combined = combined + [tokenizer.eos_token_id] * pad_len

            prompt_len = len(prompt_ids)
            real_len = len(prompt_ids) + len(response_ids)

            # Sécurité : s'assurer qu'il reste au moins 1 token de réponse
            if prompt_len >= len(combined):
                prompt_len = len(combined) - 1

            input_ids = torch.tensor(combined, dtype=torch.long, device=device)

            # Forward + backward
            loss_val = train_step(model, optimizer, input_ids, prompt_len, real_len, schedule_cfg)
            accum_loss += loss_val

            # Gradient accumulation
            if (i + 1) % SFT_CONFIG["grad_accum_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    SFT_CONFIG["max_grad_norm"]
                )
                optimizer.step()
                optimizer.zero_grad()

                global_step += 1

                # Update LR
                lr = get_lr(global_step)
                for g in optimizer.param_groups:
                    g["lr"] = lr

                # Log
                avg_loss = accum_loss / SFT_CONFIG["grad_accum_steps"]
                accum_loss = 0.0

                if global_step % 10 == 0:
                    elapsed = time.perf_counter() - step_start
                    print(f"[SFT] step {global_step:04d}/{SFT_CONFIG['max_steps']} | "
                          f"loss: {avg_loss:.4f} | lr: {lr:.2e} | time: {elapsed:.0f}s")

                # Évaluation + early stop
                if global_step % SFT_CONFIG["eval_every"] == 0:
                    val_loss = eval_step(model, val_ds.select(range(min(50, len(val_ds)))), schedule_cfg)
                    print(f"[SFT] === Val step {global_step} | train_loss: {avg_loss:.4f} | val_loss: {val_loss:.4f} ===")

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        steps_since_improvement = 0
                        # Sauvegarde le meilleur checkpoint
                        ckpt_path = os.path.join(SFT_CONFIG["output_dir"], "sft_best.pt")
                        torch.save({
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "step": global_step,
                        }, ckpt_path)
                        print(f"[SFT] 🏆 Nouveau meilleur modèle: {ckpt_path}")
                    else:
                        steps_since_improvement += SFT_CONFIG["eval_every"]
                        if steps_since_improvement >= 1000:  # patience de 1000 steps
                            print(f"[SFT] ⚠️ Early stop: val loss stagnante (best: {best_val_loss:.4f}, current: {val_loss:.4f})")
                            global_step = SFT_CONFIG["max_steps"]
                            break

                # Sauvegarde périodique + checkpoint de reprise
                if global_step % SFT_CONFIG["save_every"] == 0:
                    ckpt_path = os.path.join(SFT_CONFIG["output_dir"], f"sft_step{global_step:05d}.pt")
                    torch.save({
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": global_step,
                    }, ckpt_path)
                    print(f"[SFT] Sauvegardé: {ckpt_path}")

                    # Checkpoint de reprise (overwrite)
                    torch.save({
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": global_step,
                    }, resume_ckpt)

                # Garde-fou absolu
                if avg_loss < 0.10:
                    print(f"[SFT] 🚨 Overfitting détecté (loss {avg_loss:.4f}), arrêt immédiat")
                    global_step = SFT_CONFIG["max_steps"]
                    break

    # Sauvegarde finale
    final_path = os.path.join(SFT_CONFIG["output_dir"], "sft_final.pt")
    torch.save({"model": model.state_dict()}, final_path)
    print(f"[SFT] ✅ Modèle final sauvegardé: {final_path}")

    # Nettoie le checkpoint de reprise
    if os.path.exists(resume_ckpt):
        os.remove(resume_ckpt)


if __name__ == "__main__":
    main()