import os
import sys

# Read the code of all the modules in this directory ASAP, for logging
_script_dir = os.path.dirname(os.path.abspath(__file__))
code = ""
for _fname in sorted(f for f in os.listdir(_script_dir) if f.endswith(".py")):
    with open(os.path.join(_script_dir, _fname), "r") as f:
        code += f"\n\n{'-'*40}\n# {_fname}\n{'-'*40}\n\n"
        code += f.read()

import copy
import gc
import time

from dist_setup import grad_accum_steps, grad_scale, master_process, world_size
from config import TRAINING_STAGES, args, training_schedule
from data import distributed_data_generator, get_bigram_hash
from model import GPT
from training import TrainingManager

import torch
import triton
import torch.distributed as dist
from torch import nn

import torch._inductor.config as inductor_config
inductor_config.triton.cudagraphs = False

# -----------------------------------------------------------------------------
# int main

# begin logging
logfile = None

def print0(s, console=False):
    if master_process:
        with open(logfile, "a") as f:
            if console:
                print(s)
            print(s, file=f)

def nvidia_smi():
    import subprocess  # avoid top level import
    return subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout

def main():
    global logfile
    if master_process:
        run_id = args.run_id
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{run_id}.txt"
        print(logfile)
        
    print0(code)
    print0("="*100)
    print0(f"Running Python {sys.version}")
    print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}")
    print0(f"Running Triton version {triton.__version__}")
    print0(nvidia_smi())
    print0("="*100)

    # 1. Initialisation du modèle (world_size = 1)
    _val_seq_len = args.val_batch_size // (grad_accum_steps * world_size)
    _train_seq_len = max(s.train_max_seq_len for s in TRAINING_STAGES)
    max_seq_len = max(
        args.val_batch_size // (grad_accum_steps * world_size),
        max(s.train_max_seq_len for s in TRAINING_STAGES) * 2
    )
    
    model: nn.Module = GPT(
        vocab_size=50257,
        num_layers=11,
        num_heads=6,
        head_dim=128,
        model_dim=768,
        max_seq_len=max_seq_len
    ).cuda()

    # 2. Passage en bfloat16 pour tout le réseau et les banques de paramètres
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

    # 3. FIX MONO-GPU : On n'exécute la synchronisation broadcast QUE si on est en Multi-GPU
    if dist.is_initialized() and world_size > 1:
        for param in model.parameters():
            dist.broadcast(param.detach(), 0)
        dist.broadcast(model.bigram_sign_table, 0)

    # 4. Quantification FP8 des MLP (Avant compilation)
    model.quantize_mlp_fp8()

    # 5. FIX COMPILE : mode="max-autotune" sans fullgraph pour Ada Lovelace (sm_89)
    print0("Compilation PyTorch Inductor (max-autotune sans CUDAGraphs)...")
    model: nn.Module = torch.compile(model, mode="default", dynamic=False)

    # 6. Lancement du Training Manager
    training_manager = TrainingManager(model)

    ########################################
    #            Warmup kernels            #
    ########################################
    print0("Compiling model and warming up kernels (~7 minutes on first execution)", console=True)
    # Warmup the training kernels, then re-initialize the state so we aren't cheating
    initial_state = dict(model=copy.deepcopy(model.state_dict()),
                         optimizer=training_manager.get_state()) # save the initial state
    train_loader = distributed_data_generator(args.train_files, TRAINING_STAGES[0].batch_size, TRAINING_STAGES[0].train_max_seq_len, grad_accum_steps=grad_accum_steps)
    val_loader = distributed_data_generator(args.val_files, args.val_batch_size, -1, grad_accum_steps=grad_accum_steps, align_to_bos=False)

    transition_steps = training_manager.get_transition_steps()
    # first and last pair of steps in each transition
    warmup_steps = sorted({0, 1} | {s + offset for s in transition_steps for offset in [-2, -1, 0, 1] if s + offset >= 2})
    print0(f"Sampling steps {warmup_steps} for warmup", console=True)
    for step in warmup_steps:
        torch.compiler.cudagraph_mark_step_begin()
        training_manager.advance_schedule(step)
        model.eval()
        with torch.no_grad():
            inputs, targets, cum_seqlens, bigram_inputs, _ = next(val_loader)
            model(inputs, targets, cum_seqlens, bigram_inputs, training_manager.get_forward_args()).mean()
        model.train()
        for idx in range(grad_accum_steps):
            send_args = training_manager.train_loader_send_args
            inputs, targets, cum_seqlens, bigram_inputs, bigram_cpu = train_loader.send(send_args)
            training_manager.sparse_index_update(step, bigram_cpu)
            loss = model(inputs, targets, cum_seqlens, bigram_inputs, training_manager.get_forward_args()).sum() * grad_scale
            training_manager.sparse_index_share(step)
            loss.backward()
            del loss

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        training_manager.step_optimizers(step)
        model.quantize_mlp_fp8()
    print0("Resetting Model", console=True)
    model.zero_grad(set_to_none=True)
    model.load_state_dict(initial_state["model"])
    training_manager.reset(initial_state["optimizer"])
    del val_loader, train_loader, initial_state
    model.quantize_mlp_fp8()
    model.train()

    # ---------------------------------------------
    # PATCH (RTX 4060 Ti) : reprise sur checkpoint si RESUME_CKPT est defini.
    # Le warmup ci-dessus reste execute (compilation/verification des kernels),
    # puis l'etat du checkpoint ecrase l'etat initial.
    start_step = 0
    resume_loader_state = None
    if args.resume_path:
        print0(f"Resuming from checkpoint: {args.resume_path}", console=True)
        ckpt = torch.load(args.resume_path, map_location="cpu")
        assert ckpt["grad_accum_steps"] == grad_accum_steps, (
            f"GRAD_ACCUM_STEPS different entre le checkpoint ({ckpt['grad_accum_steps']}) et ce run "
            f"({grad_accum_steps}) : la position dans les donnees ne serait plus valide.")
        model.load_state_dict(ckpt["model"])
        # L'optimizer state du checkpoint est indexe par NOM de parametre : les id()
        # de NorMuonAndAdam.state_dict() ne survivent pas a un nouveau processus.
        name_to_param = dict(model.named_parameters())
        optimizer = training_manager.optimizer
        for name, saved_state in ckpt["optimizer"].items():
            p_state = optimizer.param_states[name_to_param[name]]
            for k, v in saved_state.items():
                if isinstance(v, torch.Tensor) and k in p_state:
                    p_state[k] = v.to(dtype=p_state[k].dtype, device=p_state[k].device)
                else:
                    p_state[k] = v
        optimizer.split_embed = ckpt["split_embed"]
        torch.set_rng_state(ckpt["rng_cpu"])
        torch.cuda.set_rng_state(ckpt["rng_cuda"])
        start_step = ckpt["step"]
        resume_loader_state = ckpt["loader"]
        model.quantize_mlp_fp8()
        print0(f"Resumed at step {start_step}/{training_schedule.total_steps}", console=True)

    ########################################
    #        Training and validation       #
    ########################################
    train_loader = distributed_data_generator(args.train_files, TRAINING_STAGES[0].batch_size, TRAINING_STAGES[0].train_max_seq_len, grad_accum_steps=grad_accum_steps)

    if resume_loader_state is not None:
        train_loader.set_state(resume_loader_state)
        # Rejoue le schedule pas a pas jusqu'au step de reprise : reproduit a l'identique
        # les transitions YaRN (fenetres glissantes) et les changements de batch size.
        for _step in range(start_step):
            training_manager.advance_schedule(_step)

    def _save_checkpoint(next_step: int):
        """Sauvegarde atomique (.tmp puis rename) pour survivre a une coupure pendant l'ecriture."""
        optimizer = training_manager.optimizer
        id_to_name = {id(p): name for name, p in model.named_parameters()}
        payload = dict(
            step=next_step,
            grad_accum_steps=grad_accum_steps,
            model=model.state_dict(),
            optimizer={id_to_name[id(p)]: state for p, state in optimizer.param_states.items()},
            split_embed=optimizer.split_embed,
            loader=train_loader.get_state(),
            rng_cpu=torch.get_rng_state(),
            rng_cuda=torch.cuda.get_rng_state(),
        )
        os.makedirs(os.path.dirname(args.ckpt_path), exist_ok=True)
        tmp_path = args.ckpt_path + ".tmp"
        torch.save(payload, tmp_path)
        os.replace(tmp_path, args.ckpt_path)
        print0(f"checkpoint saved: {args.ckpt_path} (next step {next_step})", console=True)

    gc.collect()

    training_time_ms = 0
    # start the clock
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    # begin training
    train_steps = training_schedule.total_steps
    for step in range(start_step, train_steps + 1):
        training_manager.advance_schedule(step)
        last_step = (step == train_steps)
        training_manager.advance_schedule(step)
        # --------------- VALIDATION SECTION -----------------
        if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            if last_step:
                training_manager.apply_final_ws_ext()
            # stop the clock
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.perf_counter() - t0)
            model.eval()
            assert args.val_tokens % args.val_batch_size == 0
            val_steps = grad_accum_steps * args.val_tokens // args.val_batch_size
            val_loader = distributed_data_generator(args.val_files, args.val_batch_size, -1, grad_accum_steps=grad_accum_steps, align_to_bos=False)
            val_loss = 0
            with torch.no_grad():
                for _ in range(val_steps):
                    inputs, targets, cum_seqlens, bigram_inputs, _ = next(val_loader)
                    val_loss += model(inputs, targets, cum_seqlens, bigram_inputs, training_manager.get_forward_args()).mean()
            val_loss /= val_steps
            del val_loader
            dist.reduce(val_loss, 0, op=dist.ReduceOp.AVG)
            print0(f"step:{step}/{train_steps} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step, 1):.2f}ms", console=True)
            model.train()
            # start the clock again
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if master_process and args.save_checkpoint:
                log = dict(step=step, code=code, model=model.state_dict(), optimizer=training_manager.get_state())
                os.makedirs(f"logs/{run_id}", exist_ok=True)
                torch.save(log, f"logs/{run_id}/state_step{step:06d}.pt")
            # the last step only has the validation loop, so break to avoid training
            break

        # --------------- TRAINING SECTION -----------------
        for idx in range(grad_accum_steps):
            inputs, targets, cum_seqlens, bigram_inputs, bigram_cpu = train_loader.send(training_manager.train_loader_send_args)
            training_manager.sparse_index_update(step, bigram_cpu)
            loss = model(inputs, targets, cum_seqlens, bigram_inputs, training_manager.get_forward_args()).sum() * grad_scale
            training_manager.sparse_index_share(step)
            loss.backward()


            if step % 10 == 0:
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.norm().item() ** 2
                print0(f"step {step} grad_norm: {total_norm**0.5:.2f}")
                
                # Logger les lambdas critiques
                print0(f"resid_lambdas max: {model.resid_lambdas.max().item():.3f}")
                print0(f"post_lambdas: {model.post_lambdas.data}")  



            del loss
        training_manager.step_optimizers(step)
        model.quantize_mlp_fp8()

        # PATCH : checkpoint periodique (permet d'interrompre/reprendre le run)
        if master_process and args.ckpt_every > 0 and (step + 1) % args.ckpt_every == 0:
            _save_checkpoint(step + 1)

        # logging
        approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
        print0(f"step:{step+1}/{train_steps} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms/(step + 1):.2f}ms", console=True)

    if args.run_evals:
        model.eval()
        from evals import hellaswag
        hellaswag.evaluate(model=model, 
                           schedule_cfg=training_manager.get_forward_args(), 
                           seq_len=args.val_batch_size // (grad_accum_steps * world_size),
                           get_bigram_hash=get_bigram_hash, 
                           print0=print0)

    print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
           f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", console=True)
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
