import copy

from config import args, get_muon_momentum, training_schedule
from dist_setup import grad_accum_steps, rank, world_size
from model import ForwardScheduleConfig
from optimizers import (NorMuonAndAdam, _sparse_comms_active,
                        sparse_comms_share_indexes, sparse_comms_start)

import numpy as np
import torch

class TrainingManager():
    """
    Manages the NorMuonAndAdam for all parameters with explicit ordering.
        1. Scalars are given higher momentum terms to smooth learning @ChrisJMcCormick
        2. Adam optimizers are only stepped on odd steps @classiclarryd
        3. Explicit scatter_order and work_order for communication scheduling (no backward hooks)
        4. Muon has a linear momentum warmup and cooldown schedule
        5. Learning rates follow a linear decay schedule
        6. Embed is tied to lm_head until split step (2/3 of training), then untied @classiclarryd
    """
    def __init__(self, model):
        self.model = model
        self.block_size = 128

        # - Ordering dictates when to launch reduce/reduce_scatter operations
        # - "sharded" parameters use reduce_scatter/all_gather and "replicated" ones use all_reduce
        # - lr_mul and wd_mul are per-parameter learning rate and weight decay multipliers
        self.param_table = {
            "qk_bank":        {"optim": "normuon", "comms": "sharded",        "adam_betas": None},
            "vo_bank":        {"optim": "normuon", "comms": "sharded",        "adam_betas": None},
            "mlp_bank":       {"optim": "normuon", "comms": "sharded",        "adam_betas": None},
            "scalars":        {"optim": "adam",    "comms": "replicated",     "adam_betas": [0.9,  0.99], "lr_mul": 5.0,  "wd_mul": 0.0},
            "smear_gate":     {"optim": "adam",    "comms": "replicated",     "adam_betas": [0.9,  0.99], "lr_mul": 0.01, "wd_mul": 0.0},
            "skip_gate":      {"optim": "adam",    "comms": "replicated",     "adam_betas": [0.9,  0.99], "lr_mul": 0.05, "wd_mul": 0.0},
            "attn_gate_bank": {"optim": "adam",    "comms": "replicated",     "adam_betas": [0.9,  0.99]},
            "ve_gate_bank":   {"optim": "adam",    "comms": "replicated",     "adam_betas": [0.9,  0.99]},
            "lm_head":        {"optim": "adam",    "comms": "sharded",        "adam_betas": [0.5,  0.95], "wd_mul": 30.},
            "bigram_embed":  {"optim": "adam",    "comms": "sharded_sparse", "adam_betas": [0.75, 0.95], "lr_mul": 25.,  "wd_mul": 5.0},
            "post_lambdas":   {"optim": "adam",    "comms": "replicated",     "adam_betas": [0.9,  0.95], "lr_mul": 1.0,  "wd_mul": 0.0},
            "x0_lambdas":     {"optim": "adam",    "comms": "replicated",     "adam_betas": [0.9,  0.95], "lr_mul": 1.0,  "wd_mul": 0.0},
            "bigram_lambdas": {"optim": "adam",    "comms": "replicated",     "adam_betas": [0.9,  0.95], "lr_mul": 1.0,  "wd_mul": 0.0},
            "resid_lambdas":  {"optim": "adam",    "comms": "replicated",     "adam_betas": [0.9,  0.95], "lr_mul": 5.0,  "wd_mul": 0.0},
            "xsa_alphas":     {"optim": "adam",    "comms": "replicated",     "adam_betas": [0.9,  0.95], "lr_mul": 1.0,  "wd_mul": 0.0},
            "value_embeds":   {"optim": "adam",    "comms": "sharded",        "adam_betas": [0.75, 0.95], "lr_mul": 25.,  "wd_mul": 5.0},
            "embed":          {"optim": "adam",    "comms": "sharded",        "adam_betas": [0.5,  0.95], "wd_mul": 30.},
        }

        # ---- MUDD parameter overrides ----
        self.param_table.update({
            "mudd_w1":    {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.99], "lr_mul": 0.25},
            "mudd_w2":    {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.99], "lr_mul": 0.25},
            "mudd_b2":    {"optim": "adam", "comms": "replicated", "adam_betas": [0.9, 0.99], "lr_mul": 0.25, "wd_mul": 0.0},
        })

        # - Process smaller/faster params first while large reduces complete
        # - lm_head must complete before embed sync (when tied)
        self.work_order = [
            "scalars", "smear_gate", "skip_gate", "attn_gate_bank", "ve_gate_bank", "mudd_b2", "xsa_alphas",
            "post_lambdas", "x0_lambdas", "bigram_lambdas", "resid_lambdas",  # Small, fast
        ] + [
            "mudd_w2",
            "value_embeds", "bigram_embed",  # Medium
            "mudd_w1",
            "lm_head", "embed",   # lm_head must complete before embed sync (when tied)
            "qk_bank", "vo_bank", "mlp_bank",  # Large, polar express - process last to maximize overlap
        ]

        adam_defaults = dict(
            lr=0.0015,
            eps=1e-8,
            weight_decay=0.005,
        )

        normuon_defaults = dict(
            lr=0.006,
            momentum=0.95,
            beta2=0.9,
            weight_decay=0.3,
        )

        self.optimizer = NorMuonAndAdam(
            model.named_parameters(),
            param_table=self.param_table,
            scatter_order=list(self.param_table),  # Dict order defines scatter priority
            work_order=self.work_order,
            adam_defaults=adam_defaults,
            normuon_defaults=normuon_defaults,
        )

        # Split embed from lm_head at 2/3 of training (on an odd step so Adam updates)
        self.split_step = training_schedule.split_step

        self.reset()

    def apply_final_ws_ext(self):
        self.ws_long = training_schedule.ws_post_yarn_ext

    def get_forward_args(self):
        return ForwardScheduleConfig(
            mtp_weights = self.mtp_weights,
            ws_short = self.ws_short * self.block_size,
            ws_long = self.ws_long * self.block_size,
            train_max_seq_len = self.train_max_seq_len
        )

    def _is_adam_step(self, step: int):
        """Adam params are only updated on odd steps."""
        return step % 2 == 1

    def get_transition_steps(self):
        return [start for start, _ in training_schedule.boundaries[1:]]

    def advance_schedule(self, step: int):
        stage, _ = training_schedule.lookup(step)
        self.ws_short, new_ws_long = stage.window_sizes
        if new_ws_long != self.ws_long:
            self.model.yarn.apply(self.ws_long * self.block_size, new_ws_long * self.block_size)
            self.model.yarn_paired_head.apply(self.ws_long * self.block_size, new_ws_long * self.block_size)

        new_batch_size = stage.batch_size
        new_train_max_seq_len = stage.train_max_seq_len
        if new_batch_size != self.batch_size or new_train_max_seq_len != self.train_max_seq_len:
            self.train_loader_send_args = (new_batch_size, new_train_max_seq_len, grad_accum_steps)
            self.batch_size = new_batch_size
            self.train_max_seq_len = new_train_max_seq_len
        else:
            self.train_loader_send_args = None

        self.ws_long = new_ws_long
        self.mtp_weights = training_schedule.mtp_weights[step]

    def step_optimizers(self, step: int):
        step_lr = training_schedule.get_lr(step)
        muon_momentum = get_muon_momentum(step)
        do_adam = self._is_adam_step(step)

        # Update learning rates and momentum for all params
        freeze_lambdas = step < 100

        for param, p_cfg in self.optimizer.param_cfgs.items():
            p_cfg.lr = p_cfg.initial_lr * step_lr
            if p_cfg.optim == "normuon":
                p_cfg.momentum = muon_momentum
            
            if freeze_lambdas and p_cfg.label in ("resid_lambdas", "post_lambdas"):
                p_cfg.lr = 0.0

        self.optimizer.step(do_adam=do_adam)

        # At split step: copy lm_head optimizer state to embed and mark as split
        if step == self.split_step:
            self.optimizer.copy_lm_state_to_embed()

    def reset(self, state=None):
        if state is not None:
            self.optimizer.load_state_dict(state)

        # Reset NorMuon momentum buffers and split_embed state
        self.optimizer.reset()

        stage, _ = training_schedule.lookup(0)
        self.ws_short, self.ws_long = stage.window_sizes
        self.batch_size = stage.batch_size
        self.train_max_seq_len = stage.train_max_seq_len
        self.model.yarn.reset()
        self.model.yarn_paired_head.reset()
        if _sparse_comms_active():
            self.row_update_mask = np.zeros(args.bigram_vocab_size, dtype=np.uint8)
            self.sparse_counts_state = None
            # buffer we use for fast GPU uploads of send indexes
            self.send_idxes_buffer = torch.empty(args.bigram_vocab_size, dtype=torch.int32, pin_memory=True)


    def get_state(self):
        return copy.deepcopy(self.optimizer.state_dict())

    def sparse_index_update(self, step, bigram_indexes):
        if not _sparse_comms_active():
            return

        self.row_update_mask[bigram_indexes] = 1

        if self._is_adam_step(step):
            with torch.no_grad():
                bigram_idx_np = np.flatnonzero(self.row_update_mask).astype(np.int32)
                send_idxes, send_counts, recv_counts, recv_counts_fut = sparse_comms_start(
                    bigram_idx_np, args.bigram_vocab_size, rank, world_size, self.send_idxes_buffer
                )
                self.sparse_counts_state = (send_idxes, send_counts, recv_counts, recv_counts_fut)

    def sparse_index_share(self, step):
        if not _sparse_comms_active() or not self._is_adam_step(step):
            return

        send_idxes, send_counts, recv_counts, recv_counts_fut = self.sparse_counts_state
        self.sparse_counts_state = None

        recv_counts_fut.wait()
        recv_idxes, sparse_state, idxes_fut = sparse_comms_share_indexes(send_idxes, send_counts, recv_counts)
        self.optimizer._reduce_futures[self.model.bigram_embed.weight] = [idxes_fut, recv_idxes]
        self.optimizer._sparse_async_data[self.model.bigram_embed.weight] = sparse_state

        self.row_update_mask.fill(0)
