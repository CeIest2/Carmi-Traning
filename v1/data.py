import glob
import threading
from pathlib import Path

from config import args
from model import next_multiple_of_n

import torch
import torch.distributed as dist
from torch import Tensor

# -----------------------------------------------------------------------------
# Distributed data loader

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True) # avoid pin_memory copy by @YouJiacheng
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy by @YouJiacheng
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

BOS_ID = 50256
TRAIN_MAX_NUM_DOCS = {16384: 64, 32768: 96, 49152: 128}

class Shard:
    def __init__(self, tokens: Tensor, world_size: int = 1):
        self.tokens = tokens
        self.size = tokens.numel()
        self.world_size = world_size
        self.i = 0

        # Partial index now, full index async
        self.bos_idx = (tokens[:6_000_000] == BOS_ID).nonzero(as_tuple=True)[0].to(torch.int64).cpu().numpy()
        self._full_idx = None
        self._loader_thread = None
        self._ready = threading.Event()
        self._loader_thread = threading.Thread(target=self._scan)
        self._loader_thread.start()

    def _scan(self):
        self._full_idx = (self.tokens == BOS_ID).nonzero(as_tuple=True)[0].to(torch.int64).cpu().numpy()
        self._ready.set()

    def _maybe_switch(self):
        # Switch to full index as soon as async scan completes
        if self.bos_idx is not self._full_idx and self._ready.is_set():
            self._loader_thread.join()
            self.bos_idx = self._full_idx

    def next_batch(self, num_tokens_local: int, max_seq_len: int):
        self._maybe_switch()
        n = len(self.bos_idx)
        starts = [[] for _ in range(self.world_size)]
        ends = [[] for _ in range(self.world_size)]

        idx = self.i
        for r in range(self.world_size):
            cur_len = 0
            while cur_len <= num_tokens_local:
                if idx >= n:
                    raise StopIteration(f"Insufficient BOS ahead; hit tail of shard.")
                cur = self.bos_idx[idx]
                starts[r].append(cur)
                idx += 1
                end = min(self.bos_idx[idx] if idx < n else self.size,
                          cur + max_seq_len,
                          cur + num_tokens_local - cur_len + 1)
                ends[r].append(end)
                cur_len += end - cur

            assert cur_len == num_tokens_local + 1
        self.i = idx
        return starts, ends

    @staticmethod
    def load_async(file: Path, world_size: int = 1):
        """Returns getter function for async shard loading"""
        result = {}
        ready = threading.Event()
        def load():
            tokens = _load_data_shard(file)
            result['shard'] = Shard(tokens, world_size)
            ready.set()
        thread = threading.Thread(target=load)
        thread.start()
        def get():
            ready.wait()
            thread.join()
            return result['shard']
        return get

def get_bigram_hash(x):
    """
    Computes bigram hash for each position using [prev_token, curr_token].
    Multiply by arbitary large ints to get even spread over int32 range.
    Position 0 is mapped to the reserved index (vocab_size - 1).
    BOS_tokens within the batch will hash based on last token of prior doc. Masking this ran slower and showed no improvement.
    """
    rand_int_1 = 36313
    rand_int_2 = 27191
    mod = args.bigram_vocab_size-1
    x = x.to(torch.int32)
    out = torch.empty_like(x, pin_memory=True)
    out.copy_(x)
    out[0] = mod
    out[1:] = torch.bitwise_xor(rand_int_1 * out[1:], rand_int_2 * out[:-1]) % mod
    return out

class DistributedDataGenerator:
    """Data loader iterateur. Meme interface que l'ancien generateur (next()/send()),
    mais avec get_state()/set_state() pour la reprise sur checkpoint.
    send(params) applique d'abord les nouveaux (num_tokens, max_seq_len, grad_accum),
    puis produit le batch suivant (semantique identique au .send() du generateur)."""

    def __init__(self, filename_pattern: str, num_tokens: int, max_seq_len: int, grad_accum_steps: int = 1, align_to_bos: bool = True):
        # align_to_bos: each sequence begins with Beginning of Sequence token, sequences truncated to max_seq_len
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        assert num_tokens % (self.world_size * grad_accum_steps) == 0, "Batch size must be divisible by world size"
        self.num_tokens = num_tokens // grad_accum_steps
        self.max_seq_len = max_seq_len
        self.align_to_bos = align_to_bos

        self.files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {filename_pattern}")

        self.cur_file_idx = 0
        self.next_file_idx = 1  # prochain fichier a (pre)charger
        tokens = _load_data_shard(self.files[0])
        if align_to_bos:
            self.shard = Shard(tokens, self.world_size)
            self.next_shard_getter = Shard.load_async(self.files[1], self.world_size) if len(self.files) > 1 else None
        else:
            self.tokens = tokens
            self.pos = 0  # for unaligned case

    def __next__(self):
        return self._produce_batch()

    def send(self, new_params):
        if new_params is not None:
            # makes it possible to receive new (num_tokens, max_seq_len, grad_accum_steps) via .send()
            new_num_tokens, new_max_seq_len, new_grad_accum_steps = new_params
            assert new_num_tokens % (self.world_size * new_grad_accum_steps) == 0, "Num tokens must be divisible by world size"
            self.num_tokens = new_num_tokens // new_grad_accum_steps
            self.max_seq_len = new_max_seq_len
        return self._produce_batch()

    def get_state(self):
        """Position exacte dans le stream de donnees (shard courant + index du prochain document)."""
        assert self.align_to_bos, "get_state n'est implemente que pour le train loader"
        return {"file_idx": self.cur_file_idx, "doc_idx": self.shard.i}

    def set_state(self, state: dict):
        """Recharge le shard sauvegarde et saute directement au prochain document a consommer."""
        assert self.align_to_bos, "set_state n'est implemente que pour le train loader"
        file_idx = state["file_idx"]
        tokens = _load_data_shard(self.files[file_idx])
        self.shard = Shard(tokens, self.world_size)
        # attendre l'index BOS complet (scan async) avant le seek : l'index partiel
        # (premiers 6M tokens) pourrait contenir moins de documents que doc_idx
        self.shard._ready.wait()
        self.shard._maybe_switch()
        self.shard.i = state["doc_idx"]
        self.cur_file_idx = file_idx
        self.next_file_idx = file_idx + 1
        self.next_shard_getter = (
            Shard.load_async(self.files[self.next_file_idx], self.world_size)
            if self.next_file_idx < len(self.files) else None
        )

    def _advance_shard(self):
        if self.next_shard_getter is None:
            # Dataset epuise : on reboucle sur le premier shard (nouvelle epoch)
            self.next_shard_getter = Shard.load_async(self.files[0], self.world_size)
            self.next_file_idx = 0
        self.shard = self.next_shard_getter()
        self.cur_file_idx = self.next_file_idx
        self.next_file_idx = self.cur_file_idx + 1
        self.next_shard_getter = (
            Shard.load_async(self.files[self.next_file_idx], self.world_size)
            if self.next_file_idx < len(self.files) else None
        )

    def _produce_batch(self):
        num_tokens_local = self.num_tokens // self.world_size
        max_num_docs = TRAIN_MAX_NUM_DOCS.get(num_tokens_local, next_multiple_of_n(num_tokens_local // 300, n=128))

        if self.align_to_bos:
            while True:
                try:
                    seq_starts, seq_ends = self.shard.next_batch(num_tokens_local, self.max_seq_len)
                    start_idxs, end_idxs = torch.tensor(seq_starts[self.rank]), torch.tensor(seq_ends[self.rank])
                    break
                except StopIteration:
                    # This shard is exhausted, load the next one and retry.
                    self._advance_shard()
            tokens = self.shard.tokens
            buf = torch.cat([tokens[i:j] for i, j in zip(start_idxs, end_idxs)])
            _inputs = buf[:-1]
            _targets = buf[1:]
            end_idxs[-1] -= 1  # last document was too long to account for _targets offset
            cum_lengths = (end_idxs - start_idxs).cumsum(0)

        else:
            if self.pos + self.num_tokens + 1 >= len(self.tokens):  # should not occur for val data
                self.tokens, self.pos = _load_data_shard(self.files[self.next_file_idx]), 0
                self.next_file_idx += 1

            pos_local = self.pos + self.rank * num_tokens_local
            buf = self.tokens[pos_local: pos_local + num_tokens_local + 1]
            _inputs = buf[:-1].view(num_tokens_local, )
            _targets = buf[1:].view(num_tokens_local, )

            cum_lengths = torch.nonzero(_inputs == BOS_ID)[:, 0]
            self.pos += self.num_tokens

        _cum_lengths = torch.full((max_num_docs,), num_tokens_local)
        _cum_lengths[0] = 0
        _cum_lengths[1:len(cum_lengths) + 1] = cum_lengths

        # Cast to int32 on CPU before transfer to avoid dtype conversion during .to()
        _inputs = _inputs.to(dtype=torch.int32)
        _targets = _targets.to(dtype=torch.int64)
        _cum_lengths = _cum_lengths.to(dtype=torch.int32)
        _bigram_inputs = get_bigram_hash(_inputs)

        return (
            _inputs.to(device="cuda", non_blocking=True),
            _targets.to(device="cuda", non_blocking=True),
            _cum_lengths.to(device="cuda", non_blocking=True),
            _bigram_inputs.to(device="cuda", non_blocking=True),
            _bigram_inputs.numpy(),
        )

def distributed_data_generator(filename_pattern: str, num_tokens: int, max_seq_len: int, grad_accum_steps: int = 1, align_to_bos: bool = True):
    return DistributedDataGenerator(filename_pattern, num_tokens, max_seq_len, grad_accum_steps, align_to_bos)
