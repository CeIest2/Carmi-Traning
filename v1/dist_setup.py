import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch.distributed as dist

torch.empty(
    1, device=f"cuda:{os.environ['LOCAL_RANK']}", requires_grad=True
).backward()  # prevents a bug on some systems

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
assert 8 % world_size == 0, "world_size must be a divisor of 8"

grad_accum_steps = int(os.environ.get("GRAD_ACCUM_STEPS", "64")) // world_size
grad_scale = 1 / grad_accum_steps # consistent grad magnitudes between different num_devices
assert torch.cuda.is_available()
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
dist.init_process_group(backend="cuda:nccl,cpu:gloo", device_id=device)
dist.barrier()
master_process = (rank == 0) # this process will do logging, checkpointing etc.
