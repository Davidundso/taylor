import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from curvlinops import GGNLinearOperator  # Assuming curvlinops is installed
import os

# Set this globally
reduction = "mean"

# Your model, loss, and data
def model_fn():
    return torch.nn.Sequential(
        torch.nn.Linear(10, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 1)
    )

def loss_fn():
    return torch.nn.MSELoss(reduction='mean')

# Dummy data (you can replace this with your real batch)
def get_data():
    x = torch.randn(64, 10)
    y = torch.randn(64, 1)
    return x, y

# This class wraps the GGN in a distributed way
class DistributedGGN:
    def __init__(self, ggn_op):
        self.ggn_op = ggn_op

    def __matmul__(self, v):
        local_result = self.ggn_op @ v
        dist.all_reduce(local_result)  # Sum across GPUs
        if reduction == 'mean':
            local_result /= dist.get_world_size()
        return local_result

    def matvec(self, v):
        return self.__matmul__(v)

# === Main DDP logic ===
def ddp_worker(rank, world_size, v, single_gpu_result):
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)

    # Setup model and data
    model = model_fn().to(rank)
    loss = loss_fn()
    x, y = get_data()
    x = x.to(rank)
    y = y.to(rank)
    model = DDP(model, device_ids=[rank])
    params = list(model.module.parameters())

    # Build GGN operator
    ggn_op = GGNLinearOperator(model, loss, params, (x, y))
    dist_ggn = DistributedGGN(ggn_op)

    # Broadcast vector to all processes
    v = v.to(rank)
    dist.broadcast(v, src=0)

    result = dist_ggn @ v

    # Only rank 0 compares
    if rank == 0:
        match = torch.allclose(result.cpu(), single_gpu_result, atol=1e-6)
        print(f"Distributed result matches single-GPU: {match}")
        if not match:
            print("Difference:", (result.cpu() - single_gpu_result).norm())

    dist.destroy_process_group()

# === Main entry ===
def main():
    # Single-GPU run
    model = model_fn().cuda(0)
    loss = loss_fn()
    x, y = get_data()
    x = x.cuda(0)
    y = y.cuda(0)
    params = list(model.parameters())
    ggn_op = GGNLinearOperator(model, loss, params, (x, y))

    # Test vector (randomly generated)
    v = torch.randn_like(torch.cat([p.view(-1) for p in params]))

    # Run single-GPU computation
    result_single = ggn_op @ v.cuda(0)
    result_single = result_single.cpu()

    # Launch multi-GPU DDP
    world_size = torch.cuda.device_count()
    mp.spawn(ddp_worker, args=(world_size, v, result_single), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
