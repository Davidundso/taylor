import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from curvlinops import GGNLinearOperator
import os

reduction = "mean"

# defining a simple model, loss function, and data generation
def model_fn():
    return torch.nn.Sequential(
        torch.nn.Linear(10, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 1)
    )

def loss_fn():
    return torch.nn.MSELoss(reduction='mean')

def get_data():
    x = torch.randn(64, 10)
    y = torch.randn(64, 1)
    return x, y

# our custom distributed GGN operator
class DistributedGGN:
    def __init__(self, ggn_op):
        self.ggn_op = ggn_op

    def __matmul__(self, v):
        local_result = self.ggn_op @ v
        dist.all_reduce(local_result)
        if reduction == 'mean':
            local_result /= dist.get_world_size()
        return local_result

    def matvec(self, v):
        return self.__matmul__(v)

# Distributed worker function: We call this after our single GPU computation is done
def ddp_worker(rank, world_size, v, initial_state_dict, result_single):
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])

    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    model = model_fn().to(local_rank)
    model.load_state_dict(initial_state_dict)
    loss = loss_fn()

    # Load full data and shard by rank
    x_full, y_full = get_data()
    share_size = x_full.size(0) // world_size
    x = x_full[rank * share_size:(rank + 1) * share_size].to(local_rank)
    y = y_full[rank * share_size:(rank + 1) * share_size].to(local_rank)

    model = DDP(model, device_ids=[local_rank])
    params = list(model.module.parameters())
    data = [(x, y)]

    ggn_op = GGNLinearOperator(model, loss, params, data)
    dist_ggn = DistributedGGN(ggn_op)

    # Broadcast v to all processes
    v_shape = v.shape
    if rank == 0:
        v = v.to(local_rank)
    else:
        v = torch.empty(v_shape, device=local_rank)
    dist.broadcast(v, src=0)

    result = dist_ggn @ v

    if rank == 0:
        match = torch.allclose(result.cpu(), result_single, atol=1e-6)
        print(f"\nDistributed result matches single-GPU: {match}")
        if not match:
            diff = (result.cpu() - result_single).norm().item()
            print(f"Difference norm: {diff:.6f}")
        # print the result for verification
        print("Distributed result:", result.cpu().numpy())
        print("Single-GPU result:", result_single.numpy())
        print("Broadcasted vector v:", v.cpu().numpy())
    # raise error to ensure all processes complete
    raise RuntimeError("Test completed.")

def main():
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    model = model_fn().cuda(0)
    model_state = model.state_dict()
    loss = loss_fn()

    x, y = get_data()
    x = x.cuda(0)
    y = y.cuda(0)

    params = list(model.parameters())
    data = [(x, y)]
    ggn_op = GGNLinearOperator(model, loss, params, data)

    v = torch.randn_like(torch.cat([p.view(-1) for p in params]))
    result_single = ggn_op @ v.cuda(0)
    result_single = result_single.cpu()

    world_size = torch.cuda.device_count()
    mp.spawn(
        ddp_worker,
        args=(world_size, v, model_state, result_single),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()
