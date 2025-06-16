import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from curvlinops import GGNLinearOperator

# --- Model, Loss, and Data Setup ---

def create_model():
    return torch.nn.Sequential(
        torch.nn.Linear(10, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 1)
    )

# We use mean reduction, the GGN should use the same reduction as the loss
def create_loss(reduction="mean"):
    return torch.nn.MSELoss(reduction=reduction)

def generate_data():
    x = torch.randn(64, 10)
    y = torch.randn(64, 1)
    return x, y

# --- Custom Distributed GGN Operator ---

# The exact class we want to use for distributed GGN computation
class DistributedGGN:
    def __init__(self, local_operator, reduction="mean"):
        self.local_operator = local_operator
        self.reduction = reduction

    def __matmul__(self, vector):
        local_result = self.local_operator @ vector
        dist.all_reduce(local_result)
        if self.reduction == "mean":
            local_result /= dist.get_world_size()
        return local_result

    def matvec(self, vector):
        return self @ vector

# --- DDP Worker Function ---
# Starting the distributed processes. This is handled in the background in AlgoPerf
def ddp_worker(rank, N_GPUS):
    torch.cuda.set_device(rank)          # assigns the current process to the correct GPU
    # Initialize the distributed process group (allows for communication between processes like all-reduce)
    dist.init_process_group(backend="nccl") 
    torch.manual_seed(0) # Set random seed for reproducibility
    torch.cuda.manual_seed(0) # consistent random seed across GPUs

    # Setup model and loss
    model = create_model().to(rank)
    loss_fn = create_loss(reduction="mean")
    params = list(model.parameters())

    # Create test vector identically across ranks
    vector = torch.randn_like(torch.cat([p.view(-1) for p in params])).to(rank)

    # Generate full data and distribute it across devices
    x_full, y_full = generate_data()
    shard_size = x_full.size(0) // N_GPUS
    x_local = x_full[rank * shard_size:(rank + 1) * shard_size].to(rank)
    y_local = y_full[rank * shard_size:(rank + 1) * shard_size].to(rank)
    local_data = [(x_local, y_local)]

    # Setup distributed model and operator
    model_ddp = DDP(model, device_ids=[rank])
    local_params = list(model_ddp.module.parameters())
    local_op = GGNLinearOperator(model_ddp, loss_fn, local_params, local_data)
    dist_ggn = DistributedGGN(local_op, reduction="mean")

    # Distributed computation
    distributed_result = dist_ggn @ vector

    # Full computation on rank 0
    if rank == 0:
        model_full = create_model().to(rank)
        model_full.load_state_dict(model_ddp.module.state_dict())
        loss_full = create_loss(reduction="mean")
        full_params = list(model_full.parameters())
        full_data = [(x_full.to(rank), y_full.to(rank))]
        full_op = GGNLinearOperator(model_full, loss_full, full_params, full_data)
        full_result = (full_op @ vector).cpu()

        # Compare
        match = torch.allclose(distributed_result.cpu(), full_result, atol=1e-6)
        print(f"\nDistributed result matches single-GPU: {match}")
        if not match:
            diff_norm = (distributed_result.cpu() - full_result).norm().item()
            print(f"Difference norm: {diff_norm:.6f}")
        print("Distributed result:", distributed_result.cpu().numpy())
        print("Single-GPU result:", full_result.numpy())

    raise RuntimeError("Computation complete on all ranks.")

# --- Main Function ---

def main():
    N_GPUS = torch.cuda.device_count()
    mp.spawn(ddp_worker, args=(N_GPUS,), nprocs=N_GPUS, join=True)

if __name__ == "__main__":
    main()
