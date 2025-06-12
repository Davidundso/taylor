"""Submission file for an NAdamW optimizer with warmup+cosine LR in PyTorch."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
import math
from typing import Dict, Iterator, List, Tuple

from absl import logging
import torch
from torch import Tensor
import torch.distributed.nn as dist_nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import LinearLR
from torch.optim.lr_scheduler import SequentialLR
# timing
import time


from algorithmic_efficiency import spec
from algorithmic_efficiency.pytorch_utils import pytorch_setup
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from curvlinops import GGNLinearOperator
import torch.nn as nn
import csv

USE_PYTORCH_DDP, RANK, DEVICE, N_GPUS = pytorch_setup()

# To change the lr schedule: (1) Change the learning rate appropriately to make it e.g. smaller
# (2) then change the step hint: an e.g. 4x smaller lr means 4x more steps are needed
HPARAMS = {
    "dropout_rate": 0.1,
    "learning_rate": 0.0017486387539278373 / 8,      # make lr 8 times smaller for using only one instead of 8 a100 GPUs (compared to AlgoPerf comptetion)
    "one_minus_beta1": 0.06733926164,
    "beta2": 0.9955159689799007,
    "weight_decay": 0.08121616522670176,
    "warmup_factor": 0.02
}


# Modified from github.com/pytorch/pytorch/blob/v1.12.1/torch/optim/adamw.py.
class NAdamW(torch.optim.Optimizer):
  r"""Implements NAdamW algorithm.

    See Table 1 in https://arxiv.org/abs/1910.05446 for the implementation of
    the NAdam algorithm (there is also a comment in the code which highlights
    the only difference of NAdamW and AdamW).
    For further details regarding the algorithm we refer to
    `Decoupled Weight Decay Regularization`_.

    Args:
      params (iterable): iterable of parameters to optimize or dicts defining
          parameter groups
      lr (float, optional): learning rate (default: 1e-3)
      betas (Tuple[float, float], optional): coefficients used for computing
          running averages of gradient and its square (default: (0.9, 0.999))
      eps (float, optional): term added to the denominator to improve
          numerical stability (default: 1e-8)
      weight_decay (float, optional): weight decay coefficient (default: 1e-2)
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
  """

  def __init__(self,
               params,
               lr=1e-3,
               betas=(0.9, 0.999),
               eps=1e-8,
               weight_decay=1e-2):
    if not 0.0 <= lr:
      raise ValueError(f'Invalid learning rate: {lr}')
    if not 0.0 <= eps:
      raise ValueError(f'Invalid epsilon value: {eps}')
    if not 0.0 <= betas[0] < 1.0:
      raise ValueError(f'Invalid beta parameter at index 0: {betas[0]}')
    if not 0.0 <= betas[1] < 1.0:
      raise ValueError(f'Invalid beta parameter at index 1: {betas[1]}')
    if not 0.0 <= weight_decay:
      raise ValueError(f'Invalid weight_decay value: {weight_decay}')
    defaults = {
        'lr': lr, 'betas': betas, 'eps': eps, 'weight_decay': weight_decay
    }
    super().__init__(params, defaults)

  def __setstate__(self, state):
    super().__setstate__(state)
    state_values = list(self.state.values())
    step_is_tensor = (len(state_values) != 0) and torch.is_tensor(
        state_values[0]['step'])
    if not step_is_tensor:
      for s in state_values:
        s['step'] = torch.tensor(float(s['step']))

  @torch.no_grad()
  def step(self, closure=None):
    """Performs a single optimization step.

        Args:
          closure (callable, optional): A closure that reevaluates the model
              and returns the loss.
    """
    self._cuda_graph_capture_health_check()

    loss = None
    if closure is not None:
      with torch.enable_grad():
        loss = closure()

    for group in self.param_groups:
      params_with_grad = []
      grads = []
      exp_avgs = []
      exp_avg_sqs = []
      state_steps = []
      beta1, beta2 = group['betas']

      for p in group['params']:
        if p.grad is None:
          continue
        params_with_grad.append(p)
        if p.grad.is_sparse:
          raise RuntimeError('NAdamW does not support sparse gradients')
        grads.append(p.grad)

        state = self.state[p]

        # State initialization
        if len(state) == 0:
          state['step'] = torch.tensor(0.)
          # Exponential moving average of gradient values
          state['exp_avg'] = torch.zeros_like(
              p, memory_format=torch.preserve_format)
          # Exponential moving average of squared gradient values
          state['exp_avg_sq'] = torch.zeros_like(
              p, memory_format=torch.preserve_format)

        exp_avgs.append(state['exp_avg'])
        exp_avg_sqs.append(state['exp_avg_sq'])
        state_steps.append(state['step'])

      nadamw(
          params_with_grad,
          grads,
          exp_avgs,
          exp_avg_sqs,
          state_steps,
          beta1=beta1,
          beta2=beta2,
          lr=group['lr'],
          weight_decay=group['weight_decay'],
          eps=group['eps'])

    return loss


def nadamw(params: List[Tensor],
           grads: List[Tensor],
           exp_avgs: List[Tensor],
           exp_avg_sqs: List[Tensor],
           state_steps: List[Tensor],
           beta1: float,
           beta2: float,
           lr: float,
           weight_decay: float,
           eps: float) -> None:
  r"""Functional API that performs NAdamW algorithm computation.
    See NAdamW class for details.
  """

  if not all(isinstance(t, torch.Tensor) for t in state_steps):
    raise RuntimeError(
        'API has changed, `state_steps` argument must contain a list of' +
        ' singleton tensors')

  for i, param in enumerate(params):
    grad = grads[i]
    exp_avg = exp_avgs[i]
    exp_avg_sq = exp_avg_sqs[i]
    step_t = state_steps[i]

    # Update step.
    step_t += 1

    # Perform stepweight decay.
    param.mul_(1 - lr * weight_decay)

    # Decay the first and second moment running average coefficient.
    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

    # Only difference between NAdamW and AdamW in this implementation.
    # The official PyTorch implementation of NAdam uses a different algorithm.
    # We undo these ops later on, which could cause numerical issues but saves
    # us from having to make an extra copy of the gradients.
    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

    step = step_t.item()

    bias_correction1 = 1 - beta1**step
    bias_correction2 = 1 - beta2**step

    step_size = lr / bias_correction1

    bias_correction2_sqrt = math.sqrt(bias_correction2)
    denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

    param.addcdiv_(exp_avg, denom, value=-step_size)
    exp_avg.sub_(grad, alpha=1 - beta1).div_(beta1)


def init_optimizer_state(workload: spec.Workload,
                         model_params: spec.ParameterContainer,
                         model_state: spec.ModelAuxiliaryState,
                         hyperparameters: spec.Hyperparameters,
                         rng: spec.RandomState) -> spec.OptimizerState:
  """Creates a NAdamW optimizer and a learning rate schedule."""
  del model_state
  del rng
  del hyperparameters

  hyperparameters = HPARAMS

  optimizer_state = {
    'optimizer': NAdamW(
        model_params.parameters(),
        lr=hyperparameters['learning_rate'],
        betas=(1.0 - hyperparameters['one_minus_beta1'],
               hyperparameters['beta2']),
        eps=1e-8,
        weight_decay=hyperparameters['weight_decay']),
}

  def pytorch_cosine_warmup(step_hint: int, hyperparameters, optimizer):
      warmup_steps = int(hyperparameters['warmup_factor'] * step_hint)
      warmup = LinearLR(
          optimizer, start_factor=1e-10, end_factor=1., total_iters=warmup_steps)
      cosine_steps = max(step_hint - warmup_steps, 1)
      cosine_decay = CosineAnnealingLR(optimizer, T_max=cosine_steps)
      return SequentialLR(
          optimizer, schedulers=[warmup, cosine_decay], milestones=[warmup_steps])

  optimizer_state['scheduler'] = pytorch_cosine_warmup(
      workload.step_hint * 8, hyperparameters, optimizer_state['optimizer'])  

  return optimizer_state



def update_params(workload: spec.Workload,
                  current_param_container: spec.ParameterContainer,
                  current_params_types: spec.ParameterTypeTree,
                  model_state: spec.ModelAuxiliaryState,
                  hyperparameters: spec.Hyperparameters,
                  batch: Dict[str, spec.Tensor],
                  loss_type: spec.LossType,
                  optimizer_state: spec.OptimizerState,
                  eval_results: List[Tuple[int, float]],
                  global_step: int,
                  rng: spec.RandomState) -> spec.UpdateReturn:
  """Return (updated_optimizer_state, updated_params, updated_model_state)."""
  del current_params_types
  del loss_type
  del eval_results
  del hyperparameters

  reduction = 'sum'  # enum: 'sum', 'mean'
  timing = True


  def get_loss_function(loss_type):
    """
    Maps a loss type to a PyTorch loss function.

    Args:
        loss_type (LossType): The loss type Enum.

    Returns:
        A PyTorch loss function (instance of nn.Module).
    """
    loss_mapping = {
        "SOFTMAX_CROSS_ENTROPY": nn.CrossEntropyLoss(),
        "SIGMOID_CROSS_ENTROPY": nn.BCEWithLogitsLoss(),
        "MEAN_SQUARED_ERROR": nn.MSELoss(),
        "CTC_LOSS": nn.CTCLoss(),  # Requires alignment inputs
        "MEAN_ABSOLUTE_ERROR": nn.L1Loss(),
    }

    # Convert Enum to string (e.g., "LossType.SOFTMAX_CROSS_ENTROPY" -> "SOFTMAX_CROSS_ENTROPY")
    loss_type_str = loss_type.name if hasattr(loss_type, 'name') else str(loss_type)

    if loss_type_str not in loss_mapping:
        raise ValueError(f"Unsupported loss type: {loss_type_str}")

    return loss_mapping[loss_type_str]

  class DistributedGGN:
    def __init__(self, ggn_op):
        self.ggn_op = ggn_op

    def __matmul__(self, v):
        local_result = self.ggn_op @ v
        torch.distributed.all_reduce(local_result)  # Sums across GPUs, use AVG later
        if reduction == 'mean':
          local_result /= N_GPUS
        return local_result

    def matvec(self, v):
        return self.__matmul__(v)


  print("Using torch.ddp:", USE_PYTORCH_DDP)
  print("device name:", DEVICE)
  print("device rank:", RANK)

  
  hyperparameters = HPARAMS

  inputs = batch['inputs']
  targets = batch['targets']

  # TEST 1: Full batch
  # Criteo: for GGN: prepare data

  Data = [(inputs.to(DEVICE), targets.view(-1,1).to(DEVICE))]
  print(f"inputs1 dtype: {inputs.dtype}, shape: {inputs.shape}")
  print(f"targets1 dtype: {targets.dtype}, shape: {targets.shape}")


  current_model = current_param_container

  optimizer_state['optimizer'].zero_grad()



  # for GGN: prepare loss
  loss_fn = get_loss_function(workload.loss_type)

  hyperparameters = HPARAMS

  # for GGN: prepare model
  current_model = current_param_container

  # for GGN: prepare params
  params_list = [param for param in current_model.parameters() if param.requires_grad] # save params before step


  if timing:
    start_time_ggn = time.time()


  print(f"[Rank {RANK}] Model device: {next(current_model.parameters()).device}")
  print(f"[Rank {RANK}] Data device: {inputs.device}")
  


  for i, p in enumerate(params_list):
    print(f"[Rank {RANK}] Param {i} device: {p.device}")


  # test vector multiplication (same on all GPUs so don't use random numbers)
  v = torch.ones_like(parameters_to_vector(params_list))

  # computes GGN on each device
  GGN_separate = GGNLinearOperator(current_model, loss_fn, params_list, Data)
  print("passed GGN initialization")

  if timing:
    step_time_ggn = time.time() - start_time_ggn
    print(f' GGN computation time: {step_time_ggn} seconds')

  # computes GGN@v on each device
  ggn_v_separate = GGN_separate @ v

  print("first five elements of ggn_v_separate:")
  print(ggn_v_separate[:5])

  # sum over GGN@v vectors
  torch.distributed.all_reduce(ggn_v_separate, op=torch.distributed.ReduceOp.SUM)  # later we want to use AVG, or compute AVG manually
  if reduction == 'mean':
    ggn_v_separate /= N_GPUS

  # rename to distinguish afterwards
  GGN_v_seperate_then_reduced = ggn_v_separate

  print("first five elements of ggn_v_sepparate_then_reduced:")
  print(GGN_v_seperate_then_reduced[:5])

  # GGN wrapper: Also computes summed product if used for vector multiplication
  GGN_reduced = DistributedGGN(GGN_separate)  
  print("passed distributed initialization")

  # compute reduced version
  GGN_v_reduced = GGN_reduced @ v
  print("first five elements of GGN_v_reduced:")
  print(GGN_v_reduced[:5])

  # compare: Do we get the same result if we use separate GGN@v's and then sum AND if we use our class which inherently reduces
  close = torch.allclose(GGN_v_seperate_then_reduced, GGN_v_reduced, rtol=1e-5, atol=1e-8)

  # compute norm of both vectors
  norm_separate = torch.norm(GGN_v_seperate_then_reduced, 2)
  norm_reduced = torch.norm(GGN_v_reduced, 2)

  print("Norm of GGN_v_separate:", norm_separate.item())
  print("Norm of GGN_v_reduced:", norm_reduced.item())

  print("Distributed reduced GGN@v equals separated and then reduced GGN@v:", close)

  # TEST 2: half batches
  # split the batch into two halves
  batch_size = inputs.size(0)
  half_batch_size = batch_size // 2

  inputs1, inputs2 = inputs[:half_batch_size], inputs[half_batch_size:]
  targets1, targets2 = targets[:half_batch_size], targets[half_batch_size:]

  # for GGN: Data
  Data_1 = [(inputs1.to(DEVICE), targets1.view(-1,1).to(DEVICE))]  # remove 'view(-1, 1)' for mnist
  Data_2 = [(inputs2.to(DEVICE), targets2.view(-1,1).to(DEVICE))]  # remove 'view(-1, 1)' for mnist

  # compute GGN on both halves
  GGN_1_separate = GGNLinearOperator(current_model, loss_fn, params_list, Data_1)
  GGN_2_separate = GGNLinearOperator(current_model, loss_fn, params_list, Data_2)

  # compute GGN@v on both halves
  ggn_v_1_separate = GGN_1_separate @ v
  ggn_v_2_separate = GGN_2_separate @ v

  # reduce both halves
  torch.distributed.all_reduce(ggn_v_1_separate, op=torch.distributed.ReduceOp.SUM)
  if reduction == 'mean':
    ggn_v_1_separate /= N_GPUS

  torch.distributed.all_reduce(ggn_v_2_separate, op=torch.distributed.ReduceOp.SUM)
  if reduction == 'mean':
    ggn_v_2_separate /= N_GPUS

  # rename to distinguish afterwards
  GGN_v_1_separate_then_reduced = ggn_v_1_separate
  GGN_v_2_separate_then_reduced = ggn_v_2_separate

  # add the two halves
  GGN_v_both_combined = GGN_v_1_separate_then_reduced + GGN_v_2_separate_then_reduced
  if reduction == 'mean':
    GGN_v_both_combined /= 2  # average the two halves

  # compare to GGN_v_reduced
  close_2 = torch.allclose(GGN_v_both_combined, GGN_v_reduced, rtol=1e-5, atol=1e-8) 

  # compute norm of both vectors
  norm_both_combined = torch.norm(GGN_v_both_combined, 2)
  norm_reduced = torch.norm(GGN_v_reduced, 2)
  print("Norm of GGN_v_both_combined:", norm_both_combined.item())
  print("Norm of GGN_v_reduced:", norm_reduced.item())
  print("Difference in norms:", torch.abs(norm_both_combined - norm_reduced).item())
  print("Distributed reduced GGN@v equals separated and then reduced GGN@v (both halves):", close_2)
  

  raise RuntimeError("Test finished")



def get_batch_size(workload_name):
  # Return the global batch size.
  if workload_name == 'criteo1tb':
    return int(262_144/8)  # smaller batch size for testing
  elif workload_name == 'fastmri':
    return int(32/8 * 2)
  elif workload_name == 'imagenet_resnet':
    return int(1024/8 * 2)
  elif workload_name == 'imagenet_resnet_silu':
    return int(512/8 * 2)
  elif workload_name == 'imagenet_resnet_gelu':
    return int(512/8 * 2)
  elif workload_name == 'imagenet_vit':
    return int(1024/8 * 2)
  elif workload_name == 'librispeech_conformer':
    return int(256/8 * 2)
  elif workload_name == 'librispeech_deepspeech':
    return int(256/8 * 2)
  elif workload_name == 'ogbg':
    return int(512/8 * 2)
  elif workload_name == 'wmt':
    return int(128/8 * 2)
  elif workload_name == 'mnist':
    return 16*2                                     # double the batch size for the two halves
  else:
    raise ValueError(f'Unsupported workload name: {workload_name}.')


def data_selection(workload: spec.Workload,
                   input_queue: Iterator[Dict[str, spec.Tensor]],
                   optimizer_state: spec.OptimizerState,
                   current_param_container: spec.ParameterContainer,
                   model_state: spec.ModelAuxiliaryState,
                   hyperparameters: spec.Hyperparameters,
                   global_step: int,
                   rng: spec.RandomState) -> Dict[str, spec.Tensor]:
  """Select data from the infinitely repeating, pre-shuffled input queue.
  Each element of the queue is a batch of training examples and labels.
  """
  del workload
  del optimizer_state
  del current_param_container
  del model_state
  del hyperparameters
  del global_step
  del rng
  batch = next(input_queue)
  return batch
