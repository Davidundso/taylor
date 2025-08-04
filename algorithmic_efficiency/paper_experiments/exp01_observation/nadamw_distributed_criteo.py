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

from algorithmic_efficiency import spec
from algorithmic_efficiency.pytorch_utils import pytorch_setup
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from curvlinops import GGNLinearOperator
import torch.nn as nn
import csv
# timing
import time
import gc

USE_PYTORCH_DDP, RANK, DEVICE, N_GPUS = pytorch_setup()
# To change the lr schedule: (1) Change the learning rate appropriately to make it e.g. smaller
# (2) then change the step hint: an e.g. 4x smaller lr means 4x more steps are needed
HPARAMS = {
    "dropout_rate": 0.1,
    "learning_rate": 0.0017486387539278373 / 8 * N_GPUS,      # make lr 8 times smaller for using only one instead of 8 a100 GPUs (compared to AlgoPerf comptetion)
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
      workload.step_hint * (8 / N_GPUS), hyperparameters, optimizer_state['optimizer'])  

  return optimizer_state


# create torch loss function from workload loss type for GGN computation
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

# distrbuted GGN class, reduction can either be mean or it sums
class DistributedGGN:
    def __init__(self, ggn_op, reduction='mean', N_GPUS=None):
        self.ggn_op = ggn_op
        self.reduction = reduction
        self.N_GPUS = N_GPUS

    def __matmul__(self, v):
        local_result = self.ggn_op @ v
        torch.distributed.all_reduce(local_result, op=torch.distributed.ReduceOp.SUM) 
        if self.reduction == 'mean':
          local_result /= self.N_GPUS
        return local_result

    def matvec(self, v):
        return self.__matmul__(v)


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

  num_consec_alphas = 100
  comp_alphas_each = 1000
  start_comp_alpha = 0            # earliest alpha computations: If set to zero alpha is computed for the first num_consec_alphas steps 
  timing = True                    # if timings should be printed
  workload_name = 'criteo1tb'      # enum: criteo1tb, imagenet_resnet, imagenet_vit
  #workload_name = 'imagenet_resnet' 
  #workload_name = 'imagenet_vit' 
  use_median_alpha = False         # if True, use the median of the last num_consec_alphas alphas to set the learning rate

    # log: num of devices, batch size, learning rate, max number of steps
  if global_step == 0 or global_step % 1000 == 0:
    logging.info('Using %d GPUs', N_GPUS)
    logging.info('Per GPU batch size before halving: %d', batch['inputs'].size(0))
    logging.info('Batch size from batch size fn: %d', get_batch_size(workload_name))
    logging.info('Learning rate: %f', HPARAMS['learning_rate'])
    logging.info('Max number of steps: %d', workload.step_hint * (8 / N_GPUS))


  # do a normal step for most steps (using one half of the batch)
  if global_step % comp_alphas_each >= num_consec_alphas or global_step < start_comp_alpha:
    if timing:
      start_normal_step = time.time()


    hyperparameters = HPARAMS

    # only use half of the batch
    batch_size = batch['inputs'].size(0)
    half_batch_size = batch_size // 2

    batch = {
        'inputs': batch['inputs'][:half_batch_size],
        'targets': batch['targets'][:half_batch_size],
        
    }
    if batch.get('weights') is not None:
      batch['weights'] = batch['weights'][:half_batch_size]


    current_model = current_param_container
    current_model.train()
    optimizer_state['optimizer'].zero_grad()

    logits_batch, new_model_state = workload.model_fn(
        params=current_model,
        augmented_and_preprocessed_input_batch=batch,
        model_state=model_state,
        mode=spec.ForwardPassMode.TRAIN,
        rng=rng,
        update_batch_norm=True)

    label_smoothing = (
        hyperparameters.label_smoothing if hasattr(hyperparameters,
                                                  'label_smoothing') else 0.0)
    if hasattr(hyperparameters, 'grad_clip'):
      grad_clip = hyperparameters.grad_clip
    else:
      grad_clip = None

    loss_dict = workload.loss_fn(
        label_batch=batch['targets'],
        logits_batch=logits_batch,
        # mask_batch=batch.get('weights'),
        label_smoothing=label_smoothing)
    summed_loss = loss_dict['summed']
    n_valid_examples = loss_dict['n_valid_examples']
    if USE_PYTORCH_DDP:
      # Use dist_nn.all_reduce to ensure correct loss and gradient scaling.
      summed_loss = dist_nn.all_reduce(summed_loss)
      n_valid_examples = dist_nn.all_reduce(n_valid_examples)
    loss = summed_loss / n_valid_examples

    loss.backward()

    if grad_clip is not None:
      torch.nn.utils.clip_grad_norm_(
          current_model.parameters(), max_norm=grad_clip)
    optimizer_state['optimizer'].step()
    if global_step < 2000 + num_consec_alphas:
      optimizer_state['scheduler'].step()

    # Log training metrics - loss, grad_norm, batch_size.
    if global_step <= 100 or global_step % 500 == 0:
      with torch.no_grad():
        parameters = [p for p in current_model.parameters() if p.grad is not None]
        grad_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), 2) for p in parameters]), 2)
      if workload.metrics_logger is not None:
        workload.metrics_logger.append_scalar_metrics(
            {
                'loss': loss.item(),
                'grad_norm': grad_norm.item(),
            }, global_step)
      logging.info('%d) loss = %0.3f, grad_norm = %0.3f',
                  global_step,
                  loss.item(),
                  grad_norm.item())
      
    if timing:
      time_normal_step = time.time() - start_normal_step
      
    if RANK == 0:  # log only on rank 0
      if timing:
        if workload_name == 'criteo1tb':
          log_dir = os.path.expandvars("$WORK/paper_experiments/exp01_criteo")
        elif workload_name == 'imagenet_resnet':
          log_dir = os.path.expandvars("$WORK/paper_experiments/exp01_imagenet_resnet")
        elif workload_name == 'imagenet_vit':
          log_dir = os.path.expandvars("$WORK/paper_experiments/exp01_imagenet_vit")

        # Ensure the directory exists
        os.makedirs(log_dir, exist_ok=True)

        # Construct the full path to the log file
        log_file_path = os.path.join(log_dir, 'timing_log.csv')

        log_data = [global_step, "", "", time_normal_step]  # Empty strings for GGN and alpha step times

        # Check if the file exists and write a header if needed
        try:
            with open(log_file_path, 'x') as log_file:  # Open in exclusive creation mode
                writer = csv.writer(log_file)
                writer.writerow(["Global step", "GGN time B1", "Alpha step time", "Normal step time"
                                ])  # Write header
        except FileExistsError:
            pass
        
        # Append the log data
        with open(log_file_path, 'a') as log_file:
            writer = csv.writer(log_file)
            writer.writerow(log_data)

    

    return (optimizer_state, current_param_container, new_model_state)
  
  # for num_consec_alphas of comp_alpha_each steps caclulate 
  # alpha debiased and biased using both halves of the batch
  if timing:
     start_time_alpha_step = time.time()


  loss_fn = get_loss_function(workload.loss_type) # get torch loss function for GGN 
  
  hyperparameters = HPARAMS
  # forward function for the GGN
  current_model = current_param_container
  # list of diff'able params for the GGN
  params_list = [param for param in current_model.parameters() if param.requires_grad] # save params before step

  # set to evaluation mode before GGN
  current_model.eval()

  # Split the batch into two halves
  inputs = batch['inputs']
  targets = batch['targets']
  weights = batch.get('weights')
  batch_size = inputs.size(0)
  half_batch_size = batch_size // 2

  inputs1, inputs2 = inputs[:half_batch_size], inputs[half_batch_size:]
  targets1, targets2 = targets[:half_batch_size], targets[half_batch_size:]

  if global_step == 0:
    print("Using device:", DEVICE)

  # First half
  optimizer_state['optimizer'].zero_grad()

  if workload_name == 'criteo1tb':
    Data_b1 = [(inputs1.to(DEVICE), targets1.view(-1,1).to(DEVICE))] # criteo needs the targets like this (due to custom forward function)
  
  if workload_name == 'imagenet_resnet' or workload_name == 'imagenet_vit':
    Data_b1 = [(inputs1.to(DEVICE), targets1.to(DEVICE))]


  if timing:
    start_ggn_b1 = time.time()
  # GGN on each device
  GGN_b1 = GGNLinearOperator(current_model, loss_fn, params_list, Data_b1, check_deterministic=False)  # GGN on each device
  # construct combined GGN
  GGN_b1_combined = DistributedGGN(GGN_b1, reduction='mean', N_GPUS=N_GPUS)  # use mean reduction 
  # Data1 not used anymore, so we can delete it (saves memory)


  del Data_b1
  torch.cuda.empty_cache()      # possibly free up memory after GGN computation

  if timing:
    time_ggn_b1 = time.time() - start_ggn_b1
  # set to train mode before step

  # compute ggn
  if workload_name == 'criteo1tb':
    Data_b2 = [(inputs2.to(DEVICE), targets2.view(-1,1).to(DEVICE))] 

  if workload_name == 'imagenet_resnet' or workload_name == 'imagenet_vit':
    Data_b2 = [(inputs2.to(DEVICE), targets2.to(DEVICE))]
  
  current_model.eval()  # set to evaluation mode before GGN
  

  GGN_b2 = GGNLinearOperator(current_model, loss_fn, params_list, Data_b2, check_deterministic=False)  # GGN on each device

  # construct combined GGN
  GGN_b2_combined = DistributedGGN(GGN_b2, reduction='mean', N_GPUS=N_GPUS)  # use mean reduction
  # Data2 not used anymore, so we can delete it (saves memory)

  del Data_b2
  torch.cuda.empty_cache()

  current_model.train()

  logits_batch1, new_model_state1 = workload.model_fn(
      params=current_model,
      augmented_and_preprocessed_input_batch={'inputs': inputs1, 'targets': targets1},
      model_state=model_state,
      mode=spec.ForwardPassMode.TRAIN,
      rng=rng,
      update_batch_norm=True)

  label_smoothing = (
      hyperparameters.label_smoothing if hasattr(hyperparameters,
                                                 'label_smoothing') else 0.0)
  grad_clip = hyperparameters.grad_clip if hasattr(hyperparameters, 'grad_clip') else None

  loss_dict1 = workload.loss_fn(
      label_batch=targets1,
      logits_batch=logits_batch1,
      # mask_batch=weights1,
      label_smoothing=label_smoothing)
  summed_loss1 = loss_dict1['summed']
  n_valid_examples1 = loss_dict1['n_valid_examples']
  if USE_PYTORCH_DDP:
    summed_loss1 = dist_nn.all_reduce(summed_loss1)
    n_valid_examples1 = dist_nn.all_reduce(n_valid_examples1)
  loss1 = summed_loss1 / n_valid_examples1

  loss1.backward()

  gradients_b1 = parameters_to_vector(param.grad for param in current_model.parameters() if param.grad is not None) # extract the gradient
  gradients_norm_b1 = torch.norm(gradients_b1, 2)

  # Clean up to free GPU memory before second GGN
  del logits_batch1, new_model_state1, targets1, inputs1
  torch.cuda.empty_cache()


  # Gradient clipping for the first half
  if grad_clip is not None:
    torch.nn.utils.clip_grad_norm_(current_model.parameters(), max_norm=grad_clip)

  # Moving to the second half
  params_list = [param for param in current_model.parameters() if param.requires_grad] # save params before step
  theta_0_b2 = parameters_to_vector([param.detach().clone() for param in params_list]) # convert to vector

  # Second half
  optimizer_state['optimizer'].zero_grad()
  
  logits_batch2, new_model_state2 = workload.model_fn(
      params=current_model,
      augmented_and_preprocessed_input_batch={'inputs': inputs2, 'targets': targets2},
      model_state=model_state,
      mode=spec.ForwardPassMode.TRAIN,
      rng=rng,
      update_batch_norm=True)
  
  label_smoothing = (
      hyperparameters.label_smoothing if hasattr(hyperparameters,
                                                  'label_smoothing') else 0.0)
  grad_clip = hyperparameters.grad_clip if hasattr(hyperparameters, 'grad_clip') else None

  loss_dict2 = workload.loss_fn(
      label_batch=targets2,
      logits_batch=logits_batch2,
      # mask_batch=weights2, removed. cannot be passed to GGN
      label_smoothing=label_smoothing)
  summed_loss2 = loss_dict2['summed']
  n_valid_examples2 = loss_dict2['n_valid_examples']
  if USE_PYTORCH_DDP:
    summed_loss2 = dist_nn.all_reduce(summed_loss2)
    n_valid_examples2 = dist_nn.all_reduce(n_valid_examples2)

  loss2 = summed_loss2 / n_valid_examples2
  loss2.backward()

  gradients_b2 = parameters_to_vector(param.grad for param in current_model.parameters() if param.grad is not None)
  gradients_norm_b2 = torch.norm(gradients_b2, 2)

  # Gradient clipping for the second half
  if grad_clip is not None:
    torch.nn.utils.clip_grad_norm_(current_model.parameters(), max_norm=grad_clip)

  # Update the parameters after the second half
  optimizer_state['optimizer'].step()
  # step the scheduler (only do this if you don't use your own curvature-based learning rate)
  optimizer_state['scheduler'].step()


  theta_1_b2 = parameters_to_vector([param.detach().clone() for param in current_param_container.parameters() if param.requires_grad])

  d_unnormalized_b2 = theta_1_b2 - theta_0_b2

  # Compute alpha using d_unnormalized from batch 2 and ggn and gradient from batch 1
  GGNd1 = GGN_b1_combined @ d_unnormalized_b2

  dGGNd_1 = torch.dot(GGNd1, d_unnormalized_b2)

  dg1 = - torch.dot(gradients_b1, d_unnormalized_b2)  # numerator: - d^T*g

  alpha_star1_unbiased = dg1 / dGGNd_1

  # compute alpha using only batch 2
  GGNd_b2 = GGN_b2_combined @ d_unnormalized_b2

  dGGNd_b2 = torch.dot(GGNd_b2, d_unnormalized_b2)

  dg_b2 = - torch.dot(gradients_b2, d_unnormalized_b2)  # numerator: - d^T*g

  alpha_star_b2_biased = dg_b2 / dGGNd_b2

  if timing:
     alpha_step_time = time.time() - start_time_alpha_step

  if RANK == 0:                                                               # log only on rank 0
    # Log training metrics - loss, grad_norm, batch_size.
    if global_step <= 100 or global_step % 500 == 0:
      with torch.no_grad():
        parameters = [p for p in current_model.parameters() if p.grad is not None]
        grad_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), 2) for p in parameters]), 2)
      if workload.metrics_logger is not None:
        workload.metrics_logger.append_scalar_metrics(
            {
                'loss': loss1.item(),
                'grad_norm': grad_norm.item(),
            }, global_step)
      logging.info('%d) loss = %0.3f, grad_norm = %0.3f',
                  global_step,
                  loss1.item(),
                  grad_norm.item())
    


    current_lr = optimizer_state['optimizer'].param_groups[0]['lr']
    # log the values of alpha_star1_unbiased, alpha_star2, alpha_star_b1, alpha_star_b2_biased into a csv file
    if workload_name == 'criteo1tb':
      log_dir = os.path.expandvars("$WORK/paper_experiments/exp01_criteo")
    elif workload_name == 'imagenet_resnet':
      log_dir = os.path.expandvars("$WORK/paper_experiments/exp01_imagenet_resnet")
    elif workload_name == 'imagenet_vit':
      log_dir = os.path.expandvars("$WORK/paper_experiments/exp01_imagenet_vit")

    # Ensure the directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Construct the full path to the log file
    log_file_path = os.path.join(log_dir, 'alpha_star_log.csv')

    log_data = [global_step, alpha_star1_unbiased.item(), alpha_star_b2_biased.item(), current_lr]

    # Check if the file exists and write a header if needed
    try:
        with open(log_file_path, 'x') as log_file:  # Open in exclusive creation mode
            writer = csv.writer(log_file)
            writer.writerow(["Global step", "alpha1 unbiased (GGN,g B1; d B2)", "alpha B2 biased", "Learning rate"
                            ])  # Write header
    except FileExistsError:
        pass  # File already exists, no need to write the header

    # Append the log data
    with open(log_file_path, 'a') as log_file:
        writer = csv.writer(log_file)
        writer.writerow(log_data)

    # safety precaution: Every 1000 steps log num_gpus, batch size from batch fn, batch size from inputs into a csv file named "batch_size_log.csv"
    if global_step % 1000 == 0:
        # Ensure the directory exists
        os.makedirs(log_dir, exist_ok=True)

        # Construct the full path to the log file
        log_file_path = os.path.join(log_dir, 'batch_size_log.csv')

        log_data = [global_step, N_GPUS, batch['inputs'].size(0), get_batch_size(workload_name)]

        # Check if the file exists and write a header if needed
        try:
            with open(log_file_path, 'x') as log_file:  # Open in exclusive creation mode
                writer = csv.writer(log_file)
                writer.writerow(["Global step", "Num GPUs", "Batch size from inputs", "Batch size from batch size fn"
                                ])  # Write header              
        except FileExistsError:
            pass

    # log the same values but each multiplied by the learning rate into a new file alpha_star_scaled_log.csv

    # Ensure the directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Construct the full path to the log file
    log_file_path = os.path.join(log_dir, 'alpha_star_scaled_log.csv')

    log_data = [global_step, alpha_star1_unbiased.item()*current_lr, alpha_star_b2_biased.item()*current_lr, current_lr]

    # Check if the file exists and write a header if needed

    try:
        with open(log_file_path, 'x') as log_file:  # Open in exclusive creation mode
            writer = csv.writer(log_file)
            writer.writerow(["Global step", "alpha1 unbiased * lr (GGN,g B1; d B2)", "alpha B2 biased * lr", "Learning rate"
                            ])  # Write header
    except FileExistsError:
        pass
    
    # Append the log data
    with open(log_file_path, 'a') as log_file:
        writer = csv.writer(log_file)
        writer.writerow(log_data)

    # log all numerators and denominators of the alphas next to the alphas in a new file alpha_star_numerators_denominators_log.csv

    # Check cosine similarity
    cosine_1 = torch.dot(gradients_b1, d_unnormalized_b2) / (gradients_b1.norm() * d_unnormalized_b2.norm())
    cosine_b2 = torch.dot(gradients_b2, d_unnormalized_b2) / (gradients_b2.norm() * d_unnormalized_b2.norm())

    # Ensure the directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Construct the full path to the log file
    log_file_path = os.path.join(log_dir, 'alpha_star_numerators_denominators_log.csv')

    log_data = [global_step, 
                alpha_star1_unbiased.item(), dg1.item(), dGGNd_1.item(), cosine_1.item(),
                alpha_star_b2_biased.item(), dg_b2.item(), dGGNd_b2.item(), cosine_b2.item(),
                    current_lr]
    
    # Check if the file exists and write a header if needed
    try:
        with open(log_file_path, 'x') as log_file:  # Open in exclusive creation mode
            writer = csv.writer(log_file)
            writer.writerow(["Global step",
              "alpha1 unbiased", "dg1 (d B2, g B1)", "dGGNd1 (d B2, GGN B1)", "cosine1 (d B2, g B1)",
              "alpha B2 biased", "dg B2 biased", "dGGNd B2 biased", "cosine2 biased",
              "Current learning rate"
                            ])  # Write header
    except FileExistsError:
        pass
    
    # Append the log data
    with open(log_file_path, 'a') as log_file:
        writer = csv.writer(log_file)
        writer.writerow(log_data)

    # next log scaled versions of the numerators and denominators
    # log the numerators and denominators of the alphas scaled by the learning rate in a new file numerators_denominators_scaled_log.csv
    log_data_scaled = [global_step,
                       alpha_star1_unbiased.item() * current_lr, dg1.item() * current_lr, dGGNd_1.item() * current_lr, cosine_1.item(),
                       alpha_star_b2_biased.item() * current_lr, dg_b2.item() * current_lr, dGGNd_b2.item() * current_lr, cosine_b2.item(),
                       current_lr]

    # Ensure the directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Construct the full path to the log file
    log_file_path = os.path.join(log_dir, 'numerators_denominators_scaled_log.csv')

    # Check if the file exists and write a header if needed
    try:
        with open(log_file_path, 'x') as log_file:  # Open in exclusive creation mode
            writer = csv.writer(log_file)
            writer.writerow(["Global step",
              "alpha1 unbiased * lr", "dg1 * lr", "dGGNd1 * lr", "cosine1 * lr",
              "alpha B2 biased * lr", "dg B2 biased * lr", "dGGNd B2 biased * lr", "cosine2 biased * lr",
              "Current learning rate"
                            ])  # Write header
    except FileExistsError:
        pass

    # Append the log data
    with open(log_file_path, 'a') as log_file:
        writer = csv.writer(log_file)
        writer.writerow(log_data_scaled)

    # log the gradient norms, d_norms and loss values in a new file gradient_d_loss_log.csv

    # Ensure the directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Construct the full path to the log file
    log_file_path = os.path.join(log_dir, 'gradient_d_loss_log.csv')

    log_data = [global_step, gradients_norm_b1.item(), gradients_norm_b2.item(),torch.norm(d_unnormalized_b2, 2).item(), loss1.item(), loss2.item()]

    # Check if the file exists and write a header if needed
    try:
        with open(log_file_path, 'x') as log_file:  # Open in exclusive creation mode
            writer = csv.writer(log_file)
            writer.writerow(["Global step", "Gradient norm B1", "Gradient norm B2", "d norm B2", "Loss B1", "Loss B2"
                            ])  # Write header
    except FileExistsError:
        pass
    
    # Append the log data
    with open(log_file_path, 'a') as log_file:
        writer = csv.writer(log_file)
        writer.writerow(log_data)
    
    # log the timing of GGN and alpha step in a new file timing_log.csv
    if timing:
      # Ensure the directory exists
      os.makedirs(log_dir, exist_ok=True)

      # Construct the full path to the log file
      log_file_path = os.path.join(log_dir, 'timing_log.csv')

      log_data = [global_step, time_ggn_b1, alpha_step_time, ""]

      # Check if the file exists and write a header if needed
      try:
          with open(log_file_path, 'x') as log_file:  # Open in exclusive creation mode
              writer = csv.writer(log_file)
              writer.writerow(["Global step", "GGN time B1", "Alpha step time", "Normal step time"
                              ])  # Write header
      except FileExistsError:
          pass
      
      # Append the log data
      with open(log_file_path, 'a') as log_file:
          writer = csv.writer(log_file)
          writer.writerow(log_data)

    if use_median_alpha:
      # declare global variable for average of alpha*lr
      global alpha_values
      
      if global_step % comp_alphas_each == 0:
        alpha_values = []

      # sum alpha values scaled by the learning rate
      alpha_values.append(alpha_star1_unbiased * current_lr)  # Store as tensors directly

      # after 50 alphas were added, calculate the average and print it
      if global_step % comp_alphas_each == num_consec_alphas - 1 and global_step > comp_alphas_each:
        tensor_alpha_values = torch.stack(alpha_values)  # Convert list of tensors to a single tensor
        # Compute median using quantile with midpoint interpolation
        median_alpha_star1 = torch.quantile(tensor_alpha_values, 0.5, interpolation='midpoint')  

        alpha_values = []                     # back to zero for the next 50 alphas

        if median_alpha_star1.item() > 0:
          for i, param_group in enumerate(optimizer_state['optimizer'].param_groups):
              # Print the current learning rate before changing
              print(f"Before change - Parameter group {i}: lr = {param_group['lr']}")
              
              # Change the learning rate
              param_group['lr'] = median_alpha_star1.item()

              # Print the new learning rate after changing
              print(f"After change - Parameter group {i}: lr = {param_group['lr']}")



  return optimizer_state, current_param_container, new_model_state2  # Return the final model state


def get_batch_size(workload_name):
  # Return the global batch size.
  if workload_name == 'criteo1tb':
    return int(262_144/8 * 2 * N_GPUS)  # original batch size / 8 (original number of GPUs) * 2 (for two halves) * 2 (for two GPUs)
  elif workload_name == 'fastmri':
    return 32
  elif workload_name == 'imagenet_resnet':
    return int(1024/8 * 2 * N_GPUS)  # original batch size / 8 (original number of GPUs) * 2 (for two halves) * 2 (for two GPUs)
  elif workload_name == 'imagenet_resnet_silu':
    return 512
  elif workload_name == 'imagenet_resnet_gelu':
    return 512
  elif workload_name == 'imagenet_vit':
    return int(1024/8 * 2 * N_GPUS)  # original batch size / 8 (original number of GPUs) * 2 (for two halves) * 2 (for two GPUs)
  elif workload_name == 'librispeech_conformer':
    return 256
  elif workload_name == 'librispeech_deepspeech':
    return 256
  elif workload_name == 'ogbg':
    return 512
  elif workload_name == 'wmt':
    return 128
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