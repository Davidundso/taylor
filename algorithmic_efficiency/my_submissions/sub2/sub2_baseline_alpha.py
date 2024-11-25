"""Submission file for an NAdamW optimizer with warmup+cosine LR in PyTorch."""
# to fix import error
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
import torch.nn as nn

import random
from algorithmic_efficiency import spec
from algorithmic_efficiency.pytorch_utils import pytorch_setup
from source.adaptive_optimizer import AdaptiveLROptimizer
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from curvlinops import GGNLinearOperator
import csv


USE_PYTORCH_DDP = pytorch_setup()[0]

HPARAMS = {
    "dropout_rate": 0.1,
    "learning_rate": 0.0017486387539278373,
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
      workload.step_hint, hyperparameters, optimizer_state['optimizer'])

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

  

  hyperparameters = HPARAMS

  current_model = current_param_container

  params_list = [param for param in current_model.parameters() if param.requires_grad]

  theta_0 = parameters_to_vector([param.detach().clone() for param in params_list])  
  p = 100

  print_bool = global_step % p == 0

  if print_bool:
    print("first 10 elements of theta_0: ", theta_0[:10])  # debugging
    print("Theta_0 lenght:", len(theta_0))  # debugging

  current_model.train()
  optimizer_state['optimizer'].zero_grad()

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


  loss_fn = get_loss_function(workload.loss_type)
  # data structure expected by GGNLinearOperator
  Data = [(batch['inputs'], batch['targets'])]

  if global_step == 0:
    print("Model architecture:\n", current_model)

  # Adjust target and input shapes
  for inputs, targets in Data:
      print("Input shape before any changes:", inputs.shape)
      print("Target shape before any changes:", targets.shape)

      # Add batch dimension to inputs if missing
      if len(inputs.shape) == 3:
          print("Adding batch dimension...")
          inputs = inputs.unsqueeze(1)
          print("Input shape after adding batch dimension:", inputs.shape)

      

      # Adjust targets to match inputs
      if len(targets.shape) == 3:  # Mismatched batch size
          print("Aligning target shape to match inputs...")
          targets = targets=targets.unsqueeze(1)  # Adjust batch dimension
          print("Target shape after adjustment:", targets.shape)

      # Ensure inputs require gradients
      #inputs.requires_grad_(True)
      #print("Does inputs require grad?", inputs.requires_grad)

  # Recreate Data structure
  Data = [(inputs, targets)]

  # Pass to GGNLinearOperator
  try:
      GGN = GGNLinearOperator(current_model, loss_fn, params_list, Data)
      print("GGNLinearOperator initialized successfully!")
  except Exception as e:
      print("Error during GGNLinearOperator initialization:", e)



  # GGN = GGNLinearOperator(current_model, loss_fn, params_list, Data)
  if global_step % p == 0:
    print("Done with GGN")  # debugging

# forrward pass through model_fn
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

  if global_step % p == 0:
    print("grad_clip: ", grad_clip)  # debugging

  loss_dict = workload.loss_fn(
      label_batch=batch['targets'],
      logits_batch=logits_batch,
      mask_batch=batch.get('weights'),
      label_smoothing=label_smoothing)
  summed_loss = loss_dict['summed']
  n_valid_examples = loss_dict['n_valid_examples']
  if USE_PYTORCH_DDP:
    # Use dist_nn.all_reduce to ensure correct loss and gradient scaling.
    summed_loss = dist_nn.all_reduce(summed_loss)
    n_valid_examples = dist_nn.all_reduce(n_valid_examples)
  loss = summed_loss / n_valid_examples

  loss.backward()
  
  gradients = parameters_to_vector(param.grad for param in current_model.parameters() if param.grad is not None)
  gradients_norm = torch.norm(gradients, 2)

  if print_bool:
    print("Done with gradients")  # debugging
    print("graduent lenght:", len(gradients))  # debugging

  if grad_clip is not None:
    torch.nn.utils.clip_grad_norm_(
        current_model.parameters(), max_norm=grad_clip)
  

  optimizer_state['optimizer'].step()
  optimizer_state['scheduler'].step()
  
  theta_1 = parameters_to_vector([param.detach().clone() for param in current_param_container.parameters() if param.requires_grad])  
  if print_bool:
    print("first 10 elements of theta_1: ", theta_1[:10])  # debugging 
    print("Theta_1 lenght:", len(theta_1))  # debugging

  d_unnormalized = theta_1 - theta_0
  d_unnormalized_norm = torch.norm(d_unnormalized, 2)
  d_normalized = d_unnormalized / d_unnormalized_norm

  if print_bool:
    print("Norm of d_normalized: ", torch.norm(d_normalized).item()) # debugging

  if print_bool:
    print("Done with d_unnormalized")  # debugging
    print("first 10 elements of d_unnormalized: ", d_unnormalized[:10])  # debugging
    # Ensure d_unnormalized is a list or a sequence
    d_unnormalized_list = list(d_unnormalized)
    # Select ten random elements
    random_elements = random.sample(d_unnormalized_list, 10)
    # Print the random elements
    print("Ten random elements of d_unnormalized: ", random_elements)  # debugging
    print("d_unnormalized lenght:", len(d_unnormalized))  # debugging

  GGNd = GGN @ d_unnormalized.detach().cpu().numpy()
  GGNd_normalized = GGN @ d_normalized.detach().cpu().numpy()

  if print_bool:
    print("Done with GGNd")  # debugging
  GGNd_tensor = torch.tensor(GGNd, device='cuda') # from_numpy()
  GGNd_normalized_tensor = torch.tensor(GGNd_normalized, device='cuda') # from_numpy()

  #if print_bool:
    #print("Done with GGNd_tensor")  # debugging
    #print("device of GGNd_tensor: ", GGNd_tensor.device)  # debugging
    #print("device of d_unnormalized: ", d_unnormalized.device)  # debugging

  dGGNd = torch.dot(GGNd_tensor, d_unnormalized)
  dGGNd_normalized = torch.dot(GGNd_normalized_tensor, d_normalized)

  if print_bool:
    print("Done with dGGNd")  # debugging      
  dg = - torch.dot(gradients, d_unnormalized)  # numerator: - d^T*g
  dg_normalized = - torch.dot(gradients, d_normalized)  # numerator: - d^T*g
  
  if print_bool:
    print("Done with dg")  # debugging

  alpha_star = dg / dGGNd
  alpha_star_normalized = dg_normalized / dGGNd_normalized

  if print_bool:
    print("dg(Zaehler): ", dg.item())  # debugging   
    print("dGGNd(Nenner): ", dGGNd.item())  # debugging
    print("alpha_star:", alpha_star.item())  # debugging
  

  

  if print_bool:
    print("Done with alpha_star")  # debugging

  current_lr = optimizer_state['optimizer'].param_groups[0]['lr']



  alpha_log_dir = "$WORK/cluster_experiments/exp02"

  # Ensure the directory exists
  os.makedirs(alpha_log_dir, exist_ok=True)

  # Construct the full path to the log file
  log_file_path = os.path.join(alpha_log_dir, 'alpha_star_log.csv')

  log_data = [global_step, alpha_star.item(), dg.item(), dGGNd.item(), d_unnormalized_norm.item(), gradients_norm.item(),
   alpha_star_normalized.item(), dg_normalized.item(), dGGNd_normalized.item(), current_lr]

  # Check if the file exists and write a header if needed
  try:
      with open(log_file_path, 'x') as log_file:  # Open in exclusive creation mode
          writer = csv.writer(log_file)
          writer.writerow(["Global Step", "Alpha Star", "Zaehler(d x g)", "Nenner (dGGNd)", "d L2 Norm", "gradients L2 Norm",
           "Alpha Star(NORMED)", "Zaehler(d x g) (NORMED)", "Nenner (dGGNd) (NORMED)", "Current LR"])  # Write header
  except FileExistsError:
      pass  # File already exists, no need to write the header

  # Append the log data
  with open(log_file_path, 'a') as log_file:
      writer = csv.writer(log_file)
      writer.writerow(log_data)
  
  
  

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

  return (optimizer_state, current_param_container, new_model_state)


def get_batch_size(workload_name):
  # Return the global batch size.
  if workload_name == 'criteo1tb':
    return 262_144
  elif workload_name == 'fastmri':
    return 32
  elif workload_name == 'imagenet_resnet':
    return 1024
  elif workload_name == 'imagenet_resnet_silu':
    return 512
  elif workload_name == 'imagenet_resnet_gelu':
    return 512
  elif workload_name == 'imagenet_vit':
    return 1024
  elif workload_name == 'librispeech_conformer':
    return 256
  elif workload_name == 'librispeech_deepspeech':
    return 256
  elif workload_name == 'ogbg':
    return 512
  elif workload_name == 'wmt':
    return 128
  elif workload_name == 'mnist':
    return 256
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
