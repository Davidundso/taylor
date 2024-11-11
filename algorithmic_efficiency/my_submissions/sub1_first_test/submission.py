"""Template submission module.

See https://github.com/mlcommons/algorithmic-efficiency/blob/main/DOCUMENTATION.md#allowed-submissions
and https://github.com/mlcommons/algorithmic-efficiency/blob/main/DOCUMENTATION.md#disallowed-submissions
for guidelines.
"""
from typing import Dict, Iterator, List, Tuple

from algorithmic_efficiency import spec

# my additional imports: only these are specific, curvelinops etc should be implicitly imported via AdaptiveLROptimizer
import torch
import torch.nn as nn
from torch.optim import SGD
from adaptive_optimizer import AdaptiveLROptimizer
import torch.optim as optim
 

def init_optimizer_state(workload: spec.Workload,
                         model_params: spec.ParameterContainer,
                         model_state: spec.ModelAuxiliaryState,
                         hyperparameters: spec.Hyperparameters,
                         rng: spec.RandomState) -> spec.OptimizerState:
    """Creates a Nesterov optimizer and a learning rate schedule.
    Returns:
     optimizer state
     optimizer_update_fn (not in all submissions, so I will skip that)
    """
    # Get learning rate and momentum from hyperparameters
    lr = hyperparameters.get("learning_rate", 0.01)  # Default to 0.01 if not provided
    momentum = hyperparameters.get("momentum", 0.9)  # Default to 0.9 if not provided
    
    # Initialize Nesterov optimizer
    optimizer = optim.SGD(model_params, lr=lr, momentum=momentum, nesterov=True)

    

    # Return optimizer state and update function
    optimizer_state = optimizer.state_dict()
    return optimizer_state


import torch
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR

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
    """
    Returns:
     (new_optimizer_state, update_fn)
     new_params
     new_model_state
    """
    # Initialize AdaptiveLROptimizer
    model = current_param_container
    loss_function = workload.loss_function
    optimizer = workload.optimizer  # Assuming it’s Adam

    adaptive_optimizer = AdaptiveLROptimizer(model, loss_function, optimizer)

    # Get the step_hint value from workload
    step_hint = workload.step_hint()

    # Adjust hyperparameters based on step_hint
    warmup_steps = hyperparameters.get("warmup_factor", 0.05) * step_hint
    total_steps = step_hint  # Use step_hint as the total number of steps

    # Get initial learning rate and minimum learning rate
    initial_lr = hyperparameters.get("initial_lr", 0.001)
    min_lr = hyperparameters.get("min_lr", 1e-6)

    # Step 1: Define warmup scheduler (Linear warmup)
    def warmup_lr_lambda(step: int):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 1.0

    # Step 2: Define cosine decay scheduler after warmup
    cosine_steps = max(total_steps - warmup_steps, 1)  # Ensure there's at least one decay step
    cosine_lr_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=min_lr)

    # Combine warmup and cosine decay
    lr_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lr_lambda)
    
    # Apply the combined learning rate schedule
    if global_step < warmup_steps:
        lr_scheduler.step()
    else:
        cosine_lr_scheduler.step()

    # Extract data and target from batch
    data = batch["data"]
    target = batch["target"]

    # Compute alpha_star and the direction
    alpha_star, direction = adaptive_optimizer.compute_alpha_star(data, target)

    # Apply the computed step
    adaptive_optimizer.apply_step(alpha_star, direction)

    # Retrieve updated states
    updated_optimizer_state = optimizer.state_dict()  # Get the updated optimizer state
    updated_variables = {name: param.detach().clone() for name, param in model.named_parameters()}
    updated_model_state = model.state_dict()

    # Update the optimizer state to include the new scheduler
    optimizer_state['scheduler'] = lr_scheduler if global_step < warmup_steps else cosine_lr_scheduler

    return updated_optimizer_state, updated_variables, updated_model_state

  


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
    return 16
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
    Tip:
    If you would just like the next batch from the input queue return next(input_queue).

    Returns:
     batch: next batch of input data
    """
    return next(input_queue)
