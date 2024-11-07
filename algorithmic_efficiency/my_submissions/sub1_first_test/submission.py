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
    model = workload.model  # or adapt as needed to access model
    loss_function = workload.loss_function
    optimizer = workload.optimizer  # Assuming it’s Adam

    adaptive_optimizer = AdaptiveLROptimizer(model, loss_function, optimizer)

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

    return updated_optimizer_state, updated_variables, updated_model_state
  


def get_batch_size(workload_name):
  """
    Gets batch size for workload. 
    Note that these batch sizes only apply during training and not during evals.
    Args: 
      workload_name (str): Valid workload_name values are: "wmt", "ogbg", 
        "criteo1tb", "fastmri", "imagenet_resnet", "imagenet_vit", 
        "librispeech_deepspeech", "librispeech_conformer" or any of the
        variants.
    Returns:
      int: batch_size 
    Raises:
      ValueError: If workload_name is not handled.
    """
  pass


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
