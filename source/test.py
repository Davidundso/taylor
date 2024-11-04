import torch
import torch.nn as nn
from torch.optim import SGD
from adaptive_optimizer import AdaptiveLROptimizer


def test_adaptive_lr_optimizer():
    # Step 1: Define a simple 1D model
    theta_0 = torch.randn(1)
    theta_old = theta_0.clone()

    class Simple1DModel(nn.Module):
        def __init__(self):
            super(Simple1DModel, self).__init__()
            self.param = nn.Parameter(theta_0)  # Start at w = 0
        
        def forward(self, x):
            return self.param * x  # Linear model: y = w * x

    # Step 2: Use built-in MSELoss, which is a quadratic loss function
    loss_function = nn.MSELoss()


    # Step 3: Set up the model, optimizer, and custom adaptive optimizer
    model = Simple1DModel()
    sgd_optimizer = SGD(model.parameters(), lr=.012)  # Dummy optimizer
    adaptive_optimizer = AdaptiveLROptimizer(model, loss_function, sgd_optimizer)

    # Step 4: 
    target = 5 * torch.randn(1) 
    x = 10 * torch.randn(1)  # Input (doesn't matter in 1D)
    print("theta before optimizer:", model.param.item())
    # Step 5: Use the AdaptiveLROptimizer to compute alpha_star and direction
    alpha_star, direction = adaptive_optimizer.compute_alpha_star(x, target)
    print(" theta_0 =", model.param.item())
    print("input x =", x.item())
    print("target =", target.item())
    print(f"Computed alpha_star: {alpha_star.item()}")
    print(f"Direction: {direction.item()}")
    print("theta_1 = theta_0 + d * alpha_star =", theta_old.item() + direction.item() * alpha_star.item())


    # Step 6: Apply the step using AdaptiveLROptimizer
    adaptive_optimizer.apply_step(alpha_star, direction)
    
    # Step 7: 
    theta_target = theta_old.item() + direction.item()* alpha_star.item()
    print(f"Updated parameter value: {model.param.item()}")
    print(f"Target value theta1: {theta_target}")

    assert torch.isclose(model.param, torch.tensor(theta_target) , atol=1e-6), "The parameter did not step to the right value!"
    print("Test passed: Parameter stepped to: ", model.param.item(), " Analytically it should be:", theta_target)

    # Assert that the parameter is now close to the target minimum (w* = 2)
    assert torch.isclose(model(x), target, atol=1e-6), "The parameter model fit is not good enough (should be perfect)!"
    print("Test passed: Model predicts", model(x).item(), "The target is:", target.item())

    assert torch.isclose(loss_function(model(x), target), torch.tensor(0, dtype=torch.float32), atol=1e-6), "The loss is not very close to 0"
    print("Test passed: Loss is :", loss_function(model(x), target).item())

# Run the test
test_adaptive_lr_optimizer()

