import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
# from torch.nn.utils import vector_to_parameters  # this is NOT inverse of params_to_vector
from torch.optim import SGD
from curvlinops import GGNLinearOperator
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from curvlinops import GGNLinearOperator


class AdaptiveLROptimizer :
    def __init__(self, model, loss_function, optimizer):
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer

    def compute_alpha_star(self, data, target):
        """
        Given data and target, computes the update direction d_unnormalized and the "optimal" step size alpha_star
        """
        # Step 1: Save the current parameters (theta_0)
        theta_0 = parameters_to_vector(self.model.parameters())  # brauche ich hier clone? Lukas fragen.

        # Step 2: Zero the gradients
        self.optimizer.zero_grad()

        # Step 3: Forward pass to compute loss
        output = self.model(data)
        loss = self.loss_function(output, target)

        # Step 4: Backward pass to compute gradients
        loss.backward()

        # Step 5: Prepare data for GGN (Generalized Gauss-Newton)
        Data = [(data, target)]  # Data as expected by GGNLinearOperator
        params = [p for p in self.model.parameters() if p.requires_grad]  # Filter for trainable parameters

        # Assuming GGNLinearOperator is already imported and available
        GGN = GGNLinearOperator(self.model, self.loss_function, params, Data)  # Instantiate GGN operator
        
        # Step 6: Extract gradients and convert to a vector, move to 5
        gradients = parameters_to_vector(param.grad for param in self.model.parameters() if param.grad is not None)

        # Step 7: Perform optimizer step (this will change the parameters temporarily)
        self.optimizer.step()

        # Step 8: Compute the direction of adjustment (d_unnormalized)
        d_unnormalized = parameters_to_vector(self.model.parameters()) - theta_0

        ## Step 9: Revert the model parameters back to the original state (theta_0)
        #for param, original_param in zip(self.model.parameters(), theta_0.split([param.numel() for param in self.model.parameters()])):
        #    param.data.copy_(original_param.data)
        vector_to_parameters(theta_0, self.model.parameters())

        GGNd = GGN @ d_unnormalized.detach().numpy()  # Multiply GGN * d, outputs np array

        # Step 10: Compute alpha_* based on the direction (this is where you would implement your custom logic)
        GGNd_tensor = torch.tensor(GGNd) # from_numpy()
        
        dGGNd = torch.dot(GGNd_tensor, d_unnormalized)
        
        dg = - torch.dot(gradients, d_unnormalized)  # numerator: - d^T*g
        
        alpha_star = dg / dGGNd

        return alpha_star, d_unnormalized
    

    def apply_step(self, alpha_star, direction):
        """
        Apply a step in the direction `direction` with step size `alpha_star`.
        This updates the model parameters by taking a step along the direction of adjustment.
        """
        # Step 1: Scale the direction by alpha_star
        step_direction = alpha_star * direction

        # Step 2: Update the parameters in the direction of `step_direction`
        with torch.no_grad():  # Ensure no gradients are tracked
            for param, step in zip(self.model.parameters(), step_direction.split([param.numel() for param in self.model.parameters()])):
                param.add_(step.view_as(param))  # Apply the update to each parameter

        # No need to return anything, as the model parameters are updated in place


