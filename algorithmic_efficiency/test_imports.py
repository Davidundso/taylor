import sys
import os
print("Current sys.path:", sys.path)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print("Updated sys.path:", sys.path)

from source.adaptive_optimizer import AdaptiveLROptimizer
from source.plotting import plot_data

# Dummy model, loss function, and optimizer for testing
import torch.nn as nn
import torch.optim as optim

class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

model = DummyModel()
loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Create an instance of AdaptiveLROptimizer
adaptive_lr_optimizer = AdaptiveLROptimizer(model, loss_function, optimizer)

print("Import and instantiation successful!")