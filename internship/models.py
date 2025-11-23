
import torch.nn as nn


# ---------------------------
# MLP model
# ---------------------------
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256 ,128], num_classes=8, dropout=0.2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], num_classes)
        )
    def forward(self, x):
        return self.model(x)