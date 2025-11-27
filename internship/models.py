
import torch.nn as nn
from setup import bi, bioiain, config

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



def get_model_class(name):
    if name == "MLP":
        if name == "MLP":
            from models import MLP
            return MLP
        else:
            bi.log("error", "Model name not recognised:", name)
            exit()
            return None