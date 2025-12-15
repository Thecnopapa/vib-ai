import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
from torchview import draw_graph


class Net32(nn.Module):
    def __init__(self):
        def __init__(self, n_features=10):
            super().__init__()
            self.conv1 = nn.Conv2d(4, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)
            self.fc3 = nn.Linear(84, n_features)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x