


import torch
import torch.nn as nn
import torch.nn.functional as F




class Net(nn.Module):
    def __init__(self, n_features=10, n_chanels=4, fig_size=64, decode=False):
        super().__init__()
        self.n_features = n_features
        self.n_chanels = n_chanels
        self.fig_size = fig_size
        print("FIG_SIZE", self.fig_size)
        print("N_CHANNELS =", n_chanels)
        assert fig_size % 32 == 0
        self.kernel1 = 4
        self.kernel2 = 4
        print("KERNELS:", self.kernel1, self.kernel2)

        self.conv1 = nn.Conv2d(n_chanels, n_chanels * 2, self.kernel1, stride=1)
        self.conv2 = nn.Conv2d(n_chanels * 2, n_chanels * 4, self.kernel2)
        self.red_fig_size = self.fig_size - self.kernel2 - self.kernel1 + 2

        print("RED_FIG_SIZE =", self.red_fig_size)
        assert self.red_fig_size % 2 == 0
        print("CONV2 flat:", n_chanels * 2 * int(self.red_fig_size / 2) * self.red_fig_size)

        self.fc1 = nn.Linear(int(self.red_fig_size / 2) * n_chanels * 2 * self.red_fig_size * n_chanels * 4,
                             fig_size * 2)
        self.fc2 = nn.Linear(fig_size * 2, fig_size)
        self.fc3 = nn.Linear(fig_size, n_features)

        self.rfc3 = nn.Linear(n_features, fig_size)
        self.rfc2 = nn.Linear(fig_size, fig_size * 2)
        self.rfc1 = nn.Linear(fig_size * 2,
                              int(self.red_fig_size / 2) * n_chanels * 2 * self.red_fig_size * n_chanels * 4)
        self.rconv2 = nn.ConvTranspose2d(n_chanels * 4, n_chanels * 2, self.kernel2)
        self.rconv1 = nn.ConvTranspose2d(n_chanels * 2, n_chanels, self.kernel1, stride=1)

    def forward(self, x):
        # [4, n_channels, 32, 32] / [4, n_channels, 100, 100]
        # print(x.shape)
        if len(x.shape) == 3:
            x = x.reshape([1, *x.shape])
        # print(x.shape)
        # print(x)
        x = F.relu(self.conv1(x))
        # print(x.shape)

        # [4, 6, 14, 14]
        # print(x.shape)
        x = F.relu(self.conv2(x))
        # [4, 16, 5, 5]
        # print(x.shape)
        x = torch.flatten(x, 1)
        # [4, 400] / [4, 7400]
        # print(x.shape)
        x = F.relu(self.fc1(x))
        # [4, 120]
        x = F.relu(self.fc2(x))
        # [4, 84]
        x = self.fc3(x)
        # [4, *n_features*]
        return x

    def decode(self, x):
        print(x.shape)
        x = self.rfc3(x)
        x = self.rfc2(x)
        x = self.rfc1(x)
        print(x.shape)
        x = torch.unflatten(x, -1, (self.n_chanels * 4, self.red_fig_size, self.red_fig_size))
        print(x.shape)
        x = self.rconv2(x)
        print(x.shape)
        x = self.rconv1(x)
        print(x.shape)
        return x

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