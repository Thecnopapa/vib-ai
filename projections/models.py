import sys, os, json


import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import torchvision




def SimpleLoss(pred, target, smoot=None, show=False):




    #print("PRED:", pred)
    #print("TARGET", target)

    # clean = pred * target
    # print("CLEAN:", clean)
    # clean_diff = clean - target
    # print(clean_diff)
    # clean_loss = torch.sub(torch.max(clean_diff), torch.max(target))
    # print(clean_loss)

    diff = torch.subtract(pred, target)
    #print("DIFF", diff)
    square = torch.pow(diff, 2)
    #print("SQUARE:", square)
    mean = torch.mean(square)
    #print("MEAN:", mean)


    loss = torch.sub(torch.Tensor([1]), torch.square(mean)).reshape(1)
    #print(loss, loss.shape)


    return loss


def SimpleImageLoss(pred, target, smoot=None, show=False):

    if show:
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[1].imshow(pred.detach().numpy()[0][0])
        ax[1].set_title("pred")
        ax[0].imshow(target.detach().numpy()[0][0])
        ax[0].set_title("target")

    diff = pred - target
    if show:
        ax[2].imshow(diff.detach().numpy()[0][0])
        ax[2].set_title("diff")
        plt.show()
        plt.close()

    loss = 1 + torch.mean(diff)
    return loss


def RotLoss(pred, target, loss_fn, smooth=0, show=True):
    #print(np.random.rand(1)[0])
    show = np.random.rand(1)[0] <= 0.001
    targets= [target]
    angles =[90, 180, 270]
    flip_axes = ((-2,), (-1,))

    for a in angles:
        r = torchvision.transforms.functional.rotate(target, a)
        targets.append(r)

    for a in flip_axes:
        f = torch.flip(target, tuple(a))
        targets.append(f)

    #print(targets)


    losses = [loss_fn(pred, t) for t in targets]
    #print(losses)

    if show:
        fig, axes = plt.subplots(1, 7, figsize=(24, 3))
        axes[0].imshow(pred.detach().numpy()[0][0])



        for ax, t, l in zip(axes[1:], targets, losses):
            #print(t.shape)
            ax.imshow(t.detach().numpy()[0][0])
            ax.set_title(f"{l:.3f}")

        #fig.savefig("/storage/emulated/0/Download/RotLoss.png")
        fig.savefig("figs/RotLoss.png")
        plt.close()


    loss = min(losses)
    #print(loss)


    return loss







def DiceLoss(pred, target, smooth=0, show=False):

    # Apply sigmoid to convert logits to probabilities

    #pred = torch.sigmoid(pred)
    #target = torch.sigmoid(pred)
    #print(pred.shape)
    #print(target.shape)

    if show:
        fig, ax = plt.subplots(1, 6, figsize=(30, 5))
        ax[1].imshow(pred.detach().numpy())
        ax[1].set_title("pred")
        ax[0].imshow(target.detach().numpy())
        ax[0].set_title("target")






    # Calculate intersection and union
    intersection = (pred * target).sum(dim=(0, 1))
    #(intersection.detach().numpy())
    #print(intersection.shape)
    if show:
        i = (pred * target)
        ax[2].imshow(i.detach().numpy())
        ax[2].set_title(f"intersection ({intersection:.3f})")


    union = pred.sum(dim=(0,1)) + target.sum(dim=(0,1))
    #print(union.detach().numpy())
    #print(union.shape)
    if show:
        u = (pred + target)
        ax[3].imshow(u.detach().numpy())
        ax[3].set_title(f"union ({union:.3f})")


    # Compute Dice Coefficient
    dice = (2. * intersection + smooth) / (union + smooth)
    #print(dice.detach().numpy())
    #print(dice.shape)
    if show:
        d = (2. * i + smooth) / (u + smooth)
        ax[4].imshow(d.detach().numpy())
        ax[4].set_title(f"dice ({dice:.3f})")


    # Return Dice Loss
    loss = 1 - dice.mean()
    #print("DICE LOSS:", loss)

    if show:
        l = (1 - d)
        ax[5].imshow(l.detach().numpy())
        ax[5].set_title(f"loss ({loss:.3f})")
        plt.show()
        plt.close()


    return loss




def test_layer(x):
    shape = x.shape

    i = nn.Linear(28 * 28 * 1, 10)
    o = nn.Linear(10, 28 * 28 * 1)

    print(x.shape)

    x = nn.Flatten()(x)
    print(x.shape)
    x = i(x)
    x = o(x)
    x = nn.Unflatten(-1, shape)(x)
    return x




class SmallNetQMINST(nn.Module):
    def __init__(self, n_features=10, n_channels=1, fig_size=28, batch_size=1, mode="normal"):
        super().__init__()
        self.n_features = n_features
        self.n_channels = n_channels
        self.fig_size = fig_size
        self.inner_figs = [int(self.fig_size/2), int(self.fig_size/4)+1]
        self.channels = [self.n_channels, 32, 128]
        self.mode = mode

        self.f_layers = [

            nn.Conv2d(self.channels[0], self.channels[1], 3, stride=2, padding=1), # 1, 32, 14, 14 (6272)
            nn.LeakyReLU(),
            nn.Conv2d(self.channels[1], self.channels[2], 2, stride=2, padding=1),  # 1, 128, 8, 8 (8192)
            nn.LeakyReLU(),
            nn.Flatten(), # 1, (flat)
            nn.Linear(self.inner_figs[1] * self.inner_figs[1] * self.channels[2], n_features),
            nn.Softmax(dim=-1)
        ]



        self.r_layers = [
            nn.Linear(n_features, self.inner_figs[1] * self.inner_figs[1] * self.channels[2]),
            nn.Unflatten(-1, (self.channels[2], self.inner_figs[1], self.inner_figs[1])),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(self.channels[2], self.channels[1], 2, stride=2, padding=1, output_padding=0),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(self.channels[1], self.channels[0], 3, stride=2, padding=1, output_padding=1),

            nn.Softmax(dim=-1)
        ]
        self.f_net = nn.Sequential(*self.f_layers)
        self.r_net = nn.Sequential(*self.r_layers)
        self.auto_f_net = nn.Sequential(*self.f_layers, *self.r_layers)
        self.auto_r_net = nn.Sequential(*self.r_layers[::-1], *self.f_layers[::-1])
    def forward(self, x, mode=None):
        if mode is None:
            mode = self.mode
        if len(x.shape) == 3:
            x = x.reshape([1, *x.shape])

        if mode == "normal":
            return self._forward(x)
        elif mode == "auto":
            return self._auto_forward(x)
        else:
            return None

    def backward(self, x, mode=None):
        if mode is None:
            mode = self.mode
        if len(x.shape) == 3:
            x = x.reshape([1, *x.shape])

        if mode == "normal":
            return self._backward(x)
        elif mode == "auto":
            return self.auto_r_net(x)
        else:
            return None


    def _forward(self, x):
        x = self.f_net(x)
        return x

    def _backward(self, x):
        x = self.r_net(x)
        return x

    def _auto_forward(self, x):
        x = self.auto_f_net(x)
        return x

    def _auto_backward(self, x):
        x = self.auto_r_net(x)
        return x


class SmallNetv3(nn.Module):
    def __init__(self, n_features=10, n_channels=1, fig_size=28):
        super().__init__()
        self.n_features = n_features
        self.n_channels = n_channels
        self.fig_size = fig_size
        print("FIG_SIZE", self.fig_size)
        print("N_CHANNELS =", n_channels)
        #assert fig_size % 32 == 0
        self.kernel1 = 5
        self.stride1 = 2
        self.kernel2 = 4
        self.stride2 = 2
        print("KERNELS:", self.kernel1, self.kernel2)
        self.red_fig_size = 30
        print("RED_FIG_SIZE =", self.red_fig_size)
        assert self.red_fig_size % 2 == 0

        f_layers = [
            nn.Conv2d(n_channels, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, n_features),
            #nn.ReLU(),
            #nn.Linear(fig_size * 4, fig_size*2),
            #nn.ReLU(),
            #nn.Linear(fig_size*2, n_features),
            #nn.Softmax(dim=1)
        ]

        print("CONV2 flat:", n_channels * 2 * int(self.red_fig_size / 2) * self.red_fig_size)

        r_layers = [
            nn.Linear(n_features, 128 * 16 *16),
            nn.ReLU(),
            #nn.Linear(fig_size*2, fig_size * 4),
            #nn.ReLU(),
            #nn.Linear(fig_size * 4, int(self.fig_size / 2) * n_channels*2 *int(self.fig_size/2)),
            nn.Unflatten(-1, (128, 16, 16)),
            #nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, 2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, 2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, 2, padding=1, output_padding=1),
            #nn.ReLU(),
            #nn.ConvTranspose2d(n_channels * 2, n_channels, self.kernel1, stride=self.stride1, padding=2, output_padding=1),
            #nn.Softmax(dim=2)

        ]
        self.f_net = nn.Sequential(*f_layers)
        self.r_net = nn.Sequential(*r_layers)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.reshape([1, *x.shape])
        x = self.f_net(x)
        return x

    def backward(self, x):
        x = self.r_net(x)
        return x




class SmallNetv2(nn.Module):
    def __init__(self, n_features=None, n_channels=1, fig_size=128):
        super().__init__()
        self.n_features = n_features
        self.n_channels = n_channels
        self.fig_size = fig_size
        print("FIG_SIZE", self.fig_size)
        print("N_CHANNELS =", n_channels)
        assert fig_size % 32 == 0
        self.kernel1 = 5
        self.stride1 = 2
        self.kernel2 = 4
        self.stride2 = 2
        print("KERNELS:", self.kernel1, self.kernel2)
        self.red_fig_size = 30
        print("RED_FIG_SIZE =", self.red_fig_size)
        assert self.red_fig_size % 2 == 0

        f_layers = [
            nn.Conv2d(n_channels, n_channels * 2, self.kernel1, stride=self.stride1),
            nn.ReLU(),
            nn.Conv2d(n_channels * 2, n_channels * 4, self.kernel2, stride=self.stride2),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(int(self.red_fig_size / 2) * n_channels * 2 * self.red_fig_size * n_channels * 4,
                      fig_size * 2),
            nn.ReLU(),

            nn.Linear(fig_size * 2, fig_size),
            nn.ReLU(),
            nn.Linear(fig_size, n_features),
            nn.Softmax(dim=1)
        ]

        print("CONV2 flat:", n_channels * 2 * int(self.red_fig_size / 2) * self.red_fig_size)

        r_layers = [
            nn.Linear(n_features, fig_size),
            nn.ReLU(),
            nn.Linear(fig_size, fig_size * 2),
            nn.ReLU(),
            nn.Linear(fig_size * 2,
                      int(self.red_fig_size / 2) * n_channels * 2 * self.red_fig_size * n_channels * 4),
            nn.Unflatten(-1, (self.n_channels * 4, self.red_fig_size, self.red_fig_size)),
            nn.ReLU(),
            nn.ConvTranspose2d(n_channels * 4, n_channels * 2, self.kernel2, stride=self.stride2),
            nn.ReLU(),
            nn.ConvTranspose2d(n_channels * 2, n_channels, self.kernel1, stride=self.stride1, output_padding=1),
            #nn.Softmax(dim=2)

        ]
        self.f_net = nn.Sequential(*f_layers)
        self.r_net = nn.Sequential(*r_layers)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.reshape([1, *x.shape])
        x = self.f_net(x)
        return x

    def backward(self, x):
        x = self.r_net(x)
        return x



class SmallNet(nn.Module):
    def __init__(self, n_features=10, n_channels=4, fig_size=64, decode=False):
        super().__init__()
        self.n_features = n_features
        self.n_channels = n_channels
        self.fig_size = fig_size
        print("FIG_SIZE", self.fig_size)
        print("N_CHANNELS =", n_channels)
        assert fig_size % 32 == 0
        self.kernel1 = 5
        self.stride1 = 2
        self.kernel2 = 4
        self.stride2 = 2
        print("KERNELS:", self.kernel1, self.kernel2)

        self.conv1 = nn.Conv2d(n_channels, n_channels * 2, self.kernel1, stride=self.stride1)
        self.conv2 = nn.Conv2d(n_channels * 2, n_channels * 4, self.kernel2, stride=self.stride2)
        #self.red_fig_size = self.fig_size - self.kernel2 - self.kernel1 +
        self.red_fig_size = 30

        print("RED_FIG_SIZE =", self.red_fig_size)
        assert self.red_fig_size % 2 == 0
        print("CONV2 flat:", n_channels * 2 * int(self.red_fig_size / 2) * self.red_fig_size)

        self.fc1 = nn.Linear(int(self.red_fig_size / 2) * n_channels * 2 * self.red_fig_size * n_channels * 4,
                             fig_size * 2)
        self.fc2 = nn.Linear(fig_size * 2, fig_size)
        self.fc3 = nn.Linear(fig_size, n_features)

        self.rfc3 = nn.Linear(n_features, fig_size)
        self.rfc2 = nn.Linear(fig_size, fig_size * 2)
        self.rfc1 = nn.Linear(fig_size * 2,
                              int(self.red_fig_size / 2) * n_channels * 2 * self.red_fig_size * n_channels * 4)
        self.rconv2 = nn.ConvTranspose2d(n_channels * 4, n_channels * 2, self.kernel2, stride=self.stride2)
        self.rconv1 = nn.ConvTranspose2d(n_channels * 2, n_channels, self.kernel1, stride=self.stride1)

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
        self.decode(x)
        return x

    def decode(self, x):

        x = self.rfc3(x)
        x = self.rfc2(x)
        x = self.rfc1(x)
        x = torch.unflatten(x, -1, (self.n_channels * 4, self.red_fig_size, self.red_fig_size))
        x = self.rconv2(x)
        x = self.rconv1(x)
        self.last_decode = x
        return x




class Net(nn.Module):
    def __init__(self, n_features=10, n_channels=4, fig_size=64, decode=False):
        super().__init__()
        self.n_features = n_features
        self.n_channels = n_channels
        self.fig_size = fig_size
        print("FIG_SIZE", self.fig_size)
        print("N_CHANNELS =", n_channels)
        assert fig_size % 32 == 0
        self.kernel1 = 4
        self.kernel2 = 4
        print("KERNELS:", self.kernel1, self.kernel2)

        self.conv1 = nn.Conv2d(n_channels, n_channels * 2, self.kernel1, stride=1)
        self.conv2 = nn.Conv2d(n_channels * 2, n_channels * 4, self.kernel2)
        self.red_fig_size = self.fig_size - self.kernel2 - self.kernel1 + 2

        print("RED_FIG_SIZE =", self.red_fig_size)
        assert self.red_fig_size % 2 == 0
        print("CONV2 flat:", n_channels * 2 * int(self.red_fig_size / 2) * self.red_fig_size)

        self.fc1 = nn.Linear(int(self.red_fig_size / 2) * n_channels * 2 * self.red_fig_size * n_channels * 4,
                             fig_size * 2)
        self.fc2 = nn.Linear(fig_size * 2, fig_size)
        self.fc3 = nn.Linear(fig_size, n_features)

        self.rfc3 = nn.Linear(n_features, fig_size)
        self.rfc2 = nn.Linear(fig_size, fig_size * 2)
        self.rfc1 = nn.Linear(fig_size * 2,
                              int(self.red_fig_size / 2) * n_channels * 2 * self.red_fig_size * n_channels * 4)
        self.rconv2 = nn.ConvTranspose2d(n_channels * 4, n_channels * 2, self.kernel2)
        self.rconv1 = nn.ConvTranspose2d(n_channels * 2, n_channels, self.kernel1, stride=1)

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
        print(x)
        x = x * self.rfc3._bias
        print(x)
        x = self.rfc3(x)
        x = x * self.rfc2._bias
        print(x)
        x = self.rfc2(x)
        x = x * self.rfc1._bias
        print(x)
        x = self.rfc1(x)
        print(x.shape)
        x = torch.unflatten(x, -1, (self.n_channels * 4, self.red_fig_size, self.red_fig_size))
        print(x)
        print(x.shape)
        for n, (t, b) in enumerate(zip(x, self.rconv2._bias)):
            x[n] = t * b
        print(x.shape)
        print(x)

        x = self.rconv2(x)
        print(x)
        print(x.shape)
        for n, (t, b) in enumerate(zip(x, self.rconv1._bias)):
            x[n] = t * b
        print(x.shape)
        print(x)
        x = self.rconv1(x)
        #print(x.shape)
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
