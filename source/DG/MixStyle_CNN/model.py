import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class MixStyle(nn.Module):
    def __init__(self, p=0.5, alpha=0.1, eps=1e-6):
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha
        self._activated = True

    def forward(self, x):
        if not self.training or not self._activated:
            return x

        if random.random() > self.p:
            return x

        B = x.size(0)

        mu = x.mean(dim=[2, 3, 4], keepdim=True)
        var = x.var(dim=[2, 3, 4], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x-mu) / sig

        lmda = self.beta.sample((B, 1, 1, 1, 1))
        lmda = lmda.to(x.device)

        perm = torch.randperm(B)

        mu2, sig2 = mu[perm], sig[perm]
        mu_mix = mu*lmda + mu2 * (1-lmda)
        sig_mix = sig*lmda + sig2 * (1-lmda)

        return x_normed*sig_mix + mu_mix

class MixStyle_CNN(nn.Module):
    def __init__(self):
        super(MixStyle_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=5,
                               out_channels=32,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               padding_mode='zeros')
        self.conv2 = nn.Conv2d(in_channels=32, 
                               out_channels=64, 
                               kernel_size=3, 
                               stride=1, 
                               padding=1, 
                               padding_mode='zeros')
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 4)
        self.mixstyle = MixStyle()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        if self.training:
            x = x.unsqueeze(-1)
            x = self.mixstyle(x)
            x = x.squeeze(-1)        
        x = self.conv2(x)
        x = F.relu(x)
        if self.training:
            x = x.unsqueeze(-1)
            x = self.mixstyle(x)
            x = x.squeeze(-1)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
