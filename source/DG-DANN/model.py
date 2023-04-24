import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class DG_DANN(nn.Module):
    def __init__(self, momentum):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(310, 256),
            nn.BatchNorm1d(num_features=256, momentum=momentum),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(num_features=128, momentum=momentum),
            nn.Dropout(),
            nn.ReLU()
        )

        self.class_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(num_features=64, momentum=momentum),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(num_features=32, momentum=momentum),
            nn.ReLU(),
            nn.Linear(32, 4),
            nn.LogSoftmax(dim=1)
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(128, 32),
            nn.BatchNorm1d(32, momentum=momentum),
            nn.ReLU(),
            nn.Linear(32, 14),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, input_data, class_alpha=1.0, domain_alpha=1.0):
        feature = self.feature_extractor(input_data)
        class_reverse_feature = ReverseLayer.apply(feature, class_alpha)
        domain_reverse_feature = ReverseLayer.apply(feature, domain_alpha)
        class_pred = self.class_classifier(class_reverse_feature)
        domain_pred = self.domain_classifier(domain_reverse_feature)
        return class_pred, domain_pred
    
class ReverseLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_outputs):
        output = grad_outputs.neg() * ctx.alpha
        return output, None