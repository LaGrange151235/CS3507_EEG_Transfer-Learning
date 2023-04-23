import torch
import torch.nn as nn

class FeatureExtractor(nn.Module):
    def __init__(self, momentum):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(310, 256), 
            nn.BatchNorm1d(256, momentum=momentum),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128,  momentum=momentum),
            nn.Dropout(),
            nn.ReLU()
        )
    
    def forward(self, input_data):
        return self.net(input_data)
        
class Classifier(nn.Module):
    def __init__(self, momentum):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64, momentum=momentum),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32, momentum=momentum),
            nn.ReLU(),
            nn.Linear(32, 4)
        )

    def forward(self, input_data):
        return self.net(input_data)

class Discriminator(nn.Module):
    def __init__(self, momentum):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(128, 32),
            nn.BatchNorm1d(32, momentum=momentum),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, input_data):
        return self.net(input_data)