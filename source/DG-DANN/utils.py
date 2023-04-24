import torch
from torch.utils.data import Dataset

class npArraryDataset(Dataset):
    def __init__(self, data, label, domain):
        self.data = torch.from_numpy(data).float()
        self.label = torch.from_numpy(label).float()
        self.domain = torch.from_numpy(domain).float()
        self.domain_list = list(set(domain))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.label[index]
        z = self.domain_list.index(self.domain[index])
        return x, y, z