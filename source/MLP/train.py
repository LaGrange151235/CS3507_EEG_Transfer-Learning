import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch

from utils import *

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(310, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )
    def forward(self, input_data):
        output = self.net(input_data)
        return output

def normalize_data(data):
    data=torch.from_numpy(data)
    mean=data.mean(dim=0, keepdim=True)
    standard = data.std(dim=0, unbiased=True, keepdim=True)
    data = (data-mean)/standard
    data = np.array(data)
    return data

def test(model, test_dataloader, device):
    model.eval()
    n_correct = 0
    n_total = 0
    accuracy = 0
    target_iter = iter(test_dataloader)
    with torch.no_grad():
        for i in range(len(test_dataloader)):
            target_data = target_iter._next_data()
            target_value, target_label = target_data
            target_value = target_value.to(device)
            target_label = target_label.long().to(device)
            batch_size = len(target_label)
                
            pred = model(target_value)
            pred = pred.data.max(1, keepdim=True)[1]
            n_correct += pred.eq(target_label.data.view_as(pred)).cpu().sum()
            n_total += batch_size
        accuracy = float(n_correct) / n_total
    return accuracy

def train(id):
    dataset_path = "./data/SEED-IV_concatenate_unfold/"+str(id)+"/"
    train_data = np.load(dataset_path+"train_data.npy")
    train_label = np.load(dataset_path+"train_label.npy")
    test_data = np.load(dataset_path+"test_data.npy")
    test_label = np.load(dataset_path+"test_label.npy")
    
    train_data = normalize_data(train_data)
    test_data = normalize_data(test_data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 256
    n_epoch = 10
    lr = 1e-4

    train_dataset = npArraryDataset(train_data, train_label)
    test_dataset = npArraryDataset(test_data, test_label)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    model = MLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_acc_list = []
    test_acc_list = []

    for epoch in range(n_epoch):
        model.train()
        for idx, train_data in enumerate(train_dataloader):
            train_value, train_label = train_data
            train_value = train_value.to(device)
            train_label = train_label.long().to(device)
            optimizer.zero_grad()
            pred = model(train_value)
            loss = criterion(pred, train_label)
            loss.backward()
            optimizer.step()
        
        train_acc = test(model, train_dataloader, device)
        test_acc = test(model, test_dataloader, device)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("test_session_id: %d, epoch: %d, train_acc: %.4f, test_acc: %.4f" % (id, epoch, train_acc, test_acc))
    
    max_train_acc = max(train_acc_list)
    max_test_acc = max(test_acc_list)
    max_train_acc_idx = train_acc_list.index(max_train_acc)
    max_test_acc_idx = test_acc_list.index(max_test_acc)

    return max_train_acc, max_test_acc_idx, max_test_acc, max_test_acc_idx

def train_process(i, max_train_acc_list, max_test_acc_list):
    max_train_acc, max_train_acc_idx, max_test_acc, max_test_acc_idx = train(i)
    max_train_acc_list.append(max_train_acc)
    max_test_acc_list.append(max_test_acc)
    print("test_session_id: %d, max_train_acc: %.4f @ epoch: %d, max_test_acc: %.4f @ epoch: %d" % (i, max_train_acc, max_train_acc_idx, max_test_acc, max_test_acc_idx))

if __name__=="__main__":
    max_train_acc_list = []
    max_test_acc_list = []

    for i in range(15):
        train_process(i, max_train_acc_list, max_test_acc_list)

    avg_train_acc = float(sum(max_train_acc_list)) / len(max_train_acc_list)
    avg_test_acc = float(sum(max_test_acc_list)) / len(max_test_acc_list)
    print("avg_trian_acc: %.4f, avg_test_acc: %.4f" % (avg_train_acc, avg_test_acc))
    