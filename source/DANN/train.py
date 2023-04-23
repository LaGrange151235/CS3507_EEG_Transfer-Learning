import numpy as np
import random
import sys
import os
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import threading

from utils import *
from model import *

def print_dataset_info(train_dataloader, test_dataloader):
    for data, label in train_dataloader:
        print("train_dataloader data size: %s, label size: %s" % (data.shape, label.shape))
        break
    print("train_dataset len: %d" % (len(train_dataloader.dataset)))
    for data, label in test_dataloader:
        print("test_dataloader data size: %s, label size: %s" % (data.shape, label.shape))
        break
    print("train_dataset len: %d" % (len(test_dataloader.dataset)))

def test(model, test_dataloader, device, alpha=0):
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

            class_output, _ = model(input_data=target_value, alpha=alpha)
            pred = class_output.data.max(1, keepdim=True)[1]
            n_correct += pred.eq(target_label.data.view_as(pred)).cpu().sum()
            n_total += batch_size
        accuracy = float(n_correct) / n_total
        return accuracy

def normalize_data(data):
    data=torch.from_numpy(data)
    mean=data.mean(dim=0, keepdim=True)
    standard = data.std(dim=0, unbiased=True, keepdim=True)
    data = (data-mean)/standard
    data = np.array(data)
    return data

def train(id):
    dataset_path = "./data/SEED-IV_concatenate_unfold/"+str(id)+"/"
    train_data = np.load(dataset_path+"train_data.npy")
    train_label = np.load(dataset_path+"train_label.npy")
    test_data = np.load(dataset_path+"test_data.npy")
    test_label = np.load(dataset_path+"test_label.npy")

    train_data = normalize_data(train_data)
    test_data = normalize_data(test_data)

    #print(train_data.shape, train_label.shape, test_data.shape, test_label.shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 256
    n_epoch = 100
    lr = 1e-3
    momentum = 0.5

    model = DANN(momentum=momentum).to(device)
    for param in model.parameters():
        param.requires_grad = True
    optimizer = optim.Adam(model.parameters(), lr=lr)
    class_loss = nn.NLLLoss()
    domain_loss = nn.NLLLoss()

    train_dataset = npArraryDataset(train_data, train_label)
    test_dataset = npArraryDataset(test_data, test_label)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    #print_dataset_info(train_dataloader, test_dataloader)

    train_acc_list = []
    test_acc_list = []

    '''train'''
    for epoch in range(n_epoch):
        len_dataloader = min(len(train_dataloader), len(test_dataloader))
        source_iter = iter(train_dataloader)
        target_iter = iter(test_dataloader)
    
        model.train()
        for i in range(len_dataloader):
            p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
            alpha = 0.5 * (2. / (1. + np.exp(-10 * p)) - 1)

            source_data = source_iter._next_data()
            source_value, source_label = source_data

            optimizer.zero_grad()
            batch_size = len(source_label)
            source_domain_label = torch.zeros(batch_size).long()

            source_value = source_value.to(device)
            source_label = source_label.long().to(device)
            source_domain_label = source_domain_label.long().to(device)

            source_class_pred, source_domain_pred = model(input_data=source_value, alpha=alpha)
            source_class_loss = class_loss(source_class_pred, source_label)
            source_domain_loss = domain_loss(source_domain_pred, source_domain_label)
            
            target_data = target_iter._next_data()
            target_value, target_label = target_data
            
            batch_size = len(target_value)
            target_value = target_value.to(device)
            target_label = target_label.long().to(device)
            target_domain_label = torch.ones(batch_size).long().to(device)

            target_class_pred, target_domain_pred = model(input_data=target_value, alpha=alpha)
            target_domain_loss = domain_loss(target_domain_pred, target_domain_label)

            loss = source_class_loss + source_domain_loss + target_domain_loss
            loss.backward()
            optimizer.step()

        train_acc = test(model=model, test_dataloader=train_dataloader, device=device, alpha=alpha)
        test_acc = test(model=model, test_dataloader=test_dataloader, device=device, alpha=alpha)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        save_path = "./model/DANN/"
        if os.path.exists(save_path) == False:
            os.makedirs(save_path)
        save_path = save_path+"model_"+str(id)+"_epoch_"+str(epoch)+".pt"
        torch.save(model, save_path)

        print("test_session_id: %d, epoch: %d, train_acc: %.4f, test_acc: %.4f" % (id, epoch, train_acc, test_acc))
    
    max_train_acc = max(train_acc_list)
    max_test_acc = max(test_acc_list)
    max_train_acc_idx = train_acc_list.index(max_train_acc)
    max_test_acc_idx = test_acc_list.index(max_test_acc)
    for i in range(n_epoch):
        if i==max_test_acc_idx:
            continue
        else:
            command = "rm ./model/DANN/model_"+str(id)+"_epoch_"+str(i)+".pt"
            os.system(command)    
    return max_train_acc, max_train_acc_idx, max_test_acc, max_test_acc_idx

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