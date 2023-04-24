import numpy as np
import random
import sys
import os
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import threading
import multiprocessing

from utils import *
from model import *

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

            class_output = model(target_value)
            pred = class_output.data.max(1, keepdim=True)[1]
            n_correct += pred.eq(target_label.data.view_as(pred)).cpu().sum()
            n_total += batch_size
        accuracy = float(n_correct) / n_total
        return accuracy

def train(id):
    dataset_path = "./data/SEED-IV_concatenate_reshape/"+str(id)+"/"
    train_data = np.load(dataset_path+"train_data.npy")
    train_label = np.load(dataset_path+"train_label.npy")
    test_data = np.load(dataset_path+"test_data.npy")
    test_label = np.load(dataset_path+"test_label.npy")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 256
    n_epoch = 100
    lr = 1e-4

    model = CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_dataset = npArraryDataset(train_data, train_label)
    test_dataset = npArraryDataset(test_data, test_label)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    train_acc_list = []
    test_acc_list = []

    for epoch in range(n_epoch):
        for i, data_portion in enumerate(train_dataloader):
                train_data, train_label = data_portion
                train_data = train_data.float().to(device)
                train_label = train_label.float().to(device)
                model.train()
                # Forward pass
                outputs = model(train_data)
                loss = criterion(outputs, train_label.long())
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        train_acc = test(model=model, test_dataloader=train_dataloader, device=device)
        test_acc = test(model=model, test_dataloader=test_dataloader, device=device)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("test_session_id: %d, epoch: %d, train_acc: %.4f, test_acc: %.4f, loss: %.4f" % (id, epoch, train_acc, test_acc, loss))

    max_train_acc = max(train_acc_list)
    max_test_acc = max(test_acc_list)
    max_train_acc_idx = train_acc_list.index(max_train_acc)
    max_test_acc_idx = test_acc_list.index(max_test_acc)
    return max_train_acc, max_train_acc_idx, max_test_acc, max_test_acc_idx

def train_process(i, max_train_acc_list, max_test_acc_list):
    max_train_acc, max_train_acc_idx, max_test_acc, max_test_acc_idx = train(i)
    max_train_acc_list.append(max_train_acc)
    max_test_acc_list.append(max_test_acc)
    print("test_session_id: %d, max_train_acc: %.4f @ epoch: %d, max_test_acc: %.4f @ epoch: %d" % (i, max_train_acc, max_train_acc_idx, max_test_acc, max_test_acc_idx))

def train_(i, train_acc_queue, test_acc_queue):
    max_train_acc, max_train_acc_idx, max_test_acc, max_test_acc_idx = train(i)
    train_acc_queue.put(max_train_acc)
    test_acc_queue.put(max_test_acc)

if __name__=="__main__":
    #max_train_acc_list = []
    #max_test_acc_list = []
#
    #for i in range(15):
    #    train_process(i, max_train_acc_list, max_test_acc_list)
#
    #avg_train_acc = float(sum(max_train_acc_list)) / len(max_train_acc_list)
    #avg_test_acc = float(sum(max_test_acc_list)) / len(max_test_acc_list)
    #print("avg_trian_acc: %.4f, avg_test_acc: %.4f" % (avg_train_acc, avg_test_acc))

    train_acc_queue = multiprocessing.Queue()
    test_acc_queue = multiprocessing.Queue()
    proc_list = []
    for i in range(0,5):
        proc = multiprocessing.Process(target=train_, args=(i, train_acc_queue, test_acc_queue))
        proc_list.append(proc)
        proc.start()
    for proc in proc_list:
        proc.join()

    for i in range(5,10):
        proc = multiprocessing.Process(target=train_, args=(i, train_acc_queue, test_acc_queue))
        proc_list.append(proc)
        proc.start()
    for proc in proc_list:
        proc.join()

    for i in range(10,15):
        proc = multiprocessing.Process(target=train_, args=(i, train_acc_queue, test_acc_queue))
        proc_list.append(proc)
        proc.start()
    for proc in proc_list:
        proc.join()

    train_acc_list = [train_acc_queue.get() for proc in proc_list]
    test_acc_list = [test_acc_queue.get() for proc in proc_list]
    avg_train_acc = sum(train_acc_list)/len(train_acc_list)
    avg_test_acc = sum(test_acc_list)/len(test_acc_list)
    std_train_acc = np.std(train_acc_list)
    std_test_acc  = np.std(test_acc_list)
    print("[MixStyle_CNN Final Result] total_models: %d, avg_train_acc: %.4f, train_acc_std:%.4f, avg_test_acc: %.4f, test_acc_std: %.4f" % (len(test_acc_list), avg_train_acc, std_train_acc, avg_test_acc, std_test_acc))