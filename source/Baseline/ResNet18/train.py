import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import multiprocessing
import os
import time
import argparse

from utils import *
from model import *

def proc_logging(id, info, train_info):
    info = "[ResNet18 Result for test_session="+str(id)+"] "+info
    path = "./log/ResNet18/"+train_info+"/"
    if os.path.exists(path) == False:
        os.makedirs(path)
    log_path = path+str(id)+".log"
    log_file = open(log_path, "a")
    log_file.writelines(info+"\n")
    print(info)

def global_logging(info, train_info):
    info = "[ResNet18 Final Result] "+info
    path = "./log/ResNet18/"+train_info+"/"
    log_path = path+"main.log"
    log_file = open(log_path, "a")
    log_file.writelines(info+"\n")
    print(info)

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

def train(id, bs=256, ne=10, lr=0.0001, train_info=""):
    dataset_path = "./data/SEED-IV_concatenate_reshape/"+str(id)+"/"
    train_data = np.load(dataset_path+"train_data.npy")
    train_label = np.load(dataset_path+"train_label.npy")
    test_data = np.load(dataset_path+"test_data.npy")
    test_label = np.load(dataset_path+"test_label.npy")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = bs
    n_epoch = ne
    lr = lr

    model = ResNet18(num_classes=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_dataset = npArraryDataset(train_data, train_label)
    test_dataset = npArraryDataset(test_data, test_label)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset)

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
        proc_logging(id=id, info=("epoch: %d, train_acc: %.4f, test_acc: %.4f" % (epoch, train_acc, test_acc)), train_info=train_info)

        '''save model'''
        save_path = "./model/ResNet18/"+train_info+"/"
        if os.path.exists(save_path) == False:
            os.makedirs(save_path)
        save_path = save_path+"model_"+str(id)+"_epoch_"+str(epoch)+".pt"
        torch.save(model, save_path)

    max_train_acc = max(train_acc_list)
    max_test_acc = max(test_acc_list)
    max_train_acc_idx = train_acc_list.index(max_train_acc)
    max_test_acc_idx = test_acc_list.index(max_test_acc)
    '''remove unnecessary models'''
    save_path = "./model/ResNet18/"+train_info+"/"
    for i in range(n_epoch):
        if i==max_test_acc_idx:
            continue
        else:
            command = "rm "+save_path+"model_"+str(id)+"_epoch_"+str(i)+".pt"
            os.system(command)
    return max_train_acc, max_train_acc_idx, max_test_acc, max_test_acc_idx

def train_(id, train_time_queue, train_acc_queue, test_acc_queue, train_info, bs, ne, lr):
    start_time = time.time()
    max_train_acc, max_train_acc_idx, max_test_acc, max_test_acc_idx = train(id, bs, ne, lr, train_info)
    end_time = time.time()
    train_time = end_time - start_time
    proc_logging(id, ("train_time: %.4f, max_train_acc: %.4f, max_train_acc_idx: %d, max_test_acc: %.4f, max_test_acc_idx: %d" % (train_time, max_train_acc, max_train_acc_idx, max_test_acc, max_test_acc_idx)), train_info)
    train_time_queue.put(train_time)
    train_acc_queue.put(max_train_acc)
    test_acc_queue.put(max_test_acc)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", type=int, default=256)
    parser.add_argument("--ne", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.0001)
    args = parser.parse_args()
    train_info = str(args.bs)+"_"+str(args.ne)+"_"+str(args.lr)
    os.system("rm -r ./log/ResNet18/"+train_info+"/")
    os.system("rm -r ./model/ResNet18/"+train_info+"/")
    train_time_queue = multiprocessing.Queue()
    train_acc_queue = multiprocessing.Queue()
    test_acc_queue = multiprocessing.Queue()
    proc_list = []
    for i in range(0,5):
        proc = multiprocessing.Process(target=train_, args=(i, train_time_queue, train_acc_queue, test_acc_queue, train_info, args.bs, args.ne, args.lr))
        proc_list.append(proc)
        proc.start()
    for proc in proc_list:
        proc.join()

    for i in range(5,10):
        proc = multiprocessing.Process(target=train_, args=(i, train_time_queue, train_acc_queue, test_acc_queue, train_info, args.bs, args.ne, args.lr))
        proc_list.append(proc)
        proc.start()
    for proc in proc_list:
        proc.join()

    for i in range(10,15):
        proc = multiprocessing.Process(target=train_, args=(i, train_time_queue, train_acc_queue, test_acc_queue, train_info, args.bs, args.ne, args.lr))
        proc_list.append(proc)
        proc.start()
    for proc in proc_list:
        proc.join()

    train_acc_list = [train_acc_queue.get() for proc in proc_list]
    test_acc_list = [test_acc_queue.get() for proc in proc_list]
    train_time_list = [train_time_queue.get() for proc in proc_list]
    avg_train_acc = sum(train_acc_list)/len(train_acc_list)
    avg_test_acc = sum(test_acc_list)/len(test_acc_list)
    avg_train_time = sum(train_time_list)/len(train_time_list)
    std_train_acc = np.std(train_acc_list)
    std_test_acc  = np.std(test_acc_list)
    std_train_time = np.std(train_time_list)
    global_logging(("total_models: %d, avg_train_time: %.4f" % (len(test_acc_list), avg_train_time)), train_info)
    global_logging(("avg_train_acc: %.4f, std_train_acc: %.4f" % (avg_train_acc, std_train_acc)), train_info)
    global_logging(("avg_test_acc: %.4f, std_test_acc: %.4f" % (avg_test_acc, std_test_acc)), train_info)