import numpy as np
import random
import sys
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

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

best_accs = []
for i in range(15):
    dataset_path = "./data/SEED-IV_concatenate_reshape/"+str(i)+"/"
    train_data = np.load(dataset_path+"train_data.npy")
    train_label = np.load(dataset_path+"train_label.npy")
    test_data = np.load(dataset_path+"test_data.npy")
    test_label = np.load(dataset_path+"test_label.npy")

    batch_size = 32
    n_epoch = 100
    lr = 1e-3

    train_dataset = npArraryDataset(train_data, train_label)
    test_dataset = npArraryDataset(test_data, test_label)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    print_dataset_info(train_dataloader, test_dataloader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNNModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_class = torch.nn.NLLLoss().to(device)
    loss_domain = torch.nn.NLLLoss().to(device)
    for param in model.parameters():
        param.requires_grad = True

    acc_list = []

    for epoch in range(n_epoch):
        len_dataloader = min(len(train_dataloader), len(test_dataloader))
        data_source_iter = iter(train_dataloader)
        data_target_iter = iter(test_dataloader)

        for i in range(len_dataloader):
            p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            data_source = data_source_iter._next_data()
            s_img, s_label = data_source
            s_img = s_img.to(device)
            s_label = s_label.long().to(device)

            model.zero_grad()
            batch_size = len(s_label)
            domain_label = torch.zeros(batch_size).long().to(device)

            class_output, domain_output = model(input_data=s_img, alpha=alpha)
            err_s_label = loss_class(class_output, s_label)
            err_s_domain = loss_domain(domain_output, domain_label)

            data_target = data_target_iter._next_data()
            t_img, _ = data_target
            t_img = t_img.to(device)
            batch_size = len(t_img)
            domain_label = torch.ones(batch_size).long().to(device)

            _, domain_output = model(input_data=t_img, alpha=alpha)
            err_t_domain = loss_domain(domain_output, domain_label)

            err = err_t_domain + err_s_domain + err_s_label
            err.backward()
            optimizer.step()

            #sys.stdout.write("epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f" % (epoch, i + 1, len_dataloader, err_s_label.data.cpu().numpy(), err_s_domain.data.cpu().numpy(), err_t_domain.data.cpu().item()))
            #sys.stdout.flush()
            print("epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f" % (epoch, i + 1, len_dataloader, err_s_label.data.cpu().numpy(), err_s_domain.data.cpu().numpy(), err_t_domain.data.cpu().item()))

        accuracy = 0
        correct = 0
        total = 0

        model.eval()
        data_target_iter = iter(test_dataloader)
        for i in range(len(test_dataloader)):
            data_target = data_target_iter._next_data()
            t_img, t_label = data_target
            t_img = t_img.to(device)
            t_label = t_label.to(device)
            batch_size = len(t_label)

            class_output, _ = model(input_data=t_img, alpha=0)
            pred = class_output.data.max(1, keepdim=True)[1]
            correct += pred.eq(t_label.data.view_as(pred)).cpu().sum()
            total += batch_size

        accuracy = correct * 1.0 / total
        acc_list.append(accuracy)
        print("acc: %.4f" % (accuracy))

    best_acc = max(acc_list)
    best_accs.append(best_acc)
    print("bast acc: %.4f" % (best_acc))

avg_best_acc = sum(best_accs)/15.0
print("avg best acc: %.4f" % (avg_best_acc))