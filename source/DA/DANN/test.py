import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse

from utils import *
from model import *

def normalize_data(data):
    data=torch.from_numpy(data)
    mean=data.mean(dim=0, keepdim=True)
    standard = data.std(dim=0, unbiased=True, keepdim=True)
    data = (data-mean)/standard
    data = np.array(data)
    return data

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

def test_(id, path):
    dir_list = os.listdir(path)
    for dir in dir_list:
        if "model_"+str(id)+"_" in dir:
            model_path = path+"/"+dir
            model = torch.load(model_path)
            dataset_path = "./data/SEED-IV_concatenate_unfold/"+str(id)+"/"
            test_data = np.load(dataset_path+"test_data.npy")
            test_data = normalize_data(test_data)
            test_label = np.load(dataset_path+"test_label.npy")
            test_dataset = npArraryDataset(test_data, test_label)
            test_dataloader = DataLoader(dataset=test_dataset)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)

            acc = test(model, test_dataloader, device)
            print("[DANN Test Result for test_session=%d] acc=%.4f" % (id, acc))
            return acc

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str)
    args = parser.parse_args()
    acc_list = []
    for i in range(15):
        acc = test_(i, args.path)
        acc_list.append(acc)

    print("[DANN Test Final Result] avg_acc: %.4f" % (sum(acc_list)/len(acc_list)))