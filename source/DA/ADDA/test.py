import torch
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

def test(classifier, feature_extractor, value, label):
    classifier.eval()
    feature_extractor.eval()
    with torch.no_grad():
        pred_class_score = classifier(feature_extractor(value))
        pred_class = pred_class_score.max(1)[1]
        acc = round((pred_class == label).float().mean().cpu().numpy().tolist(), 4)
    return acc


def test_(id, path):
    dir_list = os.listdir(path)
    classifier_dir = ""
    target_feature_extractor_dir = ""
    for dir in dir_list:
        if "classifier_"+str(id)+"_" in dir:
            classifier_dir = dir
        if "target_feature_extractor_"+str(id)+"_" in dir:
            target_feature_extractor_dir = dir

    classifier_path = path+"/"+classifier_dir
    target_feature_extractor_path = path+"/"+target_feature_extractor_dir
    classifier = torch.load(classifier_path)
    target_feature_extractor = torch.load(target_feature_extractor_path)
    dataset_path = "./data/SEED-IV_concatenate_unfold/"+str(id)+"/"
    test_data = np.load(dataset_path+"test_data.npy")
    test_data = normalize_data(test_data)
    test_label = np.load(dataset_path+"test_label.npy")
    test_dataset = npArraryDataset(test_data, test_label)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = classifier.to(device)
    target_feature_extractor = target_feature_extractor.to(device)
    target_value, target_label = test_dataset[:]
    target_value = target_value.to(device)
    target_label = target_label.long().to(device)
    acc = test(classifier, target_feature_extractor, target_value, target_label)
    print("[ADDA Test Result for test_session=%d] acc=%.4f" % (id, acc))
    return acc

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str)
    args = parser.parse_args()
    acc_list = []
    for i in range(15):
        acc = test_(i, args.path)
        acc_list.append(acc)

    print("[ADDA Test Final Result] avg_acc: %.4f" % (sum(acc_list)/len(acc_list)))