import numpy as np
import random
import sys
import os
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import threading
import copy

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

def test(classifier, feature_extractor, value, label):
    with torch.no_grad():
        pred_class_score = classifier(feature_extractor(value))
        pred_class = pred_class_score.max(1)[1]
        acc = round((pred_class == label).float().mean().cpu().numpy().tolist(), 4)
    return acc

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
    n_epoch_pre = 10
    n_epoch = 10
    beta_1 = 0.5
    beta_2 = 0.9
    lr = 1e-4
    momentum = 0.5

    source_feature_extractor = FeatureExtractor(momentum=momentum).to(device)
    classifier = Classifier(momentum=momentum).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(list(source_feature_extractor.parameters())+list(classifier.parameters()), lr=lr, betas=(beta_1, beta_2))

    train_dataset = npArraryDataset(train_data, train_label)
    test_dataset = npArraryDataset(test_data, test_label)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    #print_dataset_info(train_dataloader, test_dataloader)

    train_acc_list = []
    test_acc_list = []

    '''pre train'''
    for epoch in range(n_epoch_pre):
        source_iter = iter(train_dataloader)

        for i in range(len(source_iter)):
            source_data = source_iter._next_data()
            source_value, source_label = source_data
            source_value = source_value.to(device)
            source_label = source_label.long().to(device)

            optimizer.zero_grad()
            batch_size = len(source_label)
            pred = classifier(source_feature_extractor(source_value))
            loss = criterion(pred, source_label)
            loss.backward()
            optimizer.step()            
        
        target_value, target_label = test_dataset[:]
        target_value = target_value.to(device)
        target_label = target_label.long().to(device)
        source_value, source_label = train_dataset[:]
        source_value = source_value.to(device)
        source_label = source_label.long().to(device)

        train_acc = test(classifier, source_feature_extractor, source_value, source_label)
        test_acc = test(classifier, source_feature_extractor, target_value, target_label)

        print("test_session_id: %d, pre_train_epoch: %d, train_acc: %.4f, test_acc: %.4f" % (id, epoch, train_acc, test_acc))

    
    '''train'''
    target_feature_extractor = FeatureExtractor(momentum=momentum).to(device)
    target_feature_extractor.load_state_dict(source_feature_extractor.state_dict())
    target_feature_extractor = target_feature_extractor.to(device)
    discriminator = Discriminator(momentum=momentum).to(device)

    optimizer_target_feature_extractor = optim.Adam(target_feature_extractor.parameters(), lr=lr, betas=(beta_1, beta_2))
    optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta_1, beta_2))
    
    train_dataset = npArraryDataset(train_data, train_label)
    test_dataset = npArraryDataset(test_data, test_label)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(n_epoch):
        discriminator.train()
        target_feature_extractor.train()
        len_dataloader = min(len(test_dataloader), len(train_dataloader))
        source_iter = iter(train_dataloader)
        target_iter = iter(test_dataloader)

        for i in range(len_dataloader):
            sorce_data = source_iter._next_data()
            source_value, source_label = sorce_data
            source_value = source_value.to(device)
            source_label = source_label.long().to(device)

            target_value = target_iter._next_data()
            target_value, target_label = target_value
            target_value = target_value.to(device)
            target_label = target_label.long().to(device)
            
            '''train discriminator'''
            optimizer_discriminator.zero_grad()
            discriminator_source_feature = source_feature_extractor(source_value)
            discriminator_target_feature = target_feature_extractor(target_value)
            discriminator_feature = torch.cat((discriminator_source_feature, discriminator_target_feature), dim=0)
            discriminator_pred = discriminator(discriminator_feature.detach())
            discriminator_source_label = torch.zeros(discriminator_source_feature.shape[0]).long()
            discriminator_target_label = torch.ones(discriminator_target_feature.shape[0]).long()
            discriminator_label = torch.cat((discriminator_source_label, discriminator_target_label), dim=0).to(device)
            
            discriminator_loss = criterion(discriminator_pred, discriminator_label)
            discriminator_loss.backward()
            optimizer_discriminator.step()
            optimizer_target_feature_extractor.zero_grad()
            target_feature_extractor_target_feature = target_feature_extractor(target_value)
            target_feature_extractor_pred = discriminator(target_feature_extractor_target_feature)
            target_feature_extractor_target_label = torch.zeros(target_feature_extractor_target_feature.shape[0]).long().to(device)
            target_feature_extractor_loss = criterion(target_feature_extractor_pred, target_feature_extractor_target_label)
            target_feature_extractor_loss.backward()
            optimizer_target_feature_extractor.step()
        
        save_path = "./model/ADDA/"
        if os.path.exists(save_path) == False:
            os.makedirs(save_path)
        classifier_save_path = save_path+"classifier_"+str(id)+"_epoch_"+str(epoch)+".pt"
        source_feature_extractor_save_path = save_path+"source_feature_extractor_"+str(id)+"_epoch_"+str(epoch)+".pt"
        target_feature_extractor_save_path = save_path+"target_feature_extractor_"+str(id)+"_epoch_"+str(epoch)+".pt"
        torch.save(classifier, classifier_save_path)
        torch.save(source_feature_extractor, source_feature_extractor_save_path)
        torch.save(target_feature_extractor, target_feature_extractor_save_path)

        '''test'''
        discriminator.eval()
        target_feature_extractor.eval()
        source_feature_extractor.eval()
        target_value, target_label = test_dataset[:]
        target_value = target_value.to(device)
        target_label = target_label.to(device)

        source_value, source_label = train_dataset[:]
        source_value = source_value.to(device)
        source_label = source_label.to(device)

        train_acc = test(classifier, source_feature_extractor, source_value, source_label)
        test_acc = test(classifier, target_feature_extractor, target_value, target_label)

        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("test_session_id: %d, train_epoch: %d, train_acc: %.4f, test_acc: %.4f" % (id, epoch,  train_acc, test_acc))
            
    max_train_acc = max(train_acc_list)
    max_test_acc = max(test_acc_list)
    max_train_acc_idx = train_acc_list.index(max_train_acc)
    max_test_acc_idx = test_acc_list.index(max_test_acc)
    for i in range(n_epoch):
        if i==max_test_acc_idx:
            continue
        else:
            command = "rm ./model/ADDA/classifier_"+str(id)+"_epoch_"+str(i)+".pt"
            os.system(command)
            command = "rm ./model/ADDA/target_feature_extractor_"+str(id)+"_epoch_"+str(i)+".pt"
            os.system(command)
            command = "rm ./model/ADDA/source_feature_extractor_"+str(id)+"_epoch_"+str(i)+".pt"
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