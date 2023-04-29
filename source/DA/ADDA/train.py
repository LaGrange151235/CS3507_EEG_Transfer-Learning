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
    info = "[ADDA Result for test_session="+str(id)+"] "+info
    path = "./log/ADDA/"+train_info+"/"
    if os.path.exists(path) == False:
        os.makedirs(path)
    log_path = path+str(id)+".log"
    log_file = open(log_path, "a")
    log_file.writelines(info+"\n")
    print(info)

def global_logging(info, train_info):
    info = "[ADDA Final Result] "+info
    path = "./log/ADDA/"+train_info+"/"
    log_path = path+"main.log"
    log_file = open(log_path, "a")
    log_file.writelines(info+"\n")
    print(info)

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

def train(id, bs=256, nep=10, ne=10, lr=0.0001, train_info=""):
    dataset_path = "./data/SEED-IV_concatenate_unfold/"+str(id)+"/"
    train_data = np.load(dataset_path+"train_data.npy")
    train_label = np.load(dataset_path+"train_label.npy")
    test_data = np.load(dataset_path+"test_data.npy")
    test_label = np.load(dataset_path+"test_label.npy")

    train_data = normalize_data(train_data)
    test_data = normalize_data(test_data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = bs
    n_epoch_pre = nep
    n_epoch = ne
    lr = lr
    beta_1 = 0.5
    beta_2 = 0.9
    momentum = 0.5

    source_feature_extractor = FeatureExtractor(momentum=momentum).to(device)
    classifier = Classifier(momentum=momentum).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(list(source_feature_extractor.parameters())+list(classifier.parameters()), lr=lr, betas=(beta_1, beta_2))

    train_dataset = npArraryDataset(train_data, train_label)
    test_dataset = npArraryDataset(test_data, test_label)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

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

        proc_logging(id, ("pre_train_epoch: %d, train_acc: %.4f, test_acc: %.4f" % (epoch, train_acc, test_acc)), train_info)

    
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
        
        '''test'''
        classifier.eval()
        discriminator.eval()
        target_feature_extractor.eval()
        source_feature_extractor.eval()
        train_dataset_ = npArraryDataset(train_data, train_label)
        test_dataset_ = npArraryDataset(test_data, test_label)
        target_value, target_label = test_dataset_[:]
        target_value = target_value.to(device)
        target_label = target_label.to(device)

        source_value, source_label = train_dataset_[:]
        source_value = source_value.to(device)
        source_label = source_label.to(device)

        train_acc = test(classifier, source_feature_extractor, source_value, source_label)
        test_acc = test(classifier, target_feature_extractor, target_value, target_label)

        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        proc_logging(id=id, info=("train_epoch: %d, train_acc: %.4f, test_acc: %.4f" % (epoch, train_acc, test_acc)), train_info=train_info)

        '''save model'''
        save_path = "./model/ADDA/"+train_info+"/"
        if os.path.exists(save_path) == False:
            os.makedirs(save_path)
        classifier_save_path = save_path+"classifier_"+str(id)+"_epoch_"+str(epoch)+".pt"
        torch.save(classifier, classifier_save_path)
        
        #source_feature_extractor_save_path = save_path+"source_feature_extractor_"+str(id)+"_epoch_"+str(epoch)+".pt"
        #torch.save(source_feature_extractor, source_feature_extractor_save_path)

        target_feature_extractor_save_path = save_path+"target_feature_extractor_"+str(id)+"_epoch_"+str(epoch)+".pt"
        torch.save(target_feature_extractor, target_feature_extractor_save_path)
            
    max_train_acc = max(train_acc_list)
    max_test_acc = max(test_acc_list)
    max_train_acc_idx = train_acc_list.index(max_train_acc)
    max_test_acc_idx = test_acc_list.index(max_test_acc)
    '''remove unnecessary models'''
    save_path = "./model/ADDA/"+train_info+"/"
    for i in range(n_epoch):
        if i==max_test_acc_idx:
            continue
        else:
            command = "rm "+save_path+"classifier_"+str(id)+"_epoch_"+str(i)+".pt"
            os.system(command)
            #command = "rm "+save_path+"source_feature_extractor_"+str(id)+"_epoch_"+str(i)+".pt"
            #os.system(command)
            command = "rm "+save_path+"target_feature_extractor_"+str(id)+"_epoch_"+str(i)+".pt"
            os.system(command)
    return max_train_acc, max_train_acc_idx, max_test_acc, max_test_acc_idx

def train_(id, train_time_queue, train_acc_queue, test_acc_queue, train_info, bs, nep, ne, lr):
    start_time = time.time()
    max_train_acc, max_train_acc_idx, max_test_acc, max_test_acc_idx = train(id, bs, nep, ne, lr, train_info)
    end_time = time.time()
    train_time = end_time - start_time
    proc_logging(id, ("train_time: %.4f, max_train_acc: %.4f, max_train_acc_idx: %d, max_test_acc: %.4f, max_test_acc_idx: %d" % (train_time, max_train_acc, max_train_acc_idx, max_test_acc, max_test_acc_idx)), train_info)
    train_time_queue.put(train_time)
    train_acc_queue.put(max_train_acc)
    test_acc_queue.put(max_test_acc)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", type=int, default=256)
    parser.add_argument("--nep", type=int, default=10)
    parser.add_argument("--ne", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.0001)
    args = parser.parse_args()
    train_info = str(args.bs)+"_"+str(args.ne)+"_"+str(args.lr)
    os.system("rm -r ./log/ADDA/"+train_info+"/")
    os.system("rm -r ./model/ADDA/"+train_info+"/")
    train_time_queue = multiprocessing.Queue()
    train_acc_queue = multiprocessing.Queue()
    test_acc_queue = multiprocessing.Queue()
    proc_list = []
    for i in range(0,5):
        proc = multiprocessing.Process(target=train_, args=(i, train_time_queue, train_acc_queue, test_acc_queue, train_info, args.bs, args.nep, args.ne, args.lr))
        proc_list.append(proc)
        proc.start()
    for proc in proc_list:
        proc.join()

    for i in range(5,10):
        proc = multiprocessing.Process(target=train_, args=(i, train_time_queue, train_acc_queue, test_acc_queue, train_info, args.bs, args.nep, args.ne, args.lr))
        proc_list.append(proc)
        proc.start()
    for proc in proc_list:
        proc.join()

    for i in range(10,15):
        proc = multiprocessing.Process(target=train_, args=(i, train_time_queue, train_acc_queue, test_acc_queue, train_info, args.bs, args.nep, args.ne, args.lr))
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