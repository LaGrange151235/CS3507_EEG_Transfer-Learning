import numpy as np
from sklearn import svm
import joblib
import multiprocessing
import os
import time
import argparse

def test(clf,test_data,test_label):
    pred = clf.predict(test_data)
    correct = np.sum(pred==test_label)
    acc = correct / len(test_label)
    return acc

def train(train_data, train_label, test_data, test_label, mode='ovo'):
    train_acc = 0
    test_acc = 0
    clf = None
    if mode == 'ovo':
        clf=svm.SVC(decision_function_shape='ovo')
        clf.fit(train_data, train_label)
        test_acc=test(clf, test_data, test_label)
        train_acc=test(clf, train_data, train_label)
    if mode == 'ovr':
        clf=svm.SVC(decision_function_shape='ovr')
        clf.fit(train_data, train_label)
        test_acc=test(clf, test_data, test_label)
        train_acc=test(clf, train_data, train_label)
    return train_acc, test_acc, clf
    
def train_(id, train_time_queue, train_acc_queue, test_acc_queue, mode='ovo'):
    dataset_path = "./data/SEED-IV_concatenate/"+str(id)+"/"
    train_data = np.load(dataset_path+"train_data.npy")
    train_label = np.load(dataset_path+"train_label.npy")
    test_data = np.load(dataset_path+"test_data.npy")
    test_label = np.load(dataset_path+"test_label.npy")
    train_data = train_data.reshape(train_data.shape[0], -1)
    test_data = test_data.reshape(test_data.shape[0], -1)

    start_time = time.time()
    train_acc, test_acc, clf = train(train_data, train_label, test_data, test_label, mode)
    end_time = time.time()
    train_time = end_time - start_time
    proc_logging(id, ("train_time: %.4f, train_acc: %.4f, test_acc: %.4f" % (train_time, train_acc, test_acc)), mode)
    train_time_queue.put(train_time)
    train_acc_queue.put(train_acc)
    test_acc_queue.put(test_acc)


    if os.path.exists("./model/SVM/"+mode+"/") == False:
        os.makedirs("./model/SVM/"+mode+"/")
    joblib.dump(clf, "./model/SVM/"+mode+"/clf_"+str(id)+".pickle")

def proc_logging(id, info, mode):
    info = "[SVM Result for test_session="+str(id)+"] "+info
    path = "./log/SVM/"+mode+"/"
    if os.path.exists(path) == False:
        os.makedirs(path)
    log_path = path+str(id)+".log"
    log_file = open(log_path, "a")
    log_file.writelines(info+"\n")
    print(info)

def global_logging(info, mode):
    info = "[SVM Final Result] "+info
    path = "./log/SVM/"+mode+"/"
    log_path = path+"main.log"
    log_file = open(log_path, "a")
    log_file.writelines(info+"\n")
    print(info)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="ovo")
    args = parser.parse_args()
    os.system("rm -r ./log/SVM/"+args.mode)
    os.system("rm -r ./model/SVM/"+args.mode)
    train_time_queue = multiprocessing.Queue()
    train_acc_queue = multiprocessing.Queue()
    test_acc_queue = multiprocessing.Queue()
    proc_list = []  
    for i in range(15):
        proc = multiprocessing.Process(target=train_, args=(i, train_time_queue, train_acc_queue, test_acc_queue, args.mode))
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
    global_logging(("total_models: %d, avg_train_time: %.4f" % (len(test_acc_list), avg_train_time)), args.mode)
    global_logging(("avg_train_acc: %.4f, std_train_acc: %.4f" % (avg_train_acc, std_train_acc)), args.mode)
    global_logging(("avg_test_acc: %.4f, std_test_acc: %.4f" % (avg_test_acc, std_test_acc)), args.mode)