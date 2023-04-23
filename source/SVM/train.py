import numpy as np
from sklearn import svm
import multiprocessing
import pickle
import os

def test(clf,test_data,test_label):
    pred = clf.predict(test_data)
    correct = np.sum(pred==test_label)
    acc = correct / len(test_label)
    return acc

def train(train_data, train_label, test_data, test_label):
    clf=svm.SVC(decision_function_shape='ovo')
    clf.fit(train_data, train_label)
    test_acc=test(clf, test_data, test_label)
    train_acc=test(clf, train_data, train_label)
    return train_acc, test_acc, clf
    
def train_(id, train_acc_queue, test_acc_queue):
    dataset_path = "./data/SEED-IV_concatenate/"+str(id)+"/"
    train_data = np.load(dataset_path+"train_data.npy")
    train_label = np.load(dataset_path+"train_label.npy")
    test_data = np.load(dataset_path+"test_data.npy")
    test_label = np.load(dataset_path+"test_label.npy")
    train_data = train_data.reshape(train_data.shape[0], -1)
    test_data = test_data.reshape(test_data.shape[0], -1)
    train_acc, test_acc, clf = train(train_data, train_label, test_data, test_label)
    print("test_session: %d, train_acc: %.4f, test_acc: %.4f" % (int(id), train_acc, test_acc))
    train_acc_queue.put(train_acc)
    test_acc_queue.put(test_acc)

    clf_data = pickle.dumps(clf)
    if os.path.exists("./model/SVM/") == False:
        os.makedirs("./model/SVM")
    clf_file = open("./model/SVM/clf_"+str(id)+".model", "wb+")
    clf_file.write(clf_data)
    clf_file.close()

    return train_acc, test_acc

if __name__=="__main__":
    train_acc_queue = multiprocessing.Queue()
    test_acc_queue = multiprocessing.Queue()
    proc_list = []
    for i in range(15):
        proc = multiprocessing.Process(target=train_, args=(i, train_acc_queue, test_acc_queue))
        proc_list.append(proc)
        proc.start()
    for proc in proc_list:
        proc.join()

    train_acc_list = [train_acc_queue.get() for proc in proc_list]
    test_acc_list = [test_acc_queue.get() for proc in proc_list]
    avg_train_acc = sum(train_acc_list)/len(train_acc_list)
    avg_test_acc = sum(test_acc_list)/len(test_acc_list)
    print("[Final Result] train_acc: %.4f, test_acc: %.4f" % (avg_train_acc, avg_test_acc))