import numpy as np
from sklearn import svm
import multiprocessing
import joblib
import os
import argparse

def test(clf,test_data,test_label):
    pred = clf.predict(test_data)
    correct = np.sum(pred==test_label)
    acc = correct / len(test_label)
    return acc

def test_(id, path):
    clf_path = path+"/clf_"+str(id)+".pickle"
    clf = joblib.load(clf_path)
    dataset_path = "./data/SEED-IV_concatenate/"+str(id)+"/"
    test_data = np.load(dataset_path+"test_data.npy")
    test_label = np.load(dataset_path+"test_label.npy")
    test_data = test_data.reshape(test_data.shape[0], -1)
    acc = test(clf, test_data, test_label)
    print("[SVM Test Result for test_session=%d] acc=%.4f" % (id, acc))
    return acc

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str)
    args = parser.parse_args()
    acc_list = []
    for i in range(15):
        acc = test_(i, args.path)
        acc_list.append(acc)

    print("[SVM Test Final Result] avg_acc: %.4f" % (sum(acc_list)/len(acc_list)))
