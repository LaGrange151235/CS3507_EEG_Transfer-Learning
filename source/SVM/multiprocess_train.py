import os

for i in range(15):
    os.system("nohup python3 /home/ubuntu/EEG_Transfer-Learning/source/SVM/train.py --id "+str(i)+" >> "+str(i)+".log 2>&1 &")