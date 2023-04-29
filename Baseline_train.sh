nohup python ./source/Baseline/SVM/train.py --mode ovr
nohup python ./source/Baseline/SVM/train.py --mode ovo

nohup python ./source/Baseline/MLP/train.py --bs 512 --ne 10 --lr 0.001
nohup python ./source/Baseline/MLP/train.py --bs 512 --ne 10 --lr 0.0001
nohup python ./source/Baseline/MLP/train.py --bs 512 --ne 10 --lr 0.00001
nohup python ./source/Baseline/MLP/train.py --bs 256 --ne 10 --lr 0.001
nohup python ./source/Baseline/MLP/train.py --bs 256 --ne 10 --lr 0.0001
nohup python ./source/Baseline/MLP/train.py --bs 256 --ne 10 --lr 0.00001
nohup python ./source/Baseline/MLP/train.py --bs 128 --ne 10 --lr 0.001
nohup python ./source/Baseline/MLP/train.py --bs 128 --ne 10 --lr 0.0001
nohup python ./source/Baseline/MLP/train.py --bs 128 --ne 10 --lr 0.00001

nohup python ./source/Baseline/CNN/train.py --bs 512 --ne 100 --lr 0.001
nohup python ./source/Baseline/CNN/train.py --bs 512 --ne 100 --lr 0.0001
nohup python ./source/Baseline/CNN/train.py --bs 512 --ne 100 --lr 0.00001
nohup python ./source/Baseline/CNN/train.py --bs 256 --ne 100 --lr 0.001
nohup python ./source/Baseline/CNN/train.py --bs 256 --ne 100 --lr 0.0001
nohup python ./source/Baseline/CNN/train.py --bs 256 --ne 100 --lr 0.00001
nohup python ./source/Baseline/CNN/train.py --bs 128 --ne 100 --lr 0.001
nohup python ./source/Baseline/CNN/train.py --bs 128 --ne 100 --lr 0.0001
nohup python ./source/Baseline/CNN/train.py --bs 128 --ne 100 --lr 0.00001

nohup python ./source/Baseline/ResNet18/train.py --bs 512 --ne 10 --lr 0.001
nohup python ./source/Baseline/ResNet18/train.py --bs 512 --ne 10 --lr 0.0001
nohup python ./source/Baseline/ResNet18/train.py --bs 512 --ne 10 --lr 0.00001
nohup python ./source/Baseline/ResNet18/train.py --bs 256 --ne 10 --lr 0.001
nohup python ./source/Baseline/ResNet18/train.py --bs 256 --ne 10 --lr 0.0001
nohup python ./source/Baseline/ResNet18/train.py --bs 256 --ne 10 --lr 0.00001
nohup python ./source/Baseline/ResNet18/train.py --bs 128 --ne 10 --lr 0.001
nohup python ./source/Baseline/ResNet18/train.py --bs 128 --ne 10 --lr 0.0001
nohup python ./source/Baseline/ResNet18/train.py --bs 128 --ne 10 --lr 0.00001