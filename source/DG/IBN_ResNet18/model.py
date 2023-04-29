import torch
import torch.nn as nn

class IBN(nn.Module):
    def __init__(self, planes):
        super(IBN, self).__init__()
        self.half1 = int(planes / 2)
        self.IN = nn.InstanceNorm2d(self.half1, affine=True)
        self.BN = nn.BatchNorm2d(self.half1)
        self.relu = nn.ReLU(inplace=True)
        self.half2 = planes - self.half1

    def forward(self, x):
        x1 = self.IN(x[:, :self.half1, :, :].contiguous())
        x2 = self.BN(x[:, self.half1:, :, :].contiguous())
        out = torch.cat((x1, x2), 1)
        out = self.relu(out)
        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.ibn_layer = IBN(planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.ibn_layer(out)
        out = nn.functional.relu(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = nn.functional.relu(out)
        return out

class IBN_ResNet18(nn.Module):
    def __init__(self, num_classes=4):
        super(IBN_ResNet18, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(5, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.linear = nn.Linear(512*BasicBlock.expansion, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = nn.functional.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def IBN_ResNet18_2D():
    return IBN_ResNet18()