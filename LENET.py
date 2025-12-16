import torch
import torch.nn as nn
import torch.nn.functional as F


# 定义LeNet网络
class LeNet(nn.Module):
    def __init__(self, ln, input_shape, n_class):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(input_shape, 32, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        ln = (ln - 4 - 1) // 1 + 1
        ln = (ln - 1 - 1) // 2 + 1
        ln = (ln - 4 - 1) // 1 + 1
        ln = (ln - 1 - 1) // 2 + 1
        self.fc2 = nn.Linear(1200, 600)
        self.fc3 = nn.Linear(600, n_class)
        self.fc1 = nn.Linear(ln * ln * 64, 1200)

    def forward(self, img):
        x = self.pool1(torch.relu(self.conv1(img)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(img.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

