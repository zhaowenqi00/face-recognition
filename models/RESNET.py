import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self,in_channel,out_channel,stride=1):
        super(ResBlock,self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
        )
        self.shortcut = nn.Sequential()
        if in_channel != out_channel or stride > 1:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(out_channel),
            )

    def forward(self,x):
        out1=self.layer(x)
        out2=self.shortcut(x)
        out=out1+out2
        out=F.relu(out)
        return out


class ResNet_1(nn.Module):
    def make_layer(self, block, out_channel, stride, num_block):
        layers_list = []
        for i in range(num_block):
            if i == 0:
                in_stride = stride
            else:
                in_stride = 1
            layers_list.append(block(self.in_channel,
                                     out_channel,
                                     in_stride))
            self.in_channel=out_channel
        return nn.Sequential(*layers_list)

    def __init__(self, ResBlock, ln, input_shape, n_class):
        super(ResNet_1, self).__init__()
        # if stn_flag:
        #     ln1 = (ln - 7) // 1 + 1
        #     ln1 = (ln1 - 1 - 1) // 2 + 1
        #     ln1 = (ln1 - 4 - 1) // 1 + 1
        #     ln1 = (ln1 - 1 - 1) // 2 + 1
        # self.f = stn_flag
        # if self.f:
        #     self.localization = nn.Sequential(
        #         nn.Conv2d(1, 8, kernel_size=7),
        #         nn.MaxPool2d(2, stride=2),
        #         nn.ReLU(True),
        #         nn.Conv2d(8, 10, kernel_size=5),
        #         nn.MaxPool2d(2, stride=2),
        #         nn.ReLU(True)
        #     )
        #
        #     self.fc_loc = nn.Sequential(
        #         nn.Linear(10 * ln1 * ln1, 32),
        #         nn.ReLU(True),
        #         nn.Linear(32, 3 * 2)
        #     )
        self.in_channel=32
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_shape, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.layer1 = \
            self.make_layer(ResBlock,
                            64, 2, 2)
        self.layer2 = \
            self.make_layer(ResBlock, 128, 2, 2)
        self.layer3 = \
            self.make_layer(ResBlock, 256, 2, 2)
        self.layer4 = \
            self.make_layer(ResBlock, 512, 2, 2)
        self.fc = nn.Linear(8192, n_class)
    #
    # def stn(self, x):
    #     xs = self.localization(x)
    #     # print(xs.shape)
    #     xs = xs.view(-1, 10 * 21 * 21)
    #     # print(xs.shape)
    #     theta = self.fc_loc(xs)
    #     # print(theta.shape)
    #     theta = theta.view(-1, 2, 3)
    #     # print(theta.shape)
    #
    #     grid = F.affine_grid(theta, x.size())
    #     x = F.grid_sample(x, grid)
    #     # print(x.shape)
    #
    #     return x

    def forward(self,x):
        # if self.f:
        #     x = self.stn(x)
        out=self.conv1(x)
        out=self.layer1(out)
        out=self.layer2(out)
        out=self.layer3(out)
        out=self.layer4(out)
        out=F.avg_pool2d(out,2)
        # print(out.shape)
        out=out.view(x.size(0),-1)
        out=self.fc(out)
        return out


def RESNET(ln, input_shape, n_class):
    return ResNet_1(ResBlock, ln, input_shape, n_class)
