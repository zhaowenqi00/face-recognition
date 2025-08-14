import torch
import torch.nn as nn
import torch.nn.functional as F

class VGGnet(nn.Module):
    def __init__(self, ln, input_shape, n_class):
        super(VGGnet, self).__init__()
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
        #第一层卷积
        self.conv1_1= nn.Sequential(
            nn.Conv2d(input_shape, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.max_pooling1 = nn.MaxPool2d(kernel_size=2, stride=2)
        ln = (ln + 2 - 2 - 1) // 1 + 1
        ln = (ln + 2 - 2 - 1) // 1 + 1
        ln = (ln - 1 - 1) // 2 + 1
        #第二层卷积
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.conv2_2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.max_pooling2 = nn.MaxPool2d(kernel_size=2, stride=2)
        ln = (ln + 2 - 2 - 1) // 1 + 1
        ln = (ln + 2 - 2 - 1) // 1 + 1
        ln = (ln - 1 - 1) // 2 + 1
        #第三层卷积
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.conv3_2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.conv3_3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.max_pooling3 = nn.MaxPool2d(kernel_size=2,
                                         stride=2,
                                         padding=1)
        ln = (ln + 2 - 2 - 1) // 1 + 1
        ln = (ln + 2 - 2 - 1) // 1 + 1
        ln = (ln + 2 - 2 - 1) // 1 + 1
        ln = (ln + 2 - 1 - 1) // 2 + 1
        #第四层卷积
        self.conv4_1 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.conv4_2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.conv4_3 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.max_pooling4 = nn.MaxPool2d(kernel_size=2,
                                         stride=2)
        ln = (ln + 2 - 2 - 1) // 1 + 1
        ln = (ln + 2 - 2 - 1) // 1 + 1
        ln = (ln + 2 - 2 - 1) // 1 + 1
        ln = (ln - 1 - 1) // 2 + 1
        #第五层卷积
        self.conv5_1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.conv5_2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.conv5_3 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        ln = (ln + 2 - 2 - 1) // 1 + 1
        ln = (ln + 2 - 2 - 1) // 1 + 1
        ln = (ln + 2 - 2 - 1) // 1 + 1
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(4096, 4096)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(4096, n_class)
        self.fc1 = nn.Linear(ln * ln * 512, 4096)

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

    def forward(self, x):
        batchsize = x.size(0)
        # if self.f:
        #     x = self.stn(x)
        out = self.conv1_1(x)
        out = self.conv1_2(out)
        out = self.max_pooling1(out)

        out = self.conv2_1(out)
        out = self.conv2_2(out)
        out = self.max_pooling2(out)

        #
        out = self.conv3_1(out)
        out = self.conv3_2(out)
        out = self.conv3_3(out)
        out = self.max_pooling3(out)

        out = self.conv4_1(out)
        out = self.conv4_2(out)
        out = self.conv4_3(out)
        out = self.max_pooling4(out)

        out = self.conv5_1(out)
        out = self.conv5_2(out)
        out = self.conv5_3(out)

        out = out.view(batchsize, -1)

        out = self.fc1(out)
        out = self.relu1(out)

        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out
