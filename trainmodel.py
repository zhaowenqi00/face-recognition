import os
import pickle
from torch import nn
import torch.utils.data as data_utils
import numpy as np
import cv2
import torch
import tqdm  # 进度条工具
from tqdm import trange
import shutil
from VGGNET import VGGnet
from RESNET import RESNET
from LENET import LeNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def get_photograph(i,name,img):
    # 配置项
    img_dir = 'E:/image1'  # 设置照片的存储位置
    name = name
    img = img
    name_path = os.path.join(img_dir, name)
    if not os.path.exists(name_path):  # 创建文件夹
        os.makedirs(name_path)
    img_name = str(i) + '.jpg'
    cv2.imwrite(os.path.join(name_path, img_name), img)


def get_face():
    if os.path.exists('E:/face_images_lian'):
        shutil.rmtree('E:/face_images_lian')
    net = cv2.dnn.readNetFromTensorflow("opencv_face_detector_uint8.pb", "opencv_face_detector.pbtxt")
    peoples = os.listdir('E:/image1/')
    for people in peoples:
        img_names = os.listdir('E:/image1/' + people)
        for img_name in img_names:
            img_path = 'E:/image1/' + people + '/' + img_name
            img = cv2.imread(img_path)
            height, width, _ = img.shape
            # 创建一个blob从图片，然后通过网络进行前向传播
            blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                         (300, 300), (104.0, 177.0, 123.0))
            net.setInput(blob)
            detections = net.forward()
            #         print(detections.shape[2])
            # 解析检测结果
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.9:
                    box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                    #                 print(box)
                    (startX, startY, endX, endY) = box.astype("int")
                    if startX < 0:
                        startX = 0
                    if startY < 0:
                        startY = 0
                    if endX < 0:
                        endX = 0
                    if endY < 0:
                        endY = 0
                    break
            face = img[startY:endY, startX:endX]
            #         cv2.imshow('1',face)
            #         cv2.waitKey(0)
            face = cv2.resize(face, (128, 128))  # 将所有的图片处理成统一的尺寸（64，64）

            if not os.path.exists('E:/face_images_lian/' + people):
                os.makedirs('E:/face_images_lian/' + people)
            cv2.imwrite('E:/face_images_lian/' + people + '/' + img_name, face)


def get_data():
    path = 'E:/face_images_lian/'
    names = os.listdir(path)
    names_dic = {i: j for i, j in enumerate(names)}
    imgs = []
    labels = []
    for i in range(len(names)):
        img_names = os.listdir(path + names[i])
        for j in range(len(img_names)):
            labels.append(i)
            img = cv2.imread(path + names[i] + '/' + img_names[j], 0)
            imgs.append(img)
    print("转化字典：", names_dic)
    pickle.dump(names_dic, open('E:/python笔记/毕业设计/qt/class', "wb"), protocol=4)

    y_train_li = [[0 for i in range(len(names_dic))] for j in range(len(labels))]
    for i in range(len(labels)):
        y_train_li[i][labels[i]] = 1

    x_train = np.array(imgs)
    y_train = np.array(y_train_li)
    # 将图片处理成模型需要的标准格式
    # （数据量，图片的宽度，图片的高度，图片的通道数）
    x_train = np.expand_dims(x_train, -1)
    print("训练集数据大小", x_train.shape)
    print("训练集标签大小", y_train.shape)
    return x_train, y_train


def train_model(model, x_train, y_train):
    batch_size = 16
    n_epoch = 100
    lr = 0.00001
    img_size = 128
    model_name = model
    opt_name = 'Adam'
    names_dic = pickle.load(open('E:/python笔记/毕业设计/qt/class', 'rb'))
    n_layers = len(names_dic)

    x_train = torch.tensor(x_train)  # (155,1,64,64)
    y_train = torch.tensor(y_train)
    x_train = x_train.view(x_train.shape[0], 1, img_size, img_size)

    # senti_dict = pickle.load(open('../result/class', 'rb'))
    # 装载训练集
    train_dataset = data_utils.TensorDataset(x_train, y_train)
    train_loader = data_utils.DataLoader(dataset=train_dataset,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         drop_last=True)

    #
    criterion = nn.CrossEntropyLoss()
    if model_name == "LeNet-5":
        model = LeNet(img_size, 1, n_layers)
    elif model_name == "VGGNet":
        model = VGGnet(img_size, 1, n_layers)
    else:
        model = RESNET(img_size, 1, n_layers)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # loss_ = {'loss': [], 'acc': []}

    with trange(n_epoch) as t:
        for epoch in t:
            t.set_description('训练进度')
            sum_loss = 0
            acc = 0
            for i, data in enumerate(train_loader):
                inputs, labels = data
                inputs = inputs.to(torch.float32)
                inputs = inputs.to(device)
                labels = labels.to(device)
                # print(labels.shape)
                optimizer.zero_grad()
                outputs = model(inputs).to(device)
                # print(outputs.shape)
                acc += (outputs.argmax(dim=1) == labels.argmax(dim=1)).sum().item()
                # print(i)
                # print(acc)

                # print(outputs[0].shape)
                # print(labels.shape)
                loss = criterion(outputs, labels.float())

                loss.backward()
                optimizer.step()
                sum_loss += loss.item()
            t.set_postfix(loss=sum_loss / (i + 1), acc=acc / ((i + 1) * batch_size))
    if os.path.exists('E:/python笔记/毕业设计/qt/' + model_name + '.pth'):
        os.remove('E:/python笔记/毕业设计/qt/' + model_name + '.pth')
    torch.save(model, 'E:/python笔记/毕业设计/qt/' + model_name + '.pth')


if __name__ == '__main__':
    # 初始化TensorFlow人脸检测模型
    net = cv2.dnn.readNetFromTensorflow("qt/opencv_face_detector_uint8.pb", "qt/opencv_face_detector.pbtxt")
    get_face()
    x_train, y_train = get_data()
    train_model("ResNet", x_train, y_train)
    # model = torch.load( 'qt/LeNet-5.pth')
    # model.eval()
