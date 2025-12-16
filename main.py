import os
import pickle
import torch.utils.data as data_utils
import numpy as np
import cv2
import torch
from matplotlib import pyplot as plt
from sympy.tensor import tensor
from torch import nn
from tqdm import trange
from VGGNET import VGGnet
from RESNET import RESNET
from LENET import LeNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
torch.cuda.empty_cache()
# 切分训练集和测试集
path = 'E:/face_images_lian/'
names = os.listdir(path)
names_dic = {}

x_train = []
x_test = []
y_train = []
y_test = []
sum_z = 0
for i in range(len(names)):
    img_names = os.listdir(path + names[i])
    len_ = len(img_names)
    if len_ > 20:
        names_dic[sum_z] = names[i]
        # len_test = int(len_ * 0.1)
        len_test = 16
        for j in range(len(img_names) - len_test):
            y_train.append(sum_z)
            img = cv2.imread(path + names[i] + '/' + img_names[j], 0)
            x_train.append(img)
        for j in range(len(img_names) - len_test, len(img_names)):
            img = cv2.imread(path + names[i] + '/' + img_names[j], 0)
            x_test.append(img)
            y_test.append(sum_z)
        sum_z += 1
print("转化字典：",names_dic)
# swapped_dict = {v: k for k, v in names_dic.items()}
# print(swapped_dict)
# pickle.dump(swapped_dict, open('../result/class_swapped', "wb"), protocol=4)
# pickle.dump(names_dic, open('../result/class', "wb"), protocol=4)
y_train_li = [[0 for i in range(len(names_dic))]for j in range(len(y_train))]
y_test_li = [[0 for i in range(len(names_dic))]for j in range(len(y_test))]
for i in range(len(y_train)):
    y_train_li[i][y_train[i]] = 1
for i in range(len(y_test)):
    y_test_li[i][y_test[i]] = 1

x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train_li)
y_test = np.array(y_test_li)
# 将图片处理成模型需要的标准格式
# （数据量，图片的宽度，图片的高度，图片的通道数）
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("训练集数据大小",x_train.shape)
print("训练集标签大小",y_train.shape)
print("测试集数据大小",x_test.shape)
print("测试集标签大小",y_test.shape)
if __name__ == '__main__':
    batch_size = 16
    # device = torch.device("cpu")
    n_epoch = 100
    lr = 0.00001
    img_size = 128
    # output_size = 1
    model_name = 'LeNet'
    # model_name = 'RESNET'
    opt_name = 'Adam'
    n_layers = len(names_dic)

    x_train = torch.tensor(x_train)#(155,1,64,64)
    y_train = torch.tensor(y_train)
    x_train = x_train.view(x_train.shape[0],1,img_size,img_size)
    x_test = torch.tensor(x_test)
    y_test = torch.tensor(y_test)
    x_test = x_test.view(x_test.shape[0], 1, img_size, img_size)

    # senti_dict = pickle.load(open('../result/class', 'rb'))
    senti_dict = names_dic
    # 装载训练集
    train_dataset = data_utils.TensorDataset(x_train, y_train)
    train_loader = data_utils.DataLoader(dataset=train_dataset,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         drop_last=True)
    # 装载测试集
    test_dataset = data_utils.TensorDataset(x_test, y_test)
    test_loader = data_utils.DataLoader(dataset=test_dataset,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        drop_last=True)
    #
    criterion = nn.CrossEntropyLoss()
    # model = LeNet(img_size, 1, n_layers)
    # model = VGGnet(img_size, 1, n_layers)
    # model = RESNET(img_size, 1, n_layers)

    # model.load_state_dict(torch.load('../result/LeNet.pth'))
    # model.eval()
    # model = model.to(device)

    model = RESNET(img_size, 1, n_layers)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_ = {'loss': [], 'acc': []}

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
            t.set_postfix(loss=sum_loss / (i+1), acc=acc / ((i+1)*batch_size))
            # t.set_postfix(acc=acc / (i+1))

            # print('Epoch: {}/{}.............'.format(epoch + 1, n_epoch), end=' ')
            # print("Loss: {:.8f}".format(sum_loss / (i+1)),end=' ')
            # print("acc: {:.8f}".format(acc / ((i+1)*batch_size)))
            loss_['loss'].append(sum_loss / (i+1))
            loss_['acc'].append(acc / ((i+1)*batch_size))

    model.eval()
    criterion = nn.CrossEntropyLoss()
    sum_loss = 0
    acc = 0
    classify = {}

    # plt.figure(figsize=(8, 6), dpi=200)
    summ = 0
    for i, data in enumerate(test_loader):
        inputs, labels = data
        inputs = inputs.to(torch.float32)
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        # print("i",i)
        # print(outputs.shape)
        # print(labels.shape)
        # outputs, h = model(inputs, h)
        # print(labels)
        acc += (outputs.argmax(dim=1) == labels.argmax(dim=1)).sum().item()
        for j in range(outputs.shape[0]):

            if outputs[j].argmax(dim=0) == labels[j].argmax(dim=0):
                # if j % 8 == 0 and summ < 30:
                #     ax = plt.subplot(5, 6, summ + 1)
                #     ax.set_title(senti_dict[int(labels[j].argmax(dim=0))], fontsize=5,verticalalignment='top')
                #     ax.axis('off')
                #     tensor_cpu = inputs[j][0].cpu()
                #     numpy_array = tensor_cpu.numpy()
                #     # print(numpy_array)
                #     ax.imshow(numpy_array,cmap="gray")
                #     summ += 1
                if senti_dict[int(labels[j].argmax(dim=0))] not in classify:
                    classify[senti_dict[int(labels[j].argmax(dim=0))]] = {senti_dict[int(outputs[j].argmax(dim=0))]: 1}
                else:
                    if senti_dict[int(outputs[j].argmax(dim=0))] in classify[senti_dict[int(labels[j].argmax(dim=0))]]:
                        classify[senti_dict[int(labels[j].argmax(dim=0))]][
                            senti_dict[int(outputs[j].argmax(dim=0))]] += 1
                    else:
                        classify[senti_dict[int(labels[j].argmax(dim=0))]][
                            senti_dict[int(outputs[j].argmax(dim=0))]] = 1
            else:
                # if j % 8 == 0 and summ < 30:
                #     ax = plt.subplot(5, 6, summ + 1)
                #     ax.set_title(senti_dict[int(outputs[j].argmax(dim=0))] + "\n"+str('(') + senti_dict[int(labels[j].argmax(dim=0))] + str(')'), color='r',
                #              fontsize=3,verticalalignment='top')
                #     ax.axis('off')
                #     tensor_cpu = inputs[j].cpu()
                #     numpy_array = tensor_cpu.numpy()
                #     # print(numpy_array)
                #     ax.imshow(numpy_array[0],cmap="gray")
                #     summ += 1
                if senti_dict[int(labels[j].argmax(dim=0))] not in classify:
                    classify[senti_dict[int(labels[j].argmax(dim=0))]] = {senti_dict[int(outputs[j].argmax(dim=0))]: 1}
                else:
                    if senti_dict[int(outputs[j].argmax(dim=0))] in classify[senti_dict[int(labels[j].argmax(dim=0))]]:
                        classify[senti_dict[int(labels[j].argmax(dim=0))]][
                            senti_dict[int(outputs[j].argmax(dim=0))]] += 1
                    else:
                        classify[senti_dict[int(labels[j].argmax(dim=0))]][
                            senti_dict[int(outputs[j].argmax(dim=0))]] = 1
        # print(accuracy_score(out,lab))
        # print(labels.shape)
        loss = criterion(outputs, labels.float())
        sum_loss += loss.item()
    # plt.show()
    print('模型测试准确率为：', acc / ((i+1)*batch_size))
    print("Loss: {:.4f}".format(sum_loss / (i + 1)))
    print('结果字典：', classify)

    torch.save({'model': model.state_dict()}, '../result/' + model_name + '.pth')
    pickle.dump(loss_, open('../result/' + model_name + '.loss', "wb"), protocol=4)
    pickle.dump(classify, open('../result/' + model_name + '.matrix', "wb"), protocol=4)


