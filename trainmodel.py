import os
import pickle
import shutil
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data_utils
from tqdm import trange

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

from data.config import (
    RAW_IMAGE_DIR, FACE_IMAGE_DIR, RESULT_DIR,
    DETECTOR_PROTO, DETECTOR_MODEL,
)
from models import LeNet, VGGnet, RESNET

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def _detector_net():
    return cv2.dnn.readNetFromTensorflow(DETECTOR_MODEL, DETECTOR_PROTO)


def get_photograph(i, name, img):
    name_path = os.path.join(RAW_IMAGE_DIR, name)
    if not os.path.exists(name_path):
        os.makedirs(name_path)
    cv2.imwrite(os.path.join(name_path, f'{i}.jpg'), img)


def get_face():
    if os.path.exists(FACE_IMAGE_DIR):
        shutil.rmtree(FACE_IMAGE_DIR)

    net = _detector_net()
    peoples = os.listdir(RAW_IMAGE_DIR)

    for people in peoples:
        people_dir = os.path.join(RAW_IMAGE_DIR, people)
        if not os.path.isdir(people_dir):
            continue
        img_names = os.listdir(people_dir)
        for img_name in img_names:
            img_path = os.path.join(people_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            height, width, _ = img.shape

            blob = cv2.dnn.blobFromImage(
                cv2.resize(img, (300, 300)), 1.0,
                (300, 300), (104.0, 177.0, 123.0)
            )
            net.setInput(blob)
            detections = net.forward()

            startX, startY, endX, endY = 0, 0, width, height
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.9:
                    box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                    startX, startY, endX, endY = np.maximum(box.astype("int"), 0)
                    break

            face = img[startY:endY, startX:endX]
            face = cv2.resize(face, (128, 128))

            out_dir = os.path.join(FACE_IMAGE_DIR, people)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            cv2.imwrite(os.path.join(out_dir, img_name), face)


def get_data():
    names = os.listdir(FACE_IMAGE_DIR)
    names_dic = {i: name for i, name in enumerate(names)}

    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
    pickle.dump(names_dic, open(os.path.join(RESULT_DIR, 'class'), 'wb'), protocol=4)

    imgs, labels = [], []
    for i, name in enumerate(names):
        name_dir = os.path.join(FACE_IMAGE_DIR, name)
        if not os.path.isdir(name_dir):
            continue
        for img_name in os.listdir(name_dir):
            labels.append(i)
            img = cv2.imread(os.path.join(name_dir, img_name), 0)
            if img is not None:
                imgs.append(img)

    y_onehot = np.zeros((len(labels), len(names_dic)), dtype=np.float32)
    for i, label in enumerate(labels):
        y_onehot[i, label] = 1

    x = np.expand_dims(np.array(imgs, dtype=np.float32), -1)
    y = y_onehot

    print("转化字典：", names_dic)
    print("训练集大小", x.shape)
    print("标签大小", y.shape)
    return x, y


def train_model(model_name, x_train, y_train):
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    batch_size = 16
    n_epoch = 100
    lr = 1e-5
    img_size = 128

    names_dic = pickle.load(open(os.path.join(RESULT_DIR, 'class'), 'rb'))
    n_layers = len(names_dic)

    x_train = torch.tensor(x_train).view(x_train.shape[0], 1, img_size, img_size)
    y_train = torch.tensor(y_train)

    train_dataset = data_utils.TensorDataset(x_train, y_train)
    train_loader = data_utils.DataLoader(
        dataset=train_dataset, batch_size=batch_size,
        shuffle=True, drop_last=True
    )

    criterion = nn.CrossEntropyLoss()
    if model_name == 'LeNet-5':
        model = LeNet(img_size, 1, n_layers)
    elif model_name == 'VGGNet':
        model = VGGnet(img_size, 1, n_layers)
    else:
        model = RESNET(img_size, 1, n_layers)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    with trange(n_epoch) as t:
        for epoch in t:
            t.set_description('训练进度')
            sum_loss, acc = 0, 0
            for i, (inputs, labels) in enumerate(train_loader):
                inputs = inputs.to(torch.float32).to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                acc += (outputs.argmax(dim=1) == labels.argmax(dim=1)).sum().item()
                loss = criterion(outputs, labels.float())
                loss.backward()
                optimizer.step()
                sum_loss += loss.item()
            t.set_postfix(loss=sum_loss / (i + 1), acc=acc / ((i + 1) * batch_size))

    model_path = os.path.join(RESULT_DIR, f'{model_name}.pth')
    torch.save(model, model_path)
    print(f"模型已保存到: {model_path}")


if __name__ == '__main__':
    get_face()
    x_train, y_train = get_data()
    train_model('ResNet', x_train, y_train)
