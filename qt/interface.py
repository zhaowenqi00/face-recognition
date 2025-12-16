import os
import pickle
import shutil
import sys
import cv2
import numpy as np
from PyQt5.QtCore import QCoreApplication, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QFileDialog
from PyQt5 import uic
import torch
from trainmodel import get_face, get_data, train_model, get_photograph

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = cv2.dnn.readNetFromTensorflow("opencv_face_detector_uint8.pb", "opencv_face_detector.pbtxt")


class Login:
    def __init__(self):
        self.ui_login = uic.loadUi('Login.ui')
        login_bt = self.ui_login.login
        login_bt.clicked.connect(self.Login_)
        register_bt = self.ui_login.register_2
        register_bt.clicked.connect(self.register_l)
        forget_bt = self.ui_login.forget
        forget_bt.clicked.connect(self.register_l)

    def Login_(self):
        username = self.ui_login.key.text()
        password = self.ui_login.value.text()
        user = pickle.load(open('user', 'rb'))
        if username in user.keys() and user[username] == password:
            self.dlg = MainWindow()
            self.dlg.ui.show()

    def register_l(self):
        self.dlg = Register()
        self.dlg.ui_register.show()


class Register:
    def __init__(self):
        self.ui_register = uic.loadUi('register.ui')
        bt = self.ui_register.pushButton
        bt.clicked.connect(self.register)

    def register(self):
        username = self.ui_register.lineEdit1.text()
        password1 = self.ui_register.lineEdit2.text()
        password2 = self.ui_register.lineEdit3.text()
        if username != "":
            if password1 == password2:
                user = pickle.load(open('user', 'rb'))
                user[username] = password1
                pickle.dump(user, open('user', "wb"), protocol=4)


class MainWindow:
    def __init__(self):
        self.ui = uic.loadUi('untitled.ui')
        bt = self.ui.pushButton
        bt1 = self.ui.pushButton1
        bt2 = self.ui.pushButton2
        self.model_ = "ResNet"
        item1Action = self.ui.actionLeNet_5
        item1Action.triggered.connect(self.item1)
        item2Action = self.ui.actionVGGNet
        item2Action.triggered.connect(self.item2)
        item3Action = self.ui.actionResNet
        item3Action.triggered.connect(self.item3)
        bt.clicked.connect(self.photo_Face_Recognition)
        bt1.clicked.connect(self.open_Face_Recognition)
        bt2.clicked.connect(self.open_Face_manage)

    def item1(self):
        self.model_ = "LeNet-5"

    def item2(self):
        self.model_ = "VGGNet"

    def item3(self):
        self.model_ = "ResNet"

    def photo_Face_Recognition(self):
        self.dlg = photo_Face_R(self.model_)
        self.dlg.ui.show()

    def open_Face_Recognition(self):
        # 实例化一个对话框类
        self.dlg = Face_Recognition(self.model_)
        # 显示对话框，代码阻塞在这里，
        # 等待对话框关闭后，才能继续往后执行
        self.dlg.ui.show()

    def open_Face_manage(self):
        self.dlg = management_data(self.model_)
        self.dlg.ui.show()


class photo_Face_R:
    def __init__(self, model):
        self.img = None
        self.ui = uic.loadUi('untitled6.ui')
        self.model_ = self.ui.label_3
        self.model_.setText(model)
        self.label = self.ui.label
        if model == "LeNet-5":
            self.model = torch.load('LeNet-5.pth')
        elif model == "VGGNet":
            self.model = torch.load('VGGNet.pth')
        else:
            self.model = torch.load('ResNet.pth')
        self.model.eval()
        self.names_dic = pickle.load(open('class', 'rb'))
        bt1 = self.ui.pushButton1
        bt2 = self.ui.pushButton2
        bt1.clicked.connect(self.open_files)
        bt2.clicked.connect(self.recognition)

    def open_files(self):
        folder_path = QFileDialog.getOpenFileName(self.ui,"选择图片", "E:/image1")
        self.img = cv2.imread(folder_path[0])
        show = cv2.resize(self.img, (670, 430))
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        showImage = QImage(show.data, show.shape[1], show.shape[0], show.shape[1] * 3, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(showImage))
        # print(folder_path[0])
    def recognition(self):
        gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGRA2GRAY)
        height, width, _ = self.img.shape

        # 创建一个blob从图片，然后通过网络进行前向传播
        blob = cv2.dnn.blobFromImage(cv2.resize(self.img, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        # 解析检测结果
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (startX, startY, endX, endY) = box.astype("int")

                # 在人脸上画矩形框
                cv2.rectangle(self.img, (startX, startY), (endX, endY), (0, 0, 255), 2)

                face = gray_img[startY:endY, startX:endX]

                face = cv2.resize(face, (128, 128))

                face = np.expand_dims(face, 0)
                face = np.expand_dims(face, -1)
                face = torch.tensor(face)
                inputs = face.view(face.shape[0], 1, 128, 128)

                inputs = inputs.to(torch.float32)
                inputs = inputs.to(device)
                outputs = self.model(inputs)
                outputs_lable = outputs.argmax(dim=1)
                # # 利用姓名字典还原真实姓名
                cv2.putText(self.img, self.names_dic[int(outputs_lable[0])], (startX, startY - 20), cv2.FONT_HERSHEY_SIMPLEX,
                            2, 255, 3)
        show = cv2.resize(self.img, (670, 430))
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        showImage = QImage(show.data, show.shape[1], show.shape[0], show.shape[1] * 3, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(showImage))


class Face_Recognition:
    def __init__(self, model):
        self.ui = uic.loadUi('untitled2.ui')
        self.model_ = self.ui.label_3
        self.model_.setText(model)
        self.timer_camera = QTimer()  # 初始化定时器
        self.cap = cv2.VideoCapture()  # 初始化摄像头
        self.CAM_NUM = 0
        if model == "LeNet-5":
            self.model = torch.load('LeNet-5.pth')
        elif model == "VGGNet":
            self.model = torch.load('VGGNet.pth')
        else:
            self.model = torch.load('ResNet.pth')
        self.model.eval()
        self.names_dic = pickle.load(open('class', 'rb'))
        # print(self.names_dic)
        bt1 = self.ui.pushButton1
        bt2 = self.ui.pushButton2
        self.timer_camera.timeout.connect(self.show_camera)
        bt1.clicked.connect(self.open_video)
        bt2.clicked.connect(self.close_video)

    def show_camera(self):
        self.label = self.ui.label
        flag, self.image = self.cap.read()
        img = self.face_detect_method(self.image)

        show = cv2.resize(img, (741, 491))
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        showImage = QImage(show.data, show.shape[1], show.shape[0], show.shape[1] * 3, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(showImage))

    def open_video(self):
        flag = self.cap.open(self.CAM_NUM)
        if flag == True:
            self.timer_camera.start(30)

    def close_video(self):
        self.timer_camera.stop()
        self.cap.release()
        self.label.clear()

    def face_detect_method(self, img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        height, width, _ = img.shape

        # 创建一个blob从图片，然后通过网络进行前向传播
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        # 解析检测结果
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (startX, startY, endX, endY) = box.astype("int")

                # 在人脸上画矩形框
                cv2.rectangle(img, (startX, startY), (endX, endY), (0, 0, 255), 2)

                face = gray_img[startY:endY, startX:endX]

                face = cv2.resize(face, (128, 128))

                face = np.expand_dims(face, 0)
                face = np.expand_dims(face, -1)
                face = torch.tensor(face)
                inputs = face.view(face.shape[0], 1, 128, 128)

                inputs = inputs.to(torch.float32)
                inputs = inputs.to(device)
                outputs = self.model(inputs)
                outputs_lable = outputs.argmax(dim=1)
                # print(outputs_lable[0])
                # # 利用姓名字典还原真实姓名
                cv2.putText(img, self.names_dic[int(outputs_lable[0])], (startX, startY - 20), cv2.FONT_HERSHEY_SIMPLEX,
                            1, 255, 2)
        return img


class management_data:
    def __init__(self, name):
        self.ui = uic.loadUi('untitled3.ui')
        self.modname = name
        bt1 = self.ui.pushButton1
        bt2 = self.ui.pushButton2
        bt3 = self.ui.pushButton3
        bt4 = self.ui.pushButton4
        self.lab = self.ui.label
        bt1.clicked.connect(self.show_face_people)
        bt2.clicked.connect(self.add_face)
        bt3.clicked.connect(self.del_face)
        bt4.clicked.connect(self.train_model)

    def show_face_people(self):
        path = 'E:/image1/'
        names = os.listdir(path)
        name = '姓名：\n'
        for i in names:
            name += str(i)
            name += "\n"
        # print(name)
        self.lab.setText(name)

    def add_face(self):
        self.dlg = Add_Face()
        self.dlg.ui.show()

    def del_face(self):
        self.dlg = Del_Face()
        self.dlg.ui.show()

    def train_model(self):
        get_face()
        x_train, y_train = get_data()
        train_model(self.modname, x_train, y_train)
        self.lab.setText(self.modname + "训练结束")


class Add_Face:
    def __init__(self):
        self.ui = uic.loadUi('untitled4.ui')
        bt = self.ui.pushButton
        self.label = self.ui.label_2
        self.timer_camera = QTimer()  # 初始化定时器
        self.cap = cv2.VideoCapture(0)  # 初始化摄像头
        self.timer_camera.timeout.connect(self.show_camera)
        self.open_video()
        bt.clicked.connect(self.close_video)

    def show_camera(self):
        flag, self.image = self.cap.read()
        show = cv2.resize(self.image, (431, 351))
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        showImage = QImage(show.data, show.shape[1], show.shape[0], show.shape[1] * 3, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(showImage))

    def open_video(self):
        flag = self.cap.open(0)
        if flag:
            self.timer_camera.start(30)

    def close_video(self):
        self.name = self.ui.lineEdit.text()
        for i in range(200):
            flag, image = self.cap.read()
            get_photograph(i, self.name, image)
        self.timer_camera.stop()
        self.cap.release()
        self.label.clear()
        self.label.setText("添加完成")


class Del_Face:
    def __init__(self):
        self.ui = uic.loadUi('untitled5.ui')
        bt = self.ui.pushButton
        self.label = self.ui.label_2
        bt.clicked.connect(self.del_)

    def del_(self):
        name = self.ui.lineEdit.text()
        if os.path.exists('E:/image1/' + name):
            shutil.rmtree('E:/image1/' + name)
            self.label.setText("成功删除")
        else:
            self.label.setText("无此人人脸数据")


app = QApplication([])
system = Login()
system.ui_login.show()
app.exec()
