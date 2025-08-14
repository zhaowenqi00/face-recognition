# Face Recognition System

基于 PyQt5 + PyTorch 的人脸识别系统，支持图片识别和实时摄像头识别。

## 项目结构

```
face-recognition/
├── ui/                  # PyQt5 界面组件
│   ├── interface.py     # 所有窗口逻辑
│   └── *.ui             # Qt Designer 界面文件
├── models/              # 神经网络模型
│   ├── LENET.py         # LeNet-5
│   ├── VGGNET.py        # VGGNet
│   └── RESNET.py        # ResNet
├── data/                # 配置文件与人脸检测模型
│   ├── config.py        # 路径配置
│   ├── opencv_face_detector_uint8.pb   # OpenCV 人脸检测模型（权重）
│   ├── opencv_face_detector.pbtxt      # OpenCV 人脸检测模型（配置）
│   └── user.pkl         # 用户凭证
├── face_data/           # 人脸图像数据
│   ├── raw/            # 原始采集图像
│   └── cropped/        # 裁剪后人脸
├── result/              # 训练输出（模型、类别映射）
├── main.py             # 独立训练脚本
├── trainmodel.py       # 训练工具函数
└── requirements.txt    # 依赖
```

## 环境配置

```bash
pip install -r requirements.txt
```

如遇网络问题，可使用国内镜像：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

主要依赖：PyTorch, OpenCV, PyQt5, NumPy, Matplotlib, tqdm, scikit-learn。

## 使用方法

### 1. 启动程序

```bash
python ui/interface.py
```
效果如下：
<img width="542" height="307" alt="image" src="https://github.com/user-attachments/assets/2f99ac8e-5db6-43f3-ae38-f5e955787ec2" />
### 2. 添加人脸数据

- 进入 **数据管理** → **添加人脸**
- 输入姓名，通过摄像头采集 200 张照片
- 照片自动保存至 `data/face_data/raw/<姓名>/`

### 3. 训练模型

- 进入 **数据管理** → **训练模型**
- 选择模型架构（LeNet-5 / VGGNet / ResNet）
- 系统自动完成人脸检测、裁剪、训练
- 模型保存至 `result/` 目录

### 4. 人脸识别

- **图片识别**：选择图片，程序自动检测人脸并识别
- **摄像头识别**：实时摄像头画面，动态检测并识别

## 模型说明

| 模型 | 架构特点 | 输入尺寸 |
|------|---------|---------|
| **LeNet-5** | 经典 CNN：2 层卷积 + 3 层全连接 | 128×128 灰度图 |
| **VGGNet** | 深层网络：5 个卷积块（最深 512 通道）+ 2 层全连接 | 128×128 灰度图 |
| **ResNet** | 残差网络：4 个残差块（含跳跃连接），防止梯度消失 | 128×128 灰度图 |

人脸检测使用 OpenCV DNN 模块（基于 ResNet 的预训练检测器）。

## 独立训练脚本

`main.py` 可独立运行训练：

```bash
python main.py
```

默认配置：RESNET 模型、100 轮训练、batch_size=16、学习率 1e-5。
