import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# -------------------- 路径配置 --------------------
# 原始人脸图像（摄像头拍照保存位置）
RAW_IMAGE_DIR = os.path.join(PROJECT_ROOT, 'face_data', 'raw')

# 裁剪后人脸图像（经 OpenCV 检测裁剪后用于训练）
FACE_IMAGE_DIR = os.path.join(PROJECT_ROOT, 'face_data', 'cropped')

# OpenCV 人脸检测模型（位于 data/ 目录）
DETECTOR_PROTO = os.path.join(BASE_DIR, 'opencv_face_detector.pbtxt')
DETECTOR_MODEL = os.path.join(BASE_DIR, 'opencv_face_detector_uint8.pb')

# 用户数据文件（位于 data/ 目录）
USER_FILE = os.path.join(BASE_DIR, 'user.pkl')

# 训练输出目录（相对于项目根目录）
RESULT_DIR = os.path.join(PROJECT_ROOT, 'result')
