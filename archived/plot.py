import copy
import pickle
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# img = cv2.imread('img.png',0)
# cv2.imshow("0", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

LeNet_data = pickle.load(open('../result/LeNet.loss', 'rb'))
VGG_data = pickle.load(open('../result/VGG.loss', 'rb'))
RESNET_data = pickle.load(open('../result/RESNET.loss', 'rb'))
print("LeNet",LeNet_data["loss"][0],LeNet_data["loss"][49],LeNet_data["loss"][99])
print("VGG",VGG_data["loss"][0],VGG_data["loss"][49],VGG_data["loss"][99])
print("RESNET",RESNET_data["loss"][0],RESNET_data["loss"][49],RESNET_data["loss"][99])
# # print(LeNet_data)
#
# plt.figure(figsize=(10,5),dpi=100)
# plt.xlabel('训练轮次')
# plt.ylabel('损失值')
# plt.ylim(-0.1, 5)
# plt.xlim(-1,100)
# plt.grid()
# my_x_ticks = np.arange(0, 101, 10)
# my_y_ticks = np.arange(0, 5.01, 0.5)
# plt.xticks(my_x_ticks)
# plt.yticks(my_y_ticks)
# plt.plot(LeNet_data["loss"],label="LeNet-5")
# plt.plot(VGG_data["loss"],label="VGGNet16")
# plt.plot(RESNET_data["loss"],label="RESNET18")
# plt.legend()
# plt.title("三种模型在训练集上的loss值对比")
# plt.show()


# plt.figure(figsize=(10,5),dpi=100)
# plt.xlabel('训练轮次')
# plt.ylabel('准确率')
# plt.ylim(0, 1.01)
# plt.xlim(0,100)
# plt.grid()
# my_x_ticks = np.arange(0, 101, 10)
# my_y_ticks = np.arange(0, 1.01, 0.1)
# plt.xticks(my_x_ticks)
# plt.yticks(my_y_ticks)
# plt.plot(LeNet_data["acc"],label="LeNet-5")
# plt.plot(VGG_data["acc"],label="VGGNet16")
# plt.plot(RESNET_data["acc"],label="RESNET18")
# plt.legend()
# plt.title("三种模型在训练集上的准确率对比")
# plt.show()


# df = pd.read_excel("C:/Users/赵文琦/Desktop/几个城市住宅样本均价价格组合数据.xlsx")
# df_col = df.columns
# plt.figure(figsize=(6,4),dpi=100)
# for i in range(1,len(df_col)):
#     plt.plot(df['时间'],df[df_col[i]],label=df_col[i].split(':')[1])
# plt.legend()
# plt.title("各城市住宅样本均价折线图")
# plt.show()
#
#
# #调整画布
# #figsize画布大小
# #dpi清晰率
# plt.figure(figsize=(10,5),dpi=500)
# #添加标题
# plt.title("y=sin(x)与y=cos(x)的图像")
# plt.rcParams['font.sans-serif']=['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
# #坐标标签
# plt.xlabel('x')
# plt.ylabel('y')
# #网格
# plt.grid()
# #坐标刻度
# plt.xlim(-np.pi,np.pi)
# #plt.ylim(-np.pi,np.pi)
# plt.plot(x,y1)
# plt.plot(x,y2)
# #添加文本
# plt.text(0,0,'lll')
# #添加图例
# plt.legend(['sin(x)','cos(x)'])
# plt.savefig('C:\\Users\\赵文琦\\Desktop\\y=sin(x)与y=cos(x)的图像')
# plt.show()
#
# import pandas as pd
# def plot_learning_curves(history):
#     pd.DataFrame(history.history).plot(figsize=(9,4))
#     plt.grid(True)
#     plt.show()
# plot_learning_curves(history)
