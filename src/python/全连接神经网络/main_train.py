import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import keras
from keras.layers import Dense
from keras.utils.np_utils import to_categorical
from sklearn.metrics import classification_report

# 解决Chart中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载数据集
dataset = pd.read_csv("breast_cancer_data.csv")
# print(dataset)

# 提取数据集中的特征
X = dataset.iloc[:, :-1]
# print(X)

# 提取数据集中的标签（根据列名提取）
Y = dataset['target']
# print(Y)

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=66)
# print(x_train)

# 将数据标签转化为one-hot向量格式
y_train_one = to_categorical(y_train, 2)
y_test_one = to_categorical(y_test, 2)
# print(y_train_one)

# 将数据特征进行归一化
sc = MinMaxScaler(feature_range=(0, 1))
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
# print(x_train)

# 利用Keras框架帮助搭建深度学习网络模型（Dense：表示全连接神经网络）
model = keras.Sequential()
model.add(Dense(10, activation="relu"))		 # 搭建第一层隐藏层，10个神经元，激活函数为：relu
model.add(Dense(10, activation="relu"))		 # 搭建第二层隐藏层，10个神经元，激活函数为：relu
model.add(Dense(2, activation="softmax"))	 # 搭建输出层，输出两个，激活函数为：softmax

# 对神经网络进行编译（损失函数：交叉熵损失函数；优化器：随机梯度下降法；评价指标：准确率）
model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])

# 训练（训练轮次epochs：110，64为一捆丢到网络中学习，verbose=2 打印训练日志，验证集）
history = model.fit(x_train, y_train_one, epochs=110, batch_size=64, verbose=2, validation_data=(x_test, y_test_one))

# 导出模型（.h5是Keras模型文件的后缀）
model.save("model.h5")

# 绘制训练集和验证集的loss值对比
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.title("全连接神经网络loss值图")
plt.legend()
plt.show()

# 绘制训练集和验证集的准确率的对比图
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.title("全连接神经网络accuracy值图")
plt.legend()
plt.show()
