import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import keras
from keras.layers import Dense
from keras.utils.np_utils import to_categorical
from keras.models import load_model
from sklearn.metrics import classification_report

# 解决Chart中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载数据集
dataset = pd.read_csv("breast_cancer_data.csv")

# 提取数据集中的特征
X = dataset.iloc[:, :-1]
# print(X)

# 提取数据集中的标签（根据列名提取）
Y = dataset['target']
# print(Y)

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=66)

# 将数据特征进行归一化
sc = MinMaxScaler(feature_range=(0, 1))
x_test = sc.fit_transform(x_test)

# 导入模型
model = load_model("model.h5")

# 利用训练好的模型进行推理
predict = model.predict(x_test)
# print(predict)  # 打印概率

# 找出矩阵中的最大值的索引
# axis=1，表明按行比较，输出每行中最大值的索引
# axis=0，则输出每列中最大值的索引。
y_pred = np.argmax(predict, axis=1)
# print(y_pred)

# 将识别的结果转化成汉字
result = []
for i in range(len(y_pred)):
	if y_pred[i] == 0:
		result.append('良性')
	else:
		result.append('恶性')

# print(result)

# 模型评价（打印模型的精确度和召回）
report = classification_report(y_test, y_pred, labels=[0, 1], target_names=["良性", '恶性'])
print(report)
