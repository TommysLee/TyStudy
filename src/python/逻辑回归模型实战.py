import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 读取CSV数据
dataset = pd.read_csv("breast_cancer_data.csv")
# print(dataset)

# 提取特征数据X（所有的行 和 指定列【第0列至倒数第二列】）
X = dataset.iloc[:, : -1]
# print(X)

# 提取标签数据Y（根据列名提取）
Y = dataset['target']
# print(Y)

# 划分数据集和测试集（shuffle为true，即随机划分：80%数据集、20%测试集）
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)
# print(x_train)

# 数据归一化处理 公式：（X - Xmin）/ Xmax - Xmin
sc = MinMaxScaler(feature_range=(0, 1))
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
# print(x_train)

# 逻辑回归模型搭建
lr = LogisticRegression()
lr.fit(x_train, y_train)

# 打印模型的参数
# print("w = ", lr.coef_)
# print("b = ", lr.intercept_)

# 利用训练好的模型进行推理测试
pre_reslut = lr.predict(x_test)
# print(pre_reslut)

# 打印预测结果的概率
pre_reslut_proba = lr.predict_proba(x_test)
# print(pre_reslut_proba)

# 获取恶性的概率
pre_list = pre_reslut_proba[:, 1]
# print(pre_list)

# 设置阈值
thresholds = 0.3

# 设置保存结果的列表
result = []
result_name = []

for v in pre_list:
	if v > thresholds:
		result.append(1)
		result_name.append("恶性")
	else:
		result.append(0)
		result_name.append("良性")

# 打印阈值调整后的结果
print(result)
print(result_name)

# 输出结果的精确率和召回还有F1值
report = classification_report(y_test, result, labels=[0, 1], target_names=['良性', '恶性'])
print("\n打印报告：\n")
print(report)
