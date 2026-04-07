# 数据（特征和标签）
x_data = [1, 2, 3]
y_data = [2, 4, 6]

# 参数初始化（W 和 学习率）
w = 4
a = 0.01


# 定义线性回归模型
def forward(x):
	return w * x


# 定义损失函数
def loss(xs, ys):
	total = 0
	for x, y in zip(xs, ys):
		total += (forward(x) - y) ** 2
	return total / len(xs)


# 定义计算梯度的函数
def gradient(xs, ys):
	total = 0
	for x, y in zip(xs, ys):
		total += 2 * x * (forward(x) - y)
	return total / len(xs)


# 更新参数 w
def updatew():
	global w
	w = w - a * gradient(x_data, y_data)


# 训练
def train():
	count = 100
	ls = loss(x_data, y_data)
	print("初始w =", w, "\t学习率：", a, "\tloss =", ls)
	print("计划训练轮次：", count, "\n")

	for i in range(count):
		updatew()
		ls = loss(x_data, y_data)
		print("轮次：", i, "\tw =", w, "\tloss =", ls)

	print("\n训练结束：w =", w, "\tloss =", ls)


# 预测
def predict(x):
	print("\n预测：x =", x, "\ty =", forward(x))


# 执行
train()
predict(4)
