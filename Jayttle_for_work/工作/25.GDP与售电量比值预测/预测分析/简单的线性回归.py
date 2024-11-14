import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 假设前三个季度的数据
quarters = np.array([1, 2, 3]).reshape(-1, 1)  # 第1季度、2季度、3季度
data = np.array([100, 120, 140])  # 第1季度、2季度、3季度的数值

# 创建线性回归模型
model = LinearRegression()

# 训练模型
model.fit(quarters, data)

# 预测第4季度的数据
predicted_value = model.predict(np.array([[4]]))  # 预测第4季度
print("预测的第4季度数据：", predicted_value[0])

# 可选：绘制数据和回归线
plt.scatter(quarters, data, color='blue')  # 原始数据点
plt.plot([1, 2, 3, 4], model.predict(np.array([[1], [2], [3], [4]])), color='red')  # 回归线
plt.xlabel("季度")
plt.ylabel("数据")
plt.show()
