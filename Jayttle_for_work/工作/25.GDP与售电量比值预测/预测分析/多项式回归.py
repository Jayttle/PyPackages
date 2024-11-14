from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np

# 假设前三个季度的数据
quarters = np.array([1, 2, 3]).reshape(-1, 1)  # 第1季度、2季度、3季度
data = np.array([100, 120, 140])  # 第1季度、2季度、3季度的数值

# 将数据转换为多项式形式
poly = PolynomialFeatures(degree=2)  # 设置多项式的阶数为2，可以调整阶数
quarters_poly = poly.fit_transform(quarters)

# 创建并训练多项式回归模型
model = LinearRegression()
model.fit(quarters_poly, data)

# 预测第4季度的数据
predicted_value = model.predict(poly.transform([[4]]))  # 预测第4季度
print("预测的第4季度数据：", predicted_value[0])

# 可选：绘制数据和多项式回归曲线
import matplotlib.pyplot as plt
plt.scatter(quarters, data, color='blue')  # 原始数据点
plt.plot([1, 2, 3, 4], model.predict(poly.transform([[1], [2], [3], [4]])), color='red')  # 回归曲线
plt.xlabel("季度")
plt.ylabel("数据")
plt.show()
