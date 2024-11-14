import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# 假设前三个季度的数据
data = [100, 120, 140]  # 第1季度、2季度、3季度的数值

# 转换为时间序列数据
data_series = pd.Series(data)

# 创建ARIMA模型 (p=1, d=1, q=0 只是示例参数，实际使用时可以通过ACF和PACF图确定)
model = ARIMA(data_series, order=(1, 1, 0))

# 拟合模型
model_fit = model.fit()

# 预测第4季度
forecast = model_fit.forecast(steps=1)
print("预测的第4季度数据：", forecast[0])

# 可选：绘制时间序列及预测值
plt.plot(data_series, label='实际数据')
plt.plot([3, 4], [data_series.iloc[-1], forecast[0]], label='预测数据', color='red')
plt.legend()
plt.show()
