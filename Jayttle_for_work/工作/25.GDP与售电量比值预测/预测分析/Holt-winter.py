import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# 假设仍然是3个季度的数据
data = [100, 120, 110]  # 第1季度、2季度、3季度的数值

quarters = pd.Series(data, index=pd.date_range(start='2023-01-01', periods=3, freq='QE'))


# 创建Holt-Winters模型（不使用季节性）
model = ExponentialSmoothing(quarters, trend='add', seasonal=None)

# 拟合模型
model_fit = model.fit()

# 预测第4季度的数据
forecast = model_fit.forecast(steps=1)
print("预测的第4季度数据：", forecast.iloc[0])


