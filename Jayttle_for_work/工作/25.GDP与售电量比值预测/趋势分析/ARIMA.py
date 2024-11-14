import pandas as pd
from pmdarima import auto_arima
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# 示例时间序列数据（示意性）
data = {
    'Quarter': ['2020-Q1', '2020-Q2', '2020-Q3', '2020-Q4',
                '2021-Q1', '2021-Q2', '2021-Q3', '2021-Q4',
                '2022-Q1', '2022-Q2', '2022-Q3', '2022-Q4'],
    'Value': [1.5, 1.6, 1.7, 1.8, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7]
}

df = pd.DataFrame(data)
df['Quarter'] = pd.to_datetime(df['Quarter'])
df.set_index('Quarter', inplace=True)


# 使用自动ARIMA来选择合适的参数
model_fit = auto_arima(df['Value'], seasonal=True, m=4,  # m=4 because there are 4 quarters per cycle
                       trace=True,
                       error_action='ignore',  
                       suppress_warnings=True)
print(model_fit.summary())

# 使用自动ARIMA来选择合适的参数
model_fit = auto_arima(df['Value'], seasonal=True, m=4,  # m=4 because there are 4 quarters per cycle
                       trace=True,
                       error_action='ignore',  
                       suppress_warnings=True)
print(model_fit.summary())
# 预测接下来的四个季度
forecast = model_fit.forecast(steps=4)
print("预测值：", forecast)

# 可视化
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['Value'], label='历史数据')
plt.plot(pd.date_range(df.index[-1], periods=4, freq='Q'), forecast, label='预测数据', color='red')
plt.legend()
plt.title('ARIMA模型预测')
plt.xlabel('日期')
plt.ylabel('值')
plt.show()