import pandas as pd
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from statsmodels.tsa.statespace.sarimax import SARIMAX
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# 计算 MAPE
def calculate_mape(actual, predicted):
    return np.mean(np.abs((actual - predicted) / actual)) * 100

# 数据加载
file_path = r"C:\Users\juntaox\Desktop\电量汇总结果.xlsx"
sheet_name = '1、小陆家嘴'
df = pd.read_excel(file_path, usecols=['日期', sheet_name])
df['日期'] = pd.to_datetime(df['日期'].str[2:], format='%Y%m')
df[sheet_name] = df[sheet_name]
df.set_index('日期', inplace=True)

# 划分训练集和测试集
df_train = df.iloc[:-6]
df_test = df.iloc[-6:]

# SARIMA 模型
model_sarima = SARIMAX(df_train[sheet_name], order=(1, 1, 2), seasonal_order=(1, 1, 0, 12))
results_sarima = model_sarima.fit(disp=False)
df_train['L_hat'] = results_sarima.predict(start=df_train.index[0], end=df_train.index[-1])

# 计算误差
df_train['E_t'] = df_train[sheet_name] - df_train['L_hat']
df_train['E_t'] = df_train['E_t']
# 数据归一化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_E = scaler.fit_transform(df_train['E_t'].values.reshape(-1, 1))

# LSTM 模型
model_lstm = Sequential()
model_lstm.add(LSTM(15, input_shape=(50, 1)))  # 隐藏层神经元个数为15，输入序列长度为50
model_lstm.add(Dense(1))
optimizer = Adam(learning_rate=0.025)  # 初始学习率为0.025
model_lstm.compile(optimizer=optimizer, loss='mean_squared_error')

# LSTM 数据预处理
X_train, y_train = scaled_E[:-1], scaled_E[1:]
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))

# LSTM 训练
model_lstm.fit(X_train, y_train, epochs=1000, batch_size=1, verbose=2)  # 迭代次数1000

# LSTM 预测 - 使用训练集的最后一个预测值开始
predictions = []
last_scaled_E = scaled_E[-50:]  # 使用最后50个值作为初始输入

for _ in range(len(df_test)):
    input_data = last_scaled_E.reshape((1, 50, 1))
    pred = model_lstm.predict(input_data)
    predictions.append(pred[0, 0])
    last_scaled_E = np.append(last_scaled_E[1:], pred[0, 0])  # 更新输入序列

# 反归一化预测结果
predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# 组合预测
df_test['N_hat'] = predictions.flatten()
df_test['L_hat'] = results_sarima.predict(start=df_test.index[0], end=df_test.index[-1])  # 预测 L_hat
df_test['Y_hat'] = df_test['L_hat'] + df_test['N_hat'] 

# 计算最终预测的 MSE
mse = mean_squared_error(df_test[sheet_name], df_test['Y_hat'])
print(f'Mean Squared Error: {mse}')

# 打印预测结果
print(df_test[['L_hat', 'N_hat', 'Y_hat', sheet_name]])
# MAPE for L_hat
mape_L_hat = calculate_mape(df_test[sheet_name], df_test['L_hat'])
print(f'MAPE for L_hat: {mape_L_hat:.2f}%')

# MAPE for Y_hat
mape_Y_hat = calculate_mape(df_test[sheet_name], df_test['Y_hat'])
print(f'MAPE for Y_hat: {mape_Y_hat:.2f}%')