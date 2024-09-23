import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error
import datetime

def read_and_prepare_data(file_path: str, columns: list):
    df = pd.read_excel(file_path, usecols=columns)
    if '日期' in df.columns:
        df['日期'] = pd.to_datetime(df['日期'].str[2:], format='%Y%m')
    else:
        raise KeyError("DataFrame 中找不到 '日期' 列")

    df.set_index('日期', inplace=True)
    df.columns = ['总和']
    
    df_train = df.iloc[:-6]
    df_test = df.iloc[-6:]

    print("训练集样本数：", len(df_train))
    print("测试集样本数：", len(df_test))
    return df_train, df_test

def train_LSTM_model(df: pd.DataFrame):
    sales = df['总和'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    sales_scaled = scaler.fit_transform(sales)

    X, y = [], []
    for i in range(len(sales_scaled) - 10):
        X.append(sales_scaled[i:i + 10, 0])
        y.append(sales_scaled[i + 10, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=15, input_shape=(X.shape[1], 1)))  # 设置隐藏层神经元个数为 15
    model.add(Dense(units=1))
    
    # 设置初始学习率为 0.025
    initial_learning_rate = 0.025
    optimizer = Adam(learning_rate=initial_learning_rate)

    # 添加学习率衰减
    lr_reduction = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=10, verbose=1)

    model.compile(optimizer=optimizer, loss='mean_squared_error')
    model.fit(X, y, epochs=1000, batch_size=32, callbacks=[lr_reduction])  # 迭代次数设为 1000

    model.save(f'model_LSTM.h5')

    train_predictions = model.predict(X)
    train_predictions = scaler.inverse_transform(train_predictions)
    true_sales = df['总和'].values[10:]
    mse = mean_squared_error(true_sales, train_predictions)
    mae = mean_absolute_error(true_sales, train_predictions)

    print(f"Mean Squared Error on Training Data: {mse}")
    print(f"Mean Absolute Error on Training Data: {mae}")

def load_model_predict_model(file_path, sheet_names, date_str):
    model = load_model(f'model_LSTM.h5')

    df = pd.read_excel(file_path)
    if '日期' in df.columns:
        df['日期'] = pd.to_datetime(df['日期'].str[2:], format='%Y%m')
    else:
        raise KeyError("DataFrame 中找不到 '日期' 列")
    
    input_date = datetime.datetime.strptime(date_str, '%Y/%m')
    product_df = df[(df['日期'] <= input_date)].copy()

    sales = product_df[sheet_names].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    sales_scaled = scaler.fit_transform(sales)

    input_sequence = sales_scaled[-10:].reshape(1, 10, 1)
    predicted_sales_scaled = model.predict(input_sequence)
    predicted_sales = scaler.inverse_transform(predicted_sales_scaled)

    return predicted_sales[0, 0]

if __name__ == '__main__':
    print('---------------------------------------------------------------')
    file_path = r"C:\Users\juntaox\Desktop\电量汇总结果.xlsx"
    sheet_names = '1、小陆家嘴'
    columns_to_read = ['日期', sheet_names]
    df_train, df_test = read_and_prepare_data(file_path, columns_to_read)
    # train_LSTM_model(df_train)
    prediction_date = '2024/9'
    predicted_sales = load_model_predict_model(file_path, sheet_names, prediction_date)
    print(predicted_sales)