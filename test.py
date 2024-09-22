import pandas as pd
import re
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def read_data(file_path):
    df = pd.read_excel(file_path)
    column_list = [col for col in df.columns if col.startswith('电量')]
    result_data = {'年月': [], '总和': []}

    for col in column_list:
        match = re.match(r'电量(\d{4})(\d{2})', col)
        if match:
            year = int(match.group(1))
            month = int(match.group(2))
            month_str = str(month).zfill(2)
            result_data['年月'].append(f'{year}{month_str}')
            total = df[col].sum()
            result_data['总和'].append(total)

    return pd.DataFrame(result_data)

def arithmetic_average(df):
    return df['总和'].mean()

def moving_average(df, window=3):
    return df['总和'].rolling(window=window).mean()

def weighted_moving_average(df, weights):
    return df['总和'].rolling(window=len(weights)).apply(lambda x: np.dot(x, weights), raw=True)

def exponential_smoothing(df, alpha=0.3):
    model = ExponentialSmoothing(df['总和'], trend=None, seasonal=None)
    fit = model.fit(smoothing_level=alpha, optimized=False)
    return fit.fittedvalues

def split_data(df, train_size=0.8):
    train_index = int(len(df) * train_size)
    train = df.iloc[:train_index]
    validation = df.iloc[train_index:]
    return train, validation

def forecast_future(train, periods=4):
    model = ARIMA(train['总和'], order=(1, 1, 1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=periods)
    return forecast



def forecast_future_sarima(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), periods=4):
    model = SARIMAX(train['总和'], order=order, seasonal_order=seasonal_order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=periods)
    return forecast




if __name__ == "__main__":
    file_path = r"C:\Users\Jayttle\Desktop\2024917_desktop\小陆家嘴.xlsx"
    result_df = read_data(file_path)

    print("算术平均法预测:", arithmetic_average(result_df))
    print("移动平均法预测:\n", moving_average(result_df))
    
    weights = [0.5, 0.3, 0.2]  # 示例权重
    print("加权移动平均预测:\n", weighted_moving_average(result_df, weights))

    print("指数平滑法预测:\n", exponential_smoothing(result_df))

    # 划分训练集和验证集
    # train_df, validation_df = split_data(result_df)
    train_df, validation_df = result_df.iloc[:-4], result_df.iloc[-4:]

    # 进行未来四个月的预测
    future_forecast = forecast_future(train_df, periods=4)

    print("未来四个月的预测数据:\n", future_forecast)
