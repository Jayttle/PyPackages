import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings("ignore")

def read_and_prepare_data(file_path: str):
    df = pd.read_excel(file_path)

    # 确保日期列是 datetime 类型
    if '日期' in df.columns:
        df['日期'] = pd.to_datetime(df['日期'])
    else:
        raise KeyError("DataFrame 中找不到 '日期' 列")

    # 设置日期列为索引
    df.set_index('日期', inplace=True)

    # 划分测试集与验证集
    df_train = df
    df_test = df.iloc[-4:]

    print("训练集样本数：", len(df_train))
    print("测试集样本数：", len(df_test))
    return df_train, df_test

def evaluate_sarima(params, df_train, df_test):
    model = SARIMAX(df_train['总和'], 
                    order=params['order'], 
                    seasonal_order=params['seasonal_order'])
    model_fit = model.fit(disp=False)
    
    forecast = model_fit.forecast(steps=len(df_test))
    
    df_test['预测'] = forecast.values
    df_test['误差'] = (df_test['预测'] - df_test['总和']) / df_test['总和']
    
    mse = mean_squared_error(df_test['总和'], df_test['预测'])
    return mse

def grid_search_sarima(df_train, df_test):
    best_params = None
    best_mse = np.inf
    
    param_grid = {
        'order': [(p, d, q) for p in range(3) for d in range(3) for q in range(3)],
        'seasonal_order': [(P, D, Q, S) for P in range(2) for D in range(2) for Q in range(2) for S in [12]]
    }
    
    for params in ParameterGrid(param_grid):
        mse = evaluate_sarima(params, df_train, df_test)
        print(f"测试参数 {params} 的均方误差: {mse}")
        
        if mse < best_mse:
            best_mse = mse
            best_params = params
    
    print(f"最佳参数组合: {best_params}")
    print(f"最佳均方误差: {best_mse}")
    
    return best_params

def train_and_predict_sarima(df_train, df_test):
    # 使用最佳参数训练模型
    model = SARIMAX(df_train['总和'], 
                    order=(1, 1, 2), 
                    seasonal_order=(1, 1, 0, 12))
    model_fit = model.fit(disp=False)
    
    # 进行预测
    forecast = model_fit.forecast(steps=len(df_test))

    # 创建未来6个月的日期索引
    future_dates = pd.date_range(start=df_train.index[-1] + pd.DateOffset(months=1), periods=len(df_test), freq='M')
    
    # 更新 df_test 的索引
    df_test.index = future_dates

    # 将预测结果添加到 df_test 中
    df_test['预测'] = forecast.values
    print(df_test['预测'])
    # 绘制实际值与预测值的图形
    plt.figure(figsize=(12, 6))
    plt.plot(df_train.index, df_train['总和'], label='训练集')
    plt.plot(df_test.index, df_test['预测'], label='预测值', color='red')
    plt.legend()
    plt.title('SARIMA预测')
    plt.show()

# 示例调用
file_path = r"E:\vscode_proj\小陆家嘴_sums.xlsx"  # 替换为实际的文件路径
df_train, df_test = read_and_prepare_data(file_path)
train_and_predict_sarima(df_train, df_test)
