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

    # 确保年月列是 datetime 类型，使用指定格式进行解析
    if '年月' in df.columns:
        df['年月'] = pd.to_datetime(df['年月'].astype(str), format='%Y%m')
    else:
        raise KeyError("DataFrame 中找不到 '年月' 列")

    # 设置日期列为索引
    df.set_index('年月', inplace=True)

    # 划分测试集与验证集
    df_train = df.iloc[:-6]
    df_test = df.iloc[-6:]

    print("训练集样本数：", len(df_train))
    print("测试集样本数：", len(df_test))
    return df_train, df_test

def evaluate_sarima(params, df_train, df_test):
    model = SARIMAX(df_train['数据中心用电量（千瓦时）'], 
                    order=params['order'], 
                    seasonal_order=params['seasonal_order'])
    model_fit = model.fit(disp=False)
    
    forecast = model_fit.forecast(steps=len(df_test))
    
    
    df_test['预测'] = forecast.values
    df_test['误差'] = (df_test['预测'] - df_test['数据中心用电量（千瓦时）']) / df_test['数据中心用电量（千瓦时）']
    
    mse = mean_squared_error(df_test['数据中心用电量（千瓦时）'], df_test['预测'])
    return mse

def grid_search_sarima(df_train, df_test):
    best_params = None
    best_mse = np.inf
    
    # param_grid = {
    #     'order': [(p, d, q) for p in range(3) for d in range(3) for q in range(3)],
    #     'seasonal_order': [(P, D, Q, S) for P in range(2) for D in range(2) for Q in range(2) for S in [12]]
    # }
    param_grid = {
        'order': [(p, d, q) for p in [2] for d in [2] for q in [0]],
        'seasonal_order': [(P, D, Q, S) for P in [1] for D in [1] for Q in [0] for S in [12]]
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
    # 使用网格搜索法找到最佳参数
    best_params = grid_search_sarima(df_train, df_test)
    
    # 使用最佳参数训练模型
    model = SARIMAX(df_train['数据中心用电量（千瓦时）'], 
                    order=best_params['order'], 
                    seasonal_order=best_params['seasonal_order'])
    model_fit = model.fit(disp=False)
    
    # 进行预测
    forecast = model_fit.forecast(steps=len(df_test))
    
    # 将预测结果与实际测试集进行比较
    df_test['预测'] = forecast.values
    
    df_test['误差'] = (df_test['预测'] - df_test['数据中心用电量（千瓦时）']) / df_test['数据中心用电量（千瓦时）']
    comparison_df = df_test[['数据中心用电量（千瓦时）', '预测', '误差']]
    
    # 设置浮点数显示格式
    pd.options.display.float_format = '{:.4f}'.format
    print("\n实际值、预测值和误差对比:")
    print(comparison_df)
    
    # 计算并打印均方误差
    mse = mean_squared_error(df_test['数据中心用电量（千瓦时）'], df_test['预测'])
    print(f"均方误差 (MSE): {mse}")

    # 绘制实际值与预测值的图形
    plt.figure(figsize=(12, 6))
    plt.plot(df_train.index, df_train['数据中心用电量（千瓦时）'], label='训练集')
    plt.plot(df_test.index, df_test['数据中心用电量（千瓦时）'], label='实际值', color='blue')
    plt.plot(df_test.index, df_test['预测'], label='预测值', color='red')
    plt.legend()
    plt.title('SARIMA预测')
    plt.show()

# 示例调用
file_path = r"C:\Users\juntaox\Desktop\上海市数据中心、人工智能行业月度用电数据.xlsx"  # 替换为实际的文件路径
df_train, df_test = read_and_prepare_data(file_path)
train_and_predict_sarima(df_train, df_test)
