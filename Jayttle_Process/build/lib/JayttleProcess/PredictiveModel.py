import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid

def read_and_prepare_data(file_path: str, date_col: str='日期', target_col: str='总和', test_size: int=6):
    """
    读取和准备数据，确保日期列为 datetime 类型，并将数据分为训练集和测试集。
    
    Parameters:
    - file_path: 数据文件路径
    - date_col: 日期列的列名
    - target_col: 目标列的列名
    - test_size: 测试集的样本数量
    
    Returns:
    - df_train: 训练集 DataFrame
    - df_test: 测试集 DataFrame
    """
    df = pd.read_excel(file_path)
    
    # 确保日期列是 datetime 类型
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
    else:
        raise KeyError(f"DataFrame 中找不到 '{date_col}' 列")
    
    # 设置日期列为索引
    df.set_index(date_col, inplace=True)
    
    # 划分训练集与测试集
    df_train = df.iloc[:-test_size]
    df_test = df.iloc[-test_size:]
    
    print(f"训练集样本数：{len(df_train)}")
    print(f"测试集样本数：{len(df_test)}")
    
    return df_train, df_test

def evaluate_sarima(params, df_train, df_test, target_col):
    """
    评估 SARIMA 模型的均方误差。
    
    Parameters:
    - params: SARIMA 模型的参数字典
    - df_train: 训练集 DataFrame
    - df_test: 测试集 DataFrame
    - target_col: 目标列的列名
    
    Returns:
    - mse: 均方误差
    """
    model = SARIMAX(df_train[target_col], 
                    order=params['order'], 
                    seasonal_order=params['seasonal_order'])
    model_fit = model.fit(disp=False)
    
    forecast = model_fit.forecast(steps=len(df_test))
    
    df_test['预测'] = forecast.values
    df_test['误差'] = (df_test['预测'] - df_test[target_col]) / df_test[target_col]
    
    mse = mean_squared_error(df_test[target_col], df_test['预测'])
    return mse

def grid_search_sarima(df_train, df_test, target_col):
    """
    使用网格搜索法找到 SARIMA 模型的最佳参数组合。
    
    Parameters:
    - df_train: 训练集 DataFrame
    - df_test: 测试集 DataFrame
    - target_col: 目标列的列名
    
    Returns:
    - best_params: 最佳参数组合
    """
    best_params = None
    best_mse = np.inf
    
    param_grid = {
        'order': [(p, d, q) for p in [1] for d in [0] for q in [2]],
        'seasonal_order': [(P, D, Q, S) for P in [1] for D in [1] for Q in range(1) for S in [12]]
    }

    for params in ParameterGrid(param_grid):
        mse = evaluate_sarima(params, df_train, df_test, target_col)
        print(f"测试参数 {params} 的均方误差: {mse}")
        
        if mse < best_mse:
            best_mse = mse
            best_params = params
    
    print(f"最佳参数组合: {best_params}")
    print(f"最佳均方误差: {best_mse}")
    
    return best_params

def train_and_predict_sarima(file_path: str, date_col: str='日期', target_col: str='总和', test_size: int=6):
    """
    从数据读取到模型训练和预测的完整过程。
    
    Parameters:
    - file_path: 数据文件路径
    - date_col: 日期列的列名
    - target_col: 目标列的列名
    - test_size: 测试集的样本数量
    """
    # 读取并准备数据
    df_train, df_test = read_and_prepare_data(file_path, date_col, target_col, test_size)
    
    # 使用网格搜索法找到最佳参数
    best_params = grid_search_sarima(df_train, df_test, target_col)
    
    # 使用最佳参数训练模型
    model = SARIMAX(df_train[target_col], 
                    order=best_params['order'], 
                    seasonal_order=best_params['seasonal_order'])
    model_fit = model.fit(disp=False)
    
    # 进行预测
    forecast = model_fit.forecast(steps=len(df_test))
    
    # 将预测结果与实际测试集进行比较
    df_test['预测'] = forecast.values
    df_test['误差'] = (df_test['预测'] - df_test[target_col]) / df_test[target_col]
    comparison_df = df_test[[target_col, '预测', '误差']]
    
    print("\n实际值、预测值和误差对比:")
    print(comparison_df)
    
    # 计算并打印均方误差
    mse = mean_squared_error(df_test[target_col], df_test['预测'])
    print(f"均方误差 (MSE): {mse}")

    # 绘制实际值与预测值的图形
    plt.figure(figsize=(12, 6))
    plt.plot(df_train.index, df_train[target_col], label='训练集')
    plt.plot(df_test.index, df_test[target_col], label='实际值', color='blue')
    plt.plot(df_test.index, df_test['预测'], label='预测值', color='red')
    plt.legend()
    plt.title('SARIMA预测')
    plt.show()
