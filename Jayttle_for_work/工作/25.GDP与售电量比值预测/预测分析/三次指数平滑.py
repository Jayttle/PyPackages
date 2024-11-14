import pandas as pd
import numpy as np
import os
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def holt_winters_forecast(data, seasonal_periods=4, trend='add', seasonal='add'):
    """
    使用 Holt-Winters 方法进行预测
    :param data: 训练数据
    :param seasonal_periods: 季节性周期（通常是季度数据，每年4个季度）
    :param trend: 趋势类型，可以是 'add' 或 'mul'（加性或乘性趋势）
    :param seasonal: 季节性类型，可以是 'add' 或 'mul'（加性或乘性季节性）
    :return: 预测的最后一个值
    """
    model = ExponentialSmoothing(data, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods)
    model_fitted = model.fit()
    forecast = model_fitted.forecast(1)  # 预测1个时刻的值
    return forecast[0]  # 返回预测值

def calculate_error(data, seasonal_periods, trend, seasonal, forecast_fn):
    """
    计算预测误差（均方误差MSE）
    """
    actual_values = data[seasonal_periods:]  # 去除季节性周期前的数据作为训练数据
    forecast_values = []
    
    # 对数据进行预测
    for i in range(seasonal_periods, len(data)):  # 从季节性周期开始预测
        forecast_values.append(forecast_fn(data[:i], seasonal_periods, trend, seasonal))  # 预测每个时刻的值
    
    # 计算均方误差
    error = np.mean((np.array(forecast_values) - np.array(actual_values)) ** 2)
    return error

def run_main(read_file_path, output_file_path, region):
    # 读取Excel文件
    df = pd.read_excel(read_file_path, skiprows=0)
    
    # 打印前几行数据来查看数据格式
    print(df.head())

    # 遍历每一行，进行 Holt-Winters 预测（忽略第一行）
    forecasts = []
    best_trends = []  # 用来记录每行数据使用的最优 trend
    best_seasonals = []   # 用来记录每行数据使用的最优 seasonal
    best_errors = []  # 用来记录每行数据的误差值

    # 假设你想尝试的趋势和季节性类型
    trend_types = ['add', 'mul']  # 'add' 为加性趋势，'mul' 为乘性趋势
    seasonal_types = ['add', 'mul']  # 'add' 为加性季节性，'mul' 为乘性季节性

    # 假设季节性周期为4（季度数据）
    seasonal_periods = 4

    # 遍历每一行数据，选择误差最小的趋势和季节性类型
    for index, row in df.iloc[1:].iterrows():  # 使用 df.iloc[1:] 跳过第一行
        industry_name = row.iloc[0]  # 获取产业名称
        data = row.iloc[1:].dropna().values  # 获取产业对应的季度数据，并且忽略缺失值

        if len(data) == 0:
            predicted_value = float('nan')  # 如果所有值都是 NaN，设置预测值为 NaN
            best_trend = float('nan')  # 如果数据为空，最优 trend 也为 NaN
            best_seasonal = float('nan')   # 如果数据为空，最优 seasonal 也为 NaN
        else:
            # 尝试不同的趋势和季节性类型，选择误差最小的那个
            min_error = float('inf')  # 初始化最小误差为无限大
            best_trend = 'add'  # 默认的 trend 类型
            best_seasonal = 'add'   # 默认的 seasonal 类型
            for trend_type in trend_types:
                for seasonal_type in seasonal_types:
                    error = calculate_error(data, seasonal_periods, trend_type, seasonal_type, holt_winters_forecast)
                    if error < min_error:
                        min_error = error
                        best_trend = trend_type
                        best_seasonal = seasonal_type

            # 使用最佳的趋势和季节性类型进行预测
            predicted_value = holt_winters_forecast(data, seasonal_periods, best_trend, best_seasonal)

        # 将预测值和最优的 trend, seasonal 添加到对应的列表中
        forecasts.append(predicted_value)
        best_trends.append(best_trend)
        best_seasonals.append(best_seasonal)
        best_errors.append(min_error)

    # 将预测结果和最优 trend、seasonal 添加到 DataFrame 中
    df['预测值'] = ['2024年第四季度'] + forecasts  # 添加预测值列（第一行设置为 None）
    df['趋势类型'] = [None] + best_trends  # 添加最佳 trend 类型列（第一行设置为 None）
    df['季节性类型'] = [None] + best_seasonals   # 添加最佳 seasonal 类型列（第一行设置为 None）
    df['预测误差'] = [None] + best_errors   # 添加误差列（第一行设置为 None）
    print(f'2024年第四季度:{forecasts}')
    
    # 保存更新后的 DataFrame 到新的 Excel 文件
    # 保存数据到新的工作表
    if not os.path.exists(output_file_path):
        # 如果文件不存在，先创建一个空的Excel文件
        with pd.ExcelWriter(output_file_path, engine='openpyxl') as writer:
            # 创建一个空的工作表
            df.to_excel(writer, sheet_name=region)
            
    else:
        # 保存更新后的 DataFrame 到新的 Excel 文件
        with pd.ExcelWriter(output_file_path, engine='openpyxl', mode='a') as writer:
            if region in writer.book.sheetnames:
                del writer.book[region]
            df.to_excel(writer, sheet_name=region, index=True)


if __name__ == '__main__':
    print('------------------------run-----------------------')
    read_file_path = rf"E:\OneDrive\PyPackages\Jayttle_for_work\工作\25.GDP与售电量比值预测\测试.xlsx"
    output_file_path = rf"E:\OneDrive\PyPackages\Jayttle_for_work\工作\25.GDP与售电量比值预测\测试_result.xlsx"
    region = '预测24年比值'
    run_main(read_file_path, output_file_path, region)
    read_file_path = rf"C:\Users\juntaox\Documents\WeChat Files\wxid_uzs67jx3j0a322\FileStorage\File\2024-11\预测-给钧涛11.14.xlsx"
    output_file_path = rf"C:\Users\juntaox\Desktop\工作\25.GDP与售电量比值预测\二次预测结果11.14.xlsx"
    region = '预测24年比值'
    run_main(read_file_path, output_file_path, region)
