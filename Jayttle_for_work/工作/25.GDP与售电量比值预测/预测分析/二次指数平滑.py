import pandas as pd
import numpy as np
import os

def double_exponential_smoothing_forecast(data, alpha=0.5, beta=0.5):
    """
    二次指数平滑的实现（包括趋势的平滑）
    """
    if len(data) < 2:
            raise ValueError("数据长度必须至少为2个点")
    
    trend = data[1] - data[0]  # 初始化趋势为第二个数据点与第一个数据点的差值
    level = data[0]  # 初始水平（第一个数据点的值）
    forecast = [level]
    
    for t in range(1, len(data)):
        last_level = level
        last_trend = trend
        
        level = alpha * data[t] + (1 - alpha) * (level + trend)
        trend = beta * (level - last_level) + (1 - beta) * last_trend
        
        forecast.append(level + trend)
    
    return forecast[-1]  # 返回最后的预测值

def calculate_error(data, alpha, beta, forecast_fn):
    """
    计算预测误差（均方误差MSE）
    """
    actual_values = data[1:]  # 去除第一个数据点作为训练数据
    forecast_values = []
    
    # 对数据进行预测
    for i in range(2, len(data) + 1):  # 从第二个数据点开始预测，确保数据长度>=2
        forecast_values.append(forecast_fn(data[:i], alpha, beta))  # 预测每个时刻的值
    
    # 计算均方误差
    error = np.mean((np.array(forecast_values) - np.array(actual_values)) ** 2)
    return error


def run_main(read_file_path, output_file_path, region):
    # 读取Excel文件
    df = pd.read_excel(read_file_path, skiprows=0)
    
    # 打印前几行数据来查看数据格式
    print(df.head())

    # 遍历每一行，进行二次指数平滑预测（忽略第一行）
    forecasts = []
    best_alphas = []  # 用来记录每行数据使用的最优 alpha
    best_betas = []   # 用来记录每行数据使用的最优 beta

    # 假设你想尝试的alpha和beta范围
    alpha_range = np.arange(0.4, 0.75, 0.05)  # alpha从0.3到0.5，以0.1为步长
    beta_range = np.arange(0.1, 0.4, 0.02)   # beta从0.3到0.5，以0.1为步长

    # 遍历每一行数据，选择误差最小的 alpha 和 beta
    for index, row in df.iloc[1:].iterrows():  # 使用 df.iloc[1:] 跳过第一行
        industry_name = row.iloc[0]  # 获取产业名称
        data = row.iloc[1:].dropna().values  # 获取产业对应的季度数据，并且忽略缺失值

        if len(data) == 0:
            predicted_value = float('nan')  # 如果所有值都是 NaN，设置预测值为 NaN
            best_alpha = float('nan')  # 如果数据为空，最优 alpha 也为 NaN
            best_beta = float('nan')   # 如果数据为空，最优 beta 也为 NaN
        else:
            # 尝试不同的 alpha 和 beta，选择误差最小的那个
            min_error = float('inf')  # 初始化最小误差为无限大
            best_alpha = 0.5  # 默认的 alpha 值
            best_beta = 0.05   # 默认的 beta 值
            for alpha in alpha_range:
                for beta in beta_range:
                    error = calculate_error(data, alpha, beta, double_exponential_smoothing_forecast)
                    if error < min_error:
                        min_error = error
                        best_alpha = alpha
                        best_beta = beta

            # 使用最佳的 alpha 和 beta 进行预测
            predicted_value = double_exponential_smoothing_forecast(data, best_alpha, best_beta)

        # 将预测值和最优的 alpha, beta 添加到对应的列表中
        forecasts.append(predicted_value)
        best_alphas.append(best_alpha)
        best_betas.append(best_beta)

    # 将预测结果和最优 alpha、beta 添加到 DataFrame 中
    df['预测值'] = ['2024年第四季度'] + forecasts  # 添加预测值列（第一行设置为 None）
    df['指数平滑alpha'] = [None] + best_alphas  # 添加最佳 alpha 列（第一行设置为 None）
    df['指数平滑beta'] = [None] + best_betas   # 添加最佳 beta 列（第一行设置为 None）
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
    read_file_path =rf"E:\OneDrive\PyPackages\Jayttle_for_work\工作\25.GDP与售电量比值预测\测试.xlsx"
    output_file_path = rf"E:\OneDrive\PyPackages\Jayttle_for_work\工作\25.GDP与售电量比值预测\测试_result.xlsx"
    region = '预测24年比值'
    run_main(read_file_path, output_file_path, region)
    read_file_path = rf"C:\Users\juntaox\Documents\WeChat Files\wxid_uzs67jx3j0a322\FileStorage\File\2024-11\预测-给钧涛11.14.xlsx"
    output_file_path = rf"C:\Users\juntaox\Desktop\工作\25.GDP与售电量比值预测\二次预测结果11.14.xlsx"
    region = '预测24年比值'
    run_main(read_file_path, output_file_path, region)
