import pandas as pd
from typing import List, Dict
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import seaborn as sns
from scipy import stats
import json
import glob
import os
from sklearn.metrics import mean_squared_error
from scipy.stats import linregress

# region 字体设置
plt.rcParams['font.sans-serif'] = ['SimSun']  # 指定宋体为默认字体
plt.rcParams['font.sans-serif'] = 'SimSun'  # 使用指定的中文字体
plt.rcParams['axes.unicode_minus']=False#用来正常显示负

def plt_SPC_UCL_LCL(result_df: pd.DataFrame, item: str):
    # 检查 item 列是否存在
    if item in result_df.columns:
        # 计算 item 列的均值和标准偏差
        mean = result_df[item].mean()
        std_dev = result_df[item].std()
        
        # 计算 SPC 上下限
        UCL = mean + 3 * std_dev  # 上控制限
        LCL = mean - 3 * std_dev  # 下控制限

        print(f"Mean: {mean}, Standard Deviation: {std_dev}")
        print(f"UCL: {UCL}, LCL: {LCL}")

        # 绘制 item 列的折线图
        plt.figure(figsize=(10, 6))  # 设置图形大小
        plt.plot(result_df[item], linestyle='-', color='b', label=item)  # 绘制折线图

        # 绘制上下控制限
        plt.axhline(UCL, color='r', linestyle='--', label=f'UCL ({UCL:.2f})')  # 上控制限
        plt.axhline(LCL, color='r', linestyle='--', label=f'LCL ({LCL:.2f})')  # 下控制限

        # 添加图表标题、标签和图例
        plt.title('Sum Column Line Chart with SPC Control Limits')  # 设置图表标题
        plt.xlabel('Index')  # 设置 x 轴标签
        plt.ylabel('Sum Value')  # 设置 y 轴标签
        plt.grid(True)  # 添加网格
        plt.legend()  # 显示图例
        plt.show()  # 显示图表
    else:
        print("item column not found in the DataFrame.")

def plt_SPC_rolling_window(result_df: pd.DataFrame, item: str, window_size: int = 20):
    # 检查 item 列是否存在
    if item in result_df.columns:
        # 使用滚动窗口计算均值和标准偏差
        rolling_mean = result_df[item].rolling(window=window_size).mean()
        rolling_std = result_df[item].rolling(window=window_size).std()
        
        # 计算全局均值和标准偏差
        global_mean = result_df[item].mean()
        global_std = result_df[item].std()
        
        # 计算上下控制限
        UCL = global_mean + 3 * global_std  # 上控制限
        LCL = global_mean - 3 * global_std  # 下控制限

        print(f"Global Mean: {global_mean}, Global Standard Deviation: {global_std}")
        print(f"Global UCL: {UCL}, Global LCL: {LCL}")

        # 绘制滚动窗口均值和标准偏差图
        plt.figure(figsize=(12, 6))  # 设置图形大小

        # 绘制原始数据
        plt.plot(result_df[item], linestyle='-', color='b', label=f'{item} Data')
        
        # 绘制滚动窗口的均值
        plt.plot(rolling_mean, linestyle='-', color='g', label=f'{item} Rolling Mean')
        
        # 绘制滚动窗口的标准偏差
        plt.plot(rolling_mean + 3 * rolling_std, linestyle='--', color='orange', label=f'{item} Rolling UCL')
        plt.plot(rolling_mean - 3 * rolling_std, linestyle='--', color='orange', label=f'{item} Rolling LCL')

        # 绘制全局UCL和LCL
        plt.axhline(UCL, color='r', linestyle='--', label=f'Global UCL ({UCL:.2f})')
        plt.axhline(LCL, color='r', linestyle='--', label=f'Global LCL ({LCL:.2f})')

        # 添加图表标题、标签和图例
        plt.title(f'{item} Rolling Mean, UCL & LCL (Window Size = {window_size})')  # 设置图表标题
        plt.xlabel('Index')  # 设置 x 轴标签
        plt.ylabel(f'{item} Value')  # 设置 y 轴标签
        plt.grid(True)  # 添加网格
        plt.legend()  # 显示图例
        plt.show()  # 显示图表
    else:
        print(f"'{item}' column not found in the DataFrame.")

def plot_line_chart(df, item):
    """
    绘制给定 DataFrame 中某一列的折线图。

    :param df: pandas DataFrame，包含要绘制的数据。
    :param item: 字符串，指定要绘制的列名。
    """
    if item not in df.columns:
        print(f"Error: The item '{item}' is not found in the DataFrame.")
        return
    # 绘制折线图
    plt.figure(figsize=(10, 6))
    plt.plot(df[item], linestyle='-', color='b', label=item)
    plt.title(f"Line Plot of {item}")
    plt.xlabel("Index")
    plt.ylabel(item)
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    # 读取 Excel 文件
    file_path = r"C:\Users\juntaox\Desktop\工作1\3.蒸汽设备能耗监测\result_df.xlsx"
    result_df = pd.read_excel(file_path)
    plot_line_chart(result_df, 'sum')
    plot_line_chart(result_df, 'mean')
    plt_SPC_UCL_LCL(result_df, 'sum')
    plt_SPC_UCL_LCL(result_df, 'mean')
    # 调用滑动窗口分析函数
    plt_SPC_rolling_window(result_df, 'sum', window_size=20)
    plt_SPC_rolling_window(result_df, 'mean', window_size=20)