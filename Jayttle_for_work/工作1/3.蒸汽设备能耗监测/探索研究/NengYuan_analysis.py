import pandas as pd
from typing import List, Dict
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import seaborn as sns
from scipy import stats
import json
import os
from sklearn.metrics import mean_squared_error
from scipy.stats import linregress

# region 字体设置
plt.rcParams['font.sans-serif'] = ['SimSun']  # 指定宋体为默认字体
plt.rcParams['font.sans-serif'] = 'SimSun'  # 使用指定的中文字体
plt.rcParams['axes.unicode_minus']=False#用来正常显示负

def NengYuan_read_csv(file_path: str) -> pd.DataFrame:
    """
    读取 CSV 文件，筛选指定列并进行清洗处理，返回处理后的 DataFrame。
    
    :param file_path: CSV 文件的路径
    :return: 处理后的 DataFrame
    """
    # 显式指定列的数据类型
    dtype_dict = {
        'recipename': 'str',  # 假设第1列应该是浮动类型
        'module': 'str',    # 假设第2列应该是整数类型
        'module_task': 'str'       # 假设第3列应该是字符串类型
    }
    df = pd.read_csv(file_path, dtype=dtype_dict)

    # 筛选掉 'module' 为 0 的行
    df_filtered = df[df['module'] != '0']

    df_filtered_cleaned = df_filtered.copy()

    # 删除包含 NaN 值的行
    df_filtered_cleaned = df_filtered_cleaned.dropna()

    # 将 'datetime' 列转换为 pandas 的 datetime 类型
    df_filtered_cleaned['datetime'] = pd.to_datetime(df_filtered_cleaned['datetime'], errors='coerce')

    # 检查是否存在无效的 'datetime' 数据，如果有的话，删除这些行
    df_filtered_cleaned = df_filtered_cleaned.dropna(subset=['datetime'])

    # 将 'datetime' 转换为精确到分钟
    df_filtered_cleaned['datetime_min'] = df_filtered_cleaned['datetime'].dt.floor('min')
    # 统计每个分钟内的数据量
    minute_counts = df_filtered_cleaned.groupby('datetime_min').size()
    # 为每个同一分钟的数据分配秒数
    def assign_seconds(group):
        n = len(group)
        # 使用 linspace 生成均匀的秒数
        group['second'] = np.linspace(0, 59, num=n, endpoint=False).astype(int)
        return group
    # 对每一分钟的数据进行秒数分配
    df_filtered_cleaned = df_filtered_cleaned.groupby('datetime_min').apply(assign_seconds)
    # 为每条数据创建一个新的 datetime，精确到秒
    df_filtered_cleaned['new_datetime'] = df_filtered_cleaned['datetime_min'] + pd.to_timedelta(df_filtered_cleaned['second'], unit='s')
    # 删除 'second' 和 'datetime_min' 列，因为它们不再需要
    df_filtered_cleaned = df_filtered_cleaned.drop(columns=['second', 'datetime_min'])
    return df_filtered_cleaned

def get_per_stats_A_HT_steam(df: pd.DataFrame):
    df['A_HT_steam_shift'] = df['A_HT_steam_total_blend'].diff()   
    # 重置索引
    A_HT_steam_df = df[df['A_HT_steam_shift'] > 0].dropna()
    A_HT_steam_df.reset_index(drop=True, inplace=True)
    A_HT_steam_df['datetime'] = pd.to_datetime(A_HT_steam_df['datetime'])
    # 设置时间戳为索引，并按分钟重新采样
    A_HT_steam_df.set_index('datetime', inplace=True)
    # 根据新的列名进行分组和聚合
    stats_df = A_HT_steam_df.groupby(['module', 'team', 'shift', 'A_HT_phase'])['A_HT_steam_shift'].agg(
        ['mean', 'max', 'min', 'std']
    ).reset_index()
    # 统计每个组合的记录数量
    count_df = A_HT_steam_df.groupby(['module', 'team', 'shift', 'A_HT_phase']).size().reset_index(name='data_count')
    # 将统计数据与记录数量合并
    final_stats_df = pd.merge(stats_df, count_df, on=['module', 'team', 'shift', 'A_HT_phase'])
    return final_stats_df

def get_per_min_A_HT_steam(df: pd.DataFrame, output_file: str) -> pd.DataFrame:
    # 删除不需要的列
    df.drop(columns=['B_HT_phase', 'B_HT_steam_total', 'B_HT_steam_total_blend',
                     'TT1142_phase', 'TT1142_steam_total', 'TT1142_steam_total_blend'], inplace=True) 

    # 计算 A_HT_steam_total_blend 与前一行的差值并生成新列 A_HT_steam_shift
    df['A_HT_steam_shift'] = df['A_HT_steam_total_blend'].diff()

    # 筛选 A_HT_steam_shift > 0 且去除 NaN 的行
    A_HT_steam_df = df[df['A_HT_steam_shift'] > 0].dropna()
    # 重置索引
    A_HT_steam_df.reset_index(drop=True, inplace=True)
    A_HT_steam_df['datetime'] = pd.to_datetime(A_HT_steam_df['datetime'])
    
    # 设置时间戳为索引，并按分钟重新采样
    A_HT_steam_df.set_index('datetime', inplace=True)
    
    # 重采样计算每分钟的平均值、总值以及保留其他信息
    resampled_df = A_HT_steam_df.resample('min').agg(
        {
            'recipename': 'first',  # 保留第一个值
            'module': 'first',  # 保留第一个值
            'module_task': 'first',  # 保留第一个值
            'shift': 'first',  # 保留第一个值
            'team': 'first',  # 保留第一个值
            'no_gap': 'first',  # 保留第一个值
            'A_HT_phase': 'first',  # 保留 A_HT_phase 的第一个值
            'A_HT_steam_shift': ['mean', 'sum', 'std'],  # 计算每分钟的平均值和总值
        }
    )
    
    # 将列扁平化（例如 ('A_HT_steam_shift', 'mean') 变成 'A_HT_steam_shift_mean'）
    resampled_df.columns = ['_'.join(col).strip() for col in resampled_df.columns.values]
    
    # 由于我们设置了 datetime 为索引，重采样时会自动生成时间戳列
    # 将 datetime 列重新添加到数据框中
    resampled_df['datetime'] = resampled_df.index
    
    # 创建 ExcelWriter 对象，准备将不同的 recipename 数据输出到不同的工作表
    with pd.ExcelWriter(output_file) as writer:
        # 遍历每个 recipename 分组，并将每个组写入到 Excel 中的不同 sheet
        for recipename, group in resampled_df.groupby('recipename_first'):
            group.to_excel(writer, sheet_name=recipename, index=False)
    
    print(f"数据已成功写入到 {output_file}")
    return resampled_df

def plt_in_module(df: pd.DataFrame):
    # 创建一个颜色映射（确保根据 A_HT_phase_first 的唯一值分配不同颜色）
    unique_phases = df['A_HT_phase_first'].unique()  # 获取所有唯一的 A_HT_phase_first 值
    num_phases = len(unique_phases)
    cmap = plt.get_cmap('tab10', num_phases)  # 如果你使用的是plt，继续使用此方法

    # 创建一个从 A_HT_phase_first 到颜色的字典
    phase_to_color = {phase: cmap(i) for i, phase in enumerate(unique_phases)}

    # 根据 'module_first' 进行分组
    for module, group in df.groupby('module_first'):
        plt.figure(figsize=(10, 6))

        # 假设数据的索引是时间戳，先计算时间差
        group['time_diff'] = group.index.to_series().diff().dt.total_seconds() / 60  # 以分钟为单位

        # 遍历每对相邻的数据点
        for i in range(1, len(group)):
            if group['time_diff'].iloc[i] <= 2:  # 如果时间差小于等于2分钟
                phase = group['A_HT_phase_first'].iloc[i]  # 获取当前点的 A_HT_phase_first 值
                color = phase_to_color[phase]  # 获取对应的颜色
                
                # 绘制相邻的数据点
                plt.plot(group.index[i-1:i+1], group['A_HT_steam_shift_mean'].iloc[i-1:i+1],
                         label=f"Module: {module} - Phase: {phase}" if i == 1 else "",
                         color=color, linewidth=2)

        # 添加标题和标签
        plt.title(f"Module: {module} - A_HT_steam_shift_mean", fontsize=14)
        plt.xlabel("时间", fontsize=12)
        plt.ylabel("A_HT_steam_shift_mean", fontsize=12)

        # 显示网格
        plt.grid(True)

        # 显示图例
        plt.legend()

        # 显示图形
        plt.show()

if __name__ == '__main__':
    file_path = r"C:\Users\juntaox\Desktop\工作1\3.蒸汽设备能耗监测\原始材料\energy数据\energy\NengYuan_20240706_20240712.csv"
    output_file = r"C:\Users\juntaox\Desktop\工作1\3.蒸汽设备能耗监pip install jupyter测\xjt-处理\A_HT_steam_recipe.xlsx"
    # 读取和处理数据
    df = NengYuan_read_csv(file_path)
    final_stats_df = get_per_stats_A_HT_steam(df)
    final_stats_df.to_excel( r"C:\Users\juntaox\Desktop\工作1\3.蒸汽设备能耗监测\xjt-处理\A_HT_steam_stats.xlsx")
    # plt_in_module(A_steam_per_min_df)
