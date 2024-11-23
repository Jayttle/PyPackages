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

def get_A_HT_steam(df: pd.DataFrame) -> pd.DataFrame:
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
 
    return A_HT_steam_df

def segments_time_threshold(df: pd.DataFrame, time_threshold: str = '2min') -> List[pd.DataFrame]:
    """
    根据时间间隔将 DataFrame 按指定时间阈值分段。

    :param df: 要分段的 DataFrame，必须包含 'datetime' 列
    :param time_threshold: 时间间隔阈值，默认 '2min'
    :return: 按时间分段后的 DataFrame 列表
    """
    # 用来存储每个时段的分段
    segments = []
    
    # 初始化时段编号
    period_number = 1
    
    # 新的时段的起始行
    segment_start = df.iloc[0]
    
    # 遍历数据框中的行，计算时间间隔
    for i in range(1, len(df)):
        # 计算当前行与上一行的时间间隔
        time_diff = df.iloc[i]['datetime'] - df.iloc[i - 1]['datetime']
        
        # 如果时间间隔大于规定阈值，分段
        if time_diff > pd.Timedelta(time_threshold):
            # 将前一个时段的部分保存
            segments.append(df[(df['datetime'] >= segment_start['datetime']) & (df['datetime'] < df.iloc[i - 1]['datetime'])])
            # 更新下一个时段的起始行
            segment_start = df.iloc[i]
            period_number += 1
    
    # 最后一个时段的数据添加
    segments.append(df[(df['datetime'] >= segment_start['datetime'])])
    
    return segments
def plt_in_A_HT_phase(one_period_df: pd.DataFrame):
    # 检查是否存在 'A_HT_steam_shift' 和 'A_HT_phase' 列
    if 'A_HT_steam_shift' not in one_period_df.columns or 'A_HT_phase' not in one_period_df.columns:
        raise ValueError("'A_HT_steam_shift' 或 'A_HT_phase' 列缺失。")
    
    # 确保索引是 datetime 类型
    if not pd.api.types.is_datetime64_any_dtype(one_period_df.index):
        raise ValueError("索引必须是 datetime 类型")

    # 获取唯一的 A_HT_phase 值，用于分组
    unique_phases = one_period_df['A_HT_phase'].unique()

    # 设置绘图
    plt.figure(figsize=(12, 6))

    # 对每个 A_HT_phase 进行分组并绘图
    for phase in unique_phases:
        # 过滤出当前 A_HT_phase 对应的数据
        phase_data = one_period_df[one_period_df['A_HT_phase'] == phase]
        
        # 绘制该组的 'A_HT_steam_shift' 随时间变化的曲线
        plt.plot(phase_data.index, phase_data['A_HT_steam_shift'], label=f'Phase {phase}')
    
    # 添加图例
    plt.legend(title="A_HT Phase")
    
    # 设置标题和轴标签
    plt.title('A_HT_steam_shift vs Time by A_HT_phase')
    plt.xlabel('Time')
    plt.ylabel('A_HT_steam_shift')

    # 格式化 x 轴时间显示
    # 使用 '%dd-hh:mm' 格式
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    # 自动旋转 x 轴标签以避免重叠
    plt.gcf().autofmt_xdate()

    # 显示图形
    plt.tight_layout()
    plt.show()


def calculate_Cpk(df: pd.DataFrame, usl: float, lsl: float):
    """
    计算 Cpk（过程能力指数）
    
    参数：
    df : pd.DataFrame : 数据框，包含目标列 'A_HT_steam_shift'
    usl : float : 上规格限
    lsl : float : 下规格限
    
    返回：
    float : Cpk 值
    """
    # 确保数据列存在
    if 'A_HT_steam_shift' not in df.columns:
        raise ValueError("'A_HT_steam_shift' 列缺失。")
    
    # 计算均值和标准差
    mean_value = df['A_HT_steam_shift'].mean()
    std_value = df['A_HT_steam_shift'].std()
    
    # 计算 Cpk
    cpk_upper = (usl - mean_value) / (3 * std_value)
    cpk_lower = (mean_value - lsl) / (3 * std_value)
    
    cpk = min(cpk_upper, cpk_lower)
    
    print(f"Cpk 值: {cpk}")
    
    return cpk


def calculate_Ppk(df: pd.DataFrame, usl: float, lsl: float):
    """
    计算 Ppk（过程表现指数）
    
    参数：
    df : pd.DataFrame : 数据框，包含目标列 'A_HT_steam_shift'
    usl : float : 上规格限
    lsl : float : 下规格限
    
    返回：
    float : Ppk 值
    """
    # 确保数据列存在
    if 'A_HT_steam_shift' not in df.columns:
        raise ValueError("'A_HT_steam_shift' 列缺失。")
    
    # 计算均值和标准差（全体数据）
    mean_value = df['A_HT_steam_shift'].mean()
    std_value = df['A_HT_steam_shift'].std()  # 这里用整个样本的标准差
    
    # 计算 Ppk
    ppk_upper = (usl - mean_value) / (3 * std_value)
    ppk_lower = (mean_value - lsl) / (3 * std_value)
    
    ppk = min(ppk_upper, ppk_lower)
    
    print(f"Ppk 值: {ppk}")
    
    return ppk


def plt_in_A_HT_team(one_period_df: pd.DataFrame):
    # 检查是否存在 'A_HT_steam_shift', 'shift', 'team' 列
    required_columns = ['A_HT_steam_shift', 'team']
    for col in required_columns:
        if col not in one_period_df.columns:
            raise ValueError(f"'{col}' 列缺失。")
    
    # 确保索引是 datetime 类型
    if not pd.api.types.is_datetime64_any_dtype(one_period_df.index):
        raise ValueError("索引必须是 datetime 类型")

    # 获取唯一的 team 值，用于分组
    unique_teams = one_period_df['team'].unique()

    # 使用 Seaborn 的 color_palette 为不同的 team 分配颜色
    team_colors = sns.color_palette("Set1", len(unique_teams))

    # 为每个组分配唯一颜色
    team_color_map = dict(zip(unique_teams, team_colors))

    # 设置绘图
    plt.figure(figsize=(12, 6))

    # 对每个 team 进行分组并绘图
    for team in unique_teams:
        team_data = one_period_df[one_period_df['team'] == team]

        # 获取当前 team 对应的颜色
        color = team_color_map[team]
        
        # 绘制该组的 'A_HT_steam_shift' 随时间变化的曲线
        plt.plot(team_data.index, team_data['A_HT_steam_shift'],
                 label=f'Team {team}',
                 color=color)
    
    # 添加图例
    plt.legend(title="Team")
    
    # 设置标题和轴标签
    plt.title('A_HT_steam_shift vs Time by Team')
    plt.xlabel('Time')
    plt.ylabel('A_HT_steam_shift')

    # 格式化 x 轴时间显示
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    # 自动旋转 x 轴标签以避免重叠
    plt.gcf().autofmt_xdate()

    # 显示图形
    plt.tight_layout()
    plt.show()

def SPC_A_HT_steam(df: pd.DataFrame):
    # 检查是否存在 'A_HT_steam_shift' 列
    if 'A_HT_steam_shift' not in df.columns:
        raise ValueError("'A_HT_steam_shift' 列缺失。")

    # 计算均值和标准差
    mean_value = df['A_HT_steam_shift'].mean()
    std_value = df['A_HT_steam_shift'].std()

    # 定义控制限，通常为均值 ± 3倍标准差
    upper_control_limit = mean_value + 3 * std_value
    lower_control_limit = mean_value - 3 * std_value

    # 标记异常点：超出控制限的点
    df['is_anomaly'] = (df['A_HT_steam_shift'] > upper_control_limit) | (df['A_HT_steam_shift'] < lower_control_limit)

    # 打印异常点的索引
    print(f"异常点的索引：\n{df[df['is_anomaly']]}")
    
    # 绘制 SPC 控制图
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['A_HT_steam_shift'], label='A_HT_steam_shift', color='blue')  # 原始数据
    plt.axhline(mean_value, color='green', linestyle='--', label='Mean')  # 均值线
    plt.axhline(upper_control_limit, color='red', linestyle='--', label='Upper Control Limit (UCL)')  # 上控制限
    plt.axhline(lower_control_limit, color='red', linestyle='--', label='Lower Control Limit (LCL)')  # 下控制限
    
    # 标记异常点
    plt.scatter(df.index[df['is_anomaly']], df['A_HT_steam_shift'][df['is_anomaly']], color='red', label='Anomalies')
    
    # 设置图例和标题
    plt.legend()
    plt.title('SPC Control Chart for A_HT_steam_shift')
    plt.xlabel('Time')
    plt.ylabel('A_HT_steam_shift')

    # 显示图形
    plt.tight_layout()
    plt.show()

def plt_window_A_HT_steam(df: pd.DataFrame, window_size=30):
    # 检查是否存在 'A_HT_steam_shift' 列
    if 'A_HT_steam_shift' not in df.columns:
        raise ValueError("'A_HT_steam_shift' 列缺失。")

    # 使用滑动窗口计算均值和标准差
    rolling_mean = df['A_HT_steam_shift'].rolling(window=window_size, min_periods=1).mean()
    rolling_std = df['A_HT_steam_shift'].rolling(window=window_size, min_periods=1).std()

    # 计算控制限：均值 ± 3倍标准差
    upper_control_limit = rolling_mean + 3 * rolling_std
    lower_control_limit = rolling_mean - 3 * rolling_std

    # 标记异常点：超出当前窗口控制限的点
    df['is_anomaly'] = (df['A_HT_steam_shift'] > upper_control_limit) | (df['A_HT_steam_shift'] < lower_control_limit)

    # 打印异常点的索引
    print(f"异常点的索引：\n{df[df['is_anomaly']]}")
    
    # 绘制 SPC 控制图
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['A_HT_steam_shift'], label='A_HT_steam_shift', color='blue')  # 原始数据
    plt.plot(df.index, rolling_mean, label='Rolling Mean', color='green', linestyle='--')  # 滚动均值线
    # plt.plot(df.index, upper_control_limit, label='Upper Control Limit (UCL)', color='red', linestyle='--')  # 上控制限
    # plt.plot(df.index, lower_control_limit, label='Lower Control Limit (LCL)', color='red', linestyle='--')  # 下控制限
    
    # 标记异常点
    plt.scatter(df.index[df['is_anomaly']], df['A_HT_steam_shift'][df['is_anomaly']], color='red', label='Anomalies')
    
    # 设置 y 轴范围，确保从 0 开始
    max_value = max(df['A_HT_steam_shift'].max(), upper_control_limit.max())
    plt.ylim(0, max_value * 1.1)  # 留一点空间以避免数据接触边界
    
    # 设置图例和标题
    plt.legend()
    plt.title(f'SPC Control Chart for A_HT_steam_shift with Window Size {window_size}')
    plt.xlabel('Time')
    plt.ylabel('A_HT_steam_shift')

    # 显示图形
    plt.tight_layout()
    plt.show()

def df_drop_duplicates(one_period_df: pd.DataFrame):
    # 获取 recname, module, module_task, shift, team, no_gap 列的唯一组合
    unique_combinations = one_period_df[['recipename', 'module', 'module_task', 'shift', 'team', 'no_gap']].drop_duplicates()
    # 打印出唯一的组合
    print(unique_combinations)    
    # 初始化一个字典来存储结果
    time_ranges = {}
    counts = {}
    # 按组合分组
    for _, combination in unique_combinations.iterrows():
        # 创建一个过滤条件，选择当前组合对应的行
        filter_condition = (
            (one_period_df['recipename'] == combination['recipename']) &
            (one_period_df['module'] == combination['module']) &
            (one_period_df['module_task'] == combination['module_task']) &
            (one_period_df['shift'] == combination['shift']) &
            (one_period_df['team'] == combination['team']) &
            (one_period_df['no_gap'] == combination['no_gap'])
        )
        
        # 获取对应组合的子集
        subset = one_period_df[filter_condition]
        
        # 计算时间范围
        start_time = subset.index.min()  # 最早的时间戳
        end_time = subset.index.max()    # 最晚的时间戳
        
        # 存储时间范围和数量
        time_ranges[tuple(combination)] = (start_time, end_time)
        counts[tuple(combination)] = len(subset)
    
    # 打印结果
    for combination, (start_time, end_time) in time_ranges.items():
        print(f"组合: {combination}")
        print(f"时间范围: {start_time} 到 {end_time}")
        print(f"记录数量: {counts[combination]}")
        print('-' * 50)

if __name__ == '__main__':
    file_path = r"C:\Users\juntaox\Desktop\工作1\3.蒸汽设备能耗监测\原始材料\energy数据\energy\NengYuan_20240706_20240712.csv"
    df = NengYuan_read_csv(file_path)
    A_HT_steam_df = get_A_HT_steam(df)
    segments = segments_time_threshold(A_HT_steam_df)
    one_period_df = segments[0]
    one_period_df.set_index('new_datetime', inplace=True)
    print(one_period_df)

    plt_in_A_HT_phase(one_period_df)
    plt_in_A_HT_team(one_period_df)
    # SPC_A_HT_steam(one_period_df)
    plt_window_A_HT_steam(one_period_df)