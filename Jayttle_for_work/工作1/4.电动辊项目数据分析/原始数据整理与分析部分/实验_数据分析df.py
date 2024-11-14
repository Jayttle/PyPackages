import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import seaborn as sns
from scipy import stats
import json
import os

# region 字体设置
plt.rcParams['font.sans-serif'] = ['SimSun']  # 指定宋体为默认字体
plt.rcParams['font.sans-serif'] = 'SimSun'  # 使用指定的中文字体
plt.rcParams['axes.unicode_minus']=False#用来正常显示负

def 统计特征分析(df):
    """
    分析 motor_run 在 True 和 False 两种情况下，motor_speed、motor_temperature 和 motor_torque 的统计特征
    """
    # 分离 motor_run 为 True 和 False 的数据子集
    df_true = df[df['motor_run'] == True]
    df_false = df[df['motor_run'] == False]

    # 计算统计特征：均值、标准差、最小值、最大值、中位数
    stats_true = df_true[['motor_speed', 'motor_temperature', 'motor_torque']].describe().T[['mean', 'std', 'min', 'max', '50%']]
    stats_false = df_false[['motor_speed', 'motor_temperature', 'motor_torque']].describe().T[['mean', 'std', 'min', 'max', '50%']]

    # 打印统计特征
    print("motor_run == True 时的统计特征：")
    print(stats_true)
    print("\n")
    
    print("motor_run == False 时的统计特征：")
    print(stats_false)
    
    return stats_true, stats_false

def 相关系数分析(df):
    """
    分析 motor_run 在 True 和 False 两种情况下，motor_speed、motor_temperature 和 motor_torque 之间的相关系数
    """

    # 分离 motor_run 为 True 和 False 的数据子集
    df_true = df[df['motor_run'] == True]
    df_false = df[df['motor_run'] == False]

    # 计算相关系数矩阵
    correlation_true = df_true[['motor_speed', 'motor_temperature', 'motor_torque']].corr()
    correlation_false = df_false[['motor_speed', 'motor_temperature', 'motor_torque']].corr()

    # 打印相关系数矩阵
    print("motor_run == True 时的相关系数矩阵：")
    print(correlation_true)
    print("\n")
    
    print("motor_run == False 时的相关系数矩阵：")
    print(correlation_false)

    # 可视化相关系数矩阵
    plt.figure(figsize=(12, 6))

    # True 的相关系数热力图
    plt.subplot(1, 2, 1)
    sns.heatmap(correlation_true, annot=True, cmap='coolwarm', center=0)
    plt.title('motor_run == True 的相关系数')

    # False 的相关系数热力图
    plt.subplot(1, 2, 2)
    sns.heatmap(correlation_false, annot=True, cmap='coolwarm', center=0)
    plt.title('motor_run == False 的相关系数')

    # 显示图形
    plt.tight_layout()
    plt.show()


def 扭矩变化规律分析(df, torque_threshold=10):
    """
    该函数分析扭矩变化规律，检测异常变化时间点。
    
    参数:
    df : pandas DataFrame
        包含'datetime'和'motor_torque'列的数据。
    torque_threshold : float, optional
        定义扭矩变化的阈值，用于识别异常变化（默认值为10）。
    
    返回:
    None
    """
    # 确保 'datetime' 列是 datetime 类型
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # 按时间排序
    df = df.sort_values(by='datetime')
    
    # 计算连续两次数据的扭矩变化（使用差值）
    df['torque_diff'] = df['motor_torque'].diff()  # 计算每一行与前一行的差值
    
    # 标记异常变化点：当差值超过阈值时视为异常
    df['异常变化'] = df['torque_diff'].apply(lambda x: '异常' if abs(x) > torque_threshold else '正常')
    
    # 输出异常时间点
    abnormal_points = df[df['异常变化'] == '异常']
    
    if not abnormal_points.empty:
        print("检测到异常变化的时间点：")
        print(abnormal_points[['datetime', 'motor_torque', 'torque_diff']])
    else:
        print("未检测到异常变化。")
    
    # 可视化：绘制扭矩随时间的变化，并标记异常点
    plt.figure(figsize=(10, 6))
    plt.plot(df['datetime'], df['motor_torque'], label='Motor Torque', color='blue')
    
    # 标记异常变化的时间点
    plt.scatter(abnormal_points['datetime'], abnormal_points['motor_torque'], color='red', label='Abnormal Points', zorder=5)
    
    plt.xlabel('Time')
    plt.ylabel('Motor Torque')
    plt.title('Motor Torque vs Time (with Abnormal Points)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)  # 旋转 x 轴的时间标签，便于显示
    plt.tight_layout()  # 自动调整布局，避免标签被遮挡
    plt.show()

def 温度与运行时间分析(df):
    # 确保 'datetime' 列是 datetime 类型
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # 过滤出电动机在运行中的数据 (motor_run == True)
    running_df = df[df['motor_run'] == True]
    
    # 检查是否有运行数据
    if running_df.empty:
        print("没有运行时段的数据，无法进行分析")
        return
    
    # 创建时间差列（以秒为单位）
    running_df.loc[:, 'time_diff'] = (running_df['datetime'] - running_df['datetime'].iloc[0]).dt.total_seconds()

    
    # 计算温度变化趋势：可以通过线性拟合来了解温度的变化趋势
    # 我们使用 np.polyfit 来拟合一个一阶多项式（直线）
    fit_params = np.polyfit(running_df['time_diff'], running_df['motor_temperature'], 1)
    slope = fit_params[0]  # 斜率（温度变化率）
    
    # 判断温度是否逐渐升高
    if slope > 0:
        print("温度逐渐升高。")
    else:
        print("温度没有显著升高。")
    
    # 绘制温度随运行时间的变化图
    plt.figure(figsize=(10, 6))
    plt.plot(running_df['time_diff'], running_df['motor_temperature'], label='Motor Temperature')
    plt.plot(running_df['time_diff'], np.polyval(fit_params, running_df['time_diff']), label='Fitted Trend Line', linestyle='--', color='red')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Motor Temperature (°C)')
    plt.title('Motor Temperature vs Running Time')
    plt.legend()
    plt.grid(True)
    plt.show()

def 温度稳定性分析(df: pd.DataFrame, threshold=5.0):
    """
    分析 motor_temperature 是否存在异常的温度升高。
    
    参数:
    df: 包含 motor_temperature 和 datetime 列的 DataFrame
    threshold: 温度变化的阈值，当温度变化超过该阈值时标记为异常
    """
    # 计算 motor_temperature 的变化率（温度差）
    df['temperature_change'] = df['motor_temperature'].diff().abs()
    
    # 标记温度升高较大的点（温度变化大于设定的阈值）
    df['large_temperature_increase'] = df['temperature_change'] > threshold
    
    # 输出温度变化率的异常点
    large_increases = df[df['large_temperature_increase']][['datetime', 'motor_temperature', 'temperature_change']]
    print("温度升高异常的时间点：")
    print(large_increases)
    
    # 绘制 motor_temperature 和 temperature_change 的时间序列图
    plt.figure(figsize=(14, 6))
    
    # 绘制 motor_temperature
    plt.subplot(2, 1, 1)
    plt.plot(range(len(df)), df['motor_temperature'], label='Motor Temperature', color='green')  # 使用行索引作为横坐标
    plt.title('Motor Temperature 时间序列', fontsize=14)
    plt.xlabel('序列号', fontsize=12)  # 改为“序列号”
    plt.ylabel('Motor Temperature', fontsize=12)
    
    # 绘制 temperature_change（温度变化率）
    plt.subplot(2, 1, 2)
    plt.plot(range(len(df)), df['temperature_change'], label='Temperature Change', color='orange')  # 使用行索引作为横坐标
    plt.title('Temperature Change 时间序列', fontsize=14)
    plt.xlabel('序列号', fontsize=12)  # 改为“序列号”
    plt.ylabel('Temperature Change', fontsize=12)
    
    # 显示图表
    plt.tight_layout()
    plt.show()


def 速度稳定性分析(df: pd.DataFrame, threshold=1.0):
    # 记录速度为0的时间点
    zero_speed_points = df[df['motor_speed'] == 0][['datetime', 'motor_speed']]
    print("速度为0的时间点：")
    print(zero_speed_points)

    # 剔除速度为0的时间点
    df = df[df['motor_speed'] != 0]

    # 计算 motor_speed 的标准差
    motor_speed_std = df['motor_speed'].std()
    
    # 计算 motor_speed 的变化率（绝对变化量）
    df['speed_change'] = df['motor_speed'].diff().abs()
    
    # 标记速度波动较大的点（变化率大于设定的阈值）
    df['large_change'] = df['speed_change'] > threshold
    
    # 输出 motor_speed 的标准差
    print(f"motor_speed 的标准差：{motor_speed_std:.2f}")
    
    # 输出速度波动较大的时间点
    large_changes = df[df['large_change']][['datetime', 'motor_speed', 'speed_change']]
    print("速度波动较大的时间点：")
    print(large_changes)
    
    # 绘制 motor_speed 和 speed_change 的时间序列图
    plt.figure(figsize=(14, 6))
    
    # 绘制 motor_speed
    plt.subplot(2, 1, 1)
    plt.plot(range(len(df)), df['motor_speed'], label='Motor Speed', color='blue')  # 使用行索引作为横坐标
    plt.title('Motor Speed 时间序列', fontsize=14)
    plt.xlabel('序列号', fontsize=12)  # 改为“序列号”
    plt.ylabel('Motor Speed', fontsize=12)
    
    # 绘制 speed_change（速度变化率）
    plt.subplot(2, 1, 2)
    plt.plot(range(len(df)), df['speed_change'], label='Speed Change', color='red')  # 使用行索引作为横坐标
    plt.title('Speed Change 时间序列', fontsize=14)
    plt.xlabel('序列号', fontsize=12)  # 改为“序列号”
    plt.ylabel('Speed Change', fontsize=12)
    
    # 显示图表
    plt.tight_layout()
    plt.show()


def 异常状况分析(df: pd.DataFrame):
    # 筛选出 motor_fault == True 的时间点
    motor_fault_true = df[df['motor_fault'] == True]
    
    # 只保留 datetime 列
    fault_timepoints = motor_fault_true[['datetime']]
    
    # 输出 motor_fault == True 的时间点
    print("motor_fault == True 的时间点：")
    print(fault_timepoints)

def 启停频率分析(df: pd.DataFrame):
    # 计算 motor_run 状态变化的标志
    df['motor_run_change'] = df['motor_run'].ne(df['motor_run'].shift())
    
    # 筛选出 motor_run 状态变化的时间点
    changes = df[df['motor_run_change']]
    
    # 提取每个变化发生的小时部分（精确到小时）
    changes['hour'] = changes['datetime'].dt.floor('H')  # floor('H') 将时间戳截断为每小时的开始
    
    # 统计每小时的状态变化次数
    hourly_changes = changes.groupby('hour').size()
    
    # 输出每小时变化的次数
    print("每小时的状态变化次数：")
    print(hourly_changes)
    
    # 绘制每小时状态变化次数的柱状图
    plt.figure(figsize=(10, 6))  # 图表大小
    hourly_changes.plot(kind='bar', color='skyblue', edgecolor='black')
    
    # 设置标题和标签
    plt.title('每小时 motor_run 状态变化次数', fontsize=14)
    plt.xlabel('时间', fontsize=12)
    plt.ylabel('状态变化次数', fontsize=12)
    
    # 设置 x 轴刻度标签旋转 45 度，避免重叠
    plt.xticks(rotation=45)
    
        # 设置日期格式化器和日期刻度定位器
    date_locator = mdates.AutoDateLocator()  # 自动选择刻度间隔
    plt.gca().xaxis.set_major_locator(date_locator)
    
    # 设置最大显示的刻度数
    plt.gcf().autofmt_xdate()  # 旋转日期标签以避免重叠
    plt.tight_layout()  # 自动调整子图间的间距和标签位置
    
    # 显示图表
    plt.tight_layout()
    plt.show()


def 连续运行时间(df):
    """
    在 df 中添加一列，表示连续的 motor_run=True 的次数
    """
    # 初始化一个空的 'continuous_motor_run' 列，默认值为 0
    df['continuous_motor_run'] = 0
    
    # 计数器，用来追踪连续的 motor_run == True 的次数
    count = 0
    
    # 遍历 DataFrame 中的 motor_run 列
    for i in range(len(df)):
        if df.loc[i, 'motor_run'] == True:
            count += 1  # 如果 motor_run 为 True，增加计数
        else:
            count = 0  # 如果 motor_run 为 False，重置计数
        
        # 将连续次数写入对应的列
        df.loc[i, 'continuous_motor_run'] = count
    
    return df

def 分析温度与连续运行关系(df):
    """
    分析 motor_temperature 与 continuous_motor_run 之间的关系，
    并计算它们的相关性。
    """
    # 绘制散点图
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='continuous_motor_run', y='motor_temperature', data=df, color='b')
    plt.title('motor_temperature vs continuous_motor_run')
    plt.xlabel('Continuous Motor Run')
    plt.ylabel('Motor Temperature')
    plt.grid(True)
    plt.show()

    # 计算皮尔逊相关系数
    correlation = df[['continuous_motor_run', 'motor_temperature']].corr().iloc[0, 1]
    print(f"Continuous Motor Run 与 Motor Temperature 的皮尔逊相关系数: {correlation}")

    # 判断相关性强弱
    if correlation > 0.7:
        print("两者之间存在较强的正相关关系。")
    elif correlation < -0.7:
        print("两者之间存在较强的负相关关系。")
    else:
        print("两者之间相关性较弱。")

    return correlation

def analyze_motor_data(file_path):
    """
    读取CSV文件，进行基础描述性统计分析，绘制电动辊速度、温度和扭矩随时间变化趋势图，
    包括移动平均线，且避免连接超过1小时的时间点。
    
    参数:
    file_path (str): CSV文件的路径。
    """
    # 读取 CSV 文件并转换 'datetime' 列为 datetime 类型
    df = pd.read_csv(file_path)
    df['datetime'] = pd.to_datetime(df['datetime'])

    # 按时间排序
    df = df.sort_values(by='datetime')

    # 描述性统计
    print("描述性统计:\n", df[['motor_speed', 'motor_temperature', 'motor_torque']].describe())

    # 绘制电动辊的速度、温度和扭矩随时间的变化趋势
    plt.figure(figsize=(15, 8))

    # Motor Speed
    plt.subplot(3, 1, 1)
    plot_with_gap(df['datetime'], df['motor_speed'], 'Motor Speed Over Time', 'Speed (rpm)', color='b')

    # Motor Temperature
    plt.subplot(3, 1, 2)
    plot_with_gap(df['datetime'], df['motor_temperature'], 'Motor Temperature Over Time', 'Temperature (°C)', color='r')

    # Motor Torque
    plt.subplot(3, 1, 3)
    plot_with_gap(df['datetime'], df['motor_torque'], 'Motor Torque Over Time', 'Torque (Nm)', color='g')

    plt.tight_layout()
    plt.show()

    # 设置一个窗口大小，计算移动平均
    window_size = 10

    df['speed_moving_avg'] = df['motor_speed'].rolling(window=window_size).mean()
    df['temperature_moving_avg'] = df['motor_temperature'].rolling(window=window_size).mean()
    df['torque_moving_avg'] = df['motor_torque'].rolling(window=window_size).mean()

    # 绘制带移动平均线的速度、温度和扭矩
    plt.figure(figsize=(15, 8))

    # Motor Speed with Moving Average
    plt.subplot(3, 1, 1)
    plot_with_gap(df['datetime'], df['motor_speed'], 'Motor Speed with Moving Average', 'Speed (rpm)', color='b', avg_data=df['speed_moving_avg'])

    # Motor Temperature with Moving Average
    plt.subplot(3, 1, 2)
    plot_with_gap(df['datetime'], df['motor_temperature'], 'Motor Temperature with Moving Average', 'Temperature (°C)', color='r', avg_data=df['temperature_moving_avg'])

    # Motor Torque with Moving Average
    plt.subplot(3, 1, 3)
    plot_with_gap(df['datetime'], df['motor_torque'], 'Motor Torque with Moving Average', 'Torque (Nm)', color='g', avg_data=df['torque_moving_avg'])

    plt.tight_layout()
    plt.show()

def plot_with_gap(times, values, title, ylabel, color, avg_data=None):
    """
    辅助函数，用于绘制带间隔的线图（不连接超过1小时的数据点）。
    
    参数：
    times (pd.Series): 时间数据
    values (pd.Series): 数据值
    title (str): 图表标题
    ylabel (str): Y轴标签
    color (str): 线条颜色
    avg_data (pd.Series, optional): 移动平均数据
    """
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(True)

    # 计算时间差并跳过超过1小时的时间段
    valid_times = [times.iloc[0]]
    valid_values = [values.iloc[0]]
    if avg_data is not None:
        valid_avg_data = [avg_data.iloc[0]]
    else:
        valid_avg_data = []

    for i in range(1, len(times)):
        time_diff = (times.iloc[i] - times.iloc[i-1]).total_seconds() / 3600  # 转换为小时
        if time_diff <= 1:
            valid_times.append(times.iloc[i])
            valid_values.append(values.iloc[i])
            if avg_data is not None:
                valid_avg_data.append(avg_data.iloc[i])
        else:
            # 跳过超过1小时的时间段
            valid_times.append(None)
            valid_values.append(None)
            if avg_data is not None:
                valid_avg_data.append(None)

    # 绘制实际数据点
    plt.plot(valid_times, valid_values, label='Data', color=color, alpha=0.7)
    
    # 如果有移动平均数据，绘制移动平均线
    if avg_data is not None:
        plt.plot(valid_times, valid_avg_data, label='Moving Avg', color='orange')
    
    plt.legend()

def 温度差(df):
    """
    在 df 中添加一列，表示每一行与上一行的 motor_temperature 差值。
    如果是第一行，则差值设置为 NaN 或者 0。
    """
    # 使用 shift 方法计算与上一行的差值，shift(1) 返回的是上一行的值
    df['temperature_difference'] = df['motor_temperature'] - df['motor_temperature'].shift(1)
    
    # 对于第一行，temperature_difference 的差值设置为 NaN（表示无前一行）
    df.loc[0, 'temperature_difference'] = None  # 使用 loc 进行赋值
    
    return df

def 统计温度差频率和对应的温度(df):
    """
    统计温度差 (temperature_difference) 各个数值的频率，
    并显示对应的 motor_temperature 值及其频率（百分比），保留两位有效数字。
    """
    # 首先计算温度差列
    df = 温度差(df)
    
    # 获取数据总行数
    total_rows = len(df)
    
    # 统计 temperature_difference 列中各个数值出现的频率（百分比），并保留两位小数
    diff_counts = df['temperature_difference'].value_counts(dropna=False, normalize=True).sort_index() * 100
    diff_counts = diff_counts.round(2)  # 保留两位小数
    
    # 创建一个字典来保存每个温度差值对应的 motor_temperature 及其频率（百分比）
    temp_dict = {}
    
    for diff in diff_counts.index:
        # 获取所有对应的 motor_temperature 值
        temp_values = df[df['temperature_difference'] == diff]['motor_temperature']
        
        # 统计 motor_temperature 值的频率，并转换为百分比
        temp_frequency = temp_values.value_counts(normalize=True).to_dict()
        temp_frequency_percent = {k: round(v * 100, 2) for k, v in temp_frequency.items()}  # 转换为百分比并保留两位小数
        
        # 保存每个温度差对应的 motor_temperature 值及其百分比频率
        temp_dict[diff] = temp_frequency_percent
    
    # 输出结果
    print("温度差的频率（百分比）：")
    print(diff_counts)

    print("\n每个温度差对应的 motor_temperature 和频率（百分比）：")
    for diff, temp_freq in temp_dict.items():
        print(f"温度差 {diff} 对应的 motor_temperature 和频率: {temp_freq}")
    
    return diff_counts, temp_dict

def check_motor_fault_and_run(df):
    """
    检查是否存在 motor_fault=True 且 motor_run=True 的记录。

    :param df: 包含 `motor_fault` 和 `motor_run` 列的数据框
    :return: 若存在匹配的记录，返回这些记录的 DataFrame；否则返回 None 或空 DataFrame。
    """
    # 过滤出符合条件的记录
    fault_and_run_df = df[(df['motor_fault'] == True) & (df['motor_run'] == True)]
    
    # 检查结果
    if not fault_and_run_df.empty:
        return fault_and_run_df
    else:
        print("没有找到 motor_fault=True 且 motor_run=True 的记录。")
        return pd.DataFrame()
    
def 每小时统计数据(df: pd.DataFrame, speed_lower_bound=None, speed_upper_bound=None, 
                      temperature_lower_bound=None, temperature_upper_bound=None,
                      torque_lower_bound=None, torque_upper_bound=None):
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = 连续运行时间(df)
    # 创建一个用于分组的小时列
    df['hour'] = df['datetime'].dt.floor('h')

    # 初始化结果字典
    results = []

    # 按小时分组并进行统计
    for hour, group in df.groupby('hour'):
        total_count = group.shape[0]
        motor_run_true_count = group[group['motor_run']].shape[0]
        motor_run_false_count = group[~group['motor_run']].shape[0]
        
        # 计算 motor_run=True 和 False 的占比
        motor_run_true_pct = motor_run_true_count / total_count * 100
        motor_run_false_pct = motor_run_false_count / total_count * 100

        # 计算 motor_fault=True 和 False 的占比
        motor_fault_true_pct = group[group['motor_fault']].shape[0] / total_count * 100
        motor_fault_false_pct = group[~group['motor_fault']].shape[0] / total_count * 100

        # 计算 motor_run 变化的次数
        motor_run_changes = group['motor_run'].diff().abs().sum()

        # 统计 motor_speed, motor_temperature, motor_torque 在不同 motor_run 状态下的均值和标准差
        stats_true = group[group['motor_run']].agg({
            'motor_speed': ['mean', 'std'],
            'motor_temperature': ['mean', 'std'],
            'motor_torque': ['mean', 'std'],
        })

        stats_false = group[~group['motor_run']].agg({
            'motor_speed': ['mean', 'std'],
            'motor_temperature': ['mean', 'std'],
            'motor_torque': ['mean', 'std'],
        })

        # 计算 continuous_motor_run 的最大值
        continuous_motor_run_max = group['continuous_motor_run'].max()

        # 统计 motor_speed 在指定范围内和范围外的个数
        if speed_lower_bound is not None and speed_upper_bound is not None:
            motor_speed_in_range = group[(group['motor_speed'] >= speed_lower_bound) & (group['motor_speed'] <= speed_upper_bound)].shape[0]
            motor_speed_out_of_range = total_count - motor_speed_in_range
        else:
            motor_speed_in_range = None
            motor_speed_out_of_range = None

        # 统计 motor_temperature 在指定范围内和范围外的个数
        if temperature_lower_bound is not None and temperature_upper_bound is not None:
            motor_temperature_in_range = group[(group['motor_temperature'] >= temperature_lower_bound) & (group['motor_temperature'] <= temperature_upper_bound)].shape[0]
            motor_temperature_out_of_range = total_count - motor_temperature_in_range
        else:
            motor_temperature_in_range = None
            motor_temperature_out_of_range = None

        # 统计 motor_torque 在指定范围内和范围外的个数
        if torque_lower_bound is not None and torque_upper_bound is not None:
            motor_torque_in_range = group[(group['motor_torque'] >= torque_lower_bound) & (group['motor_torque'] <= torque_upper_bound)].shape[0]
            motor_torque_out_of_range = total_count - motor_torque_in_range
        else:
            motor_torque_in_range = None
            motor_torque_out_of_range = None

        # 收集这一小时的统计数据
        results.append({
            'hour': hour,
            'motor_run_true_pct': motor_run_true_pct,
            'motor_run_false_pct': motor_run_false_pct,
            'motor_fault_true_pct': motor_fault_true_pct,
            'motor_fault_false_pct': motor_fault_false_pct,
            'motor_run_changes': motor_run_changes,
            'motor_speed_mean_true': stats_true['motor_speed']['mean'] if not stats_true.empty else None,
            'motor_speed_std_true': stats_true['motor_speed']['std'] if not stats_true.empty else None,
            'motor_temperature_mean_true': stats_true['motor_temperature']['mean'] if not stats_true.empty else None,
            'motor_temperature_std_true': stats_true['motor_temperature']['std'] if not stats_true.empty else None,
            'motor_torque_mean_true': stats_true['motor_torque']['mean'] if not stats_true.empty else None,
            'motor_torque_std_true': stats_true['motor_torque']['std'] if not stats_true.empty else None,
            'motor_speed_mean_false': stats_false['motor_speed']['mean'] if not stats_false.empty else None,
            'motor_speed_std_false': stats_false['motor_speed']['std'] if not stats_false.empty else None,
            'motor_temperature_mean_false': stats_false['motor_temperature']['mean'] if not stats_false.empty else None,
            'motor_temperature_std_false': stats_false['motor_temperature']['std'] if not stats_false.empty else None,
            'motor_torque_mean_false': stats_false['motor_torque']['mean'] if not stats_false.empty else None,
            'motor_torque_std_false': stats_false['motor_torque']['std'] if not stats_false.empty else None,
            'continuous_motor_run_max': continuous_motor_run_max,
            'motor_speed_in_range': motor_speed_in_range,
            'motor_speed_out_of_range': motor_speed_out_of_range,
            'motor_temperature_in_range': motor_temperature_in_range,
            'motor_temperature_out_of_range': motor_temperature_out_of_range,
            'motor_torque_in_range': motor_torque_in_range,
            'motor_torque_out_of_range': motor_torque_out_of_range
        })
    
    # 创建 DataFrame 并舍入两位小数
    hour_df = pd.DataFrame(results)
    hour_df = hour_df.round(2)
    
    return hour_df

def calculate_confidence_interval(data, confidence=0.95):
    """
    计算数据的t分布置信区间。

    :param data: 一个包含数据的数组或Series
    :param confidence: 置信水平
    :return: (均值, 下限, 上限)
    """
    n = len(data)
    mean = np.mean(data)
    sem = stats.sem(data)  # 样本标准误
    print(f"n={n}\tsem={sem}\tt={stats.t.ppf((1 + confidence) / 2., n-1)}")
    if n > 1:  # 检查数据点数是否多于1个
        h = sem * stats.t.ppf((1 + confidence) / 2., n-1)  # t值 * 标准误
    else:
        h = np.nan  # 无法计算置信区间
    return mean, mean - h, mean + h

def calculate_percentile_range(data, lower_percentile=1.5, upper_percentile=98.5):
    """
    计算数据在指定分位数范围内的区间。

    :param data: 一个包含数据的数组或Series
    :param lower_percentile: 分位数的下限（默认5）
    :param upper_percentile: 分位数的上限（默认95）
    :return: (5th percentile, 95th percentile)
    """
    lower_bound = np.percentile(data, lower_percentile)
    upper_bound = np.percentile(data, upper_percentile)
    return lower_bound, upper_bound

def 绘制散点图_一天(df: pd.DataFrame, key: str, lower_bound: float, upper_bound: float):
    plt.figure(figsize=(19, 14.4))
    plt.scatter(df['datetime'], df[key], alpha=0.5, label=key.capitalize())
    plt.axhspan(lower_bound, upper_bound, color='orange', alpha=0.3, label='97%数据范围')
    plt.title(f'{key.capitalize()}在motor_run=true下散点图及97%数据范围', fontsize=20)  # 放大标题字体
    plt.xlabel('Datetime', fontsize=16)  # 放大x轴字体
    plt.ylabel(key.capitalize(), fontsize=16)  # 放大y轴字体
    
    # 关闭右边和上边的边框线
    ax = plt.gca()  # 获取当前的坐标轴
    ax.spines['right'].set_visible(False)  # 关闭右边的边框
    ax.spines['top'].set_visible(False)  # 关闭上边的边框
    
    # 设置纵坐标从0开始
    plt.ylim(bottom=0)
    
    # 在图上添加 lower_bound 和 upper_bound 的标签
    plt.text(df['datetime'].iloc[0], lower_bound, f'下限: {lower_bound}', fontsize=14, color='black', verticalalignment='bottom')
    plt.text(df['datetime'].iloc[0], upper_bound, f'上限: {upper_bound}', fontsize=14, color='black', verticalalignment='top')

    # 格式化横坐标显示为 hh:mm 格式
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  # 设置日期时间格式为小时:分钟
    
    # 调整 x 轴的标签显示，避免重叠
    plt.xticks(rotation=45, ha='right', fontsize=12)  # 旋转日期标签，避免重叠

    # 放大图例字体
    plt.legend(loc='upper right', fontsize=14)
    
    plt.show()

def 绘制散点图(df: pd.DataFrame, key: str, lower_bound: float, upper_bound: float, save_path: str):
    plt.figure(figsize=(19, 14.4))
    plt.scatter(df['datetime'], df[key], alpha=0.5, label=key.capitalize())
    plt.axhspan(lower_bound, upper_bound, color='orange', alpha=0.3, label='数据范围')
    plt.title(f'{key.capitalize()}在motor_run=true下散点图及数据范围', fontsize=20)  # 放大标题字体
    plt.xlabel('Datetime', fontsize=16)  # 放大x轴字体
    plt.ylabel(key.capitalize(), fontsize=16)  # 放大y轴字体
    
    # 关闭右边和上边的边框线
    ax = plt.gca()  # 获取当前的坐标轴
    ax.spines['right'].set_visible(False)  # 关闭右边的边框
    ax.spines['top'].set_visible(False)  # 关闭上边的边框
    
    # 设置纵坐标从0开始
    plt.ylim(bottom=0)
    
    # 在图上添加 lower_bound 和 upper_bound 的标签
    plt.text(df['datetime'].iloc[0], lower_bound, f'下限: {lower_bound}', fontsize=14, color='black', verticalalignment='bottom')
    plt.text(df['datetime'].iloc[0], upper_bound, f'上限: {upper_bound}', fontsize=14, color='black', verticalalignment='top')

    # 放大图例字体
    plt.legend(loc='upper right', fontsize=14)

    save_file_path = os.path.join(save_path, f'{key.capitalize()}.png')
    plt.savefig(save_file_path, dpi=300, bbox_inches='tight')  # 保存图像，设置高分辨率，去掉多余的空白区域
    
    # 关闭图像，避免内存占用
    plt.close()

def 获取目标文件夹下子文件夹csv_无条件获取(root_folder: str):
    file_list = []
    
    # 遍历指定文件夹下的每个子文件夹
    for subfolder in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, subfolder)
        
        # 确保是文件夹
        if os.path.isdir(subfolder_path):
            # 获取子文件夹中所有的 .csv 文件
            csv_files = [f for f in os.listdir(subfolder_path) if os.path.isfile(os.path.join(subfolder_path, f)) and f.endswith('.csv')]
            
            # 判断文件数量
            if len(csv_files) == 3:
                # 如果子文件夹中有3个文件，选择merge_df.csv（如果存在）
                merge_file_path = os.path.join(subfolder_path, 'merged.csv')
                if os.path.exists(merge_file_path):
                    # 读取 merge_df.csv 文件
                    file_list.append(merge_file_path)
                    print(f"读取 {merge_file_path}")
                    # 在这里可以对 df 进行处理
                else:
                    print(f"在 {subfolder_path} 找不到 merge_df.csv 文件")
            elif len(csv_files) == 1:
                # 如果只有1个文件，且该文件是csv文件
                single_file_path = os.path.join(subfolder_path, csv_files[0])
                if single_file_path.endswith('.csv'):  # 检查该文件是否是 .csv 文件
                    file_list.append(single_file_path)
                    print(f"读取 {single_file_path}")
                else:
                    print(f"{subfolder_path} 中的唯一文件不是csv文件，跳过该文件夹")
            else:
                print(f"{subfolder_path} 中的文件数量不是1或3，跳过该文件夹")
    
    return file_list


def 获取目标文件夹下子文件夹csv(root_folder: str):
    file_list = []
    
    # 遍历指定文件夹下的每个子文件夹
    for subfolder in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, subfolder)
        
        # 确保是文件夹
        if os.path.isdir(subfolder_path):
            # 获取子文件夹中的所有文件（包括 .csv 和其他文件）
            all_files = [f for f in os.listdir(subfolder_path) if os.path.isfile(os.path.join(subfolder_path, f))]
            
            # 判断文件数量
            if len(all_files) == 3:
                # 如果子文件夹中有3个文件，选择merge_df.csv（如果存在）
                merge_file_path = os.path.join(subfolder_path, 'merged.csv')
                if os.path.exists(merge_file_path):
                    # 读取 merge_df.csv 文件
                    file_list.append(merge_file_path)
                    print(f"读取 {merge_file_path}")
                    # 在这里可以对 df 进行处理
                else:
                    print(f"在 {subfolder_path} 找不到 merge_df.csv 文件")
            elif len(all_files) == 1:
                # 如果只有1个文件，且该文件是csv文件
                single_file_path = os.path.join(subfolder_path, all_files[0])
                if single_file_path.endswith('.csv'):  # 检查该文件是否是 .csv 文件
                    file_list.append(single_file_path)
                    print(f"读取 {single_file_path}")
                else:
                    print(f"{subfolder_path} 中的唯一文件不是csv文件，跳过该文件夹")
            else:
                print(f"{subfolder_path} 中的文件数量不是1或3，跳过该文件夹")
    
    return file_list
def 批量处理_速度(df:pd.DataFrame, save_path):
    df_motor_running = df[df['motor_run'] == True]
    data = df_motor_running['motor_speed'].dropna()  # 去除缺失值
    lower_bound, upper_bound = calculate_percentile_range(data, lower_percentile=0.5, upper_percentile=99.5)
    绘制散点图(df, 'motor_speed', lower_bound, upper_bound, save_path)
    return lower_bound, upper_bound

def 批量处理_温度(df:pd.DataFrame, save_path):
    # 从df中过滤出motor_run为True的数据
    df_motor_running = df[df['motor_run'] == True]
    data = df_motor_running['motor_temperature'].dropna()  # 去除缺失值
    lower_bound, upper_bound = calculate_percentile_range(data, lower_percentile=1.5, upper_percentile=98.5)
    绘制散点图(df, 'motor_temperature', lower_bound, upper_bound, save_path)
    return lower_bound, upper_bound

def 批量处理_扭矩(df:pd.DataFrame, save_path):
    # 从df中过滤出motor_run为True的数据
    df_motor_running = df[df['motor_run'] == True]
    data = df_motor_running['motor_torque'].dropna()  # 去除缺失值
    lower_bound, upper_bound = calculate_percentile_range(data, lower_percentile=1, upper_percentile=99)
    绘制散点图(df, 'motor_torque', lower_bound, upper_bound, save_path)
    return lower_bound, upper_bound

def 读取数据csv(file_path: str):
    df = pd.read_csv(file_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values(by='datetime')
    return df

def 计算总量运行的统计特征(df: pd.DataFrame):
    # 筛选出 motor_run 为 True 的数据
    df_motor_running = df[df['motor_run'] == True]
    
    # 计算 motor_speed, motor_temperature 和 motor_torque 的统计指标
    statistics = {}
    
    # 对 motor_speed 计算统计值
    statistics['motor_speed'] = {
        'median': df_motor_running['motor_speed'].median(),
        'max': df_motor_running['motor_speed'].max(),
        'mean': df_motor_running['motor_speed'].mean(),
        'std': df_motor_running['motor_speed'].std()
    }
    
    # 对 motor_temperature 计算统计值
    statistics['motor_temperature'] = {
        'median': df_motor_running['motor_temperature'].median(),
        'max': df_motor_running['motor_temperature'].max(),
        'mean': df_motor_running['motor_temperature'].mean(),
        'std': df_motor_running['motor_temperature'].std()
    }
    
    # 对 motor_torque 计算统计值
    statistics['motor_torque'] = {
        'median': df_motor_running['motor_torque'].median(),
        'max': df_motor_running['motor_torque'].max(),
        'mean': df_motor_running['motor_torque'].mean(),
        'std': df_motor_running['motor_torque'].std()
    }
    statistics = convert_int64(statistics)    
    return statistics


def 批量处理(root_folder: str):
    file_list = 获取目标文件夹下子文件夹csv(root_folder)
    for file in file_list:
        # 获取文件夹路径
        folder_path = os.path.dirname(file)
        print(folder_path)
        print(file)
        df = 读取数据csv(file)
        speed_lower_bound, speed_upper_bound = 批量处理_速度(df, folder_path)
        temperature_lower_bound, temperatured_upper_bound = 批量处理_温度(df, folder_path)
        torque_lower_bound, torque_upper_bound = 批量处理_扭矩(df, folder_path)
        hour_df = 每小时统计数据(df, speed_lower_bound, speed_upper_bound, temperature_lower_bound, temperatured_upper_bound, torque_lower_bound, torque_upper_bound)
        hour_df.to_excel(os.path.join(folder_path, "hour_df.xlsx"))
         # 将界定范围保存为JSON文件
        boundary_data = {
            "speed_lower_bound": speed_lower_bound,
            "speed_upper_bound": speed_upper_bound,
            "temperature_lower_bound": temperature_lower_bound,
            "temperature_upper_bound": temperatured_upper_bound,
            "torque_lower_bound": torque_lower_bound,
            "torque_upper_bound": torque_upper_bound
        }

        # 创建JSON文件并保存
        json_file_path = os.path.join(folder_path, "界定范围.json")
        with open(json_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(boundary_data, json_file, ensure_ascii=False, indent=4)

def convert_int64(data):
    if isinstance(data, dict):
        return {k: convert_int64(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_int64(v) for v in data]
    elif isinstance(data, np.int64):  # 检查 int64 类型
        return int(data)
    return data


def 批量处理2(root_folder: str):
    file_list = 获取目标文件夹下子文件夹csv(root_folder)
    for file in file_list:
        # 获取文件夹路径
        folder_path = os.path.dirname(file)
        print(folder_path)
        print(file)
        df = 读取数据csv(file)
        speed_lower_bound, speed_upper_bound = 批量处理_速度(df, folder_path)
        temperature_lower_bound, temperatured_upper_bound = 批量处理_温度(df, folder_path)
        torque_lower_bound, torque_upper_bound = 批量处理_扭矩(df, folder_path)
        hour_df = 每小时统计数据(df, speed_lower_bound, speed_upper_bound, temperature_lower_bound, temperatured_upper_bound, torque_lower_bound, torque_upper_bound)
        hour_df.to_excel(os.path.join(folder_path, "hour_df.xlsx"))
        # 获取统计特征
        statistics = 计算总量运行的统计特征(df)
        # 展平统计数据
        flattened_statistics = {
            "speed_lower_bound": speed_lower_bound,
            "speed_upper_bound": speed_upper_bound,
            "temperature_lower_bound": temperature_lower_bound,
            "temperature_upper_bound": temperatured_upper_bound,
            "torque_lower_bound": torque_lower_bound,
            "torque_upper_bound": torque_upper_bound,
            'motor_speed_median': statistics['motor_speed']['median'],
            'motor_speed_max': statistics['motor_speed']['max'],
            'motor_speed_mean': statistics['motor_speed']['mean'],
            'motor_speed_std': statistics['motor_speed']['std'],
            'motor_temperature_median': statistics['motor_temperature']['median'],
            'motor_temperature_max': statistics['motor_temperature']['max'],
            'motor_temperature_mean': statistics['motor_temperature']['mean'],
            'motor_temperature_std': statistics['motor_temperature']['std'],
            'motor_torque_median': statistics['motor_torque']['median'],
            'motor_torque_max': statistics['motor_torque']['max'],
            'motor_torque_mean': statistics['motor_torque']['mean'],
            'motor_torque_std': statistics['motor_torque']['std']
        }

        # 将统计数据保存到JSON文件中
        json_file_path = os.path.join(folder_path, "界定范围.json")
        with open(json_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(flattened_statistics, json_file, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    批量处理2(r'C:\Users\Jayttle\Desktop\电动辊原始数据')



    # 读取和处理数据
    # item_key = '4702-1'
    # read_file_path = rf"C:\Users\Jayttle\Desktop\电动辊原始数据\{item_key}\merged.csv"
    # df = pd.read_csv(read_file_path)
    # df['datetime'] = pd.to_datetime(df['datetime'])
    # df = df.sort_values(by='datetime')

    # # 从df中过滤出motor_run为True的数据
    # df_motor_running = df[df['motor_run'] == True]

    # # 指定日期（例如：2024-11-10）
    # specified_date = pd.to_datetime('2024-09-13').date()  # 将指定日期转换为 datetime.date 类型

    # # 筛选出指定日期的数据
    # df_motor_running = df_motor_running[df_motor_running['datetime'].dt.date == specified_date]

    # # 提取需要计算的列
    # columns_to_analyze = ['motor_speed', 'motor_temperature', 'motor_torque']

    # # 计算各列的1.5th到98.5th percentile范围
    # for column in columns_to_analyze:
    #     data = df_motor_running[column].dropna()  # 去除缺失值
    #     lower_bound, upper_bound = calculate_percentile_range(data, lower_percentile=1.5, upper_percentile=98.5)
    #     print(f"{column}: 1.5 Percentile={lower_bound:.2f}, 98.5 Percentile={upper_bound:.2f}")

    # # 对motor_speed绘制散点图并标记1.5th到98.5th百分位范围
    # lower_bound, upper_bound = calculate_percentile_range(df_motor_running['motor_speed'].dropna(), lower_percentile=1.5, upper_percentile=98.5)
    # 绘制散点图_一天(df_motor_running, 'motor_speed', lower_bound, upper_bound)
    # lower_bound, upper_bound = calculate_percentile_range(df_motor_running['motor_temperature'].dropna(), lower_percentile=1.5, upper_percentile=98.5)
    # 绘制散点图_一天(df_motor_running, 'motor_temperature', lower_bound, upper_bound)
    # lower_bound, upper_bound = calculate_percentile_range(df_motor_running['motor_torque'].dropna(), lower_percentile=1.5, upper_percentile=98.5)
    # 绘制散点图_一天(df_motor_running, 'motor_torque', lower_bound, upper_bound)


    # 
    # df = 温度差(df)

    # 扭矩变化规律分析(df)
    
    # 调用分析函数
    # 分析温度与连续运行关系(df)
    # 统计特征分析(df)
    # analyze_motor_data(read_file_path)w mei