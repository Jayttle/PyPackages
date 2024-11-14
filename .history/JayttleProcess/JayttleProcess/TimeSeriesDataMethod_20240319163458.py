import math
import random
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm
import pandas as pd
plt.rcParams['font.sans-serif'] = ['SimSun']  # 指定宋体为默认字体
plt.rcParams['font.sans-serif'] = 'SimSun'  # 使用指定的中文字体

class TimeSeriesData:
    def __init__(self, value, datetime):
        self.value = value
        self.datetime = datetime

    def __str__(self):
        return f"Value: {self.value}, Datetime: {self.datetime}"

def create_time_series_data(values, datetimes):
    """给values和datetimes return一个time_series_data"""
    time_series_data = []
    for value, datetime in zip(values, datetimes):
        data = TimeSeriesData(value, datetime)
        time_series_data.append(data)
    return time_series_data


def plot(TimeSeriesData):
    """绘制TimeSeriesData对象的时间序列图"""
    values = [data.value for data in TimeSeriesData]
    datetimes = [data.datetime for data in TimeSeriesData]

    # 修改标签和标题的文本为中文
    plt.xlabel('日期')
    plt.ylabel('数值')
    plt.title('时间序列数据')
    # 设置日期格式化器和日期刻度定位器
    date_fmt = mdates.DateFormatter("%m-%d")  # 仅显示月-日
    date_locator = mdates.DayLocator()  # 每天显示一个刻度

    plt.gca().xaxis.set_major_formatter(date_fmt)
    plt.gca().xaxis.set_major_locator(date_locator)

    plt.plot(datetimes, values)
    plt.show()

# region 基础统计分析
def calculate_mean(TimeSeriesData):
    """计算TimeSeriesData对象的平均值"""
    total = sum([data.value for data in TimeSeriesData])
    mean = total / len(TimeSeriesData)
    return mean

def calculate_median(TimeSeriesData):
    """计算TimeSeriesData对象的中位数"""
    sorted_data = sorted(TimeSeriesData, key=lambda data: data.value)
    n = len(sorted_data)
    if n % 2 == 0:
        median = (sorted_data[n // 2 - 1].value + sorted_data[n // 2].value) / 2
    else:
        median = sorted_data[n // 2].value
    return median

def calculate_variance(TimeSeriesData):
    """计算TimeSeriesData对象的方差"""
    mean = calculate_mean(TimeSeriesData)
    squared_diff = [(data.value - mean) ** 2 for data in TimeSeriesData]
    variance = sum(squared_diff) / len(TimeSeriesData)
    return variance

def calculate_standard_deviation(TimeSeriesData):
    """计算TimeSeriesData对象的标准差"""
    variance = calculate_variance(TimeSeriesData)
    standard_deviation = math.sqrt(variance)
    return standard_deviation
# endregion

def fourier_transform(TimeSeriesData):
    """对TimeSeriesData对象进行傅里叶变换"""
    values = [data.value for data in TimeSeriesData]
    transformed_values = np.fft.fft(values)
    return transformed_values

def moving_average(TimeSeriesData, window_size):
    """计算移动平均值"""
    values = [data.value for data in TimeSeriesData]
    datetimes = [data.datetime for data in TimeSeriesData]
    n = len(values)
    moving_avg = []

    for i in range(n - window_size + 1):
        window_values = values[i : i + window_size]
        avg = sum(window_values) / window_size
        moving_avg.append(avg)

    return moving_avg

def plot_moving_average(TimeSeriesData, window_size):
    """绘制移动平均线"""
    avg_values = moving_average(TimeSeriesData, window_size)
    datetimes = [data.datetime for data in TimeSeriesData]
    moving_date=datetimes[window_size - 1:]

    plot(TimeSeriesData)
    plt.plot(moving_date, avg_values, label="移动平均")
    plt.xlabel('日期')  # 指定中文标签
    plt.ylabel('数值') # 指定中文标签
    plt.title('移动平均线')  # 指定中文标签

    # 设置日期格式化器和日期刻度定位器
    date_fmt = mdates.DateFormatter("%m-%d")  # 仅显示月-日
    date_locator = mdates.DayLocator()  # 每天显示一个刻度

    plt.gca().xaxis.set_major_formatter(date_fmt)
    plt.gca().xaxis.set_major_locator(date_locator)
    plt.show()

def plot_time_series_decomposition(time_series):
    """进行季节性分解"""
    # 将 TimeSeriesData 转换为 Pandas 的 Series 对象
    values = [data.value for data in time_series]
    datetimes = [data.datetime for data in time_series]
    ts = pd.Series(values, index=datetimes)
  
    # 进行季节性分解
    decomposition = sm.tsa.seasonal_decompose(ts, model='additive')

    # 提取分解后的各部分
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    # 绘制分解后的组成部分
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    ts.plot(ax=axes[0])
    axes[0].set_ylabel('原始数据')
    trend.plot(ax=axes[1])
    axes[1].set_ylabel('趋势')
    seasonal.plot(ax=axes[2])
    axes[2].set_ylabel('季节性')
    residual.plot(ax=axes[3])
    axes[3].set_ylabel('残差')

    plt.xlabel('日期')
    plt.tight_layout()
    plt.show()

def kendall_change_point_detection(input_data):
    """时序数据的kendall突变点检测"""
    n = len(input_data)
    Sk = [0]
    UFk = [0]
    s = 0
    Exp_value = [0]
    Var_value = [0]

    for i in range(1, n):
        for j in range(i):
            if input_data[i].value > input_data[j].value:
                s += 1
        Sk.append(s)
        Exp_value.append((i + 1) * (i + 2) / 4.0)
        Var_value.append((i + 1) * i * (2 * (i + 1) + 5) / 72.0)
        UFk.append((Sk[i] - Exp_value[i]) / math.sqrt(Var_value[i]))

    Sk2 = [0]
    UBk = [0]
    UBk2 = [0]
    s2 = 0
    Exp_value2 = [0]
    Var_value2 = [0]
    input_data_t = list(reversed(input_data))

    for i in range(1, n):
        for j in range(i):
            if input_data_t[i].value > input_data_t[j].value:
                s2 += 1
        Sk2.append(s2)
        Exp_value2.append((i + 1) * (i + 2) / 4.0)
        Var_value2.append((i + 1) * i * (2 * (i + 1) + 5) / 72.0)
        UBk.append((Sk2[i] - Exp_value2[i]) / math.sqrt(Var_value2[i]))
        UBk2.append(-UBk[i])

    UBkT = list(reversed(UBk2))
    diff = [x - y for x, y in zip(UFk, UBkT)]
    K = []

    for k in range(1, n):
        if diff[k - 1] * diff[k] < 0:
            K.append(input_data[k])

    # 绘图代码可以在这里添加，如果需要的话

    return K

# 生成随机的 TimeSeriesData 数据
def generate_random_data():
    """生成随机的 TimeSeriesData 数据
    # 创建 TimeSeriesData 实例
    ts_data = TimeSeriesData(value=generate_random_data(), datetime="")
    """
    start_date = datetime(2024, 3, 13, 0, 0, 0)
    data = []

    for i in range(50):
        value = random.randint(10, 30)
        current_date = start_date + timedelta(hours=i)
        data_point = TimeSeriesData(value=value, datetime=current_date)
        data.append(data_point)

    return data



# # 创建 TimeSeriesData 实例
# ts_data = generate_random_data()

# plot_time_series_decomposition(ts_data)

values = [10, 20, 30, 40]
datetimes = ['2022-01-01', '2022-01-02', '2022-01-03', '2022-01-04']

time_series = create_time_series_data(values, datetimes)
print(time_series)