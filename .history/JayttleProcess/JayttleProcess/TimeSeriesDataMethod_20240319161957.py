import math
import random
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
plt.rcParams['font.sans-serif'] = ['SimSun']  # 指定宋体为默认字体
plt.rcParams['font.sans-serif'] = 'SimSun'  # 使用指定的中文字体

class TimeSeriesData:
    def __init__(self, value, datetime):
        self.value = value
        self.datetime = datetime

    def __str__(self):
        return f"Value: {self.value}, Datetime: {self.datetime}"
    
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
    n = len(datetimes)
    for i in range(window_size - 1, n ):
        moving_date.append(datetimes[i])

    plt.plot(TimeSeriesData.datetime, TimeSeriesData.value, label="原始数据")
    plt.plot(datetimes, avg_values, label="移动平均")
    plt.xlabel('日期')  # 指定中文标签
    plt.ylabel('数值') # 指定中文标签
    plt.title('移动平均线')  # 指定中文标签
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



# 创建 TimeSeriesData 实例
ts_data = generate_random_data()

plot(ts_data)
# 计算统计指标
mean = calculate_mean(ts_data)
print("Mean:", mean)


median = calculate_median(ts_data)
print("Median:", median)

variance = calculate_variance(ts_data)
print("Variance:", variance)

std_deviation = calculate_standard_deviation(ts_data)
print("Standard Deviation:", std_deviation)

for data in ts_data:
    print("Value:", data.value,"DateTime:", data.datetime)

transformed_values = fourier_transform(ts_data)
print(transformed_values)

plot_moving_average(ts_data,5)