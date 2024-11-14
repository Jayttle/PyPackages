import math
import random
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean

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
def analyze_fourier_transform_results(data):
    """
    函数 analyze_fourier_transform_results 用于计算和可视化 TimeSeriesData 数据的振幅谱，并识别数据中的主要周期分量。
    
    :param data: TimeSeriesData 类型的列表，包含 value 和 datetime 属性。
    """
    # 提取 value 值和构建时间序列
    values = [entry.value for entry in data]
    datetime = [entry.datetime for entry in data]

    # 计算采样率
    sampling_rate = (data[-1].datetime - data[0].datetime).total_seconds() / len(data)

    # 进行傅里叶变换
    transformed_values = np.fft.fft(values)

    # 构建频率轴
    N = len(data)  # 数据的长度
    frequencies = np.fft.fftfreq(N, d=sampling_rate)


    # 可视化振幅谱
    plt.figure()
    plt.plot(frequencies, np.abs(transformed_values))
    plt.xlabel('频率')
    plt.ylabel('振幅')
    plt.title('振幅谱')
    plt.grid(True)
    plt.show()


    """
    通过绘制频谱的幅度谱图，可以观察不同频率成分的能量分布情况。从图中你可以获取以下信息：
    峰值表示在该频率上存在主要的周期性成分。
    频谱中的宽峰表示存在多个相关频率的周期性成分。
    幅度谱中较低的值表示在该频率上不存在明显的周期性成分。
    """
# region 移动平均

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
# endregion
# region statsmodels的运用
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

def stl_decomposition(time_series):
    """进行季节性分解时序回归模型（STL）"""
    # 将 TimeSeriesData 转换为 Pandas 的 Series 对象
    values = [data.value for data in time_series]
    datetimes = [data.datetime for data in time_series]
    ts = pd.Series(values, index=datetimes)
    # 进行季节性分解时序回归模型（STL）
    result = sm.tsa.STL(ts, seasonal=13).fit()  # 以13为季节周期，可以根据需要进行调整

    # 提取分解后的各部分
    trend = result.trend
    seasonal = result.seasonal
    residual = result.resid

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


# 创建ARMA模型
def create_arma_model(time_series, order):
     # 将 TimeSeriesData 转换为 Pandas 的 Series 对象
    values = [data.value for data in time_series]
    datetimes = [data.datetime for data in time_series]
    ts = pd.Series(values, index=datetimes)
    model = sm.tsa.ARMA(ts, order=order).fit()
    return model

# 创建ARIMA模型
def create_arima_model(time_series, order):
    # 将 TimeSeriesData 转换为 Pandas 的 Series 对象
    values = [data.value for data in time_series]
    datetimes = [data.datetime for data in time_series]
    ts = pd.Series(values, index=datetimes)
    model = sm.tsa.ARIMA(ts, order=order).fit()
    return model

def predict_analyze_evaluate(time_series, order=(2, 1,1)):
    # 将 TimeSeriesData 转换为 Pandas 的 Series 对象
    values = [data.value for data in time_series]
    datetimes = [data.datetime for data in time_series]
    ts = pd.Series(values, index=datetimes)
    
    # 创建ARIMA模型
    arima_model = sm.tsa.ARIMA(ts, order=order).fit()
    
    arima_predictions = arima_model.predict(start=len(ts), end=len(ts)+2)
    
    print("ARIMA模型预测结果:", arima_predictions)
    
    # 残差分析
    arima_residuals = arima_model.resid
    
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(arima_residuals)
    plt.xlabel('时间')
    plt.ylabel('残差')
    plt.title('ARIMA模型残差图')

    plt.tight_layout()
    plt.show()
    
    # 模型评估
    arima_aic = arima_model.aic

    print("ARIMA模型AIC:", arima_aic)
# endregion
# region 突变检测
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
# endregion
# region scikit-learn使用
# 生成随机的 TimeSeriesData 数据
def generate_random_data():
    """生成随机的 TimeSeriesData 数据
    # 创建 TimeSeriesData 实例
    ts_data = TimeSeriesData(value=generate_random_data(), datetime="")
    """
    start_date = datetime(2024, 3, 13, 0, 0, 0)
    data = []

    for i in range(200):
        value = random.randint(10, 30)
        current_date = start_date + timedelta(hours=i)
        data_point = TimeSeriesData(value=value, datetime=current_date)
        data.append(data_point)

    return data

def calculate_similarity(ts1, ts2, similarity_metric='euclidean'):
    """
    计算两个时间序列之间的相似性或差异性

    Args:
        ts1 (list or numpy array): 第一个时间序列
        ts2 (list or numpy array): 第二个时间序列
        similarity_metric (str, optional): 相似性度量方法，默认为'euclidean'（欧氏距离）。可选值包括'euclidean'（欧氏距离），
                                            'pearson'（皮尔逊相关系数）。

    Returns:
        float: 两个时间序列之间的相似性或差异性值
    """
    if similarity_metric == 'euclidean':
        # 计算欧氏距离
        similarity = euclidean(ts1, ts2)
    elif similarity_metric == 'pearson':
        # 计算皮尔逊相关系数
        similarity = np.corrcoef(ts1, ts2)[0, 1]
    else:
        raise ValueError("不支持的相似性度量方法")

    return similarity

def time_series_clustering(ts_data, num_clusters):
    # 取出时间序列数据的值
    values = [data.value for data in ts_data]
    
    # 转换为numpy数组
    X = np.array(values).reshape(-1, 1)
    
    # 使用K-means进行聚类
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(X)
    
    # 获取每个时间序列的聚类标签
    labels = kmeans.labels_
    
    # 返回聚类结果
    return labels
# endregion
# # 创建 TimeSeriesData 实例
ts_data = generate_random_data()

sampling_rate = 1

# 进行傅里叶变换并绘制幅度谱图
analyze_fourier_transform_results(ts_data)
