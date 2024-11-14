import math
import random
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ar_model import AutoReg
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import root_mean_squared_error
from scipy.spatial.distance import euclidean
from scipy.signal import hilbert
from PyEMD import EMD,EEMD, CEEMDAN


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
# region 频域分析
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

    # 获取右半边的频率和对应的振幅值
    frequencies_right = frequencies[:N//2+1]
    transformed_values_right = transformed_values[:N//2+1]

    # 可视化振幅谱
    plt.figure()
    plt.plot(frequencies_right, np.abs(transformed_values_right))
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
# endregion
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

def pettitt_change_point_detection(data):
    """
    使用Pettitt突变检测方法检测时间序列数据中的突变点。

    :param data: TimeSeriesData 类型的列表，包含 value 和 datetime 属性。
    :return: 突变点的位置和统计量。
    """
    # 提取 value 值
    values = [entry.value for entry in data]

    # 计算累积和
    cumulative_sum = np.cumsum(values)

    # 突变点的位置和统计量
    change_point = 0
    max_test_statistic = 0

    n = len(values)

    for i in range(n):
        current_statistic = abs(cumulative_sum[i] - cumulative_sum[n-i-1])
        if current_statistic > max_test_statistic:
            max_test_statistic = current_statistic
            change_point = i

    return change_point, max_test_statistic

# endregion
# region scikit-learn使用
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
# region 归一化处理
def min_max_normalization(data):
    """
    使用最小-最大归一化对时序数据进行归一化处理。

    :param data: 要归一化的时间序列数据，TimeSeriesData对象列表。
    :return: 归一化后的TimeSeriesData对象列表。
    """
    values = [entry.value for entry in data]
    min_val = min(values)
    max_val = max(values)
    normalized_values = [(val - min_val) / (max_val - min_val) for val in values]
  
    normalized_data = [
        TimeSeriesData(value, entry.datetime)
        for value, entry in zip(normalized_values, data)
    ]
    return normalized_data

def standardization(data):
    """
    使用标准化对时序数据进行归一化处理。

    :param data: 要归一化的时间序列数据，TimeSeriesData对象列表。
    :return: 归一化后的TimeSeriesData对象列表。
    """
    values = [entry.value for entry in data]
    mean_val = np.mean(values)
    std_dev = np.std(values)
    normalized_values = [(val - mean_val) / std_dev for val in values]
  
    normalized_data = [
        TimeSeriesData(value, entry.datetime)
        for value, entry in zip(normalized_values, data)
    ]
    return normalized_data

def decimal_scaling_normalization(data):
    """
    使用小数定标归一化对时序数据进行归一化处理。

    :param data: 要归一化的时间序列数据，TimeSeriesData对象列表。
    :return: 归一化后的TimeSeriesData对象列表。
    """
    values = [entry.value for entry in data]
    max_abs = max(abs(min(values)), abs(max(values)))

    normalized_data = [
        TimeSeriesData(value / max_abs, entry.datetime)
        for value, entry in zip(values, data)
    ]
    return normalized_data

def log_normalization(data):
    """
    使用对数归一化对时序数据进行归一化处理。

    :param data: 要归一化的时间序列数据，TimeSeriesData对象列表。
    :return: 归一化后的TimeSeriesData对象列表。
    """
    values = [entry.value for entry in data]
    min_val = min(values)
    normalized_values = np.log(values) - np.log(min_val)
  
    normalized_data = [
        TimeSeriesData(value, entry.datetime)
        for value, entry in zip(normalized_values, data)
    ]
  
    return normalized_data

def l1_normalization(data):
    """
    使用L1范数归一化对时序数据进行归一化处理。

    :param data: 要归一化的时间序列数据，TimeSeriesData对象列表。
    :return: 归一化后的TimeSeriesData对象列表。
    """
    values = [entry.value for entry in data]
    l1_norm = np.sum(np.abs(values))

    normalized_values = [value / l1_norm for value in values]
  
    normalized_data = [
        TimeSeriesData(value, entry.datetime)
        for value, entry in zip(normalized_values, data)
    ]
  
    return normalized_data

def l2_normalization(data):
    """
    使用L2范数归一化对时序数据进行归一化处理。

    :param data: 要归一化的时间序列数据，TimeSeriesData对象列表。
    :return: 归一化后的TimeSeriesData对象列表。
    """
    values = [entry.value for entry in data]
    l2_norm = np.linalg.norm(values)

    normalized_values = [value / l2_norm for value in values]
  
    normalized_data = [
        TimeSeriesData(value, entry.datetime)
        for value, entry in zip(normalized_values, data)
    ]
  
    return normalized_data
# endregion
# region 模态分解
def hilbert_transform(time_series):
    """对时间序列进行希尔伯特变换
    amplitude_envelope, instantaneous_phase = hilbert_transform(data)
    process_and_visualize_hilbert_transform(amplitude_envelope, instantaneous_phase)
    """
    values = [data.value for data in time_series]
    analytic_signal = hilbert(values)
    amplitude_envelope = np.abs(analytic_signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    
    return amplitude_envelope, instantaneous_phase

def process_and_visualize_hilbert_transform(amplitude_envelope, instantaneous_phase):
    """处理和可视化希尔伯特变换的结果"""
    # 可视化振幅包络
    plt.figure(figsize=(8, 4))
    plt.plot(amplitude_envelope)
    plt.title("振幅包络")
    plt.xlabel("时间")
    plt.ylabel("振幅")
    plt.show()

    # 可视化瞬时相位
    plt.figure(figsize=(8, 4))
    plt.plot(instantaneous_phase)
    plt.title("瞬时相位")
    plt.xlabel("时间")
    plt.ylabel("相位")
    plt.show()

def empirical_mode_decomposition(time_series_data):
    """对时序数据进行经验模态分解
    imfs = empirical_mode_decomposition(data)
    """
    values = np.array([data.value for data in time_series_data])
    
    # 创建EMD对象，并进行分解
    emd = EMD()
    imfs = emd.emd(values)
    
    return imfs

def plot_imfs(imfs):
    """绘制IMFs"""
    num_imfs = len(imfs)

    # 创建子图布局
    fig, axes = plt.subplots(num_imfs, 1, figsize=(8, 2*num_imfs), sharex=True)

    # 绘制每个IMF的图形
    for i, imf in enumerate(imfs):
        axes[i].plot(imf)
        axes[i].set_ylabel(f"IMF {i+1}")

    # 设置横坐标标签
    axes[-1].set_xlabel("Time")

    # 调整子图之间的间距
    plt.tight_layout()

    # 显示图形
    plt.show()

def eemd(time_series_data, num_trials=100):
    """实现EEMD"""
    values = np.array([data.value for data in time_series_data])
    eemd = EEMD()
    eemd.trials = num_trials
    imfs = eemd.eemd(values)
  
    return imfs

def ceemd(time_series_data, num_trials=100):
    """实现CEEMD"""
    values = np.array([data.value for data in time_series_data])
    ceemdan = CEEMDAN(trials=num_trials)
    imfs = ceemdan.ceemdan(values)
  
    return imfs

# endregion

def ar_model_forecast(TimeSeriesData, lags=1, future_steps=10):
    """
    自回归模型(Autoregressive Model, AR)
    lags = 1
    future_steps = 10
    future_forecast, mse = ar_model_forecast(time_series_data, lags, future_steps)
    print("未来预测的均方根误差：", mse)
    在AR模型中，lags参数表示使用多少个滞后观测值作为特征来预测当前观测值。它控制了模型的阶数，也就是自回归的程度
    future_steps参数表示未来要预测的步数，即要预测多少个未来观测值。通过使用拟合的AR模型和历史观测值，可以对未来指定步数的观测值进行预测
    """
    # 提取时序数据的观测值
    X = np.array([data.value for data in TimeSeriesData])

    # 拟合AR模型
    model = AutoReg(X, lags=lags)
    model_fit = model.fit()

    # 进行未来预测
    future_forecast = model_fit.forecast(steps=future_steps)

    # 计算均方根误差
    mse = root_mean_squared_error(X[-future_steps:], future_forecast)

    # 绘制原始数据和预测结果
    plt.plot(X, label='原始数据')
    plt.plot(np.arange(len(X), len(X) + future_steps), future_forecast, label='预测结果')
    plt.xlabel('时间')
    plt.ylabel('观测值')
    plt.legend()
    plt.title('AR模型预测结果')
    plt.show()

    return future_forecast, mse

def ma_model_forecast(TimeSeriesData, q=1, future_steps=10):
    """
    移动平均模型(Moving Average Model, MA)
    future_forecast, mse = ma_model_forecast(time_series_data, q, future_steps)
    我们使用 ARIMA 类从 statsmodels 库中拟合一个 MA 模型，并使用 order=(0, 0, q) 指定使用 q 阶移动平均模型
    我们使用 ARIMA 类从 statsmodels 库中拟合一个 MA 模型，并使用 order=(0, 0, q) 指定使用 q 阶移动平均模型
    """
    # 提取时序数据的观测值
    X = np.array([data.value for data in TimeSeriesData])
    
    # 拟合MA模型
    model = ARIMA(X, order=(0, 0, q))
    model_fit = model.fit()
    
    # 进行未来预测
    future_forecast = model_fit.forecast(steps=future_steps)
    
    mse = root_mean_squared_error(np.array(X[-future_steps:]), np.array(future_forecast))
    
    # 绘制原始数据和预测结果
    plt.plot(X, label='原始数据')
    plt.plot(np.arange(len(X), len(X) + future_steps), future_forecast, label='预测结果')
    plt.xlabel('时间')
    plt.ylabel('观测值')
    plt.legend()
    plt.title('MA模型预测结果')
    plt.show()
    
    return future_forecast, mse

def arma_model_forecast(TimeSeriesData, p, q, future_steps):
    """
    自回归移动平均模型(Autoregressive Moving Average Model,ARM）
    并指定合适的p、q以及未来预测的步数future_steps，然后该函数将返回未来预测的结果以及均方根误差。
    """
    # 提取时序数据的观测值
    X = [data.value for data in TimeSeriesData]
    
    # 拟合ARMA模型
    model = ARIMA(X, order=(p, 0, q))
    model_fit = model.fit()
    
    # 进行未来预测
    future_forecast = model_fit.forecast(steps=future_steps)[0]
    
    # 计算均方根误差
    mse = root_mean_squared_error(X[-future_steps:], future_forecast, squared=False)
    
    # 绘制原始数据和预测结果
    plt.plot(X, label='原始数据')
    plt.plot(np.arange(len(X), len(X) + future_steps), future_forecast, label='预测结果')
    plt.xlabel('时间')
    plt.ylabel('观测值')
    plt.legend()
    plt.title('ARMA模型预测结果')
    plt.show()
    
    return future_forecast, mse

def plot_ACF(TimeSeriesData):
    """绘制自相关函数（ACF）图表"""
    # 提取时序数据的观测值
    X = [data.value for data in TimeSeriesData]

    # 计算自相关函数（ACF）
    acf = sm.tsa.acf(X, nlags=len(data))
    # 绘制自相关函数（ACF）图表
    plt.figure(figsize=(10, 6))
    plt.stem(acf)
    plt.xlabel('滞后阶数')
    plt.ylabel('相关系数')
    plt.title('自相关函数（ACF）')
    plt.show()

def plot_PACF(TimeSeriesData,lags=48):
    """绘制偏自相关函数（PACF）图表"""
    # 提取时序数据的观测值
    X = [data.value for data in TimeSeriesData]

    # 计算偏自相关函数（PACF）
    pacf = sm.tsa.pacf(X, nlags=lags)
    # 绘制偏自相关函数（PACF）图表
    plt.figure(figsize=(10, 6))
    plt.stem(pacf)
    plt.xlabel('滞后阶数')
    plt.ylabel('相关系数')
    plt.title('偏自相关函数（PACF）')
    plt.show()

def evaluate_arma_model(data):
    # 提取时序数据的观测值
    X = [point.value for point in data]

    # 选择一系列可能的模型阶数
    p_values = range(1, 5)  # 自回归阶数
    q_values = range(1, 5)  # 移动平均阶数

    # 用网格搜索方式寻找最佳的ARMA模型
    best_aic = np.inf
    best_params = None

    for p in p_values:
        for q in q_values:
            try:
                model = sm.tsa.ARMA(X, order=(p, q))
                results = model.fit()
                aic = results.aic
                bic = results.bic
                if aic < best_aic:
                    best_aic = aic
                    best_bic = bic
                    best_params = (p, q)
            except:
                continue

    if best_params is not None:
        print("最佳模型的参数：p={}, q={}".format(best_params[0], best_params[1]))
        print("最佳模型的AIC值：{}".format(best_aic))
        print("最佳模型的BIC值：{}".format(best_bic))
    else:
        print("未找到最佳模型")

def fit_arima_model(data, p=1, d=1, q=1, num_steps=10,output_file="arima_results.txt"):
    """
    差分自回归移动平均模型(ARIMA)
    该函数接受一个时序数据作为输入（例如TimeSeriesData实例的列表），并设置ARIMA模型的阶数（p、d、q）以及预测步数（num_steps）。
    它使用此时序数据来执行ARIMA模型的拟合，并打印出模型的统计摘要和预测的未来数据点。
    """
    X = [point.value for point in data]

    # 创建ARIMA模型对象
    model = sm.tsa.ARIMA(X, order=(p, d, q))

    # 拟合ARIMA模型
    results = model.fit()

    # 将结果写入txt文件
    with open(output_file, 'w') as file:
        file.write(results.summary().as_text())

        # 预测未来的数据点
        forecast = results.forecast(steps=num_steps)
        file.write("\n\n预测结果：\n")
        file.write(str(forecast))

# 生成随机的 TimeSeriesData 数据
def generate_random_data():
    """生成随机的 TimeSeriesData 数据
    # 创建 TimeSeriesData 实例
    ts_data = TimeSeriesData(value=generate_random_data(), datetime="")
    """
    start_date = datetime(2024, 3, 13, 0, 0, 0)
    data = []

    for i in range(150):
        value = random.randint(10, 30)
        current_date = start_date + timedelta(hours=i)
        data_point = TimeSeriesData(value=value, datetime=current_date)
        data.append(data_point)

    return data
# 创建 TimeSeriesData 实例
data = generate_random_data()
# 拟合 ARIMA 模型并预测未来数据点
fit_arima_model(data)

# # 打印IMFs
# for i, imf in enumerate(imfs):
#     print(f"IMF {i+1}:")
#     for data in imf:
#         print(data)
# plot_imfs(imfs)

