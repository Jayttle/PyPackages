# 标准库导入
import math
import random
from datetime import datetime, timedelta
import time
import warnings

# 相关第三方库导入
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm
from collections import defaultdict
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering 
from sklearn.linear_model import Ridge , Lasso, LassoCV, ElasticNet, LinearRegression
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score, mean_squared_error, silhouette_score, davies_bouldin_score, calinski_harabasz_score, v_measure_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from scipy.spatial.distance import euclidean
from scipy import signal, fft
from scipy.cluster import hierarchy
from scipy.stats import t, shapiro, pearsonr, f_oneway, gaussian_kde
from scipy.signal import hilbert, find_peaks
from PyEMD import EMD, EEMD, CEEMDAN
from typing import List, Optional, Tuple, Union
import pywt
"""
python:
O(n)算法: 输入n: 1,000,000 耗时: 15.312433242797852 ms
O(n^2)算法输入n: 10,000 耗时: 1610.5492115020752 ms
O(nlogn)算法输入n: 10,000 耗时: 5.4988861083984375 ms 
"""
# region 字体设置
plt.rcParams['font.sans-serif'] = ['SimSun']  # 指定宋体为默认字体
plt.rcParams['font.sans-serif'] = 'SimSun'  # 使用指定的中文字体
# 禁用特定警告
warnings.filterwarnings('ignore', category=UserWarning, append=True)
# 或者关闭所有警告
warnings.filterwarnings("ignore")
# endregion


class TimeSeriesData:
    def __init__(self, value: float, datetime_input):
        self.value = value
        self.datetime = self._parse_datetime(datetime_input)  # 使用内部函数进行日期时间解析

    def _parse_datetime(self, datetime_input):
        # Check if the input is a datetime object or a string
        if isinstance(datetime_input, datetime):
            return datetime_input
        elif isinstance(datetime_input, str):
            # Parse the datetime string
            try:
                return datetime.strptime(datetime_input, "%Y-%m-%d %H:%M:%S.%f")
            except ValueError:
                # If parsing with microseconds fails, try parsing without microseconds
                return datetime.strptime(datetime_input, "%Y-%m-%d %H:%M:%S")
        else:
            raise TypeError("datetime_input must be a datetime object or a datetime string in the format '%Y-%m-%d %H:%M:%S' or '%Y-%m-%d %H:%M:%S.%f'")

    def value_half(self):
        # 返回 value 的一半
        return self.value / 2

    def __str__(self):
        return f"Value: {self.value}, Datetime: {self.datetime}"

class Datapoint:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

class ExtendedTimeSeriesData(TimeSeriesData):
    def __init__(self, datapoint: Datapoint, datetime_input):
        # 将 Datapoint 转换为 float 值
        value = datapoint.x  # 或者其他方式将 Datapoint 转换为 float 值
        # 调用父类的初始化方法
        super().__init__(value, datetime_input)
        # 添加额外的属性
        self.datapoint = datapoint

    def value_half(self):
        # 重写父类的 value_half 方法，在子类中返回 value 的一半
        return self.value / 2
    
    def set_value_to_y(self):
        # 将 value 设置为 datapoint.y
        self.value = self.datapoint.y

    def __str__(self):
        # 重写父类的 __str__ 方法，添加额外的信息
        return f"Datapoint: {self.datapoint.x}, {self.datapoint.y}, Datetime: {self.datetime}"
    

# datapoint = Datapoint(10.5, 20.3)
# datetime_input = datetime.now()
# extended_data = ExtendedTimeSeriesData(datapoint, datetime_input)

# # 将 value 设置为 datapoint.y
# extended_data.set_value_to_y()

# # 现在 value 应该是 datapoint.y
# print("Value 设置为 datapoint.y:", extended_data.value)
# print("Value 设置为 datapoint.y:", extended_data.value_half())


class TimeSeriesDataList:
    def __init__(self, data_list):
        self.data_list = data_list

    def calculate_mean(self):
        """计算TimeSeriesData对象的均值"""
        total = sum(data.value for data in self.data_list)
        mean = total / len(self.data_list)
        return mean

    def calculate_variance(self):
        """计算TimeSeriesData对象的方差"""
        mean = self.calculate_mean()
        squared_diff = [(data.value - mean) ** 2 for data in self.data_list]
        variance = sum(squared_diff) / len(self.data_list)
        return variance

    def calculate_standard_deviation(self):
        """计算TimeSeriesData对象的标准差"""
        variance = self.calculate_variance()
        standard_deviation = math.sqrt(variance)
        return standard_deviation

    def calculate_change_rate(self):
        """计算TimeSeriesData对象的变化率"""
        change_rates = []
        for i in range(1, len(self.data_list)):
            current_value = self.data_list[i].value
            previous_value = self.data_list[i-1].value
            change_rate = (current_value - previous_value) / previous_value
            change_rates.append(change_rate)
        return change_rates

# # 创建一个包含 TimeSeriesData 对象的列表
# data_list = [
#     TimeSeriesData(10.5, "2024-05-27 08:30:00"),
#     TimeSeriesData(20.3, "2024-05-27 09:00:00"),
#     TimeSeriesData(15.8, "2024-05-27 09:30:00")
# ]

# # 创建 TimeSeriesDataList 对象
# time_series_data_list = TimeSeriesDataList(data_list)

print("hello world")