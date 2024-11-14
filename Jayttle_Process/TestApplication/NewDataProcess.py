# 标准库导入
import math
import random
from datetime import datetime, timedelta
import time
import warnings
import os
# 相关第三方库导入
import numpy as np
from math import floor
import chardet
import pandas as pd
from scipy.stats import skew, kurtosis
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from matplotlib.font_manager import FontProperties
import statsmodels.api as sm
from collections import defaultdict
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.linear_model import Ridge, Lasso, LassoCV, ElasticNet, LinearRegression
from sklearn.metrics import (root_mean_squared_error, mean_absolute_error, r2_score,
                             mean_squared_error, silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, v_measure_score)
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.mixture import GaussianMixture
from sklearn.svm import OneClassSVM
from sklearn.model_selection import GridSearchCV, cross_val_score
from scipy.spatial.distance import euclidean
from scipy import signal, fft, interpolate, stats
from scipy.cluster import hierarchy
from scipy.stats import t, shapiro, pearsonr, f_oneway, gaussian_kde, chi2_contingency
from scipy.signal import hilbert, find_peaks
from PyEMD import EMD, EEMD, CEEMDAN
from typing import List, Optional, Tuple, Union, Set, Dict
import pywt
from itertools import groupby
from operator import attrgetter
from dataclasses import dataclass

# 自定义库导入
from pyswarm import pso
from JayttleProcess import ListFloatDataMethod as LFDM
from JayttleProcess import TimeSeriesDataMethod as TSM, TBCProcessCsv, CommonDecorator
from JayttleProcess.TimeSeriesDataMethod import TimeSeriesData
from scipy.interpolate import interp1d
"""
python:
O(n)算法: 输入n: 1,000,000 耗时: 15.312433242797852 ms
O(n^2)算法输入n: 10,000 耗时: 1610.5492115020752 ms
O(nlogn)算法输入n: 10,000 耗时: 5.4988861083984375 ms 
"""
# region 字体设置
plt.rcParams['font.sans-serif'] = ['SimSun']  # 指定宋体为默认字体
plt.rcParams['font.sans-serif'] = 'SimSun'  # 使用指定的中文字体
# 创建一个FontProperties对象，设置字体大小为6号
font_prop = FontProperties(fname=None, size=6)
# 禁用特定警告
warnings.filterwarnings('ignore', category=UserWarning, append=True)
# 或者关闭所有警告
warnings.filterwarnings("ignore")
# endregion

class Met:
    def __init__(self, datetime_obj: datetime, temperature: float, humidness: float, pressure: float, wind_speed: float, wind_direction: int):
        self.datetime_obj = datetime_obj
        self.temperature = temperature
        self.humidness = humidness
        self.pressure = pressure
        self.wind_speed = wind_speed
        self.wind_direction = wind_direction

    def __str__(self) -> str:
        return f"DateTime: {self.datetime_obj}, Temperature: {self.temperature}, Humidity: {self.humidness}, Pressure: {self.pressure}, WindSpeed: {self.wind_speed}, WindDirection: {self.wind_direction}"

@CommonDecorator.log_function_call
def read_time_series_data(file_path: str) -> List[Met]:
    met_data_list: List[Met] = []
    with open(file_path, 'r', encoding='utf-8') as file:
        next(file)  # Skip the header
        for line in file:
            parts = line.strip().split('\t')
            datetime_str: str = parts[0]
            
            # Check if fractions of seconds are present in datetime_str
            if '.' in datetime_str:
                datetime_format = '%Y-%m-%d %H:%M:%S.%f'
            else:
                datetime_format = '%Y-%m-%d %H:%M:%S'
            
            datetime_obj: datetime = datetime.strptime(datetime_str, datetime_format)
            temperature: float = float(parts[2])
            humidness: float = float(parts[3])
            pressure: float = float(parts[4])
            wind_speed: float = float(parts[5])
            wind_direction: int = int(parts[6])  # Assuming WindDirection is an integer

            # Create a Met object and append it to the list
            met_data_list.append(Met(datetime_obj, temperature, humidness, pressure, wind_speed, wind_direction))

    return met_data_list


def read_weather_data(file_path: str) -> dict:
    weather_data = {}

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')
            date = parts[0]
            max_temp = int(parts[1].split('°')[0].split()[-1])
            min_temp = int(parts[3].split('°')[0].split()[-1])
            weather_data[date] = (max_temp, min_temp)

    return weather_data

def read_precipitation_data(file_path: str) -> dict:
    precipitation_data = {}

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')
            date = parts[0]
            precipitation = float(parts[2].split(':')[1].strip().split()[0])
            precipitation_data[date] = precipitation

    return precipitation_data

def read_humidity_data(file_path: str) -> dict:
    humidity_data = {}

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')
            date = parts[0]
            humidity = float(parts[2].split(':')[1].strip().split('%')[0])
            humidity_data[date] = humidity

    return humidity_data

def read_wind_data(file_path: str) -> dict:
    wind_data = {}

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')
            date = parts[0]
            avg_wind_speed = float(parts[2].split(':')[1].strip().split()[0])
            wind_data[date] = avg_wind_speed
    return wind_data

@CommonDecorator.log_function_call
def read_tianmeng_met() -> Dict[str, Tuple[float, float]]:
    # File path
    input_file_path: str = r"C:\Users\Jayttle\Desktop\tianmeng_met.txt"

    # Read data into Met objects
    met_data: List[Met] = read_time_series_data(input_file_path)

    # Sort met_data based on datetime_obj
    met_data.sort(key=attrgetter('datetime_obj'))

    # Group Met objects by date
    grouped_data = groupby(met_data, key=lambda met: met.datetime_obj.date())

    # Create a dictionary to store temperature data
    temperature_data: Dict[str, Tuple[float, float]] = {}

    # Create a dictionary to store time of max and min temperatures
    temperature_get_maxmin_time: Dict[str, Tuple[str, str]] = {}

    # Iterate over each group
    for date, group in grouped_data:
        # Extract temperatures with corresponding timestamps from the group
        temperature_timestamps = [(met.temperature, met.datetime_obj.time()) for met in group]
        
        # Calculate maximum and minimum temperatures
        max_temp, max_temp_time = max(temperature_timestamps, key=lambda x: x[0])
        min_temp, min_temp_time = min(temperature_timestamps, key=lambda x: x[0])
        
        # Store max and min temperatures along with their times in the dictionaries
        temperature_data[str(date)] = (max_temp, min_temp)
        
        # Classify max_temp_time and min_temp_time
        max_temp_label = "Label 1" if datetime.strptime(str(max_temp_time), '%H:%M:%S.%f').hour in range(10, 19) else "Label 2"
        min_temp_label = "Label 1" if datetime.strptime(str(min_temp_time), '%H:%M:%S.%f').hour in range(10, 19) else "Label 2"
        # Store data in temperature_get_maxmin_time
        temperature_get_maxmin_time[str(date)] = ((max_temp_label, max_temp_time), (min_temp_label, min_temp_time))

    with open('date.txt', 'w') as file:
        for date in temperature_get_maxmin_time:
            max_temp_label, max_temp_time = temperature_get_maxmin_time[date][0]
            min_temp_label, min_temp_time = temperature_get_maxmin_time[date][1]
            if max_temp_label == "Label 2" and min_temp_label == "Label 1":
                file.write(f"{date}\n")

    
    return temperature_data, temperature_get_maxmin_time


def check_change_match_ratio(list_float1: list[float], list_float2: list[float], threshold: float = 0):
    # 计算变化率
    list_changes1 = [list_float1[i+1] - list_float1[i] for i in range(len(list_float1)-1)]
    list_changes2 = [list_float2[i+1] - list_float2[i] for i in range(len(list_float2)-1)]

    # 标识变化方向
    change_directions1 = [1 if change > 0 else -1 if change < 0 else 0 for change in list_changes1]
    change_directions2 = [1 if change > 0 else -1 if change < 0 else 0 for change in list_changes2]

    # 判断变化率是否小于阈值，并将其归为变化方向为0
    for i in range(len(list_changes1)):
        if abs(list_changes1[i]) < threshold:
            change_directions1[i] = 0

    for i in range(len(list_changes2)):
        if abs(list_changes2[i]) < threshold:
            change_directions2[i] = 0

    # 比较两组数据的变化方向
    matches = [1 if dir1 == dir2 else 0 for dir1, dir2 in zip(change_directions1, change_directions2)]

    # 计算匹配的时间点数量与总时间点数量的比例
    matching_ratio = sum(matches) / len(matches)

    print("匹配比例:", matching_ratio)

    # 构建观察频数表格
    observed_table = [[sum(matches), len(matches) - sum(matches)],
                    [len(matches) - sum(matches), sum(matches)]]

    # 执行卡方检验
    chi2, p_value, _, _ = chi2_contingency(observed_table)

    print("卡方值:", chi2)
    print("p值:", p_value)


def run_compare_temperature():
    # File path
    file_path = r"D:\Program Files (x86)\Software\OneDrive\PyPackages_DataSave\weather_temperature.txt"   
    # Read weather data
    weather_data: Dict[str, Tuple[float, float]] = read_weather_data(file_path)
    temperature_data, temperature_get_maxmin_time = read_tianmeng_met()

    # Lists to store temperature data
    weather_max_temps: List[float] = []
    weather_min_temps: List[float] = []
    temperature_max_temps: List[float] = []
    temperature_min_temps: List[float] = []

    # Iterate over the keys of both dictionaries
    for date_str in weather_data.keys():
        if date_str in temperature_data:
            # Append temperatures to respective lists
            weather_max_temp, weather_min_temp = weather_data[date_str]
            temperature_max_temp, temperature_min_temp = temperature_data[date_str]
            weather_max_temps.append(weather_max_temp)
            weather_min_temps.append(weather_min_temp)
            temperature_max_temps.append(temperature_max_temp)
            temperature_min_temps.append(temperature_min_temp)

    correlation_max_temp, _ = pearsonr(weather_max_temps, temperature_max_temps)
    correlation_min_temp, _ = pearsonr(weather_min_temps, temperature_min_temps)


    print(f"当天最高温度pearson相关系数: {correlation_max_temp}")
    print(f"当天最低温度pearson相关系数: {correlation_min_temp}")

    mse_max_temp = mean_squared_error(weather_max_temps, temperature_max_temps)
    mse_min_temp = mean_squared_error(weather_min_temps, temperature_min_temps)
    mae_max_temp = mean_absolute_error(weather_max_temps, temperature_max_temps)
    mae_min_temp = mean_absolute_error(weather_min_temps, temperature_min_temps)

    print(f"当天最高温度--均方误差: {mse_max_temp}")
    print(f"当天最低温度--均方误差: {mse_min_temp}")
    print(f"当天最高温度--平均绝对误差: {mae_max_temp}")
    print(f"当天最低温度--平均绝对误差: {mae_min_temp}")

    # plt.figure(figsize=(9,6))

    # # Plot normalized weather_min_temps
    # plt.plot(range(len(weather_max_temps)), weather_max_temps, label='微软天气最高温度', color='blue')

    # # Plot normalized temperature_min_temps
    # plt.plot(range(len(temperature_max_temps)), temperature_max_temps, label='传感器最高温度', color='red')

    # # Adding labels and title
    # plt.xlabel('Days')
    # plt.ylabel('当天最高温度(℃)')
    # plt.title('微软天气与气象仪数据对比--当天最高温度')
    # plt.legend()

    # # Show plot
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    # Calculate differences
    temp_differences = [weather_temp - met_temp for weather_temp, met_temp in zip(weather_min_temps, temperature_min_temps)]
    # Calculate standard deviation
    std_deviation = np.std(temp_differences)

    # Calculate mean
    mean_temp_diff = np.mean(temp_differences)

    # Calculate coefficient of variation
    coefficient_variation = (std_deviation / mean_temp_diff) * 100
    
    print(f"最低温度差值的平均值为: {mean_temp_diff}")
    print(f"最低温度差值的标准差为: {std_deviation}")
    print(f"最低温度差值的变异系数为: {coefficient_variation}%")

        # Calculate differences

    temp_differences = [weather_temp - met_temp for weather_temp, met_temp in zip(weather_max_temps, temperature_max_temps)]
    # Calculate standard deviation
    std_deviation = np.std(temp_differences)

    # Calculate mean
    mean_temp_diff = np.mean(temp_differences)

    # Calculate coefficient of variation
    coefficient_variation = (std_deviation / mean_temp_diff) * 100
    
    print(f"最高温度差值的平均值为: {mean_temp_diff}")
    print(f"最高温度差值的标准差为: {std_deviation}")
    print(f"最高温度差值的变异系数为: {coefficient_variation}%")
    

    mse_max_temp = mean_squared_error(weather_min_temps, temperature_min_temps)
    mae_max_temp = mean_absolute_error(weather_min_temps, temperature_min_temps)

    print(f"均方误差: {mse_max_temp}")
    # print(f"当天最低温度--均方误差: {mse_min_temp}")
    print(f"平均绝对误差: {mae_max_temp}")
    # # Plotting
    # plt.figure(figsize=(10, 6))

    # # Plot temp_differences
    # plt.plot(range(len(temp_differences)), temp_differences, label='温度差值', color='green')
    # plt.axhline(y=mean_temp_diff, color='red', linestyle='--', label='平均值')


    # # Adding labels and title
    # plt.xlabel('Days')
    # plt.ylabel('温度差值 (°C)')
    # plt.title('微软天气与气象仪数据对比--当天最低温度差值')
    # plt.legend()

    # # Show plot
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
@CommonDecorator.log_function_call
def run_compare_precipitation():
    # File path
    file_path = r"D:\Program Files (x86)\Software\OneDrive\PyPackages_DataSave\weather_precipitation.txt"
    precipitation_data = read_precipitation_data(file_path)
    date_list: list[str] = []

    input_file_path: str = r"C:\Users\Jayttle\Desktop\tianmeng_met.txt"
    met_data: List[Met] = read_time_series_data(input_file_path)
    met_data.sort(key=attrgetter('datetime_obj'))

    grouped_data = groupby(met_data, key=lambda met: met.datetime_obj.date())
    temperature_data: Dict[str, Tuple[float, float]] = {}
    temperature_get_maxmin_time: Dict[str, Tuple[str, str]] = {}

    # Iterate over each group
    for date, group in grouped_data:
        # Extract temperatures with corresponding timestamps from the group
        temperature_timestamps = [(met.temperature, met.datetime_obj.time()) for met in group]
        
        # Calculate maximum and minimum temperatures
        max_temp, max_temp_time = max(temperature_timestamps, key=lambda x: x[0])
        min_temp, min_temp_time = min(temperature_timestamps, key=lambda x: x[0])
        
        # Store max and min temperatures along with their times in the dictionaries
        temperature_data[str(date)] = (max_temp, min_temp)
        
        # Classify max_temp_time and min_temp_time
        max_temp_label = "Label 1" if datetime.strptime(str(max_temp_time), '%H:%M:%S.%f').hour in range(10, 19) else "Label 2"
        min_temp_label = "Label 1" if datetime.strptime(str(min_temp_time), '%H:%M:%S.%f').hour in range(10, 19) else "Label 2"
        # Store data in temperature_get_maxmin_time
        temperature_get_maxmin_time[str(date)] = ((max_temp_label, max_temp_time), (min_temp_label, min_temp_time))


    # 遍历降水量数据
    for date, precipitation in precipitation_data.items():
        if precipitation > 1:
            date_list.append(date)
    
def run_compare_humidity():
        # File path
    file_path = r"D:\Program Files (x86)\Software\OneDrive\PyPackages_DataSave\weather_humidity.txt"
    weather_humidity_data = read_humidity_data(file_path)
    weather_humidity_data_list: list[float] = []

    input_file_path: str = r"C:\Users\Jayttle\Desktop\tianmeng_met.txt"
    met_data: List[Met] = read_time_series_data(input_file_path)
    met_data.sort(key=attrgetter('datetime_obj'))

    grouped_data = groupby(met_data, key=lambda met: met.datetime_obj.date())
    humidity_data: Dict[str, Tuple[float, float]] = {}
    humidity_get_maxmin_time: Dict[str, Tuple[str, str]] = {}

    humidity_max_temps: List[float] = []
    humidity_min_temps: List[float] = []
    humidity_mean_temps: List[float] = []

    # Iterate over each group
    for date, group in grouped_data:
        # Extract temperatures with corresponding timestamps from the group
        humidity_timestamps = [(met.humidness, met.datetime_obj.time()) for met in group]
        
        # Calculate maximum and minimum temperatures
        max_temp, max_temp_time = max(humidity_timestamps, key=lambda x: x[0])
        min_temp, min_temp_time = min(humidity_timestamps, key=lambda x: x[0])

        humidity_max_temps.append(max_temp)
        humidity_min_temps.append(min_temp)
        
        weather_humidity_data_list.append(weather_humidity_data[str(date)])
        # Store max and min temperatures along with their times in the dictionaries
        humidity_data[str(date)] = (max_temp, min_temp)

        
        # Calculate average humidity
        avg_humidity = sum(val[0] for val in humidity_timestamps) / len(humidity_timestamps)
        humidity_mean_temps.append(avg_humidity)



    correlation_max_temp, _ = pearsonr(humidity_max_temps, weather_humidity_data_list)
    correlation_min_temp, _ = pearsonr(humidity_min_temps, weather_humidity_data_list)
    correlation_mean_temp, _ = pearsonr(humidity_mean_temps, weather_humidity_data_list)

    print(f"传感器最高湿度与微软天气湿度pearson相关系数: {correlation_max_temp}")
    print(f"传感器最低湿度与微软天气湿度pearson相关系数: {correlation_min_temp}")
    print(f"传感器平均湿度与微软天气湿度pearson相关系数: {correlation_mean_temp}")


    mse_max_temp = mean_squared_error(weather_humidity_data_list, humidity_mean_temps)
    mae_max_temp = mean_absolute_error(weather_humidity_data_list, humidity_mean_temps)

    print(f"均方误差: {mse_max_temp}")
    # print(f"当天最低温度--均方误差: {mse_min_temp}")
    print(f"平均绝对误差: {mae_max_temp}")

    # plt.figure(figsize=(9,6))
    # # plt.plot(range(len(humidity_max_temps)), humidity_max_temps, label='传感器最低湿度', color='green')
    # plt.plot(range(len(humidity_mean_temps)), humidity_mean_temps, label='传感器平均湿度', color='red')
    # # plt.plot(range(len(humidity_min_temps)), humidity_min_temps, label='传感器最高湿度', color='red')
    # plt.plot(range(len(weather_humidity_data_list)), weather_humidity_data_list, label='微软天气湿度', color='blue')
    # # plt.bar(range(len(weather_humidity_data_list)), weather_humidity_data_list, label='微软天气湿度', color='blue', alpha=0.1)
    # plt.grid(color='#95a5a6',linestyle='--',linewidth=1,axis='y',alpha=0.6)

    # plt.xlabel('Days')
    # plt.ylabel('湿度(%)')
    # plt.title('微软天气与气象仪数据对比--平均湿度')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # Calculate differences
    temp_differences = [weather_temp - met_temp for weather_temp, met_temp in zip(humidity_mean_temps, weather_humidity_data_list)]
    # Calculate standard deviation
    std_deviation = np.std(temp_differences)

    # Calculate mean
    mean_temp_diff = np.mean(temp_differences)

    # Calculate coefficient of variation
    coefficient_variation = (std_deviation / mean_temp_diff) * 100
    
    print(f"差值的平均值为: {mean_temp_diff}")
    print(f"差值的标准差为: {std_deviation}")
    print(f"差值的变异系数为: {coefficient_variation}%")
def run_filter_humidity():
    input_file_path: str = r"C:\Users\Jayttle\Desktop\tianmeng_met.txt"
    met_data: List[Met] = read_time_series_data(input_file_path)
    met_data.sort(key=attrgetter('datetime_obj'))

    grouped_data = groupby(met_data, key=lambda met: met.datetime_obj.date())
    humidity_data: Dict[str, Tuple[float, float]] = {}
    humidity_get_maxmin_time: Dict[str, Tuple[str, str]] = {}

    humidity_max_temps: List[float] = []
    humidity_min_temps: List[float] = []
    humidity_mean_temps: List[float] = []

    # Iterate over each group
    for date, group in grouped_data:
        # Extract temperatures with corresponding timestamps from the group
        humidity_timestamps = [(met.humidness, met.datetime_obj.time()) for met in group]
        
        # Calculate maximum and minimum temperatures
        max_temp, max_temp_time = max(humidity_timestamps, key=lambda x: x[0])
        min_temp, min_temp_time = min(humidity_timestamps, key=lambda x: x[0])

        humidity_max_temps.append(max_temp)
        humidity_min_temps.append(min_temp)
        
        # Store max and min temperatures along with their times in the dictionaries
        humidity_data[str(date)] = (max_temp, min_temp)

        # Calculate average humidity
        avg_humidity = sum(val[0] for val in humidity_timestamps) / len(humidity_timestamps)
        humidity_mean_temps.append(avg_humidity)

    high_humidity_dates = [date for date, (max_temp, min_temp) in humidity_data.items() if max_temp > 90 or min_temp > 90]
    if high_humidity_dates:
        print("湿度超过90的日期：")
        for date in high_humidity_dates:
            print(date)
    else:
        print("没有湿度超过90的日期。")

def run_compare_wind():
        # File path
    file_path = r"D:\Program Files (x86)\Software\OneDrive\PyPackages_DataSave\weather_wind.txt"
    weather_wind_data = read_wind_data(file_path)
    weather_wind_data_list: list[float] = []

    input_file_path: str = r"C:\Users\Jayttle\Desktop\tianmeng_met.txt"
    met_data: List[Met] = read_time_series_data(input_file_path)
    met_data.sort(key=attrgetter('datetime_obj'))

    grouped_data = groupby(met_data, key=lambda met: met.datetime_obj.date())
    wind_data: Dict[str, Tuple[float, float]] = {}

    wind_max_temps: List[float] = []
    wind_min_temps: List[float] = []
    wind_mean_temps: List[float] = []

    dates_with_high_wind: list[str] = []  # 保存平均湿度大于50的日期
    # Iterate over each group
    for date, group in grouped_data:
        # Extract temperatures with corresponding timestamps from the group
        wind_timestamps = [(met.wind_speed, met.datetime_obj.time()) for met in group]
        
        # Calculate maximum and minimum temperatures
        max_temp, max_temp_time = max(wind_timestamps, key=lambda x: x[0])
        min_temp, min_temp_time = min(wind_timestamps, key=lambda x: x[0])
        # Calculate average humidity
        avg_wind = sum(val[0] for val in wind_timestamps) / len(wind_timestamps) *3.6
        # 如果平均湿度大于50，则将日期添加到列表中
        if avg_wind > 50:
            dates_with_high_wind.append(str(date))

        else:
            wind_max_temps.append(max_temp)
            wind_min_temps.append(min_temp)

            # Store max and min temperatures along with their times in the dictionaries
            wind_data[str(date)] = (max_temp, min_temp)
            

            wind_mean_temps.append(avg_wind)
            weather_wind_data_list.append(weather_wind_data[str(date)])

    check_change_match_ratio(wind_mean_temps, weather_wind_data_list, 0)

    for date in dates_with_high_wind:
        print(f"高速风：{date}")
    correlation_max_temp, _ = pearsonr(wind_max_temps, weather_wind_data_list)
    correlation_min_temp, _ = pearsonr(wind_min_temps, weather_wind_data_list)
    correlation_mean_temp, _ = pearsonr(wind_mean_temps, weather_wind_data_list)

    print(f"传感器最高风速与微软天气湿度pearson相关系数: {correlation_max_temp}")
    print(f"传感器最低湿度与微软天气湿度pearson相关系数: {correlation_min_temp}")
    print(f"传感器平均湿度与微软天气湿度pearson相关系数: {correlation_mean_temp}")

    # plt.figure(figsize=(9,6))
    # # plt.plot(range(len(humidity_max_temps)), humidity_max_temps, label='传感器最低湿度', color='green')
    # plt.plot(range(len(wind_mean_temps)), wind_mean_temps, label='传感器平均风速', color='red')
    # # plt.plot(range(len(humidity_min_temps)), humidity_min_temps, label='传感器最高湿度', color='red')
    # plt.plot(range(len(weather_wind_data_list)), weather_wind_data_list, label='微软天气风速', color='blue')
    # # plt.bar(range(len(weather_humidity_data_list)), weather_humidity_data_list, label='微软天气湿度', color='blue', alpha=0.1)
    # plt.grid(color='#95a5a6',linestyle='--',linewidth=1,axis='y',alpha=0.6)

    # plt.xlabel('Days')
    # plt.ylabel('风速(km/h)')
    # plt.title('微软天气与气象仪数据对比--平均风速')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    mse_max_temp = mean_squared_error(weather_wind_data_list, wind_mean_temps)
    mae_max_temp = mean_absolute_error(weather_wind_data_list, wind_mean_temps)

    print(f"均方误差: {mse_max_temp}")
    # print(f"当天最低温度--均方误差: {mse_min_temp}")
    print(f"平均绝对误差: {mae_max_temp}")
    # print(f"当天最低温度--平均绝对误差: {mae_min_temp}")


    # Calculate differences
    temp_differences = [weather_temp - met_temp for weather_temp, met_temp in zip(wind_mean_temps, weather_wind_data_list)]
    # Calculate standard deviation
    std_deviation = np.std(temp_differences)

    # Calculate mean
    mean_temp_diff = np.mean(temp_differences)

    # Calculate coefficient of variation
    coefficient_variation = (std_deviation / mean_temp_diff) * 100
    
    print(f"差值的平均值为: {mean_temp_diff}")
    print(f"差值的标准差为: {std_deviation}")
    print(f"差值的变异系数为: {coefficient_variation}%")

def run_check_met(): 
    input_file_path: str = r"C:\Users\Jayttle\Desktop\tianmeng_met.txt"
    met_data: List[Met] = read_time_series_data(input_file_path)
    met_data.sort(key=attrgetter('datetime_obj'))

    # Filter Met objects with wind_speed > 20
    filtered_met_data = [met for met in met_data if met.wind_speed > 30]

    # # Write filtered Met objects to a text file
    # output_file_path = r"C:\Users\Jayttle\Desktop\high_wind_speed_data.txt"
    # with open(output_file_path, 'w') as file:
    #     for met in filtered_met_data:
    #         file.write(f"{met.datetime_obj}\t{met.temperature}\t{met.humidness}\t{met.pressure}\t{met.wind_speed}\t{met.wind_direction}\n")

    # 创建一个极坐标图
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    
    # 遍历距离和方位角，绘制每个点
    for met in filtered_met_data:
        # 将方位角转换为弧度
        theta = np.deg2rad(met.wind_direction)
        # 绘制点
        ax.plot(theta, met.wind_speed, 'o', color='blue')
    
    # 设置极坐标图的标题
    ax.set_title(f'风速极坐标')
    
    # 显示图形
    plt.show()


def find_met_data_by_date(met_data: List[Met], dates: List[str]) -> List[Met]:
    filtered_met_data = []
    for met in met_data:
        for target_date in dates:
            if met.datetime_obj.date() == datetime.strptime(target_date, "%Y-%m-%d").date():
                filtered_met_data.append(met)
    return filtered_met_data


def plot_data_with_datetimes(value: List[float], datetimes:List[datetime], data_type: str):
    plt.figure(figsize=(6, 4))
    color='black'
    # 设置日期格式化器和日期刻度定位器
    date_fmt = mdates.DateFormatter("%H:00")  # 仅显示月-日-时
    date_locator = mdates.AutoDateLocator()  # 自动选择刻度间隔

    # 隐藏右边框和上边框
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.xlabel('2023年8月10日气象仪数据', fontproperties='SimSun', fontsize=10)
    plt.ylabel(f'{data_type}', fontproperties='SimSun', fontsize=10)

    plt.gca().xaxis.set_major_formatter(date_fmt)
    plt.gca().xaxis.set_major_locator(date_locator)

    # 设置刻度朝向内部，并调整刻度与坐标轴的距离
    ax.tick_params(axis='x', direction='in', pad=10)  # pad 参数用于控制刻度与坐标轴的距离
    ax.tick_params(axis='y', direction='in', pad=10)
    # 绘制折线图，根据阈值连接或不连接线段，并使用不同颜色
    prev_datetime = None
    prev_value = None
    prev_month = None
    for datetime, value in zip(datetimes, value):
        month = datetime.month
        if prev_datetime is not None:
            time_diff = datetime - prev_datetime
            if time_diff < timedelta(hours=1):  # 如果时间间隔小于阈值，则连接线段
                plt.plot([prev_datetime, datetime], [prev_value, value], linestyle='-', color=color)
            else:  # 否则不连接线段
                plt.plot([prev_datetime, datetime], [prev_value, value], linestyle='', color=color)
        prev_datetime = datetime
        prev_value = value
        prev_month = month

    # 调整底部边界向上移动一点
    plt.subplots_adjust(bottom=0.15)
    plt.show()
    # 显示图形

def plot_data_with_datetimes_interp1d(value: List[float], datetimes: List[datetime], color='blue'):
    plt.figure(figsize=(14.4, 9.6))

    # 线性插值函数
    interpolate_func = interp1d([mdates.date2num(dt) for dt in datetimes], value, kind='linear')

    # 创建新的日期范围，每分钟一个时间点
    min_datetime = min(datetimes)
    max_datetime = max(datetimes)
    new_datetimes = [min_datetime + timedelta(minutes=i) for i in range(int((max_datetime - min_datetime).total_seconds() / 60) + 1)]

    # 应用插值函数获取新的数值序列
    new_values = interpolate_func([mdates.date2num(dt) for dt in new_datetimes])

    # 设置日期格式化器和日期刻度定位器
    date_fmt = mdates.DateFormatter("%m-%d:%H")  # 仅显示月-日-时
    date_locator = mdates.AutoDateLocator()  # 自动选择刻度间隔
    plt.xlabel('日期')
    plt.ylabel('数值-百分比')
    plt.title('湿度')

    plt.gca().xaxis.set_major_formatter(date_fmt)
    plt.gca().xaxis.set_major_locator(date_locator)
    
    # 绘制插值后的数据
    plt.plot(new_datetimes, new_values, linestyle='-', color='green')
    # 绘制折线图，根据阈值连接或不连接线段，并使用不同颜色
    prev_datetime = None
    prev_value = None
    prev_month = None
    for datetime, value in zip(datetimes, value):
        month = datetime.month
        if prev_datetime is not None:
            time_diff = datetime - prev_datetime
            if time_diff < timedelta(hours=1):  # 如果时间间隔小于阈值，则连接线段
                plt.plot([prev_datetime, datetime], [prev_value, value], linestyle='-', color=color)
            else:  # 否则不连接线段
                plt.plot([prev_datetime, datetime], [prev_value, value], linestyle='', color=color)
        prev_datetime = datetime
        prev_value = value
        prev_month = month

    plt.show()

class GgkxData:
    def __init__(self, time: str, station_id: int, receiver_id: int, latitude: float, longitude: float, geo_height: float, fix_mode: int, satellite_num: int, pdop: float, sigma_e: float, sigma_n: float, sigma_u: float, prop_age: float):
        self.time = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
        self.station_id = station_id
        self.receiver_id = receiver_id
        self.latitude = latitude
        self.longitude = longitude
        self.geo_height = geo_height
        self.fix_mode = fix_mode
        self.satellite_num = satellite_num
        self.pdop = pdop
        self.sigma_e = sigma_e
        self.sigma_n = sigma_n
        self.sigma_u = sigma_u
        self.prop_age = prop_age

    @staticmethod
    @CommonDecorator.log_function_call
    def read_ggkx_data(file_path: str) -> List['GgkxData']:
        """
        读取效率: 4150506  30s 一周
        138350个/s 
        """
        ggkx_data_list: List['GgkxData'] = []
        with open(file_path, 'r') as file:
            next(file)  # Skip the header row
            for line in file:
                data = line.strip().split('\t')
                # 数据类型转换
                data[0] = str(data[0])
                data[1] = int(data[1])
                data[2] = int(data[2])
                data[3] = float(data[3])
                data[4] = float(data[4])
                data[5] = float(data[5])
                data[6] = int(data[6])
                data[7] = int(data[7])
                data[8] = float(data[8])
                data[9] = float(data[9])
                data[10] = float(data[10])
                data[11] = float(data[11])
                data[12] = float(data[12])
                ggkx_data_list.append(GgkxData(*data))
        return ggkx_data_list
    
    @classmethod
    def filter_by_date(cls, ggkx_data_list: List['GgkxData'], date: str) -> List['GgkxData']:
        filtered_data: List['GgkxData'] = [data for data in ggkx_data_list if data.time.strftime('%Y-%m-%d') == date]
        return filtered_data
    

    @classmethod
    def filter_data_to_files(cls, ggkx_data_list: List['GgkxData'], output_folder: str = r"C:\Users\Jayttle\Desktop\output_data") -> None:
        # 创建一个 defaultdict 以便按照 station_id 和 receiver_id 进行分类
        classified_data = defaultdict(list)
        for data in ggkx_data_list:
            key = (data.station_id, data.receiver_id)
            classified_data[key].append(data)

        # 确保输出文件夹存在
        os.makedirs(output_folder, exist_ok=True)

        # 将分类后的数据写入到文件
        for key, data_list in classified_data.items():
            station_id, receiver_id = key
            output_file = os.path.join(output_folder, f"ggkx_R0{station_id}{receiver_id}.txt")
            with open(output_file, 'w') as file:
                file.write("Time\tStationID\tReceiverID\tLat\tLon\tGeoHeight\tFixMode\tSateNum\tPDOP\tSigmaE\tSigmaN\tSigmaU\tPropAge\n")
                for data in data_list:
                    file.write(f"{data.time}\t{data.station_id}\t{data.receiver_id}\t{data.latitude}\t{data.longitude}\t{data.geo_height}\t{data.fix_mode}\t{data.satellite_num}\t{data.pdop}\t{data.sigma_e}\t{data.sigma_n}\t{data.sigma_u}\t{data.prop_age}\n")

    @classmethod
    def filter_data_to_files_in_specified_date(cls, ggkx_data_list: List['GgkxData'], output_folder: str, date: str) -> None:
        specified_date_data = [data for data in ggkx_data_list if data.time.date() == datetime.strptime(date, '%Y-%m-%d').date()]

        if not specified_date_data:
            print("Error: No data found for the specified date.")
            return

        classified_data = defaultdict(list)
        for data in specified_date_data:
            key = (data.station_id, data.receiver_id)
            classified_data[key].append(data)

        os.makedirs(output_folder, exist_ok=True)

        for key, data_list in classified_data.items():
            station_id, receiver_id = key
            output_file = os.path.join(output_folder, f"ggkx_R0{station_id}{receiver_id}_{date}.txt")
            with open(output_file, 'w') as file:
                file.write("Time\tStationID\tReceiverID\tLat\tLon\tGeoHeight\tFixMode\tSateNum\tPDOP\tSigmaE\tSigmaN\tSigmaU\tPropAge\n")
                for data in data_list:
                    file.write(f"{data.time}\t{data.station_id}\t{data.receiver_id}\t{data.latitude}\t{data.longitude}\t{data.geo_height}\t{data.fix_mode}\t{data.satellite_num}\t{data.pdop}\t{data.sigma_e}\t{data.sigma_n}\t{data.sigma_u}\t{data.prop_age}\n")

    @classmethod
    def convert_to_coordinates(cls, ggkx_data_list: List['GgkxData']) -> List['GgkxDataWithCoordinates']:
        # 提取纬度和经度到列表中
        latitudes = [data.latitude for data in ggkx_data_list]
        longitudes = [data.longitude for data in ggkx_data_list]
        
        # 批量进行坐标转换
        east_coords = []
        north_coords = []
            
        for lat, lon in zip(latitudes, longitudes):
            # 对纬度和经度进行坐标转换
            east_coord, north_coord = TBCProcessCsv.convert_coordinates(lat, lon)
            east_coords.append(east_coord)
            north_coords.append(north_coord)

        ggkx_data_with_coords_list = []
        for i, data in enumerate(ggkx_data_list):
            ggkx_data_with_coords = GgkxDataWithCoordinates(
                time=data.time.strftime('%Y-%m-%d %H:%M:%S'),
                station_id=data.station_id,
                receiver_id=data.receiver_id,
                east_coordinate=east_coords[i],
                north_coordinate=north_coords[i],
                geo_height=data.geo_height,
                fix_mode=data.fix_mode,
                satellite_num=data.satellite_num,
                pdop=data.pdop,
                sigma_e=data.sigma_e,
                sigma_n=data.sigma_n,
                sigma_u=data.sigma_u,
                prop_age=data.prop_age
            )
            ggkx_data_with_coords_list.append(ggkx_data_with_coords)
        return ggkx_data_with_coords_list
    
    @classmethod
    def convert_to_coordinates_in_specified_date(cls, ggkx_data_list: List['GgkxData'], specified_date: str) -> List['GgkxDataWithCoordinates']:
        specified_date_data = [data for data in ggkx_data_list if data.time.date() == datetime.strptime(specified_date, '%Y-%m-%d').date()]
        if not specified_date_data:
            print("Error: Specified date not found in the data.")
            return []

        east_coords = []
        north_coords = []
        
        for data in specified_date_data:
            # 对纬度和经度进行坐标转换
            east_coord, north_coord = TBCProcessCsv.convert_coordinates(data.latitude, data.longitude)
            east_coords.append(east_coord)
            north_coords.append(north_coord)

        ggkx_data_with_coords_list = []
        for data, east, north in zip(specified_date_data, east_coords, north_coords):
            ggkx_data_with_coords = GgkxDataWithCoordinates(
                time=data.time.strftime('%Y-%m-%d %H:%M:%S'),
                station_id=data.station_id,
                receiver_id=data.receiver_id,
                east_coordinate=east,
                north_coordinate=north,
                geo_height=data.geo_height,
                fix_mode=data.fix_mode,
                satellite_num=data.satellite_num,
                pdop=data.pdop,
                sigma_e=data.sigma_e,
                sigma_n=data.sigma_n,
                sigma_u=data.sigma_u,
                prop_age=data.prop_age
            )
            ggkx_data_with_coords_list.append(ggkx_data_with_coords)
        return ggkx_data_with_coords_list

class GgkxDataWithCoordinates:
    def __init__(self, time: str, station_id: int, receiver_id: int, east_coordinate: float, north_coordinate: float, geo_height: float, fix_mode: int, satellite_num: int, pdop: float, sigma_e: float, sigma_n: float, sigma_u: float, prop_age: float):
        try:
            self.time = datetime.strptime(time, '%Y-%m-%d %H:%M:%S.%f')  # 尝试包含毫秒部分的格式
        except ValueError:
            self.time = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')  # 失败时尝试不包含毫秒部分的格式
        self.station_id = station_id
        self.receiver_id = receiver_id
        self.east_coordinate = east_coordinate
        self.north_coordinate = north_coordinate
        self.geo_height = geo_height
        self.fix_mode = fix_mode
        self.satellite_num = satellite_num
        self.pdop = pdop
        self.sigma_e = sigma_e
        self.sigma_n = sigma_n
        self.sigma_u = sigma_u
        self.prop_age = prop_age

    @staticmethod
    def read_ggkx_data_with_coordinates(file_path: str) -> List['GgkxDataWithCoordinates']:
        ggkx_data_with_coords_list: List['GgkxDataWithCoordinates'] = []
        with open(file_path, 'r') as file:
            next(file)  # Skip the header row
            for line in file:
                data = line.strip().split('\t')
                data[1] = int(data[1])  # Station ID
                data[2] = int(data[2])  # Receiver ID
                data[3] = float(data[3])  # East coordinate
                data[4] = float(data[4])  # North coordinate
                data[5] = float(data[5])  # Geo height
                data[6] = int(data[6])  # Fix mode
                data[7] = int(data[7])  # Satellite number
                data[8] = float(data[8])  # PDOP
                data[9] = float(data[9])  # Sigma e
                data[10] = float(data[10])  # Sigma n
                data[11] = float(data[11])  # Sigma u
                data[12] = float(data[12])  # Prop age

                ggkx_data_with_coords_list.append(GgkxDataWithCoordinates(*data))
        return ggkx_data_with_coords_list

    @staticmethod
    def read_ggkx_data_in_marker_name(file_path: str, marker_name: str) -> List['GgkxDataWithCoordinates']:
        locations = {
            "B011": (35.474852, 118.072091),
            "B021": (35.465000, 118.035312),
            "R031": (35.473642676796, 118.054358431073),
            "R032": (35.473666223407, 118.054360237421),
            "R051": (35.473944469154, 118.048584306326),
            "R052": (35.473974942138, 118.048586858521),
            "R071": (35.474177696631, 118.044201562812),
            "R072": (35.474204534806, 118.044203691212),
            "R081": (35.474245973695, 118.042930824340),
            "R082": (35.474269576552, 118.042932741649)
        }

        # 定义一个字典来存储每个位置对应的东北坐标
        east_north_coordinates = {}
        # 批量转换经纬度坐标为东北坐标
        for location, (lat, lon) in locations.items():
            easting, northing = TBCProcessCsv.convert_coordinates(lat, lon)
            east_north_coordinates[location] = (easting, northing)     

        ggkx_data_with_coords_list: List['GgkxDataWithCoordinates'] = []
        east_marker, north_marker = east_north_coordinates[marker_name]
        with open(file_path, 'r') as file:
            next(file)  # Skip the header row
            for line in file:
                data = line.strip().split('\t')
                data[1] = int(data[1])  # Station ID
                data[2] = int(data[2])  # Receiver ID
                data[3] = (float(data[3]) - east_marker) * 1000  # East coordinate
                data[4] = (float(data[4]) - north_marker) * 1000 # North coordinate
                data[5] = float(data[5])  # Geo height
                data[6] = int(data[6])  # Fix mode
                data[7] = int(data[7])  # Satellite number
                data[8] = float(data[8])  # PDOP
                data[9] = float(data[9])  # Sigma e
                data[10] = float(data[10])  # Sigma n
                data[11] = float(data[11])  # Sigma u
                data[12] = float(data[12])  # Prop age

                ggkx_data_with_coords_list.append(GgkxDataWithCoordinates(*data))
        return ggkx_data_with_coords_list
    
    @classmethod
    def filter_by_date_with_coordinates(cls, ggkx_data_list: List['GgkxDataWithCoordinates'], date: str) -> List['GgkxDataWithCoordinates']:
        filtered_data: List['GgkxDataWithCoordinates'] = [data for data in ggkx_data_list if data.time.strftime('%Y-%m-%d') == date]
        return filtered_data

    @classmethod
    def filter_data_to_files_with_coordinates(cls, ggkx_data_list: List['GgkxDataWithCoordinates'], output_folder: str, title: str) -> None:
        # 创建一个 defaultdict 以便按照 station_id 和 receiver_id 进行分类
        classified_data = defaultdict(list)
        for data in ggkx_data_list:
            key = (data.station_id, data.receiver_id)
            classified_data[key].append(data)

        # 确保输出文件夹存在
        os.makedirs(output_folder, exist_ok=True)

        # 将分类后的数据写入到文件
        for key, data_list in classified_data.items():
            station_id, receiver_id = key
            output_file = os.path.join(output_folder, f"ggkx_R0{station_id}{receiver_id}_{title}.txt")
            with open(output_file, 'w') as file:
                file.write("Time\tStationID\tReceiverID\tEastCoord\tNorthCoord\tGeoHeight\tFixMode\tSateNum\tPDOP\tSigmaE\tSigmaN\tSigmaU\tPropAge\n")
                for data in data_list:
                    file.write(f"{data.time}\t{data.station_id}\t{data.receiver_id}\t{data.east_coordinate}\t{data.north_coordinate}\t{data.geo_height}\t{data.fix_mode}\t{data.satellite_num}\t{data.pdop}\t{data.sigma_e}\t{data.sigma_n}\t{data.sigma_u}\t{data.prop_age}\n")

    @classmethod
    def plot_GgkxData_coordinates(cls, ggkx_data_list: List['GgkxDataWithCoordinates'], title: str = None):  
        # 提取 east 和 north 坐标数据以及时间数据
        east_data = [data.east_coordinate for data in ggkx_data_list]
        north_data = [data.north_coordinate for data in ggkx_data_list]
        time_list = [data.time for data in ggkx_data_list]
        
        # 创建三个子图
        plt.figure(figsize=(12, 9))

        # 设置日期格式化器和日期刻度定位器
        date_fmt = mdates.DateFormatter("%H:%M")  # 仅显示月-日 时
        date_locator = mdates.AutoDateLocator()  # 自动选择刻度间隔

        # 绘制 east 坐标时序图
        plt.subplot(3, 1, 1)
        plt.plot(time_list, east_data, color='blue')
        plt.title(f'{title}东坐标')
        plt.xlabel('日期')
        plt.ylabel('东坐标')
        plt.gca().xaxis.set_major_formatter(date_fmt)
        plt.gca().xaxis.set_major_locator(date_locator)

        # 绘制 north 坐标时序图
        plt.subplot(3, 1, 2)
        plt.plot(time_list, north_data, color='green')
        plt.title(f'{title}北坐标')
        plt.xlabel('日期')
        plt.ylabel('北坐标')
        plt.gca().xaxis.set_major_formatter(date_fmt)
        plt.gca().xaxis.set_major_locator(date_locator)

        # 计算每个点与原点之间的距离
        distances = [calculate_distance(0, 0, east, north) for east, north in zip(east_data, north_data)]
        
        # 绘制距离随时间的变化图
        plt.subplot(3, 1, 3)
        plt.plot(time_list, distances, color='red')
        plt.title(f'{title}距离随时间变化')
        plt.xlabel('日期')
        plt.ylabel('距离')
        plt.gca().xaxis.set_major_formatter(date_fmt)
        plt.gca().xaxis.set_major_locator(date_locator)

        plt.tight_layout()  # 调整子图布局以防止重叠
        if title is not None:
            plt.savefig(f'D:\Program Files (x86)\Software\OneDrive\PyPackages_img\ggkx\{title}.png')
        plt.close()
            

    @classmethod
    def plot_GgkxData_coordinates_in_specified_date(cls, ggkx_data_list: List['GgkxDataWithCoordinates'], specified_date: str):  
        # 提取 east 和 north 坐标数据以及时间数据
        east_data = [data.east_coordinate for data in ggkx_data_list]
        north_data = [data.north_coordinate for data in ggkx_data_list]
        time_list = [data.time for data in ggkx_data_list]
        
        # Check if the specified date exists in the data
        specified_date_exists = False
        specified_date_data_indices = []
        for idx, time in enumerate(time_list):
            if time.date() == datetime.strptime(specified_date, '%Y-%m-%d').date():
                specified_date_exists = True
                specified_date_data_indices.append(idx)
        
        # If the specified date does not exist, print an error message and return
        if not specified_date_exists:
            print("Error: Specified date not found in the data.")
            return
        
        # Print out the available dates in the data
        available_dates = set(time.date() for time in time_list)
        print("Available dates in the data:", available_dates)
        
        # 创建三个子图
        plt.figure(figsize=(12, 9))

        # 设置日期格式化器和日期刻度定位器
        date_fmt = mdates.DateFormatter("%m-%d-%H")  # 仅显示月-日-时
        date_locator = mdates.AutoDateLocator()  # 自动选择刻度间隔

        # 绘制 east 坐标时序图
        plt.subplot(3, 1, 1)
        plt.plot([time_list[i] for i in specified_date_data_indices], [east_data[i] for i in specified_date_data_indices], color='blue')
        plt.title('东坐标')
        plt.xlabel('日期')
        plt.ylabel('东坐标')
        plt.gca().xaxis.set_major_formatter(date_fmt)
        plt.gca().xaxis.set_major_locator(date_locator)

        # 绘制 north 坐标时序图
        plt.subplot(3, 1, 2)
        plt.plot([time_list[i] for i in specified_date_data_indices], [north_data[i] for i in specified_date_data_indices], color='green')
        plt.title('北坐标')
        plt.xlabel('日期')
        plt.ylabel('北坐标')
        plt.gca().xaxis.set_major_formatter(date_fmt)
        plt.gca().xaxis.set_major_locator(date_locator)

        # 计算每个点与原点之间的距离
        distances = [calculate_distance(0, 0, east, north) for east, north in zip(east_data, north_data)]
        
        # 绘制距离随时间的变化图
        plt.subplot(3, 1, 3)
        plt.plot([time_list[i] for i in specified_date_data_indices], [distances[i] for i in specified_date_data_indices], color='red')
        plt.title('距离随时间变化')
        plt.xlabel('日期')
        plt.ylabel('距离')
        plt.gca().xaxis.set_major_formatter(date_fmt)
        plt.gca().xaxis.set_major_locator(date_locator)

        plt.tight_layout()  # 调整子图布局以防止重叠
        plt.show()


def calculate_distance(x1, y1, x2, y2):
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

def calculate_bearing(x, y):
    # 计算角度（弧度）
    angle_radians = math.atan2(y, x)
    # 将弧度转换为角度
    angle_degrees = math.degrees(angle_radians)
    # 确保角度在 [0, 360) 范围内
    bearing = (angle_degrees + 360) % 360
    return bearing

    
class TiltmeterDataAvage:
    def __init__(self,pitch: float, roll: float, num: int):
        self.pitch = pitch
        self.roll = roll
        self.num = num
    
    def read_tiltmeter_data(file_path: str) -> Dict[datetime.date, 'TiltmeterDataAvage']:
        tiltmeter_data = {}

        with open(file_path, 'r') as f:
            lines = f.readlines()

            for line in lines:
                if line.strip():  # Ensure line is not empty
                    if line.startswith('日期: '):  # Skip headers or any other specific formatting
                        continue

                    parts = line.strip().split('\t')
                    if len(parts) == 4:
                        try:
                            date_obj = datetime.strptime(parts[0], '%Y-%m-%d').date()
                            pitch = float(parts[1])
                            roll = float(parts[2])
                            num = int(parts[3])
                            tiltmeter_data[date_obj] = TiltmeterDataAvage(pitch, roll, num)
                        except ValueError:
                            print(f"Issue parsing line: {line}")
        return tiltmeter_data
    
    def find_common_dates(dict1: Dict[datetime.date, 'TiltmeterDataAvage'], dict2: Dict[datetime.date, 'TiltmeterDataAvage']) -> Dict[datetime.date, Tuple['TiltmeterDataAvage', 'TiltmeterDataAvage']]:
        common_dates = {}
        
        for date_obj in dict1:
            if date_obj in dict2:
                common_dates[date_obj] = (dict1[date_obj], dict2[date_obj])
        
        return common_dates

class DataPoint:
    def __init__(self, point_id: str, north_coordinate: float, east_coordinate: float, elevation: float,
                 latitude: float, longitude: float, horizontal_quality: str, vertical_quality: str,
                 start_point_id: str, end_point_id: str, start_time: str, end_time: str, duration: str,
                 pdop: float, rms: float, horizontal_accuracy: float, vertical_accuracy: float,
                 satellites: int, measurement_interval: int, vector_length: float, x_increment: float,
                 y_increment: float, z_increment: float, north_error: float, east_error: float,
                 elevation_error: float, height_error: float, x_error: float, y_error: float,
                 z_error: float):
        self.point_id = point_id
        self.north_coordinate = north_coordinate
        self.east_coordinate = east_coordinate
        self.elevation = elevation
        self.latitude = latitude
        self.longitude = longitude
        self.horizontal_quality = horizontal_quality
        self.vertical_quality = vertical_quality
        self.start_point_id = start_point_id
        self.end_point_id = end_point_id
        self.start_time = datetime.strptime(start_time, '%Y/%m/%d %H:%M:%S')
        self.end_time = datetime.strptime(end_time, '%Y/%m/%d %H:%M:%S')
        self.duration = duration
        self.pdop = pdop
        self.rms = rms
        self.horizontal_accuracy = horizontal_accuracy
        self.vertical_accuracy = vertical_accuracy
        self.satellites = satellites
        self.measurement_interval = measurement_interval
        self.vector_length = vector_length
        self.x_increment = x_increment
        self.y_increment = y_increment
        self.z_increment = z_increment
        self.north_error = north_error
        self.east_error = east_error
        self.elevation_error = elevation_error
        self.height_error = height_error
        self.x_error = x_error
        self.y_error = y_error
        self.z_error = z_error
            
    @staticmethod
    def read_data_points(file_path: str) -> List['DataPoint']:
        data_points: List['DataPoint'] = []
        # 使用 chardet 检测文件编码
        with open(file_path, 'rb') as file:
            result = chardet.detect(file.read())
            encoding = result['encoding']
        # 使用检测到的编码打开文件
        with open(file_path, 'r', encoding=encoding) as file:
            next(file)  # Skip the header row
            line = next(file)  # Read the second line
            data = line.strip().split('\t')
            point_id = data[0]
            north_coordinate = float(data[1])
            east_coordinate = float(data[2])
            elevation = float(data[3])
            latitude = float(data[4])
            longitude = float(data[5])
            horizontal_quality = data[6]
            vertical_quality = data[7]
            start_point_id = data[8]
            end_point_id = data[9]
            start_time = data[10]
            end_time = data[11]
            duration = data[12]
            pdop = float(data[13])
            rms = float(data[14])
            horizontal_accuracy = float(data[15])
            vertical_accuracy = float(data[16])
            satellites = int(data[17])
            measurement_interval = int(data[18])
            vector_length = float(data[19])
            x_increment = float(data[20])
            y_increment = float(data[21])
            z_increment = float(data[22])
            north_error = float(data[23])
            east_error = float(data[24])
            elevation_error = float(data[25])
            height_error = float(data[26])
            x_error = float(data[27])
            y_error = float(data[28])
            z_error = float(data[29])

            data_point = DataPoint(point_id, north_coordinate, east_coordinate, elevation,
                                   latitude, longitude, horizontal_quality, vertical_quality,
                                   start_point_id, end_point_id, start_time, end_time,
                                   duration, pdop, rms, horizontal_accuracy, vertical_accuracy,
                                   satellites, measurement_interval, vector_length,
                                   x_increment, y_increment, z_increment, north_error,
                                   east_error, elevation_error, height_error, x_error,
                                   y_error, z_error)
            data_points.append(data_point)
        return data_points


def check_data_points(folder_path: str):
    to_delete_list = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            csv_file_path = os.path.join(folder_path, filename)
            # 使用 chardet 检测文件编码
            with open(csv_file_path, 'rb') as file:
                result = chardet.detect(file.read())
                encoding = result['encoding']
            # 使用检测到的编码打开文件
            with open(csv_file_path, 'r', encoding=encoding) as file:
                next(file)  # Skip the header row
                line = next(file)  # Read the second line
                data = line.strip().split('\t')
                if len(data) != 30:
                    print(f"Error:{csv_file_path} got:{len(data)}.")
                    to_delete_list.append(csv_file_path)
    for item in to_delete_list:
        os.remove(item)


def load_csv_data(folder_path: str) -> List['DataPoint']:
    """
    Load CSV data from the specified folder.

    Args:
        folder_path (str): The path to the folder containing CSV files.

    Returns:
        List[DataPoint]: A list of data points loaded from the CSV files.
    """
    data_points = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            csv_file_path = os.path.join(folder_path, filename)
            data_points.extend(DataPoint.read_data_points(csv_file_path))
    return data_points


def load_DataPoints_return_dict() -> dict[str, list[DataPoint]]:
    data_save_path = {
        'R031_0407': r'D:\Program Files (x86)\Software\OneDrive\PyPackages_DataSave\new_data\R031_0407',
        'R032_0407': r'D:\Program Files (x86)\Software\OneDrive\PyPackages_DataSave\new_data\R032_0407',
        'R051_0407': r'D:\Program Files (x86)\Software\OneDrive\PyPackages_DataSave\new_data\R051_0407',
        'R052_0407': r'D:\Program Files (x86)\Software\OneDrive\PyPackages_DataSave\new_data\R052_0407',
        'R071_0407': r'D:\Program Files (x86)\Software\OneDrive\PyPackages_DataSave\new_data\R071_0407',
        'R072_0407': r'D:\Program Files (x86)\Software\OneDrive\PyPackages_DataSave\new_data\R072_0407',
        'R081_0407': r'D:\Program Files (x86)\Software\OneDrive\PyPackages_DataSave\new_data\R081_0407',
        'R082_0407': r'D:\Program Files (x86)\Software\OneDrive\PyPackages_DataSave\new_data\R082_0407',

        'R031_1215': r'D:\Program Files (x86)\Software\OneDrive\PyPackages_DataSave\new_data\R031_1215',
        'R032_1215': r'D:\Program Files (x86)\Software\OneDrive\PyPackages_DataSave\new_data\R032_1215',
        'R051_1215': r'D:\Program Files (x86)\Software\OneDrive\PyPackages_DataSave\new_data\R051_1215',
        'R052_1215': r'D:\Program Files (x86)\Software\OneDrive\PyPackages_DataSave\new_data\R052_1215',
        'R071_1215': r'D:\Program Files (x86)\Software\OneDrive\PyPackages_DataSave\new_data\R071_1215',
        'R072_1215': r'D:\Program Files (x86)\Software\OneDrive\PyPackages_DataSave\new_data\R072_1215',
        'R081_1215': r'D:\Program Files (x86)\Software\OneDrive\PyPackages_DataSave\new_data\R081_1215',
        'R082_1215': r'D:\Program Files (x86)\Software\OneDrive\PyPackages_DataSave\new_data\R082_1215',
    }
    # data_save_path = {
    #     'R031_0407': r'D:\Program Files (x86)\Software\OneDrive\PyPackages_DataSave\new_data\R031_0407',
    #     'R031_1215': r'D:\Program Files (x86)\Software\OneDrive\PyPackages_DataSave\new_data\R031_1215',
    #     'R032_1215': r'D:\Program Files (x86)\Software\OneDrive\PyPackages_DataSave\new_data\R032_1215',
    #     'R051_1215': r'D:\Program Files (x86)\Software\OneDrive\PyPackages_DataSave\new_data\R051_1215',
    #     'R052_1215': r'D:\Program Files (x86)\Software\OneDrive\PyPackages_DataSave\new_data\R052_1215',
    #     'R071_1215': r'D:\Program Files (x86)\Software\OneDrive\PyPackages_DataSave\new_data\R071_1215',
    #     'R072_1215': r'D:\Program Files (x86)\Software\OneDrive\PyPackages_DataSave\new_data\R072_1215',
    #     'R081_1215': r'D:\Program Files (x86)\Software\OneDrive\PyPackages_DataSave\new_data\R081_1215',
    #     'R082_1215': r'D:\Program Files (x86)\Software\OneDrive\PyPackages_DataSave\new_data\R082_1215',
    # }
    sorted_keys = sorted(data_save_path.keys())
    Receiver_DataPoints = {}

    for key in sorted_keys:
        print(f"{key}: {data_save_path[key]}")
        Receiver_DataPoints[key] = load_csv_data(data_save_path[key])
        print(f"num:{len(Receiver_DataPoints[key])}")
    return Receiver_DataPoints

def DataPoint_transform_return_xy(Receiver_DataPoints: dict[str, list[DataPoint]]) -> Tuple[dict[str, List[float]], dict[str, List[float]]]:
    x_DataPoints = {}
    y_DataPoints = {}
    
    for key in Receiver_DataPoints:
        x_DataPoints[key] = []
        y_DataPoints[key] = []
        
        for item in Receiver_DataPoints[key]:
            x, y = ropeway_transform_coordinates(item.east_coordinate, item.north_coordinate)
            x_DataPoints[key].append(x)
            y_DataPoints[key].append(y)
    
    return x_DataPoints, y_DataPoints

def transform_ListDataPoint_to_xy(data_points: List[DataPoint]) -> Tuple[List[float], List[float]]:
    x_coordinates = []
    y_coordinates = []
    
    for item in data_points:
        x, y = ropeway_transform_coordinates(item.east_coordinate, item.north_coordinate)
        x_coordinates.append(x)
        y_coordinates.append(y)
    
    return x_coordinates, y_coordinates


def filter_close_coordinates(x_coords: List[float], y_coords: List[float], threshold: float = 0.1) -> Tuple[List[float], List[float]]:
    assert len(x_coords) == len(y_coords), "x_coords and y_coords must have the same length"
    
    filtered_x_coords = []
    filtered_y_coords = []
    
    n = len(x_coords)
    used = [False] * n
    for i in range(n):
        if not used[i] and abs(x_coords[i] - y_coords[i]) >= threshold:
            used[i] = True
    
    for i in range(n):
        if  used[i]:
            filtered_x_coords.append(x_coords[i])
            filtered_y_coords.append(y_coords[i])

    return filtered_x_coords, filtered_y_coords


def transform_coordinates(x1, y1, x2, y2, x, y):
    # Calculate vector AB
    ABx = x2 - x1
    ABy = y2 - y1
    
    # Calculate the magnitude of AB
    magnitude_AB = math.sqrt(ABx**2 + ABy**2)
    
    # Unit vector along AB
    if magnitude_AB != 0:
        AB_unit_x = ABx / magnitude_AB
        AB_unit_y = ABy / magnitude_AB
    else:
        raise ValueError("Points (x1, y1) and (x2, y2) are the same, cannot determine a valid axis.")
    
    # Calculate vector AC
    ACx = x - x1
    ACy = y - y1
    
    # Coordinate transformation
    x_prime = ACx * AB_unit_x + ACy * AB_unit_y
    y_prime = -ACx * AB_unit_y + ACy * AB_unit_x
    
    return x_prime, y_prime


def ropeway_transform_coordinates(x,y):
    R031_east, R031_north= 595694.317320648, 3927652.154545162
    R032_east, R032_north= 595694.4533681379, 3927654.7689213157

    R081_east, R081_north= 594656.3945218798, 3927708.0756872687
    R082_east, R082_north= 594656.540875324, 3927710.696388756
    x_prime, y_prime = transform_coordinates(R081_east, R081_north,  R031_east, R031_north, x, y)
    return x_prime, y_prime




def load_DataPoints() -> dict[str, list[DataPoint]]:
    locations = {
        "B011": (35.474852, 118.072091),
        "B021": (35.465000, 118.035312),
        "R031": (35.473642676796, 118.054358431073),
        "R032": (35.473666223407, 118.054360237421),
        "R051": (35.473944469154, 118.048584306326),
        "R052": (35.473974942138, 118.048586858521),
        "R071": (35.474177696631, 118.044201562812),
        "R072": (35.474204534806, 118.044203691212),
        "R081": (35.474245973695, 118.042930824340),
        "R082": (35.474269576552, 118.042932741649)
    }

    # 定义一个字典来存储每个位置对应的东北坐标
    east_north_coordinates = {}
    # 批量转换经纬度坐标为东北坐标
    for location, (lat, lon) in locations.items():
        easting, northing = TBCProcessCsv.convert_coordinates(lat, lon)
        east_north_coordinates[location] = (easting, northing)


    Receiver_DataPoints = load_DataPoints_return_dict()
    grouped_DataPoints = {}  # 存储组合后的数据点
    for key, DataPoints in Receiver_DataPoints.items():
        prefix = key[:9]  # 假设按前缀进行分组
        marker_name = key[:4] 
        
        if prefix not in grouped_DataPoints:
            grouped_DataPoints[prefix] = []

        # Subtract easting and northing from DataPoints
        for datapoint in DataPoints:
            easting, northing = east_north_coordinates[marker_name]
            datapoint.east_coordinate -= easting
            datapoint.north_coordinate -= northing
            datapoint.east_coordinate *= 1000
            datapoint.north_coordinate *= 1000
            
        grouped_DataPoints[prefix].extend(DataPoints)  # 将数据点添加到相应的列表中

    # 打印结果
    for i, (prefix, DataPoints) in enumerate(grouped_DataPoints.items()):
        print(f"索引: {i}, 前缀: {prefix}, 数据点数: {len(DataPoints)}")

    return grouped_DataPoints

def load_DataPoints_in_ropeway() -> Tuple[dict[str, List[float]], dict[str, List[float]], dict[str, list[DataPoint]]]:
    locations = {
        "B011": (35.474852, 118.072091),
        "B021": (35.465000, 118.035312),
        "R031": (35.473642676796, 118.054358431073),
        "R032": (35.473666223407, 118.054360237421),
        "R051": (35.473944469154, 118.048584306326),
        "R052": (35.473974942138, 118.048586858521),
        "R071": (35.474177696631, 118.044201562812),
        "R072": (35.474204534806, 118.044203691212),
        "R081": (35.474245973695, 118.042930824340),
        "R082": (35.474269576552, 118.042932741649)
    }
    locations_ropeway ={}
    # 定义一个字典来存储每个位置对应的东北坐标
    east_north_coordinates = {}
    # 批量转换经纬度坐标为东北坐标
    for location, (lat, lon) in locations.items():
        easting, northing = TBCProcessCsv.convert_coordinates(lat, lon)
        east_north_coordinates[location] = (easting, northing)
        x_ropeway, y_ropeway = ropeway_transform_coordinates(easting, northing)
        locations_ropeway[location] = (x_ropeway, y_ropeway)


    Receiver_DataPoints = load_DataPoints_return_dict()
    x_DataPoints, y_DataPoints=DataPoint_transform_return_xy(Receiver_DataPoints)

    # for key in x_DataPoints.keys():
    #     x_DataPoints[key] = [(value - locations_ropeway[key[:4]][0]) * 1000 for value in x_DataPoints[key]]
    #     y_DataPoints[key] = [(value - locations_ropeway[key[:4]][1]) * 1000 for value in y_DataPoints[key]]
    
    return x_DataPoints, y_DataPoints, Receiver_DataPoints


def match_DataPoints_in_ropeway(Receiver_DataPoints: Dict[str, List[DataPoint]], marker_name1: str, marker_name2: str) -> Tuple[List[DataPoint], List[DataPoint]]:
    matched_list1: List[DataPoint] = []
    matched_list2: List[DataPoint] = []
    start_dates_marker1: Dict[datetime.date, List[DataPoint]] = {}
    
    try:
        # Populate start_dates_marker1 from Receiver_DataPoints[marker_name1]
        for dp in Receiver_DataPoints[marker_name1]:
            dp_date = dp.start_time.date()
            if dp_date not in start_dates_marker1:
                start_dates_marker1[dp_date] = []
            start_dates_marker1[dp_date].append(dp)
    except KeyError:
        pass  # Handle the case where marker_name1 is not found
    
    # Iterate over data points in marker_name2 and check if the start_date exists in start_dates_marker1
    for dp in Receiver_DataPoints[marker_name2]:
        dp_date = dp.start_time.date()
        if dp_date in start_dates_marker1:
            matched_list1.extend(start_dates_marker1[dp_date])
            matched_list2.append(dp)
    
    return matched_list1, matched_list2

def plot_points(x_points, y_points, title=None):
    # Convert lists to numpy arrays
    y_points = np.array(y_points)
    
    plt.figure(figsize=(9, 6))  # 设置图的大小，单位是英寸
    
    # 绘制散点图
    plt.scatter(x_points, y_points)
    plt.xlabel('索道坐标X(mm)')  # 设置x轴标签
    plt.ylabel('索道坐标Y(mm)')  # 设置y轴标签
    
    # 隐藏右边框和上边框
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)  # 隐藏下边框
    ax.spines['left'].set_visible(False)    # 隐藏左边框
    
    # 添加箭头，指向坐标轴的正方向
    ax.annotate('', xy=(1, 0.001), xytext=(0, 0.001),
                arrowprops=dict(arrowstyle='->', lw=1, color='black'),
                xycoords='axes fraction', textcoords='axes fraction')
    ax.annotate('', xy=(0.001, 1), xytext=(0.001, 0),
                arrowprops=dict(arrowstyle='->', lw=1, color='black'),
                xycoords='axes fraction', textcoords='axes fraction')
    
    # 设置刻度朝向内部，并调整刻度与坐标轴的距离
    ax.tick_params(axis='x', direction='in', pad=10)  # pad 参数用于控制刻度与坐标轴的距离
    ax.tick_params(axis='y', direction='in', pad=10)
    # 调整底部边界向上移动一点
    plt.subplots_adjust(bottom=0.15)
    plt.subplots_adjust(left=0.15)


    # 设置标题
    if title is None:
        plt.suptitle('时间序列数据', y=0.0)  # 设置标题，默认为'时间序列数据'，y参数用于调整标题在垂直方向上的位置
    else:
        plt.suptitle(title, y=0.0)  # 设置标题为传入的参数title，y参数用于调整标题在垂直方向上的位置
    
    plt.tight_layout()  # 自动调整子图间的间距和标签位置
    plt.show()


def plot_ListFloat_x(ListFloat: list[float], isShow: bool = True, SaveFilePath: Optional[str] = None, title: str = None) -> None:
    """绘制 ListFloat 对象的时间序列图"""
    values = ListFloat
    datetimes = range(len(ListFloat))  # 使用数据长度生成简单的序号作为 x 轴

    plt.figure(figsize=(6, 4))  # 设置图的大小，单位是英寸

    # 隐藏右边框和上边框
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xlabel('数据期数/期', fontproperties='SimSun', fontsize=12)  # 设置字体为宋体
    plt.ylabel('索道坐标x轴方向位移监测/m', fontproperties='SimSun', fontsize=12)  # 设置字体为宋体

    # 设置最大显示的刻度数
    plt.tight_layout()  # 自动调整子图间的间距和标签位置
    # 设置刻度朝向内部，并调整刻度与坐标轴的距离
    ax.tick_params(axis='x', direction='in', pad=10)  # pad 参数用于控制刻度与坐标轴的距离
    ax.tick_params(axis='y', direction='in', pad=10)
    
    # 绘制线条和散点，设置颜色为黑色
    plt.plot(datetimes, values, color='black')
    plt.scatter(datetimes, values, s=10, color='black')

    # 调整底部边界向上移动一点
    plt.subplots_adjust(bottom=0.15)
    plt.subplots_adjust(left=0.15)

    if SaveFilePath is not None:
        plt.savefig(SaveFilePath)
    elif isShow:
        plt.show()
    plt.close()

def plot_ListFloat_with_time_x(ListFloat: list[float], ListTime: list[datetime], isShow: bool = True, SaveFilePath: Optional[str] = None, title: str = None) -> None:
    """绘制 ListFloat 对象的时间序列图"""
    values = ListFloat
    datetimes = ListTime  # 使用数据长度生成简单的序号作为 x 轴

    plt.figure(figsize=(6, 4))  # 设置图的大小，单位是英寸

    # 隐藏右边框和上边框
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xlabel('2023年8月11日RTK位移数据', fontproperties='SimSun', fontsize=12)  # 设置字体为宋体
    plt.ylabel('索道坐标x轴方向位移监测/m', fontproperties='SimSun', fontsize=12)  # 设置字体为宋体

    # 设置日期格式化器和日期刻度定位器
    date_fmt = mdates.DateFormatter("%H:00")  # 仅显示月-日
    date_locator = mdates.AutoDateLocator()  # 自动选择刻度间隔
    plt.gca().xaxis.set_major_formatter(date_fmt)
    plt.gca().xaxis.set_major_locator(date_locator)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y:.3f}'))
    # 设置最大显示的刻度数
    plt.tight_layout()  # 自动调整子图间的间距和标签位置
    # 设置刻度朝向内部，并调整刻度与坐标轴的距离
    ax.tick_params(axis='x', direction='in', pad=10)  # pad 参数用于控制刻度与坐标轴的距离
    ax.tick_params(axis='y', direction='in', pad=10)
    
    # 绘制线条和散点，设置颜色为黑色
    plt.plot(datetimes, values, color='black')
    # plt.scatter(datetimes, values, s=10, color='black')
    # 调整底部边界向上移动一点
    plt.subplots_adjust(bottom=0.15)
    plt.subplots_adjust(left=0.15)

    if SaveFilePath is not None:
        plt.savefig(SaveFilePath)
    elif isShow:
        plt.show()
    plt.close()

def plot_ListFloat_with_time_y(ListFloat: list[float], ListTime: list[datetime], isShow: bool = True, SaveFilePath: Optional[str] = None, title: str = None) -> None:
    """绘制 ListFloat 对象的时间序列图"""
    values = ListFloat
    datetimes = ListTime  # 使用数据长度生成简单的序号作为 x 轴

    plt.figure(figsize=(6, 4))  # 设置图的大小，单位是英寸

    # 隐藏右边框和上边框
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xlabel('2023年8月11日RTK位移数据', fontproperties='SimSun', fontsize=12)  # 设置字体为宋体
    plt.ylabel('索道坐标y轴方向位移监测/m', fontproperties='SimSun', fontsize=12)  # 设置字体为宋体

    # 设置日期格式化器和日期刻度定位器
    date_fmt = mdates.DateFormatter("%H:00")  # 仅显示月-日
    date_locator = mdates.AutoDateLocator()  # 自动选择刻度间隔
    plt.gca().xaxis.set_major_formatter(date_fmt)
    plt.gca().xaxis.set_major_locator(date_locator)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y:.3f}'))
    # 设置最大显示的刻度数
    plt.tight_layout()  # 自动调整子图间的间距和标签位置
    # 设置刻度朝向内部，并调整刻度与坐标轴的距离
    ax.tick_params(axis='x', direction='in', pad=10)  # pad 参数用于控制刻度与坐标轴的距离
    ax.tick_params(axis='y', direction='in', pad=10)
    
    # 绘制线条和散点，设置颜色为黑色
    plt.plot(datetimes, values, color='black')
    # plt.scatter(datetimes, values, s=10, color='black')
    # 调整底部边界向上移动一点
    plt.subplots_adjust(bottom=0.15)
    plt.subplots_adjust(left=0.15)

    if SaveFilePath is not None:
        plt.savefig(SaveFilePath)
    elif isShow:
        plt.show()
    plt.close()

def plot_ListFloat_Compare_x(ListFloat1: list[float], ListFloat2: list[float], SaveFilePath: Optional[str] = None, isShow: bool = True, title: str = None) -> None:
    """绘制两个ListFloat对象的时间序列图"""
    """绘制 ListFloat 对象的时间序列图"""
    LFDM.calculate_similarity(ListFloat1, ListFloat2)
    values1 = ListFloat1
    values2 = ListFloat2
    datetimes = range(len(ListFloat1))  # 使用数据长度生成简单的序号作为 x 轴

    plt.figure(figsize=(6, 4))  # 设置图的大小，单位是英寸

    # 隐藏右边框和上边框
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xlabel(f'{title}数据期数/期', fontproperties='SimSun', fontsize=12)  # 设置字体为宋体
    plt.ylabel('索道坐标x轴方向位移监测/m', fontproperties='SimSun', fontsize=12)  # 设置字体为宋体

    # 设置最大显示的刻度数
    plt.tight_layout()  # 自动调整子图间的间距和标签位置
    # 设置刻度朝向内部，并调整刻度与坐标轴的距离
    ax.tick_params(axis='x', direction='in', pad=10)  # pad 参数用于控制刻度与坐标轴的距离
    ax.tick_params(axis='y', direction='in', pad=10)
    
    # 绘制线条和散点，设置颜色为黑色
    plt.plot(datetimes, values1, color='blue',linestyle='-', label='白天位移')
    plt.plot(datetimes, values2, color='red',linestyle='-.',label='晚上位移')

    # 调整底部边界向上移动一点
    plt.subplots_adjust(bottom=0.15)
    plt.subplots_adjust(left=0.15)
    plt.legend()
    
    if SaveFilePath is not None:
        plt.savefig(SaveFilePath)
    elif isShow:
        plt.show()
    plt.close()

def plot_ListFloat_pitch(ListFloat1: list[float], SaveFilePath: Optional[str] = None, isShow: bool = True, title: str = None) -> None:
    """绘制两个ListFloat对象的时间序列图"""
    """绘制 ListFloat 对象的时间序列图"""
    values1 = ListFloat1

    datetimes = range(len(ListFloat1))  # 使用数据长度生成简单的序号作为 x 轴


    plt.figure(figsize=(6, 4))  # 设置图的大小，单位是英寸

    # 隐藏右边框和上边框
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xlabel(f'数据期数/期', fontproperties='SimSun', fontsize=12)  # 设置字体为宋体
    plt.ylabel('pitch/°', fontproperties='SimSun', fontsize=12)  # 设置字体为宋体

    # 设置最大显示的刻度数
    plt.tight_layout()  # 自动调整子图间的间距和标签位置
    # 设置刻度朝向内部，并调整刻度与坐标轴的距离
    ax.tick_params(axis='x', direction='in', pad=10)  # pad 参数用于控制刻度与坐标轴的距离
    ax.tick_params(axis='y', direction='in', pad=10)
    

    # 设置 y 轴坐标格式为两位有效数字
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # 绘制线条和散点，设置颜色为黑色
    plt.plot(datetimes, values1, color='blue',linestyle='-', label='数据')

    # 调整底部边界向上移动一点
    plt.subplots_adjust(bottom=0.15)
    plt.subplots_adjust(left=0.15)
    plt.legend()
    
    if SaveFilePath is not None:
        plt.savefig(SaveFilePath)
    elif isShow:
        plt.show()
    plt.close()

def plot_ListFloat_roll(ListFloat1: list[float], SaveFilePath: Optional[str] = None, isShow: bool = True, title: str = None) -> None:
    """绘制两个ListFloat对象的时间序列图"""
    """绘制 ListFloat 对象的时间序列图"""
    values1 = ListFloat1

    datetimes = range(len(ListFloat1))  # 使用数据长度生成简单的序号作为 x 轴


    plt.figure(figsize=(6, 4))  # 设置图的大小，单位是英寸

    # 隐藏右边框和上边框
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xlabel(f'数据期数/期', fontproperties='SimSun', fontsize=12)  # 设置字体为宋体
    plt.ylabel('roll/°', fontproperties='SimSun', fontsize=12)  # 设置字体为宋体

    # 设置最大显示的刻度数
    plt.tight_layout()  # 自动调整子图间的间距和标签位置
    # 设置刻度朝向内部，并调整刻度与坐标轴的距离
    ax.tick_params(axis='x', direction='in', pad=10)  # pad 参数用于控制刻度与坐标轴的距离
    ax.tick_params(axis='y', direction='in', pad=10)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # 绘制线条和散点，设置颜色为黑色
    plt.plot(datetimes, values1, color='blue',linestyle='-', label='数据')

    # 调整底部边界向上移动一点
    plt.subplots_adjust(bottom=0.15)
    plt.subplots_adjust(left=0.15)
    plt.legend()
    
    if SaveFilePath is not None:
        plt.savefig(SaveFilePath)
    elif isShow:
        plt.show()
    plt.close()


def plot_ListFloat_yaw(ListFloat1: list[float], SaveFilePath: Optional[str] = None, isShow: bool = True) -> None:
    """绘制两个ListFloat对象的时间序列图"""
    """绘制 ListFloat 对象的时间序列图"""
    values1 = ListFloat1

    datetimes = range(len(ListFloat1))  # 使用数据长度生成简单的序号作为 x 轴


    plt.figure(figsize=(6, 4))  # 设置图的大小，单位是英寸

    # 隐藏右边框和上边框
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xlabel(f'数据期数/期', fontproperties='SimSun', fontsize=12)  # 设置字体为宋体
    plt.ylabel('yaw/°', fontproperties='SimSun', fontsize=12)  # 设置字体为宋体
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # 设置最大显示的刻度数
    plt.tight_layout()  # 自动调整子图间的间距和标签位置
    # 设置刻度朝向内部，并调整刻度与坐标轴的距离
    ax.tick_params(axis='x', direction='in', pad=10)  # pad 参数用于控制刻度与坐标轴的距离
    ax.tick_params(axis='y', direction='in', pad=10)
    
    # 绘制线条和散点，设置颜色为黑色
    plt.plot(datetimes, values1, color='blue',linestyle='-', label='数据')

    # 调整底部边界向上移动一点
    plt.subplots_adjust(bottom=0.15)
    plt.subplots_adjust(left=0.15)
    plt.legend()
    
    if SaveFilePath is not None:
        plt.savefig(SaveFilePath)
    elif isShow:
        plt.show()
    plt.close()


def plot_ListFloat_Compare_pitch_coor(ListFloat1: list[float], ListFloat2: list[float], SaveFilePath: Optional[str] = None, isShow: bool = True, title: str = None) -> None:
    """绘制两个ListFloat对象的时间序列图"""
    """绘制 ListFloat 对象的时间序列图"""
    LFDM.calculate_similarity(ListFloat1, ListFloat2)
    # values1 = ListFloat1
    # values2 = ListFloat2

    values1 = LFDM.min_max_normalization(ListFloat1)
    values2 = LFDM.min_max_normalization(ListFloat2)

    datetimes = range(len(ListFloat1))  # 使用数据长度生成简单的序号作为 x 轴


    plt.figure(figsize=(6, 4))  # 设置图的大小，单位是英寸

    # 隐藏右边框和上边框
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xlabel(f'{title}数据期数/期', fontproperties='SimSun', fontsize=12)  # 设置字体为宋体
    plt.ylabel('pitch/°', fontproperties='SimSun', fontsize=12)  # 设置字体为宋体

    # 设置最大显示的刻度数
    plt.tight_layout()  # 自动调整子图间的间距和标签位置
    # 设置刻度朝向内部，并调整刻度与坐标轴的距离
    ax.tick_params(axis='x', direction='in', pad=10)  # pad 参数用于控制刻度与坐标轴的距离
    ax.tick_params(axis='y', direction='in', pad=10)
    
    # 绘制线条和散点，设置颜色为黑色
    plt.plot(datetimes, values1, color='blue',linestyle='-', label='数据1')
    plt.plot(datetimes, values2, color='red',linestyle='-.',label='数据2')

    # 调整底部边界向上移动一点
    plt.subplots_adjust(bottom=0.15)
    plt.subplots_adjust(left=0.15)
    plt.legend()
    
    if SaveFilePath is not None:
        plt.savefig(SaveFilePath)
    elif isShow:
        plt.show()
    plt.close()

def plot_ListFloat_Compare_roll(ListFloat1: list[float], ListFloat2: list[float], SaveFilePath: Optional[str] = None, isShow: bool = True, title: str = None) -> None:
    """绘制两个ListFloat对象的时间序列图"""
    """绘制 ListFloat 对象的时间序列图"""
    LFDM.calculate_similarity(ListFloat1, ListFloat2)
    values1 = ListFloat1
    values2 = ListFloat2
    datetimes = range(len(ListFloat1))  # 使用数据长度生成简单的序号作为 x 轴

    plt.figure(figsize=(6, 4))  # 设置图的大小，单位是英寸

    # 隐藏右边框和上边框
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xlabel(f'{title}数据期数/期', fontproperties='SimSun', fontsize=12)  # 设置字体为宋体
    plt.ylabel('横滚角/°', fontproperties='SimSun', fontsize=12)  # 设置字体为宋体

    # 设置最大显示的刻度数
    plt.tight_layout()  # 自动调整子图间的间距和标签位置
    # 设置刻度朝向内部，并调整刻度与坐标轴的距离
    ax.tick_params(axis='x', direction='in', pad=10)  # pad 参数用于控制刻度与坐标轴的距离
    ax.tick_params(axis='y', direction='in', pad=10)
    
    # 绘制线条和散点，设置颜色为黑色
    plt.plot(datetimes, values1, color='black',linestyle='-', label='白天数据')
    plt.plot(datetimes, values2, color='black',linestyle='-.',label='晚上数据')

    # 调整底部边界向上移动一点
    plt.subplots_adjust(bottom=0.15)
    plt.subplots_adjust(left=0.15)
    plt.legend()
    
    if SaveFilePath is not None:
        plt.savefig(SaveFilePath)
    elif isShow:
        plt.show()
    plt.close()



def plot_ListFloat_Compare_y(ListFloat1: list[float], ListFloat2: list[float], SaveFilePath: Optional[str] = None, isShow: bool = True, title: str = None) -> None:
    """绘制两个ListFloat对象的时间序列图"""
    """绘制 ListFloat 对象的时间序列图"""
    LFDM.calculate_similarity(ListFloat1, ListFloat2)
    values1 = ListFloat1
    values2 = ListFloat2
    datetimes = range(len(ListFloat1))  # 使用数据长度生成简单的序号作为 x 轴

    plt.figure(figsize=(6, 4))  # 设置图的大小，单位是英寸

    # 隐藏右边框和上边框
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xlabel(f'数据期数/期', fontproperties='SimSun', fontsize=12)  # 设置字体为宋体
    plt.ylabel('索道坐标y轴方向位移监测/m', fontproperties='SimSun', fontsize=12)  # 设置字体为宋体

    # 设置最大显示的刻度数
    plt.tight_layout()  # 自动调整子图间的间距和标签位置
    # 设置刻度朝向内部，并调整刻度与坐标轴的距离
    ax.tick_params(axis='x', direction='in', pad=10)  # pad 参数用于控制刻度与坐标轴的距离
    ax.tick_params(axis='y', direction='in', pad=10)
    
    # 绘制线条和散点，设置颜色为黑色
    plt.plot(datetimes, values1, color='blue',linestyle='-', label='白天位移')
    plt.plot(datetimes, values2, color='red',linestyle='-.',label='晚上位移')

    # 调整底部边界向上移动一点
    plt.subplots_adjust(bottom=0.15)
    plt.subplots_adjust(left=0.15)
    plt.legend()
    if SaveFilePath is not None:
        plt.savefig(SaveFilePath)
    elif isShow:
        plt.show()
    plt.close()

    plt.close()


def plot_ListFloat_Compare_without_marker(ListFloat1: list[float], ListFloat2: list[float], to_marker_idx: list[int] = []) -> None:
    """绘制两个ListFloat对象的时间序列图"""
    """绘制 ListFloat 对象的时间序列图"""
    values1 = [value for idx,value in enumerate(ListFloat1) if idx not in to_marker_idx]
    values2 = [value for idx,value in enumerate(ListFloat2) if idx not in to_marker_idx]

    pearson_corr, spearman_corr, kendall_corr = LFDM.calculate_similarity(values1,values2)
    datetimes = range(len(values1))  # 使用数据长度生成简单的序号作为 x 轴

    plt.figure(figsize=(6, 4))  # 设置图的大小，单位是英寸

    # 隐藏右边框和上边框
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xlabel('数据期数/期', fontproperties='SimSun', fontsize=12)  # 设置字体为宋体
    plt.ylabel('索道坐标x轴方向位移监测/m', fontproperties='SimSun', fontsize=12)  # 设置字体为宋体

    # 设置最大显示的刻度数
    plt.tight_layout()  # 自动调整子图间的间距和标签位置
    # 设置刻度朝向内部，并调整刻度与坐标轴的距离
    ax.tick_params(axis='x', direction='in', pad=10)  # pad 参数用于控制刻度与坐标轴的距离
    ax.tick_params(axis='y', direction='in', pad=10)
    
    # 绘制线条和散点，设置颜色为黑色
    plt.plot(datetimes, values1, color='black')
    plt.plot(datetimes, values2, color='blue')

    plt.scatter(datetimes, values1, s=10, color='black')

    plt.show()
    plt.close()

def plot_ListFloat_with_marker(ListFloat: list[float], to_marker_idx: list[int] = []) -> None:
    """绘制两个ListFloat对象的时间序列图"""
    """绘制 ListFloat 对象的时间序列图"""
    """绘制 ListFloat 对象的时间序列图"""
    values = ListFloat
    datetimes = range(len(ListFloat))  # 使用数据长度生成简单的序号作为 x 轴

    plt.figure(figsize=(6, 4))  # 设置图的大小，单位是英寸

    # 隐藏右边框和上边框
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xlabel('数据期数/期', fontproperties='SimSun', fontsize=12)  # 设置字体为宋体
    plt.ylabel('索道坐标x轴方向位移监测/m', fontproperties='SimSun', fontsize=12)  # 设置字体为宋体

    # 设置最大显示的刻度数
    plt.tight_layout()  # 自动调整子图间的间距和标签位置
    # 设置刻度朝向内部，并调整刻度与坐标轴的距离
    ax.tick_params(axis='x', direction='in', pad=10)  # pad 参数用于控制刻度与坐标轴的距离
    ax.tick_params(axis='y', direction='in', pad=10)
    
    # 绘制线条和散点，设置颜色为黑色
    plt.plot(datetimes, values, color='black')
    for idx in range(len(values)):
        if idx not in to_marker_idx:
            plt.scatter(idx, values[idx], s=10, color='black')
        else:
            plt.scatter(idx, values[idx], s=10, color='red', marker='x')

    plt.show()
    plt.close()


def plot_ListFloat_y(ListFloat: list[float], isShow: bool = True, SaveFilePath: Optional[str] = None, title: str = None) -> None:
    """绘制 ListFloat 对象的时间序列图"""
    values = ListFloat
    datetimes = range(len(ListFloat))  # 使用数据长度生成简单的序号作为 x 轴

    plt.figure(figsize=(6, 4))  # 设置图的大小，单位是英寸

    # 隐藏右边框和上边框
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xlabel('数据期数/期', fontproperties='SimSun', fontsize=12)  # 设置字体为宋体
    plt.ylabel('索道坐标y轴方向位移监测/m', fontproperties='SimSun', fontsize=12)  # 设置字体为宋体

    # 设置最大显示的刻度数
    plt.tight_layout()  # 自动调整子图间的间距和标签位置
    # 设置刻度朝向内部，并调整刻度与坐标轴的距离
    ax.tick_params(axis='x', direction='in', pad=10)  # pad 参数用于控制刻度与坐标轴的距离
    ax.tick_params(axis='y', direction='in', pad=10)
        # 调整底部边界向上移动一点
    plt.subplots_adjust(bottom=0.15)
    plt.subplots_adjust(left=0.15)

    # 绘制线条和散点，设置颜色为黑色
    plt.plot(datetimes, values, color='black')
    plt.scatter(datetimes, values, s=10, color='black')

    if SaveFilePath is not None:
        plt.savefig(SaveFilePath)
    elif isShow:
        plt.show()
    plt.close()


def plot_points_with_markeridx(x_points: list[float], y_points: list[float], title: str = None, to_marker_idx: list[int] = []):
    # Convert lists to numpy arrays
    y_points = np.array(y_points)
    
    plt.figure(figsize=(6, 4))  # 设置图的大小，单位是英寸
    plt.xlabel('索道坐标x轴方向位移监测/m')  # 设置x轴标签
    plt.ylabel('索道坐标y轴方向位移监测/m')  # 设置y轴标签
    
    # 绘制正常点的散点图
    for idx, point in enumerate(x_points):
        if idx not in to_marker_idx:
            plt.scatter(x_points[idx], y_points[idx], color='blue', marker='o', s=20)  
    
    # 收集异常点的位置
    exception_points_x = []
    exception_points_y = []
    for idx in to_marker_idx:
        if 0 <= idx < len(x_points) and 0 <= idx < len(y_points):
            exception_points_x.append(x_points[idx])
            exception_points_y.append(y_points[idx])
    
    # 绘制异常点
    if exception_points_x and exception_points_y:
        plt.scatter(exception_points_x, exception_points_y, color='red', marker='x', s=20)
    
    # 隐藏右边框和上边框
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)  # 隐藏下边框
    ax.spines['left'].set_visible(False)    # 隐藏左边框
    
    # 添加箭头，指向坐标轴的正方向
    ax.annotate('', xy=(1, 0.001), xytext=(0, 0.001),
                arrowprops=dict(arrowstyle='->', lw=1, color='black'),
                xycoords='axes fraction', textcoords='axes fraction')
    ax.annotate('', xy=(0.001, 1), xytext=(0.001, 0),
                arrowprops=dict(arrowstyle='->', lw=1, color='black'),
                xycoords='axes fraction', textcoords='axes fraction')
    
    # 设置刻度朝向内部，并调整刻度与坐标轴的距离
    ax.tick_params(axis='x', direction='in', pad=10)  # pad 参数用于控制刻度与坐标轴的距离
    ax.tick_params(axis='y', direction='in', pad=10)
    
    # 设置标题
    if title is None:
        plt.suptitle('时间序列数据', y=1.0)  # 设置标题，默认为'时间序列数据'，y参数用于调整标题在垂直方向上的位置
    else:
        plt.suptitle(title, y=1.0)  # 设置标题为传入的参数title，y参数用于调整标题在垂直方向上的位置
    
    plt.tight_layout()  # 自动调整子图间的间距和标签位置
    plt.show()


#TODO: 1.用双天线的数据来估计一个索道支架的位置，
#TODO: 2.用气象仪数据做一个相关性分析和可视化比较
#TODO: 3.做全年的s
def main_TODO2():
    """
    坐标转换
    """
    x_DataPoints, y_DataPoints ,Receiver_DataPoints= load_DataPoints_in_ropeway()
    # LFDM.plot_ListFloat(x_DataPoints['R051_1215'], isShow=True, title='R051_1215_x')
    # LFDM.plot_ListFloat(y_DataPoints['R051_1215'], isShow=True, title='R051_1215_y')

    # item_R1_list = ['R071_1215']
    # item_R2_list = ['R072_1215']
    item_R1_list = ['R031_0407', 'R031_1215', 'R051_0407','R051_1215','R071_0407', 'R071_1215','R081_0407', 'R081_1215']
    item_R2_list = ['R032_0407','R032_1215', 'R052_0407','R052_1215','R072_0407', 'R072_1215','R082_0407', 'R082_1215']
    for idx in range(len(item_R1_list)):
        matched_list1, matched_list2 = match_DataPoints_in_ropeway(Receiver_DataPoints, item_R1_list[idx], item_R2_list[idx])
        x_coordinates_R031, y_coordinates_R031 = transform_ListDataPoint_to_xy(matched_list1)
        x_coordinates_R032, y_coordinates_R032 = transform_ListDataPoint_to_xy(matched_list2)
        
        filtered_x_coords_R1 = []
        filtered_y_coords_R1 = []
        filtered_x_coords_R2 = []
        filtered_y_coords_R2 = []
        dt_list = []
        n = len(y_coordinates_R031)
        used = [False] * n
        for i in range(n):
            if not used[i] and abs(y_coordinates_R031[i] - y_coordinates_R032[i]) >= 0.1:
                used[i] = True
        for i in range(n):
            if  used[i]:
                filtered_x_coords_R1.append(x_coordinates_R031[i])
                filtered_y_coords_R1.append(y_coordinates_R031[i])
                filtered_x_coords_R2.append(x_coordinates_R032[i])
                filtered_y_coords_R2.append(y_coordinates_R032[i])
                dt_list.append(matched_list1[i].start_time)
        print(len(filtered_x_coords_R1))
        print(f'------------{item_R1_list[idx]}------------')
        removed_points  = LFDM.detect_knn_anomaly_xy_with_while(filtered_x_coords_R1, filtered_y_coords_R1)
        print(f"{item_R1_list[idx]}:{removed_points}")
        # to_marker_point = detected_outliers.tolist()

        # plot_ListFloat_x(filtered_x_coords_R1)
        # plot_ListFloat_y(filtered_y_coords_R1)
        # plot_ListFloat_with_marker(filtered_x_coords_R1, to_marker_point)
        # plot_ListFloat_with_marker(filtered_y_coords_R1, to_marker_point)

        # plot_ListFloat_Compare_without_marker(filtered_x_coords_R1, filtered_x_coords_R2, to_marker_idx=to_marker_point)
        # # 调用 calculate_similarity 并接受返回的元组
        # for marker in to_marker_point:
        #     print(dt_list[marker])

        # plot_points_with_markeridx(filtered_x_coords_R1, filtered_y_coords_R1, '7号支架主天线', to_marker_point)

        print(f'------------{item_R2_list[idx]}------------')


        removed_points  = LFDM.detect_knn_anomaly_xy_with_while(filtered_x_coords_R2, filtered_y_coords_R2)
        # print(f"{item_R2_list[idx]}:{removed_points}")
        # # # 调用 calculate_similarity 并接受返回的元组
        # for marker in removed_points:
        #     print(dt_list[marker])

        
        # plot_points_with_markeridx(filtered_x_coords_R2, filtered_y_coords_R2, '7号支架副天线', to_marker_point)
        # LFDM.calculate_similarity(x_DataPoints['R031_1215'], y_DataPoints['R031_1215'])
    
    # LFDM.plot_ListFloat_with_markeridx(ListFloat=x_DataPoints['R031_1215'], isShow=True,to_marker_idx=to_marker_point)
    # plot_ListFloat_x(filtered_x_coords_R1)

    # plot_ListFloat_y(filtered_y_coords_R1)

    # plot_ListFloat_Compare(filtered_x_coords_R1, filtered_x_coords_R2)
    # plot_ListFloat_Compare(filtered_y_coords_R1, filtered_y_coords_R2)
    # plot_points_with_markeridx(filtered_x_coords_R1, filtered_y_coords_R2, 'marker', to_marker_point)


def main_compare():
    x_DataPoints, y_DataPoints = load_DataPoints_in_ropeway()


    data_file = r"D:\Program Files (x86)\Software\OneDrive\PyPackages_DataSave\new_data\tiltmeter_0407.txt"
    tiltmeter_data_dict: Dict[datetime.date, List['TiltmeterDataAvage']] = TiltmeterDataAvage.read_tiltmeter_data(data_file)
    list_pitch = []
    list_roll = []

    for date_obj, avg_data in tiltmeter_data_dict.items(): 
        # print(f"日期: {date_obj}, Pitch 平均值: {avg_data.pitch}, Roll 平均值: {avg_data.roll}, 样本数量: {avg_data.num}")
        list_pitch.append(avg_data.pitch)
        list_roll.append(avg_data.roll)
    
    list_pitch = LFDM.min_max_normalization(list_pitch)
    list_roll = LFDM.min_max_normalization(list_roll)
    LFDM.plot_ListFloat(ListFloat=list_pitch, isShow=True, title="pitch")
    LFDM.plot_ListFloat(ListFloat=list_roll, isShow=True, title="pitch")
    LFDM.plot_ListFloat_Compare(ListFloat1=list_pitch,ListFloat2=list_roll,title="compare")
    
def main_ggkx():
    data_file = r"C:\Users\Jayttle\Desktop\temp_Coor\ggkx_R071_ggkx_R071_2023-08-11.txt.txt"
    ggkx_data_with_coords_list = GgkxDataWithCoordinates.read_ggkx_data_with_coordinates(data_file)
    filtered_data = GgkxDataWithCoordinates.filter_by_date_with_coordinates(ggkx_data_with_coords_list, '2023-08-11')
    x_list = []
    y_list = []
    time_list = [] 
    for idx in range(len(filtered_data)):
        x_prime, y_prime = ropeway_transform_coordinates(filtered_data[idx].east_coordinate, filtered_data[idx].north_coordinate)
        x_list.append(x_prime)
        y_list.append(y_prime)
        time_list.append(filtered_data[idx].time)

    # 计算相邻差值
    x_diff = [abs(x_list[i] - x_list[i-1]) for i in range(1, len(x_list))]

    # 定义阈值（示例阈值）
    threshold = 0.01  # 根据你的数据特点自行调整

    # 打印或处理大突变点
    print("大突变点的时间和位置：")
    for i in range(len(x_diff)):
        if x_diff[i] > threshold:
            print(f"时间：{time_list[i+1]}, 索引：{i+1}, X坐标变化：{x_diff[i]}")

    plot_ListFloat_with_time_x(x_list, time_list)
    plot_ListFloat_with_time_y(y_list, time_list)
    # plot_points(x_list,y_list)

def process_data_points(Receiver_DataPoints, item_R1_list, item_R2_list, filter_outliers_dict):
    matched_list1, matched_list2 = match_DataPoints_in_ropeway(Receiver_DataPoints, item_R1_list, item_R2_list)
    x_coordinates_R031, y_coordinates_R031 = transform_ListDataPoint_to_xy(matched_list1)
    x_coordinates_R032, y_coordinates_R032 = transform_ListDataPoint_to_xy(matched_list2)
    
    filtered_x_coords_R1 = []
    filtered_y_coords_R1 = []
    filtered_x_coords_R2 = []
    filtered_y_coords_R2 = []
    dt_list = []
    n = len(y_coordinates_R031)
    used = [False] * n
    for i in range(n):
        if not used[i] and abs(y_coordinates_R031[i] - y_coordinates_R032[i]) >= 0.1:
            used[i] = True
    for i in range(n):
        if used[i]:
            filtered_x_coords_R1.append(x_coordinates_R031[i])
            filtered_y_coords_R1.append(y_coordinates_R031[i])
            filtered_x_coords_R2.append(x_coordinates_R032[i])
            filtered_y_coords_R2.append(y_coordinates_R032[i])
            dt_list.append(matched_list1[i].start_time.date())
    
    return filtered_x_coords_R1, filtered_y_coords_R1, filtered_x_coords_R2, filtered_y_coords_R2, dt_list

def process_outliers(filtered_x_coords_R1, filtered_y_coords_R1, filtered_x_coords_R2, filtered_y_coords_R2, dt_list, filter_outliers_dict, marker_name):
    x_coords_R1 = []
    y_coords_R1 = []
    x_coords_R2 = []
    y_coords_R2 = []
    tlist = []
    if marker_name in filter_outliers_dict:
        for i in range(len(filtered_x_coords_R1)):
            if i not in filter_outliers_dict[marker_name]:
                x_coords_R1.append(filtered_x_coords_R1[i])
                y_coords_R1.append(filtered_y_coords_R1[i])
                x_coords_R2.append(filtered_x_coords_R2[i]) 
                y_coords_R2.append(filtered_y_coords_R2[i])
                tlist.append(dt_list[i])
        return x_coords_R1, y_coords_R1, x_coords_R2 , y_coords_R2, tlist
    else:
        return filtered_x_coords_R1, filtered_y_coords_R1, filtered_x_coords_R2, filtered_y_coords_R2, dt_list


def analyze_variability(data:list[float]):
    data = np.array(data)   # 将输入数据转换为NumPy数组以便进行计算
    mean_value = np.mean(data)
    variance = np.var(data)
    std_dev = np.std(data)
    data_range = np.max(data) - np.min(data)
    q75, q25 = np.percentile(data, [75 ,25])
    iqr = q75 - q25
    cv = std_dev / mean_value if mean_value != 0 else 0
    data_skewness = skew(data)
    data_kurtosis = kurtosis(data)

    print(f"平均值: {mean_value:.4f}")
    print(f"方差: {variance:.4f}")
    print(f"标准差: {std_dev:.4f}")
    print(f"范围: {data_range:.4f}")
    print(f"四分位距: {iqr:.4f}")
    print(f"变异系数: {cv:.4f}")
    print(f"偏度: {data_skewness:.4f}")
    print(f"峰度: {data_kurtosis:.4f}")

    results = {
        "平均值": mean_value,
        "方差": variance,
        "标准差": std_dev,
        "范围": data_range,
        "四分位距": iqr,
        "变异系数": cv,
        "偏度": data_skewness,
        "峰度": data_kurtosis
    }
    
    return results

def detailed_compare_lists(ListFloat1, ListFloat2):
    assert len(ListFloat1) == len(ListFloat2), "列表长度不一致"

    # 逐元素比较
    differences = []
    num_differences = 0
    for a, b in zip(ListFloat1, ListFloat2):
        if a != b:
            num_differences += 1
            differences.append(abs(a - b))

    print(f"总共有 {num_differences} 处不同。")

    # 差异分析
    if differences:
        print(f"平均差异: {np.mean(differences):.2f}")
        print(f"最大差异: {max(differences):.2f}")
        print(f"最小差异: {min(differences):.2f}")
        print(f"差异标准差: {np.std(differences):.2f}")
        print(f"差异中位数: {np.median(differences):.2f}")
    else:
        print("两个列表完全相同。")

    # 相关性分析
    correlation_coefficient, p_value = pearsonr(ListFloat1, ListFloat2)
    print(f"相关系数 (Pearson): {correlation_coefficient:.2f}")
    print(f"P值: {p_value:.4f} (P值小说明相关性显著)")

    # 统计分析
    print(f"ListFloat1 偏度: {skew(ListFloat1):.2f}, 峰度: {kurtosis(ListFloat1):.2f}")
    print(f"ListFloat2 偏度: {skew(ListFloat2):.2f}, 峰度: {kurtosis(ListFloat2):.2f}")

    # 排序后比较
    if sorted(ListFloat1) == sorted(ListFloat2):
        print("排序后两个列表相同。")
    else:
        print("排序后两个列表不同。")

    # 绘制直方图和箱型图
    plt.figure(figsize=(12, 6))

    plt.subplot(121)
    plt.hist(ListFloat1, bins=20, alpha=0.7, label='ListFloat1')
    plt.hist(ListFloat2, bins=20, alpha=0.7, label='ListFloat2')
    plt.title("直方图比较")
    plt.legend()

    plt.subplot(122)
    plt.boxplot([ListFloat1, ListFloat2], labels=['ListFloat1', 'ListFloat2'])
    plt.title("箱型图比较")
    plt.tight_layout()
    plt.show()


def main_load_GNSS_data():
    print("---------------------main_load_GNSS_data-------------------")
    x_DataPoints, y_DataPoints ,Receiver_DataPoints= load_DataPoints_in_ropeway()
    item_R1_list_0407 = ['R031_0407', 'R051_0407', 'R071_0407','R081_0407']
    item_R2_list_0407 = ['R032_0407', 'R052_0407', 'R072_0407','R082_0407']
    item_R1_list_1215 = ['R031_1215', 'R051_1215', 'R071_1215', 'R081_1215']
    item_R2_list_1215 = ['R032_1215', 'R052_1215', 'R072_1215', 'R082_1215']
    
    item_R1_list = ['R031_1215', 'R051_1215', 'R071_1215', 'R081_1215', 'R031_0407', 'R051_0407', 'R071_0407','R081_0407']
    item_R2_list = ['R032_1215', 'R052_1215', 'R072_1215', 'R082_1215', 'R032_0407', 'R052_0407', 'R072_0407','R082_0407']
    filter_outliers_dict = {
        'R031_0407': [68, 77],
        'R032_0407': [68, 77],
        'R031_1215':[64, 78],
        'R032_1215':[64, 78],
        'R051_1215':[4, 22],
        'R052_1215':[4, 22],
        'R071_1215':[76],
        'R072_1215':[76],
        'R081_1215':[121],
        'R082_1215':[121],
    }
    all_data = {}

    for idx in range(len(item_R1_list)):
        filtered_x_coords_R1, filtered_y_coords_R1, filtered_x_coords_R2, filtered_y_coords_R2, dt_list = \
            process_data_points(Receiver_DataPoints, item_R1_list[idx], item_R2_list[idx], filter_outliers_dict)
        
        x_coords_R1, y_coords_R1, x_coords_R2 , y_coords_R2, tlist = \
            process_outliers(filtered_x_coords_R1, filtered_y_coords_R1, filtered_x_coords_R2, filtered_y_coords_R2, dt_list, filter_outliers_dict, item_R1_list[idx])
        
        # Initialize the key in all_data if it doesn't exist
        if item_R1_list[idx] not in all_data:
            all_data[item_R1_list[idx]] = []
        all_data[item_R1_list[idx]].append(x_coords_R1)
        all_data[item_R1_list[idx]].append(y_coords_R1)
        all_data[item_R1_list[idx]].append(tlist)
        all_data[item_R1_list[idx]].append([data.elevation for data in Receiver_DataPoints[item_R1_list[idx]]])
        if item_R2_list[idx] not in all_data:
            all_data[item_R2_list[idx]] = []
        all_data[item_R2_list[idx]].append(x_coords_R2)
        all_data[item_R2_list[idx]].append(y_coords_R2)
        all_data[item_R2_list[idx]].append(tlist)
        all_data[item_R2_list[idx]].append([data.elevation for data in Receiver_DataPoints[item_R2_list[idx]]])
        # plot_ListFloat_x(filtered_x_coords_R1)
        # plot_ListFloat_with_marker(filtered_x_coords_R1, to_marker_point)

    for key in all_data:
        print(f"{key}:{len(all_data[key][0])}\t{len(all_data[key][1])}\t{len(all_data[key][2])}")
    return all_data

def main_process_GNSS_data():
    all_data = main_load_GNSS_data()
    print("---------------------main_process_GNSS_data-------------------")

    
    to_process_itemR1 = ['R031_0407', 'R032_0407', 'R051_0407', 'R052_0407', 'R071_0407', 'R072_0407', 'R081_0407','R082_0407']
    to_process_itemR2 = ['R031_1215','R032_1215', 'R051_1215', 'R052_1215','R071_1215', 'R072_1215', 'R081_1215', 'R082_1215']

    for idx in range(len(to_process_itemR1)):
        aligned_list1_x = []
        aligned_list2_x = []

        aligned_list1_y = []
        aligned_list2_y = []
        idx1 = 0
        idx2 = 0
        dt_list1 = all_data[to_process_itemR1[idx]][2]
        dt_list2 = all_data[to_process_itemR2[idx]][2]
        ListFloat1_x = all_data[to_process_itemR1[idx]][0]
        ListFloat2_x = all_data[to_process_itemR2[idx]][0]

        ListFloat1_y = all_data[to_process_itemR1[idx]][1]
        ListFloat2_y = all_data[to_process_itemR2[idx]][1]
        while idx1 < len(dt_list1) and idx2 < len(dt_list2):
            if dt_list1[idx1] == dt_list2[idx2]:
                aligned_list1_x.append(ListFloat1_x[idx1])
                aligned_list2_x.append(ListFloat2_x[idx2])
                aligned_list1_y.append(ListFloat1_y[idx1])
                aligned_list2_y.append(ListFloat2_y[idx2])

                idx1 += 1
                idx2 += 1
            elif dt_list1[idx1] < dt_list2[idx2]:
                idx1 += 1
            else:
                idx2 += 1

        print(len(aligned_list1_x))
        print(len(aligned_list2_x))
        print(f'----------{to_process_itemR1[idx]}&{to_process_itemR2[idx]}-----------')
        save_path_x = f'D:\Program Files (x86)\Software\OneDrive\PyPackages_img\Compare\{to_process_itemR1[idx][:4]}x_compare.png'
        save_path_y = f'D:\Program Files (x86)\Software\OneDrive\PyPackages_img\Compare\{to_process_itemR1[idx][:4]}y_compare.png'
        plot_ListFloat_Compare_x(ListFloat1=aligned_list1_x, ListFloat2=aligned_list2_x,SaveFilePath=save_path_x,isShow=False,title=f'{to_process_itemR1[idx][2]}号支架')
        plot_ListFloat_Compare_y(ListFloat1=aligned_list1_y, ListFloat2=aligned_list2_y,SaveFilePath=save_path_y,isShow=False,title=f'{to_process_itemR1[idx][2]}号支架')

def main_process_GNSS_data_half():
    all_data = main_load_GNSS_data()
    print("---------------------main_process_GNSS_data-------------------")

    to_process_itemR1 = ['R031_0407', 'R032_0407', 'R051_0407', 'R052_0407', 'R071_0407', 'R072_0407', 'R081_0407', 'R082_0407']
    to_process_itemR2 = ['R031_1215', 'R032_1215', 'R051_1215', 'R052_1215', 'R071_1215', 'R072_1215', 'R081_1215', 'R082_1215']

    for idx in range(len(to_process_itemR1)):
        aligned_list1_x = []
        aligned_list2_x = []

        aligned_list1_y = []
        aligned_list2_y = []
        idx1 = 0
        idx2 = 0
        dt_list1 = all_data[to_process_itemR1[idx]][2]
        dt_list2 = all_data[to_process_itemR2[idx]][2]
        ListFloat1_x = all_data[to_process_itemR1[idx]][0]
        ListFloat2_x = all_data[to_process_itemR2[idx]][0]

        ListFloat1_y = all_data[to_process_itemR1[idx]][1]
        ListFloat2_y = all_data[to_process_itemR2[idx]][1]
        while idx1 < len(dt_list1) and idx2 < len(dt_list2):
            if dt_list1[idx1] == dt_list2[idx2]:
                aligned_list1_x.append(ListFloat1_x[idx1])
                aligned_list2_x.append(ListFloat2_x[idx2])
                aligned_list1_y.append(ListFloat1_y[idx1])
                aligned_list2_y.append(ListFloat2_y[idx2])

                idx1 += 1
                idx2 += 1
            elif dt_list1[idx1] < dt_list2[idx2]:
                idx1 += 1
            else:
                idx2 += 1

        # Split aligned lists into halves
        # Ensure aligned lists have enough elements for splitting
        if len(aligned_list1_x) < 2 or len(aligned_list2_x) < 2:
            print(f"Skipping {to_process_itemR1[idx]} and {to_process_itemR2[idx]} due to insufficient data.")
            continue

        if len(aligned_list1_x) % 2 == 0:
            mid_index = len(aligned_list1_x) // 2

            # 根据中点索引将对齐列表分割为两半
            first_half_list1_x = aligned_list1_x[:mid_index]
            second_half_list1_x = aligned_list1_x[mid_index:]

            # 确保所有对应的列表都以相同的方式分割
            first_half_list1_y = aligned_list1_y[:mid_index]
            second_half_list1_y = aligned_list1_y[mid_index:]

            first_half_list2_x = aligned_list2_x[:mid_index]
            second_half_list2_x = aligned_list2_x[mid_index:]

            first_half_list2_y = aligned_list2_y[:mid_index]
            second_half_list2_y = aligned_list2_y[mid_index:]
        else:
            # 计算中点索引以进行分割，如果是奇数长度，则自动舍去中间值
            mid_index = floor(len(aligned_list1_x) / 2)

            # 根据中点索引将对齐列表分割为两半
            first_half_list1_x = aligned_list1_x[:mid_index]
            second_half_list1_x = aligned_list1_x[mid_index+1:]  # 从mid_index+1开始以舍去中间的值

            # 确保所有对应的列表都以相同的方式分割
            first_half_list1_y = aligned_list1_y[:mid_index]
            second_half_list1_y = aligned_list1_y[mid_index+1:]

            first_half_list2_x = aligned_list2_x[:mid_index]
            second_half_list2_x = aligned_list2_x[mid_index+1:]

            first_half_list2_y = aligned_list2_y[:mid_index]
            second_half_list2_y = aligned_list2_y[mid_index+1:]

        # Calculate Pearson correlation coefficients
        corr_first_half_x, _ = pearsonr(first_half_list1_x, first_half_list2_x)
        corr_second_half_x, _ = pearsonr(second_half_list1_x, second_half_list2_x)
        corr_first_half_y, _ = pearsonr(first_half_list1_y, first_half_list2_y)
        corr_second_half_y, _ = pearsonr(second_half_list1_y, second_half_list2_y)

        print(f'----------{to_process_itemR1[idx]} & {to_process_itemR2[idx]}-----------')
        print(f"num: {len(aligned_list1_x)}")
        print(f'Correlation (first half) - X: {corr_first_half_x}')
        print(f'Correlation (second half) - X: {corr_second_half_x}')
        print(f'Correlation (first half) - Y: {corr_first_half_y}')
        print(f'Correlation (second half) - Y: {corr_second_half_y}')

        # Saving plots for X and Y comparisons
        save_path_halfx1 = f'D:\Program Files (x86)\Software\OneDrive\PyPackages_img\Compare_half\{to_process_itemR1[idx]}_half1_x_compare.png'
        save_path_halfy1 = f'D:\Program Files (x86)\Software\OneDrive\PyPackages_img\Compare_half\{to_process_itemR1[idx]}_half1_y_compare.png'
        save_path_halfx2 = f'D:\Program Files (x86)\Software\OneDrive\PyPackages_img\Compare_half\{to_process_itemR1[idx]}_half2_x_compare.png'
        save_path_halfy2 = f'D:\Program Files (x86)\Software\OneDrive\PyPackages_img\Compare_half\{to_process_itemR1[idx]}_half2_y_compare.png'

        plot_ListFloat_Compare_x(ListFloat1=first_half_list1_x, ListFloat2=second_half_list1_x, SaveFilePath=save_path_halfx1, isShow=False,title=f'{to_process_itemR1[idx][2]}号支架')
        plot_ListFloat_Compare_y(ListFloat1=first_half_list1_y, ListFloat2=second_half_list1_y, SaveFilePath=save_path_halfy1, isShow=False,title=f'{to_process_itemR1[idx][2]}号支架')
        plot_ListFloat_Compare_x(ListFloat1=first_half_list2_x, ListFloat2=second_half_list2_x, SaveFilePath=save_path_halfx2, isShow=False,title=f'{to_process_itemR1[idx][2]}号支架')
        plot_ListFloat_Compare_y(ListFloat1=first_half_list2_y, ListFloat2=second_half_list2_y, SaveFilePath=save_path_halfy2, isShow=False,title=f'{to_process_itemR1[idx][2]}号支架')

def main_process_GNSS_data_compare():
    all_data = main_load_GNSS_data()
    print("---------------------main_process_GNSS_data_compare-------------------")

    
    to_process_itemR1 = ['R031_0407', 'R032_0407', 'R051_0407', 'R052_0407', 'R071_0407', 'R072_0407', 'R081_0407','R082_0407']
    to_process_itemR2 = ['R031_1215','R032_1215', 'R051_1215', 'R052_1215','R071_1215', 'R072_1215', 'R081_1215', 'R082_1215']

    for idx in range(len(to_process_itemR1)):
        aligned_list1_x = []
        aligned_list2_x = []

        aligned_list1_y = []
        aligned_list2_y = []
        idx1 = 0
        idx2 = 0
        dt_list1 = all_data[to_process_itemR1[idx]][2]
        dt_list2 = all_data[to_process_itemR2[idx]][2]
        ListFloat1_x = all_data[to_process_itemR1[idx]][0]
        ListFloat2_x = all_data[to_process_itemR2[idx]][0]

        ListFloat1_y = all_data[to_process_itemR1[idx]][1]
        ListFloat2_y = all_data[to_process_itemR2[idx]][1]
        while idx1 < len(dt_list1) and idx2 < len(dt_list2):
            if dt_list1[idx1] == dt_list2[idx2]:
                aligned_list1_x.append(ListFloat1_x[idx1])
                aligned_list2_x.append(ListFloat2_x[idx2])
                aligned_list1_y.append(ListFloat1_y[idx1])
                aligned_list2_y.append(ListFloat2_y[idx2])

                idx1 += 1
                idx2 += 1
            elif dt_list1[idx1] < dt_list2[idx2]:
                idx1 += 1
            else:
                idx2 += 1

        print(len(aligned_list1_x))
        print(len(aligned_list2_x))
        print(f'----------{to_process_itemR1[idx]}&{to_process_itemR2[idx]}-----------')
        print(f'{to_process_itemR1[idx]}:')
        results1 = analyze_variability(aligned_list1_y)
        print()
        print(f'{to_process_itemR2[idx]}:')
        results2 = analyze_variability(aligned_list2_y)
        print()
        # 比较两个结果的差异
        for key in results1:
            difference = results1[key] - results2[key]
            print(f"{key}: 差异 = {difference:.2f}")
# 计算两两之间的Pearson相关系数并保留两位小数
def calculate_pearson(x_list, y_list):
    corr, _ = pearsonr(x_list, y_list)
    return round(corr, 3)
def main_process_GNSS_data_compare_with_tiltmeter():
    print("---------------------main_process_GNSS_data_compare_with_tiltmeter-------------------")
    tiltmeter_data_0407 = TiltmeterDataAvage.read_tiltmeter_data(r"D:\Program Files (x86)\Software\OneDrive\PyPackages_DataSave\new_data\tiltmeter_0407.txt")
    tiltmeter_data_1215 = TiltmeterDataAvage.read_tiltmeter_data(r"D:\Program Files (x86)\Software\OneDrive\PyPackages_DataSave\new_data\tiltmeter_1215.txt")
    common_dates = TiltmeterDataAvage.find_common_dates(tiltmeter_data_0407,tiltmeter_data_1215)
    # 打印共同日期的数据
    # 初始化列表
    pitch_0407 = []
    roll_0407 = []
    pitch_1215 = []
    roll_1215 = []
    dt_list = []
    # # 提取共同日期的数据并将值存入列表
    for date_obj, (data1, data2) in common_dates.items():
        pitch_0407.append(data1.pitch)
        roll_0407.append(data1.roll)
        pitch_1215.append(data2.pitch)
        roll_1215.append(data2.roll)
        dt_list.append(date_obj)


    print("---------------------main_load_GNSS_data-------------------")
    to_process_itemR1 = ['R081_0407' ]#'R032_0407', 'R051_0407', 'R052_0407', 'R071_0407', 'R072_0407', 'R081_0407','R082_0407']
    to_process_itemR2 = ['R081_1215' ]#'R032_1215', 'R051_1215', 'R052_1215','R071_1215', 'R072_1215', 'R081_1215', 'R082_1215']
    all_data = main_load_GNSS_data()
    for idx in range(len(to_process_itemR1)):
        aligned_list1_x = []
        aligned_list2_x = []

        aligned_list1_y = []
        aligned_list2_y = []

        aligned_list_time = []
        idx1 = 0
        idx2 = 0
        dt_list1 = all_data[to_process_itemR1[idx]][2]
        dt_list2 = all_data[to_process_itemR2[idx]][2]
        ListFloat1_x = all_data[to_process_itemR1[idx]][0]
        ListFloat2_x = all_data[to_process_itemR2[idx]][0]

        ListFloat1_y = all_data[to_process_itemR1[idx]][1]
        ListFloat2_y = all_data[to_process_itemR2[idx]][1]
        while idx1 < len(dt_list1) and idx2 < len(dt_list2):
            if dt_list1[idx1] == dt_list2[idx2]:
                aligned_list1_x.append(ListFloat1_x[idx1])
                aligned_list2_x.append(ListFloat2_x[idx2])
                aligned_list1_y.append(ListFloat1_y[idx1])
                aligned_list2_y.append(ListFloat2_y[idx2])
                aligned_list_time.append(dt_list1[idx1])
                idx1 += 1
                idx2 += 1
            elif dt_list1[idx1] < dt_list2[idx2]:
                idx1 += 1
            else:
                idx2 += 1

        x1_list = []
        y1_list = []
        x2_list = []
        y2_list = []
        t_list = []
        p1_list = []
        r1_list = []
        p2_list = []
        r2_list = []
        for dt_idx in range(len(aligned_list_time)):
            if aligned_list_time[dt_idx] in dt_list:
                x1_list.append(aligned_list1_x[dt_idx])
                y1_list.append(aligned_list1_y[dt_idx])
                x2_list.append(aligned_list2_x[dt_idx])
                y2_list.append(aligned_list2_y[dt_idx])
                t_list.append(aligned_list_time[dt_idx])
                p1_list.append(common_dates[aligned_list_time[dt_idx]][0].pitch)
                r1_list.append(common_dates[aligned_list_time[dt_idx]][0].roll)
                p2_list.append(common_dates[aligned_list_time[dt_idx]][1].pitch)
                r2_list.append(common_dates[aligned_list_time[dt_idx]][1].roll)
        
        # 初始化相关系数列表
        correlations = []

        # 生成所有变量之间的相关系数表格
        # 生成所有变量之间的相关系数表格
        data = {
            ('X1', 'P1'): calculate_pearson(x1_list, p1_list),
            ('X1', 'P2'): calculate_pearson(x1_list, p2_list),
            ('X1', 'R1'): calculate_pearson(x1_list, r1_list),
            ('X1', 'R2'): calculate_pearson(x1_list, r2_list),
            ('X2', 'P1'): calculate_pearson(x2_list, p1_list),
            ('X2', 'P2'): calculate_pearson(x2_list, p2_list),
            ('X2', 'R1'): calculate_pearson(x2_list, r1_list),
            ('X2', 'R2'): calculate_pearson(x2_list, r2_list),
            ('Y1', 'P1'): calculate_pearson(y1_list, p1_list),
            ('Y1', 'P2'): calculate_pearson(y1_list, p2_list),
            ('Y1', 'R1'): calculate_pearson(y1_list, r1_list),
            ('Y1', 'R2'): calculate_pearson(y1_list, r2_list),
            ('Y2', 'P1'): calculate_pearson(y2_list, p1_list),
            ('Y2', 'P2'): calculate_pearson(y2_list, p2_list),
            ('Y2', 'R1'): calculate_pearson(y2_list, r1_list),
            ('Y2', 'R2'): calculate_pearson(y2_list, r2_list),
            ('p1', 'r1'): calculate_pearson(p1_list, r1_list),
            ('p2', 'r2'): calculate_pearson(p2_list, r2_list),
        }

        # 创建DataFrame来展示相关系数
        df = pd.DataFrame(data.values(), index=data.keys(), columns=['Pearson Correlation'])

        # 打印DataFrame
        print(df)

def main_process_GNSS_data_compare_with_met():
    print("---------------------main_process_GNSS_data_compare_with_met-------------------")
    # File path
    file_path = r"D:\Program Files (x86)\Software\OneDrive\PyPackages_DataSave\weather_temperature.txt"   
    # Read weather data
    weather_data: Dict[str, Tuple[float, float]] = read_weather_data(file_path)
    temperature_data, temperature_get_maxmin_time = read_tianmeng_met()

def main_tiltmeter_compare():
    print("---------------------main_process_GNSS_data_compare_with_tiltmeter-------------------")
    tiltmeter_data_0407 = TiltmeterDataAvage.read_tiltmeter_data(r"D:\Program Files (x86)\Software\OneDrive\PyPackages_DataSave\new_data\tiltmeter_0407.txt")
    tiltmeter_data_1215 = TiltmeterDataAvage.read_tiltmeter_data(r"D:\Program Files (x86)\Software\OneDrive\PyPackages_DataSave\new_data\tiltmeter_1215.txt")
    common_dates = TiltmeterDataAvage.find_common_dates(tiltmeter_data_0407,tiltmeter_data_1215)
    # 打印共同日期的数据
    # 初始化列表
    pitch_0407 = []
    roll_0407 = []
    pitch_1215 = []
    roll_1215 = []
    dt_list = []
    # # 提取共同日期的数据并将值存入列表
    for date_obj, (data1, data2) in common_dates.items():
        pitch_0407.append(data1.pitch)
        roll_0407.append(data1.roll)
        pitch_1215.append(data2.pitch)
        roll_1215.append(data2.roll)
        dt_list.append(date_obj)

    print(f'roll_0407')
    results1 = analyze_variability(pitch_0407)
    print()
    print(f'roll_1215:')
    results2 = analyze_variability(pitch_1215)
    print()
    # 比较两个结果的差异
    for key in results1:
        difference = results1[key] - results2[key]
        print(f"{key}: 差异 = {difference:.4f}")

    # plot_ListFloat_Compare_pitch_coor(ListFloat1=roll_0407,ListFloat2=roll_1215,title='')

def calculate_attitude_angles(point1, point2):
    # 计算两点间的差向量
    vector = point2 * 1000- point1 * 1000
    print(vector)
    # 计算向量的模长
    magnitude = np.linalg.norm(vector)
    

    # 避免除以零的错误
    if magnitude == 0:
        return None

    # 计算各个角度
    # 俯仰角 Pitch: arctan(dy/dz) 如果dz为零，则处理特殊情况
    pitch = np.arctan2(vector[1], vector[2]) if vector[2] != 0 else np.pi/2 if vector[1] > 0 else -np.pi/2

    # 偏航角 Yaw: arctan(dx/dz) 如果dz为零，则处理特殊情况
    yaw = np.arctan2(vector[0], vector[2]) if vector[2] != 0 else np.pi/2 if vector[0] > 0 else -np.pi/2

    # 翻滚角 Roll: arctan(dy/dx) 如果dx为零，则处理特殊情况
    roll = np.arctan2(vector[1], vector[0]) if vector[0] != 0 else np.pi/2 if vector[1] > 0 else -np.pi/2

    # 转换为度
    pitch_deg = np.degrees(pitch)
    yaw_deg = np.degrees(yaw)
    roll_deg = np.degrees(roll)


    # 计算与90度的差值
    # pitch_deg = np.abs(90 - pitch_deg)
    # roll_deg = np.abs(90 - roll_deg)

    return {'Pitch': pitch_deg, 'Yaw': yaw_deg, 'Roll': roll_deg}

def main_process_GNSS_data_pitchrow():
    all_data = main_load_GNSS_data()

    
    print("---------------------main_process_GNSS_data_compare-------------------")

    
    to_process_itemR1 = ['R071_1215']
    to_process_itemR2 = ['R072_1215']


    aligned_list1_x = []
    aligned_list2_x = []

    aligned_list1_y = []
    aligned_list2_y = []

    aligned_list1_z = []
    aligned_list2_z = []

    aligned_dt = []
    for idx in range(len(to_process_itemR1)):

        idx1 = 0
        idx2 = 0

        dt_list1 = all_data[to_process_itemR1[idx]][2]
        dt_list2 = all_data[to_process_itemR2[idx]][2]

        ListFloat1_x = all_data[to_process_itemR1[idx]][0]
        ListFloat2_x = all_data[to_process_itemR2[idx]][0]

        ListFloat1_y = all_data[to_process_itemR1[idx]][1]
        ListFloat2_y = all_data[to_process_itemR2[idx]][1]

        ListFloat1_z = all_data[to_process_itemR1[idx]][3]
        ListFloat2_z = all_data[to_process_itemR2[idx]][3]

        while idx1 < len(dt_list1) and idx2 < len(dt_list2):
            if dt_list1[idx1] == dt_list2[idx2]:
                aligned_list1_x.append(ListFloat1_x[idx1])
                aligned_list2_x.append(ListFloat2_x[idx2])

                aligned_list1_y.append(ListFloat1_y[idx1])
                aligned_list2_y.append(ListFloat2_y[idx2])

                aligned_list1_z.append(ListFloat1_z[idx1])
                aligned_list2_z.append(ListFloat2_z[idx2])

                aligned_dt.append(dt_list1[idx1])

                idx1 += 1
                idx2 += 1
            elif dt_list1[idx1] < dt_list2[idx2]:
                idx1 += 1
            else:
                idx2 += 1

    attitude_angles_list = []
    for i in range(len(aligned_list1_x)):
        point1 = np.array([aligned_list1_x[i], aligned_list1_y[i], aligned_list1_z[i]])
        point2 = np.array([aligned_list2_x[i], aligned_list2_y[i], aligned_list2_z[i]])
        attitude_angles = calculate_attitude_angles(point1, point2)
        if attitude_angles:
            attitude_angles_list.append(attitude_angles)


    pitch_list = []
    roll_list = []
    yaw_list = []
    for i in range(len(attitude_angles_list)):
        pitch_list.append(attitude_angles_list[i]['Pitch'])
        roll_list.append(attitude_angles_list[i]['Roll'])
        yaw_list.append(attitude_angles_list[i]['Yaw'])
    # plot_ListFloat_pitch(pitch_list)
    # plot_ListFloat_yaw(yaw_list)
    # plot_ListFloat_roll(roll_list)



    tiltmeter_data_1215: Dict[datetime.date, 'TiltmeterDataAvage'] = TiltmeterDataAvage.read_tiltmeter_data(r"D:\Program Files (x86)\Software\OneDrive\PyPackages_DataSave\new_data\tiltmeter_1215.txt")
    pitch_1215 = []
    roll_1215 = []
    dt_list = []
    for date in tiltmeter_data_1215:
        dt_list.append(date)
        pitch_1215.append(tiltmeter_data_1215[date].pitch)
        roll_1215.append(tiltmeter_data_1215[date].roll)
    # 打印共同日期的数据
    # 初始化列表

    common_pitch = []
    common_roll = []

    com_pitch = []
    com_yaw= []
    com_roll = []
    # # 提取共同日期的数据并将值存入列表

    idx1 = 0
    idx2 = 0


    while idx1 < len(aligned_dt) and idx2 < len(dt_list):
        print(f'idx1: {aligned_dt[idx1]} ,idx2: {dt_list[idx2]}')
        if aligned_dt[idx1] == dt_list[idx2]:
            common_pitch.append(pitch_1215[idx2])
            common_roll.append(roll_1215[idx2])

            com_pitch.append(pitch_list[idx1])
            com_yaw.append(yaw_list[idx1])
            com_roll.append(roll_list[idx1])
            idx1 += 1
            idx2 += 1
        elif aligned_dt[idx1] < dt_list[idx2]:
            idx1 += 1
        else:
            idx2 += 1

    plot_ListFloat_Compare_pitch_coor(common_pitch,com_roll,title=' ')
    #     ile.write(f"{aligned_dt[i]}\t{attitude_angles_list[i]['Pitch']}\t{attitude_angles_list[i]['Yaw']}\t{attitude_angles_list[i]['Roll']}\n")

    # with open('date.txt', 'w') as file:
    #     for i in range(len(attitude_angles_list)):
    #         file.write(f"{aligned_dt[i]}\t{attitude_angles_list[i]['Pitch']}\t{attitude_angles_list[i]['Yaw']}\t{attitude_angles_list[i]['Roll']}\n")


if __name__ == "__main__":
    print("---------------------run-------------------")
    main_process_GNSS_data_pitchrow()