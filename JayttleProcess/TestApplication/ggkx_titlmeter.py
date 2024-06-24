# 标准库导入
import math
import random
from datetime import datetime, timedelta
import time
import warnings
import os
import shutil

# 相关第三方库导入
import pyproj
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
from typing import List, Optional, Tuple, Union, Set, Dict
import pywt
from JayttleProcess import TimeSeriesDataMethod, TBCProcessCsv, CommonDecorator
from JayttleProcess.TimeSeriesDataMethod import TimeSeriesData
from JayttleProcess import ListFloatDataMethod as LFDM

# region 字体设置
plt.rcParams['font.sans-serif'] = ['SimSun']  # 指定宋体为默认字体
plt.rcParams['font.sans-serif'] = 'SimSun'  # 使用指定的中文字体


class TiltmeterData:
    def __init__(self, time: str, station_id: int, pitch: float, roll: float):
        try:
            self.time = datetime.strptime(time, '%Y-%m-%d %H:%M:%S.%f')  # 尝试包含毫秒部分的格式
        except ValueError:
            self.time = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')  # 失败时尝试不包含毫秒部分的格式
        self.station_id = station_id
        self.pitch = pitch
        self.roll = roll

    @classmethod
    @CommonDecorator.log_function_call
    def from_file(cls, file_path: str) -> List['TiltmeterData']:
        """
        数据读取 129904 24.2381s
        效率 5361个/s
        8757913 356.5892s (有判断的)
        """
        tiltmeter_data_list: List['TiltmeterData'] = []
        with open(file_path, 'r') as file:
            next(file)  # Skip the header row
            for line in file:
                data = line.strip().split('\t')
                if len(data) == 4:
                    data[0] = str(data[0])
                    data[1] = int(data[1])  
                    data[2] = float(data[2])  
                    data[3] = float(data[3]) 
                    tiltmeter_data_list.append(TiltmeterData(*data))

        return tiltmeter_data_list

    @classmethod
    @CommonDecorator.log_function_call
    def from_file_with_date(cls, file_path: str, date: str) -> List['TiltmeterData']:
        """
        数据读取 129904 24.2381s
        效率 5361个/s
        8757913 356.5892s (有判断的)
        """
        tiltmeter_data_list: List['TiltmeterData'] = []
        with open(file_path, 'r') as file:
            next(file)  # Skip the header row
            for line in file:
                data = line.strip().split('\t')
                if len(data) == 4:
                    data[0] = str(data[0])
                    data[1] = int(data[1])  
                    data[2] = float(data[2])  
                    data[3] = float(data[3]) 
                    try:
                        time = datetime.strptime(data[0], '%Y-%m-%d %H:%M:%S.%f')  # 尝试包含毫秒部分的格式
                    except ValueError:
                        time = datetime.strptime(data[0], '%Y-%m-%d %H:%M:%S')  # 失败时尝试不包含毫秒部分的格式
                    if time.strftime('%Y-%m-%d') == date:
                        tiltmeter_data_list.append(TiltmeterData(*data))

        return tiltmeter_data_list
    def from_file_return_dict(file_path: str) -> Dict[datetime.date, list['TiltmeterData']]:
        tiltmeter_data_dict: Dict[datetime.date, 'TiltmeterData'] = defaultdict(list)
        with open(file_path, 'r') as file:
            next(file)  # Skip the header row
            for line in file:
                data = line.strip().split('\t')
                if len(data) == 4:
                    try:
                        date = datetime.strptime(data[0], '%Y-%m-%d %H:%M:%S.%f').date()
                    except ValueError:
                        date = datetime.strptime(data[0], '%Y-%m-%d %H:%M:%S').date()
                    data[0] = str(data[0])
                    data[1] = int(data[1])
                    data[2] = float(data[2])
                    data[3] = float(data[3])
                    tiltmeter_data = TiltmeterData(*data)  # Skip the date in data[0]
                    tiltmeter_data_dict[date].append(tiltmeter_data)

        return tiltmeter_data_dict
    

    def from_file_return_dict_in_hour_range(file_path: str) -> Dict[datetime.date, List['TiltmeterData']]:
        tiltmeter_data_dict: Dict[datetime.date, List[TiltmeterData]] = defaultdict(list)
        with open(file_path, 'r') as file:
            next(file)  # Skip the header row
            for line in file:
                data = line.strip().split('\t')
                if len(data) == 4:
                    try:
                        date = datetime.strptime(data[0], '%Y-%m-%d %H:%M:%S.%f').date()
                    except ValueError:
                        try:
                            date = datetime.strptime(data[0], '%Y-%m-%d %H:%M:%S').date()
                        except ValueError as e:
                            print(f"Error parsing date from line '{line}': {e}")
                            continue

                    # Check if hour (%H) is 20, 21, 22, or 23
                    hour = int(data[0].split()[1].split(':')[0])
                    if hour in [20, 21, 22, 23]:
                        data[1] = int(data[1])
                        data[2] = float(data[2])
                        data[3] = float(data[3])
                        tiltmeter_data = TiltmeterData(*data[1:])  # Skip the date in data[0]
                        tiltmeter_data_dict[date].append(tiltmeter_data)
        return tiltmeter_data_dict


 
    @classmethod
    def filter_by_date(cls, tiltmeter_data_list: List['TiltmeterData'], date: str) -> List['TiltmeterData']:
        filtered_data: List['TiltmeterData'] = [data for data in tiltmeter_data_list if data.time.strftime('%Y-%m-%d') == date]
        return filtered_data

    @classmethod
    def save_to_file(cls, tiltmeter_data_list: List['TiltmeterData'], file_path: str) -> None:
        with open(file_path, 'w') as file:
            for data in tiltmeter_data_list:
                file.write(f"{data.time.strftime('%Y-%m-%d %H:%M:%S.%f')}\t{data.station_id}\t{data.pitch}\t{data.roll}\n")
  
    @classmethod
    def plot_tiltmeter_data(cls, tiltmeter_data_list: List['TiltmeterData']):
        # 提取 pitch 和 roll 数据以及时间数据
        pitch_data = [data.pitch for data in tiltmeter_data_list]
        roll_data = [data.roll for data in tiltmeter_data_list]
        time_list = [data.time for data in tiltmeter_data_list]

        # 创建两个子图
        plt.figure(figsize=(14.4, 9))

        # 设置日期格式化器和日期刻度定位器
        date_fmt = mdates.DateFormatter("%m-%d:%H")  # 仅显示月-日-时
        date_locator = mdates.AutoDateLocator()  # 自动选择刻度间隔

        # 绘制 pitch 时序图
        plt.subplot(2, 1, 1)
        plt.plot(time_list, pitch_data, color='blue')
        plt.title('时间轴上的俯仰角变化')
        plt.xlabel('日期')
        plt.ylabel('俯仰角')
        plt.gca().xaxis.set_major_formatter(date_fmt)
        plt.gca().xaxis.set_major_locator(date_locator)
        plt.yticks(plt.yticks()[0][::len(plt.yticks()[0]) // 5])  # 设置 y 轴刻度为五个

        # 绘制 roll 时序图
        plt.subplot(2, 1, 2)
        plt.plot(time_list, roll_data, color='green')
        plt.title('时间轴上的横滚角变化')
        plt.xlabel('日期')
        plt.ylabel('横滚角')
        plt.gca().xaxis.set_major_formatter(date_fmt)
        plt.gca().xaxis.set_major_locator(date_locator)
        plt.yticks(plt.yticks()[0][::len(plt.yticks()[0]) // 5])  # 设置 y 轴刻度为五个

        plt.tight_layout(pad=3.0)  # 调整子图布局以防止重叠，并设置较大的pad值以确保y轴标签不会被截断
        plt.show()


    @classmethod
    def plot_tiltmeter_data_with_marker(cls, tiltmeter_data_list: List['TiltmeterData']):
        # 提取 pitch 和 roll 数据以及时间数据
        pitch_data = [data.pitch for data in tiltmeter_data_list]
        roll_data = [data.roll for data in tiltmeter_data_list]
        time_list = [data.time for data in tiltmeter_data_list]

        # 创建两个子图
        plt.figure(figsize=(14.4, 9))

        # 设置日期格式化器和日期刻度定位器
        date_fmt = mdates.DateFormatter("%m-%d:%H")  # 仅显示月-日-时
        date_locator = mdates.AutoDateLocator()  # 自动选择刻度间隔

        # 绘制 pitch 时序图
        plt.subplot(2, 1, 1)
        plt.plot(time_list, pitch_data, color='blue')
        plt.title('时间轴上的俯仰角变化')
        plt.xlabel('日期')
        plt.ylabel('俯仰角')
        plt.gca().xaxis.set_major_formatter(date_fmt)
        plt.gca().xaxis.set_major_locator(date_locator)
        plt.yticks(plt.yticks()[0][::len(plt.yticks()[0]) // 5])  # 设置 y 轴刻度为五个

        # 在图上标记特殊时间点
        special_times = ['2023-08-10 08:12', '2023-08-10 13:48']
        special_idx = []
        for time_str in special_times:
            special_time_dt = datetime.strptime(time_str, '%Y-%m-%d %H:%M')
            for idx,dt in enumerate(time_list):
                if dt.minute == special_time_dt.minute and dt.hour == special_time_dt.hour:
                    special_idx.append(idx)
        for idx in special_idx:
            plt.scatter(time_list[idx], pitch_data[idx], color='red', s=10, marker='x')
        # 绘制 roll 时序图
        plt.subplot(2, 1, 2)
        plt.plot(time_list, roll_data, color='green')
        plt.title('时间轴上的横滚角变化')
        plt.xlabel('日期')
        plt.ylabel('横滚角')
        plt.gca().xaxis.set_major_formatter(date_fmt)
        plt.gca().xaxis.set_major_locator(date_locator)
        plt.yticks(plt.yticks()[0][::len(plt.yticks()[0]) // 5])  # 设置 y 轴刻度为五个

        plt.tight_layout(pad=3.0)  # 调整子图布局以防止重叠，并设置较大的pad值以确保y轴标签不会被截断
        plt.show()


    @classmethod
    def plot_tiltmeter_data_with_vibration_detection(cls, tiltmeter_data_list: List['TiltmeterData'], specified_date=None, window_size=20, threshold=0.1):
        # 提取 pitch 和 roll 数据以及时间数据
        pitch_data = [data.pitch for data in tiltmeter_data_list]
        roll_data = [data.roll for data in tiltmeter_data_list]
        time_list = [data.time for data in tiltmeter_data_list]

        # 检查指定日期是否存在于数据中
        specified_date_exists = False
        specified_date_data_indices = []
        for idx, time in enumerate(time_list):
            if time.date() == datetime.strptime(specified_date, '%Y-%m-%d').date():
                specified_date_exists = True
                specified_date_data_indices.append(idx)
        
        # 如果指定日期不存在，则打印错误消息并返回
        if not specified_date_exists:
            print("错误：数据中未找到指定的日期。")
            return

        # 提取指定日期的数据
        specified_date_tiltmeter_data = [pitch_data[i] for i in specified_date_data_indices]

        if not specified_date_tiltmeter_data:
            print("指定日期无可用数据。")
            return

        # 提取 pitch 数据
        pitch_data = np.array(specified_date_tiltmeter_data)
        
        # 计算振动幅度
        vibration_magnitude = []
        num_windows = len(pitch_data) - window_size + 1
        for i in range(num_windows):
            window_pitch_data = pitch_data[i:i+window_size]
            window_std = np.std(window_pitch_data)  # 使用标准差作为振动幅度的度量
            vibration_magnitude.append(window_std)

        # 根据振动幅度的大小识别不同部分
        non_vibration_indices = np.where(np.array(vibration_magnitude) <= threshold)[0]
        vibration_indices = np.where(np.array(vibration_magnitude) > threshold)[0]
        
        # 打印第一个振动索引对应的时间
        if vibration_indices.size > 0:
            idx = vibration_indices[0]
            print(f"第一个振动索引为 {idx} 对应的时间为 {time_list[idx]}")

        # 可视化结果
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(pitch_data)), pitch_data, color='blue', label='Pitch Data')
        plt.scatter(non_vibration_indices, pitch_data[non_vibration_indices], c='green', marker='o', label='非振动')
        plt.scatter(vibration_indices, pitch_data[vibration_indices], c='red', marker='x', label='振动')
        plt.xlabel('数据索引')
        plt.ylabel('Pitch')
        plt.title('带振动检测的倾斜计数据')
        plt.legend()
        plt.show() 

    @classmethod
    def plot_tiltmeter_data_in_specified_date_with_vibration_detection(cls, tiltmeter_data_list: List['TiltmeterData'], specified_date=None, window_size=30, threshold=0.1):
        # 提取 pitch 和 roll 数据以及时间数据
        pitch_data = [data.pitch for data in tiltmeter_data_list]
        roll_data = [data.roll for data in tiltmeter_data_list]
        time_list = [data.time for data in tiltmeter_data_list]

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

        # 创建两个子图
        plt.figure(figsize=(14.4, 9.6))

        # 设置日期格式化器和日期刻度定位器
        date_fmt = mdates.DateFormatter("%m-%d %H:00")  # 仅显示月-日-时
        date_locator = mdates.AutoDateLocator()  # 自动选择刻度间隔

        # 计算振动幅度
        vibration_magnitude_pitch = []
        vibration_magnitude_roll = []
        num_windows = len(pitch_data) - window_size + 1
        for i in range(num_windows):
            window_pitch_data = pitch_data[i:i+window_size]
            window_roll_data = roll_data[i:i+window_size]
            window_std_pitch = np.std(window_pitch_data)  # 使用标准差作为振动幅度的度量
            window_std_roll = np.std(window_roll_data)
            vibration_magnitude_pitch.append(window_std_pitch)
            vibration_magnitude_roll.append(window_std_roll)

        # 根据振动幅度的大小识别不同部分
        non_vibration_indices_pitch = np.where(np.array(vibration_magnitude_pitch) <= threshold)[0]
        vibration_indices_pitch = np.where(np.array(vibration_magnitude_pitch) > threshold)[0]

        threshold = 0.05
        non_vibration_indices_roll = np.where(np.array(vibration_magnitude_roll) <= threshold)[0]
        vibration_indices_roll = np.where(np.array(vibration_magnitude_roll) > threshold)[0]

        # 绘制 pitch 时序图
        plt.subplot(2, 1, 1)
        plt.plot([time_list[i] for i in specified_date_data_indices], [pitch_data[i] for i in specified_date_data_indices], color='blue', label='俯仰角')
        plt.title('带振动检测的俯仰角变化')
        plt.xlabel('日期')
        plt.ylabel('俯仰角(°)')
        # 在第一个子图中添加震荡检测结果
        plt.scatter([time_list[i] for i in specified_date_data_indices if i in vibration_indices_pitch], [pitch_data[i] for i in specified_date_data_indices if i in vibration_indices_pitch], c='red', marker='x', label='振动点')
        plt.legend()
        plt.gca().xaxis.set_major_formatter(date_fmt)
        plt.gca().xaxis.set_major_locator(date_locator)

        # 绘制 roll 时序图
        plt.subplot(2, 1, 2)
        plt.plot([time_list[i] for i in specified_date_data_indices], [roll_data[i] for i in specified_date_data_indices], color='green', label='横滚角')
        plt.title('带振动检测的横滚角变化')
        plt.xlabel('日期')
        plt.ylabel('横滚角(°)')
        plt.gca().xaxis.set_major_formatter(date_fmt)
        plt.gca().xaxis.set_major_locator(date_locator)

        # 在第二个子图中添加震荡检测结果
        plt.scatter([time_list[i] for i in specified_date_data_indices if i in vibration_indices_roll], [roll_data[i] for i in specified_date_data_indices if i in vibration_indices_roll], c='red', marker='x', label='振动点')
        plt.legend()

        plt.tight_layout()  # 调整子图布局以防止重叠
        plt.show()

    @classmethod
    def plot_tiltmeter_data_in_specified_date(cls, tiltmeter_data_list: List['TiltmeterData'], specified_date=None):
        # 提取 pitch 和 roll 数据以及时间数据
        pitch_data = [data.pitch for data in tiltmeter_data_list]
        roll_data = [data.roll for data in tiltmeter_data_list]
        time_list = [data.time for data in tiltmeter_data_list]

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

        # 创建两个子图
        plt.figure(figsize=(14.4, 9.6))

        # # 设置日期格式化器和日期刻度定位器
        date_fmt = mdates.DateFormatter("%m-%d %H:00")  # 仅显示月-日-时
        date_locator = mdates.AutoDateLocator()  # 自动选择刻度间隔

        # 绘制 pitch 时序图
        plt.subplot(2, 1, 1)
        plt.plot([time_list[i] for i in specified_date_data_indices], [pitch_data[i] for i in specified_date_data_indices], color='blue')
        plt.title('俯仰角变化')
        plt.xlabel('日期')
        plt.ylabel('俯仰角(°)')
        plt.gca().xaxis.set_major_formatter(date_fmt)
        plt.gca().xaxis.set_major_locator(date_locator)

        # 绘制 roll 时序图
        plt.subplot(2, 1, 2)
        plt.plot([time_list[i] for i in specified_date_data_indices], [roll_data[i] for i in specified_date_data_indices], color='green')
        plt.title('横滚角变化')
        plt.xlabel('日期')
        plt.ylabel('横滚角(°)')
        plt.gca().xaxis.set_major_formatter(date_fmt)
        plt.gca().xaxis.set_major_locator(date_locator)

        plt.tight_layout()  # 调整子图布局以防止重叠
        plt.show()

    @classmethod
    def normalize_tiltmeter_data(cls, tiltmeter_data_list: List['TiltmeterData']) -> List['TiltmeterData']:
        # 找到pitch和roll的最大值和最小值
        max_pitch = max(data.pitch for data in tiltmeter_data_list)
        min_pitch = min(data.pitch for data in tiltmeter_data_list)
        max_roll = max(data.roll for data in tiltmeter_data_list)
        min_roll = min(data.roll for data in tiltmeter_data_list)

        # 归一化处理
        normalized_data_list = []
        for data in tiltmeter_data_list:
            normalized_pitch = (data.pitch - min_pitch) / (max_pitch - min_pitch)
            normalized_roll = (data.roll - min_roll) / (max_roll - min_roll)
            normalized_data_list.append(TiltmeterData(data.time, data.station_id, normalized_pitch, normalized_roll))

        return normalized_data_list
    
    @classmethod
    def get_all_dates(cls, tiltmeter_data_list: List['TiltmeterData']) -> Set[str]:
        all_dates: Set[str] = set()
        for data in tiltmeter_data_list:
            date_str = data.time.strftime('%Y-%m-%d')
            all_dates.add(date_str)
        return all_dates
    
    @classmethod
    def filter_by_time_range(cls, tiltmeter_data_list: List['TiltmeterData'], list_hour: list[int]) -> List['TiltmeterData']:
        """
        根据时间范围过滤数据，筛选出时间在特定小时范围内的数据。
        """
        filtered_data: List['TiltmeterData'] = []
        for data in tiltmeter_data_list:
            data_time = data.time
            if data_time.hour in list_hour:
                filtered_data.append(data)
        return filtered_data
    
    @classmethod
    def calculate_daily_average(cls, tiltmeter_data_list: List['TiltmeterData'], date: str) -> tuple:
        filtered_data = cls.filter_by_date(tiltmeter_data_list, date)
        if not filtered_data:
            return (0.0, 0.0)  # Return (0.0, 0.0) if no data for the given date

        num_data_points = len(filtered_data)
        total_pitch = sum(data.pitch for data in filtered_data)
        total_roll = sum(data.roll for data in filtered_data)

        average_pitch = total_pitch / num_data_points
        average_roll = total_roll / num_data_points

        return (average_pitch, average_roll)
    
    @classmethod
    def save_filtered_data_to_file(cls, tiltmeter_data_list: List['TiltmeterData'], file_path: str) -> None:
        """
        将过滤后的数据保存到文件。
        """
        with open(file_path, 'w') as file:
            for data in tiltmeter_data_list:
                file.write(f"{data.time.strftime('%Y-%m-%d %H:%M:%S.%f')}\t{data.station_id}\t{data.pitch}\t{data.roll}\n")

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

class TiltmeterDataAvage:
    def __init__(self,pitch: float, roll: float, num: int):
        self.pitch = pitch
        self.roll = roll
        self.num = num


    @classmethod
    def calculate_average(cls, tiltmeter_data_list: List[TiltmeterData]) -> 'TiltmeterDataAvage':
        total_pitch = sum(data.pitch for data in tiltmeter_data_list)
        total_roll = sum(data.roll for data in tiltmeter_data_list)
        num_samples = len(tiltmeter_data_list)
        
        if num_samples > 0:
            pitch_avg = total_pitch / num_samples
            roll_avg = total_roll / num_samples
        else:
            pitch_avg = 0.0
            roll_avg = 0.0
        
        return cls(pitch_avg, roll_avg, num_samples)
    
    def read_tiltmeter_data(file_path: str) -> Dict[datetime.date, List['TiltmeterDataAvage']]:
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

class GgkxDataWithCoordinates:
    def __init__(self, time: str, station_id: int, receiver_id: int, east_coordinate: float, north_coordinate: float, geo_height: float, fix_mode: int, satellite_num: int, pdop: float, sigma_e: float, sigma_n: float, sigma_u: float, prop_age: float):
        self.time = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
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


def split_ggkx_data_by_week(ggkx_data: List[GgkxData], output_folder: str):
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 按照时间排序数据
    sorted_data = sorted(ggkx_data, key=lambda x: x.time)
    
    # 计算第一个数据点的日期
    start_date = sorted_data[0].time.date()
    
    # 初始化七天的时间间隔
    week_start = start_date
    week_end = start_date + timedelta(days=6)
    
    # 分割数据并写入文件
    current_week_data = []
    for data in sorted_data:
        data_date = data.time.date()
        if data_date > week_end:
            # 写入当前周的数据到文件
            output_file = os.path.join(output_folder, f"{week_start}_{week_end}.txt")
            write_ggkx_data(current_week_data, output_file)
            
            # 重置当前周的数据和时间间隔
            current_week_data = []
            week_start = data_date
            week_end = week_start + timedelta(days=6)
        
        # 添加数据到当前周
        current_week_data.append(data)

    # 写入最后一周的数据到文件
    output_file = os.path.join(output_folder, f"{week_start}_{week_end}.txt")
    write_ggkx_data(current_week_data, output_file)

def write_ggkx_data(ggkx_data: List[GgkxData], file_path: str):
    with open(file_path, 'w') as file:
        # Write header
        file.write("Time\tStationID\tReceiverID\tLat\tLon\tGeoHeight\tFixMode\tSateNum\tPDOP\tSigmaE\tSigmaN\tSigmaU\tPropAge\n")
        # Write data
        for data in ggkx_data:
            file.write(f"{data.time.strftime('%Y-%m-%d %H:%M:%S')}\t{data.station_id}\t{data.receiver_id}\t{data.latitude}\t{data.longitude}\t{data.geo_height}\t{data.fix_mode}\t{data.satellite_num}\t{data.pdop}\t{data.sigma_e}\t{data.sigma_n}\t{data.sigma_u}\t{data.prop_age}\n")

def plot_data_on_same_day(ggkx_data: List[GgkxData], tiltmeter_data_list: List[TiltmeterData], date: str) -> None:
    # 过滤出日期为给定日期的 ggkx_data 数据
    filtered_ggkx_data = GgkxData.filter_by_date(ggkx_data, date)

    # 找出时间为 2023-8-2 的数据
    filtered_tiltmeter_data: List['TiltmeterData'] = TiltmeterData.filter_by_date(tiltmeter_data_list, date)
    # 提取经度、纬度数据
    ggkx_longitude = [data.longitude for data in filtered_ggkx_data]
    ggkx_latitude = [data.latitude for data in filtered_ggkx_data]
    ggkx_time = [data.time for data in filtered_ggkx_data]


    tiltmeter_pitch = [data.pitch for data in filtered_tiltmeter_data]
    tiltmeter_roll = [data.roll for data in filtered_tiltmeter_data]
    tiltmeter_time = [data.time for data in filtered_tiltmeter_data]

    # 创建两个子图
    plt.figure(figsize=(14.4, 9.6))

    # 设置日期格式化器和日期刻度定位器
    date_fmt = mdates.DateFormatter("%m-%d-%H")  # 仅显示月-日-时
    date_locator = mdates.AutoDateLocator()  # 自动选择刻度间隔

    # 绘制 pitch 时序图
    plt.subplot(4, 1, 1)
    plt.plot(ggkx_time, ggkx_longitude, color='blue')
    plt.title('时间轴上的俯仰角变化')
    plt.xlabel('日期')
    plt.ylabel('俯仰角')
    plt.gca().xaxis.set_major_formatter(date_fmt)
    plt.gca().xaxis.set_major_locator(date_locator)

    # 绘制 roll 时序图
    plt.subplot(4, 1, 2)
    plt.plot(ggkx_time, ggkx_latitude, color='green')
    plt.title('时间轴上的横滚角变化')
    plt.xlabel('日期')
    plt.ylabel('横滚角')
    plt.gca().xaxis.set_major_formatter(date_fmt)
    plt.gca().xaxis.set_major_locator(date_locator)


    # 绘制 roll 时序图
    plt.subplot(4, 1, 3)
    plt.plot(tiltmeter_time, tiltmeter_pitch, color='green')
    plt.title('时间轴上的横滚角变化')
    plt.xlabel('日期')
    plt.ylabel('横滚角')
    plt.gca().xaxis.set_major_formatter(date_fmt)
    plt.gca().xaxis.set_major_locator(date_locator)

    
    # 绘制 roll 时序图
    plt.subplot(4, 1, 4)
    plt.plot(tiltmeter_time, tiltmeter_roll, color='green')
    plt.title('时间轴上的横滚角变化')
    plt.xlabel('日期') 
    plt.ylabel('横滚角')
    plt.gca().xaxis.set_major_formatter(date_fmt)
    plt.gca().xaxis.set_major_locator(date_locator)

    plt.tight_layout()  # 调整子图布局以防止重叠
    plt.show()

def run_main1():
    to_process_item_list = ['R031', 'R032', 'R051', 'R052', 'R071','R072', 'R081', 'R082']
    for item in to_process_item_list:
        data_file: str = rf"C:\Users\Jayttle\Desktop\output_data\ggkx_{item}.txt"
        ggkx_data: List[GgkxData] = GgkxData.read_ggkx_data(data_file)
        ggkx_coordinate_list = GgkxData.convert_to_coordinates(ggkx_data)
        GgkxDataWithCoordinates.filter_data_to_files_with_coordinates(ggkx_coordinate_list, r"C:\Users\Jayttle\Desktop\output_data\Coor8m")


def run_main2():
    data_file: str = rf"C:\Users\Jayttle\Desktop\20230704\ggkx_R031_2023-7-4.txt"
    ggkx_data: List[GgkxData] = GgkxData.read_ggkx_data(data_file)
    ggkx_coordinate_list = GgkxData.convert_to_coordinates(ggkx_data)
    output_folder = rf"C:\Users\Jayttle\Desktop\output_data"
    GgkxDataWithCoordinates.filter_data_to_files_with_coordinates(ggkx_coordinate_list, output_folder, '20230811')
    # GgkxData.filter_data_to_files_in_specified_date(ggkx_data, r'C:\Users\Jayttle\Desktop\20230704', '2023-7-4')

def run_main3():
    to_process_folder_list = ['0107', '0814', '1521', '2228', '2904']
    to_process_file_list = ['R031', 'R032', 'R051', 'R052', 'R071','R072', 'R081', 'R082']

    for folder in to_process_folder_list:
        folder_path = rf"C:\Users\Jayttle\Desktop\output_data\latlon8_{folder}"
        for file_name in to_process_file_list:
            data_file = os.path.join(folder_path, f'ggkx_{file_name}.txt')
            if os.path.exists(data_file):
                ggkx_data = GgkxData.read_ggkx_data(data_file)
                ggkx_coordinate_list = GgkxData.convert_to_coordinates(ggkx_data)
                output_folder = rf"C:\Users\Jayttle\Desktop\output_data\Coor8m_{folder}"
                GgkxDataWithCoordinates.filter_data_to_files_with_coordinates(ggkx_coordinate_list, output_folder)

def run_main4():
    folder_path = r"C:\Users\Jayttle\Desktop\temp"
    # 确保路径存在
    if os.path.exists(folder_path):
        # 列出文件夹下的所有文件
        files = os.listdir(folder_path)
        # 筛选出以".txt"结尾的文件并打印
        txt_files = [file for file in files if file.endswith('R082_2023-8-12.txt')]
        if txt_files:
            for file in txt_files:
                # 提取文件名作为标题
                print(os.path.join(folder_path, file))
                title = os.path.basename(file)
                ggkx_data: List[GgkxData] = GgkxData.read_ggkx_data(os.path.join(folder_path, file))
                ggkx_coordinate_list = GgkxData.convert_to_coordinates(ggkx_data)
                GgkxDataWithCoordinates.filter_data_to_files_with_coordinates(ggkx_coordinate_list, r"C:\Users\Jayttle\Desktop\temp_Coor", title)

def run_main5():
    folder_path = r"D:\Program Files (x86)\Software\OneDrive\PyPackages_img\ggkx"
    # 确保路径存在
    if os.path.exists(folder_path):
        # 列出文件夹下的所有文件
        files = os.listdir(folder_path)
        # 筛选出以".txt"结尾的文件并打印
        png_files = [file for file in files if file.endswith('.png')]
        if png_files:
            for file in png_files:
                # 提取文件名的子串作为目标文件夹名
                subfolder_name = file[5:15]
                # 构建目标文件夹路径
                target_folder_path = os.path.join(folder_path, subfolder_name)
                # 确保目标文件夹存在，如果不存在则创建
                if not os.path.exists(target_folder_path):
                    os.makedirs(target_folder_path)
                # 构建源文件路径和目标文件路径
                source_file_path = os.path.join(folder_path, file)
                target_file_path = os.path.join(target_folder_path, file)
                # 移动文件
                shutil.move(source_file_path, target_file_path)
                print(f"Moved {file} to {target_folder_path}")
    
def day_of_year(year, month, day):
    # 导入datetime模块
    import datetime
    
    # 构建日期对象
    date_obj = datetime.date(year, month, day)
    
    # 获取当年的第几天
    day_of_year = date_obj.timetuple().tm_yday
    
    # 打印结果
    print(f"The day {date_obj} is the {day_of_year}th day of the year {year}.")


def calculate_average_per_day(tiltmeter_data_dict: Dict[datetime.date, List[TiltmeterData]]) -> Dict[datetime.date, TiltmeterDataAvage]:
    avg_dict: Dict[datetime.date, TiltmeterDataAvage] = {}
    for date_obj, data_list in tiltmeter_data_dict.items():
        avg_data = TiltmeterDataAvage.calculate_average(data_list)
        avg_dict[date_obj] = avg_data
    
    return avg_dict


def run_main6():
    data_file = r"D:\Program Files (x86)\Software\OneDrive\PyPackages\tiltmeter20230811"
    tiltmeter_data_list = TiltmeterData.from_file_with_date(data_file, '2023-08-11')
    # tiltmeter_data_dict: Dict[datetime.date, List['TiltmeterDataAvage']] = TiltmeterDataAvage.read_tiltmeter_data(data_file)
    list_pitch = []
    list_roll = []
    list_time = []
    for data in tiltmeter_data_list:
        list_pitch.append(data.pitch)
        list_roll.append(data.roll)
        list_time.append(data.time)
    # for date_obj, avg_data in tiltmeter_data_dict.items(): 
    #     # print(f"日期: {date_obj}, Pitch 平均值: {avg_data.pitch}, Roll 平均值: {avg_data.roll}, 样本数量: {avg_data.num}")
    #     list_pitch.append(avg_data.pitch)
    #     list_roll.append(avg_data.roll)
    # LFDM.plot_ListFloat_with_time(ListFloat=list_pitch, ListTime=list_time, isShow=True, title="pitch")
    TiltmeterData.plot_tiltmeter_data_with_vibration_detection(ListFloat=list_pitch, ListTime=list_time, isShow=True, title="pitch")
    # LFDM.plot_ListFloat_with_time_roll(ListFloat=list_roll, ListTime=list_time, isShow=True, title="roll")
    # TiltmeterData.plot_tiltmeter_data_in_specified_date_with_vibration_detection(tiltmeter_data_list, '2023-08-11')
    # TiltmeterData.plot_tiltmeter_data_with_marker(tiltmeter_data_list)
    # TiltmeterData.plot_tiltmeter_data_with_vibration_detection(tiltmeter_data_list, '2023-08-10')

def run_main7():
    data_file = r"C:\Users\Jayttle\Desktop\tianmeng_tiltmeter.txt"
    tiltmeter_data_dict = TiltmeterData.from_file_return_dict_in_hour_range(data_file)
    avg_dict = calculate_average_per_day(tiltmeter_data_dict)
    file_path = 'tiltmeter1215.txt'
    with open(file_path, 'w') as file:
        for key, data in avg_dict.items():
            file.write(f"{key}\t{data.pitch}\t{data.roll}\t{data.num}\n")
if __name__ == "__main__":
    print("---------------------run-------------------")
    run_main7()

#TODO: 1.用平均值的横滚角和俯仰角来表示索道支架姿态
#TODO：2.绘制曲线和相关系数表示环境数据和
    