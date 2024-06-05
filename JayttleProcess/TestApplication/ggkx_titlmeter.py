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
from typing import List, Optional, Tuple, Union
import pywt
from JayttleProcess import TimeSeriesDataMethod, TBCProcessCsv, CommonDecorator
from JayttleProcess.TimeSeriesDataMethod import TimeSeriesData

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
        """
        tiltmeter_data_list: List['TiltmeterData'] = []
        with open(file_path, 'r') as file:
            next(file)  # Skip the header row
            for line in file:
                data = line.strip().split('\t')
                data[0] = str(data[0])
                data[1] = int(data[1])  
                data[2] = float(data[2])  
                data[3] = float(data[3]) 
                tiltmeter_data_list.append(TiltmeterData(*data))

        return tiltmeter_data_list

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
        date_fmt = mdates.DateFormatter("%m-%d-%H")  # 仅显示月-日-时
        date_locator = mdates.AutoDateLocator()  # 自动选择刻度间隔

        # 绘制 pitch 时序图
        plt.subplot(2, 1, 1)
        plt.plot([time_list[i] for i in specified_date_data_indices], [pitch_data[i] for i in specified_date_data_indices], color='blue')
        plt.title('时间轴上的俯仰角变化')
        plt.xlabel('日期')
        plt.ylabel('俯仰角')
        plt.gca().xaxis.set_major_formatter(date_fmt)
        plt.gca().xaxis.set_major_locator(date_locator)

        # 绘制 roll 时序图
        plt.subplot(2, 1, 2)
        plt.plot([time_list[i] for i in specified_date_data_indices], [roll_data[i] for i in specified_date_data_indices], color='green')
        plt.title('时间轴上的横滚角变化')
        plt.xlabel('日期')
        plt.ylabel('横滚角')
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
    data_file: str = rf"C:\Users\Jayttle\Desktop\ggkx_8m.txt"
    ggkx_data: List[GgkxData] = GgkxData.read_ggkx_data(data_file)
    GgkxData.filter_data_to_files_in_specified_date(ggkx_data, r'C:\Users\Jayttle\Desktop\temp', '2023-8-12')

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

def run_main6():
    data_file = r"C:\Users\Jayttle\Desktop\tiltmeter_0707_0710.txt"
    tiltmeter_data_list = TiltmeterData.from_file(data_file)
    # iltmeterData.plot_tiltmeter_data(tiltmeter_data_list)
    # normalized_data_list = TiltmeterData.normalize_tiltmeter_data(tiltmeter_data_list)
    TiltmeterData.plot_tiltmeter_data_in_specified_date(tiltmeter_data_list, "2023-07-09")

    # # 初始化最大值和最小值为第一个数据点的值
    # min_pitch = tiltmeter_data_list[0].pitch
    # max_pitch = tiltmeter_data_list[0].pitch
    # min_roll = tiltmeter_data_list[0].roll
    # max_roll = tiltmeter_data_list[0].roll

    # # 遍历数据列表，找到最大最小值
    # for data_point in tiltmeter_data_list:
    #     min_pitch = min(min_pitch, data_point.pitch)
    #     max_pitch = max(max_pitch, data_point.pitch)
    #     min_roll = min(min_roll, data_point.roll)
    #     max_roll = max(max_roll, data_point.roll)

    # # 打印最大最小值
    # print("Pitch 最小值:", min_pitch)
    # print("Pitch 最大值:", max_pitch)
    # print("Roll 最小值:", min_roll)
    # print("Roll 最大值:", max_roll)

if __name__ == "__main__":
    print("---------------------run-------------------")
    run_main6()