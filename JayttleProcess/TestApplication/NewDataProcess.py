# 标准库导入
import math
import random
from datetime import datetime, timedelta
import time
import warnings
import os 

# 相关第三方库导入
from pyswarm import pso
from JayttleProcess import ListFloatDataMethod as LFDM
from JayttleProcess import TBCProcessCsv
import numpy as np
import chardet
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.font_manager import FontProperties
import statsmodels.api as sm
from collections import defaultdict
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering 
from sklearn.linear_model import Ridge , Lasso, LassoCV, ElasticNet, LinearRegression
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score, mean_squared_error, silhouette_score, davies_bouldin_score, calinski_harabasz_score, v_measure_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.mixture import GaussianMixture
from sklearn.svm import OneClassSVM
from sklearn.model_selection import GridSearchCV, cross_val_score
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
# 创建一个FontProperties对象，设置字体大小为6号
font_prop = FontProperties(fname=None, size=6)
# 禁用特定警告
warnings.filterwarnings('ignore', category=UserWarning, append=True)
# 或者关闭所有警告
warnings.filterwarnings("ignore")
# endregion

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
        'R031_1215': r'D:\Program Files (x86)\Software\OneDrive\PyPackages_DataSave\new_data\R031_1215',
        'R032_1215': r'D:\Program Files (x86)\Software\OneDrive\PyPackages_DataSave\new_data\R032_1215',
        'R051_1215': r'D:\Program Files (x86)\Software\OneDrive\PyPackages_DataSave\new_data\R051_1215',
        'R052_1215': r'D:\Program Files (x86)\Software\OneDrive\PyPackages_DataSave\new_data\R052_1215',
        'R071_1215': r'D:\Program Files (x86)\Software\OneDrive\PyPackages_DataSave\new_data\R071_1215',
    }
    sorted_keys = sorted(data_save_path.keys())
    Receiver_DataPoints = {}

    for key in sorted_keys:
        print(f"{key}: {data_save_path[key]}")
        Receiver_DataPoints[key] = load_csv_data(data_save_path[key])
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

def load_DataPoints_in_ropeway() -> Tuple[dict[str, List[float]], dict[str, List[float]]]:
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

    for key in x_DataPoints.keys():
        x_DataPoints[key] = [(value - locations_ropeway[key[:4]][0]) * 1000 for value in x_DataPoints[key]]
        y_DataPoints[key] = [(value - locations_ropeway[key[:4]][1]) * 1000 for value in y_DataPoints[key]]
    
    return x_DataPoints, y_DataPoints


#TODO: 1.用双天线的数据来估计一个索道支架的位置，
#TODO: 2.用气象仪数据做一个相关性分析和可视化比较
#TODO: 3.做全年的s
def main_TODO2():
    """
    坐标转换
    """
    x_DataPoints, y_DataPoints = load_DataPoints_in_ropeway()
    # LFDM.plot_ListFloat(x_DataPoints['R051_1215'], isShow=True, title='R051_1215_x')
    # LFDM.plot_ListFloat(y_DataPoints['R051_1215'], isShow=True, title='R051_1215_y')
    outliers, anomaly_scores, threshold = LFDM.detect_knn_anomaly_xy(x_DataPoints['R031_1215'], y_DataPoints['R031_1215'])
    print(outliers)
    to_marker_point = outliers.tolist()
    LFDM.plot_ListFloat_with_markeridx(ListFloat=x_DataPoints['R031_1215'], isShow=True,to_marker_idx=to_marker_point)
    LFDM.plot_points_with_markeridx(x_DataPoints['R031_1215'], y_DataPoints['R031_1215'], 'marker', to_marker_point)

if __name__ == "__main__":
    print("---------------------run-------------------")
    main_TODO2()
