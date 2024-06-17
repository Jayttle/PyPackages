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


def lv_dbscan_anomaly_detection(data: list[float], eps, min_samples, lof_threshold):
    # 将数据转换为NumPy数组
    data_array = np.array(data).reshape(-1, 1)

    # 数据标准化
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_array)

    # 使用LV-DBSCAN算法进行聚类
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(scaled_data)

    # 获取每个样本的聚类标签
    labels = dbscan.labels_

    # 计算每个样本的局部异常因子（LOF）
    lof_scores = LocalOutlierFactor(n_neighbors=min_samples + 1).fit_predict(scaled_data)

    # 标记异常点
    outliers = np.where(lof_scores == -1)[0]

    # 打印异常点
    print("Detected anomalies:")
    for outlier in outliers:
        print(f"Data point {outlier}: Cluster label: {labels[outlier]}, LOF score: {lof_scores[outlier]}")


def optimize_dbscan_params(data):
    # 目标函数，用于优化
    def dbscan_objective(params):
        eps, min_samples = params
        min_samples = int(min_samples)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(data.reshape(-1, 1))
        # 优化轮廓系数，尽量不要有噪声点
        if len(set(labels)) == 1 or -1 in labels:
            return -1  # 一个簇或者有噪声，轮廓系数无法计算
        score = silhouette_score(data.reshape(-1, 1), labels)
        return -score  # pyswarm 是最小化目标函数

    # 参数范围
    lb = [0.1, 2]  # eps的最小值，min_samples的最小值
    ub = [2, 20]   # eps的最大值，min_samples的最大值

    # 粒子群优化
    xopt, fopt = pso(dbscan_objective, lb, ub, swarmsize=50, maxiter=100)

    return xopt

def ipsodbscan_anomaly_detection(data):
    # 数据标准化
    scaler = StandardScaler()
    # 将列表转换为NumPy数组，并调整形状为(n_samples, n_features)
    scaled_data = scaler.fit_transform(np.array(data).reshape(-1, 1))
    
    # 使用IPSO优化DBSCAN参数
    eps, min_samples = optimize_dbscan_params(scaled_data)
    min_samples = int(min_samples)

    # 使用优化后的参数运行DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(scaled_data)

    # 异常点标记
    labels = dbscan.labels_
    outliers = np.where(labels == -1)[0]

    # 打印结果
    print(f"Optimal eps: {eps}, Optimal min_samples: {min_samples}")
    print("Detected anomalies:", outliers)

    return outliers, labels


def knn_anomaly_detection(data, k=5, outlier_fraction=0.01):
    # 确保数据为NumPy数组
    data = np.array(data).reshape(-1, 1)  # 转换数据为正确的形状

    # 数据标准化
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # 训练K近邻模型
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors.fit(data_scaled)
    
    # 计算每个点到其k个最近邻居的距离
    distances, indices = neighbors.kneighbors(data_scaled)

    # 计算每个点的异常分数（平均距离）
    anomaly_scores = distances.mean(axis=1)

    # 确定异常分数的阈值
    threshold = np.percentile(anomaly_scores, 100 * (1 - outlier_fraction))

    # 检测异常点
    outliers = np.where(anomaly_scores > threshold)[0]
    return outliers, anomaly_scores, threshold


def plot_float_with_idx(data: list[float], startidx: int, endidx: int):
    # 修改标签和标题的文本为中文
    values = data[startidx: endidx]
    plt.figure(figsize=(14.4, 9.6)) # 单位是英寸
    plt.xlabel('idx')
    plt.ylabel('数值')
    plt.title(f'{startidx}-{endidx}索引数据可视化')
    
    plt.plot(range(len(values)), values)
    # 设置最大显示的刻度数
    plt.gcf().autofmt_xdate()  # 旋转日期标签以避免重叠
    plt.tight_layout()  # 自动调整子图间的间距和标签位置

    plt.show()
    plt.close()

def plot_float_with_mididx(data: list[float], mididx: int):
    # 确保 mididx 不超过索引范围
    if mididx < 0 or mididx >= len(data):
        print("Error: mididx is out of range.")
        return
    
    # 确定 startidx 和 endidx
    startidx = max(mididx - 3, 0)
    endidx = min(mididx + 3, len(data))

    values = data[startidx: endidx+1]
    plt.figure(figsize=(14.4, 9.6)) # 单位是英寸
    plt.xlabel('idx')
    plt.ylabel('数值')
    plt.title(f'{startidx}-{endidx}索引数据可视化')
    
    plt.plot(range(len(values)), values)
    # 设置最大显示的刻度数
    plt.gcf().autofmt_xdate()  # 旋转日期标签以避免重叠
    plt.tight_layout()  # 自动调整子图间的间距和标签位置

    plt.show()
    plt.close()



def find_best_k(data, k_values=[3, 5, 7, 9, 11], outlier_fraction=0.05):
    """
    使用交叉验证方法找出最佳的 K 值。
    
    参数：
        - data: 一维数据，用于异常检测
        - k_values: 待评估的 K 值列表，默认为 [3, 5, 7, 9, 11]
        - outlier_fraction: 异常值比例，默认为 0.05
        
    返回值：
        - best_k: 最佳的 K 值
    """
    # 定义一个函数来评估给定的 K 值
    def evaluate_knn(k):
        outliers, _, _ = knn_anomaly_detection(data, k=k, outlier_fraction=outlier_fraction)
        return -len(outliers)  # 使用异常点的数量作为性能指标，负号表示越少越好

    best_k = None
    best_score = float('-inf')  # 初始化最佳得分

    for k in k_values:
        score = cross_val_score(evaluate_knn(k), np.zeros_like(data), cv=5).mean()  # 使用 5 折交叉验证
        if score > best_score:
            best_score = score
            best_k = k

    return best_k

def calculate_distance(x1, y1, x2, y2):
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance


def detect_anomalies_with_OneClassSVM(data: list[float]):
    # 转换数据为 NumPy 数组以便处理
    data_array = np.array(data).reshape(-1, 1)
    
    # 初始化 OneClassSVM 模型
    # 参数 nu 控制模型假定的异常点的比例
    # 参数 gamma 是 RBF 内核的系数，如果不确定可以设置为 'auto'
    ocsvm = OneClassSVM(nu=0.05, kernel='rbf', gamma='auto')
    
    # 训练模型
    ocsvm.fit(data_array)
    
    # 预测数据点，返回 1 表示正常，-1 表示异常
    predictions = ocsvm.predict(data_array)
    
    # 获取异常点的索引
    anomaly_indices = np.where(predictions == -1)[0]
    
    # 返回异常点的索引和值
    anomalies = [(index, data[index]) for index in anomaly_indices]
    return anomalies


def detect_anomalies_with_ema(data: list[float], alpha: float = 0.1, threshold_factor: float = 2.0):
    # 转换数据为 NumPy 数组
    data_array = np.array(data)
    
    # 计算指数移动平均
    ema = np.zeros_like(data_array)
    ema[0] = data_array[0]  # 将第一个 EMA 值设为第一个数据点
    for i in range(1, len(data_array)):
        ema[i] = alpha * data_array[i] + (1 - alpha) * ema[i - 1]
    
    # 计算每个数据点与其 EMA 的差的绝对值
    deviations = np.abs(data_array - ema)
    
    # 计算标准差和设定异常阈值
    std_deviation = np.std(data_array)
    threshold = std_deviation * threshold_factor
    
    # 确定异常值
    anomalies = [(index, data_array[index]) for index, deviation in enumerate(deviations) if deviation > threshold]
    
    return anomalies, ema

def detect_anomalies_with_mad(data: list[float], threshold_factor: float = 3.0):
    data_array = np.array(data)
    median = np.median(data_array)
    mad = np.median(np.abs(data_array - median))
    
    # 转换 MAD 为正态标准差等价值
    mad_std_equivalent = 1.4826 * mad
    
    # 计算异常阈值
    upper_threshold = median + threshold_factor * mad_std_equivalent
    lower_threshold = median - threshold_factor * mad_std_equivalent
    
    # 检测异常值
    anomalies = [(index, value) for index, value in enumerate(data_array)
                 if value > upper_threshold or value < lower_threshold]
    return anomalies

def detect_anomalies_with_IsolationForest(data: list[float]):
    # 转换数据格式，因为隔离森林需要二维数组格式
    data_reshaped = np.array(data).reshape(-1, 1)

    # 创建隔离森林模型
    # n_estimators 表示树的数量，contamination 表示异常数据比例的估计
    model = IsolationForest(n_estimators=100, contamination=0.1)

    # 训练模型
    model.fit(data_reshaped)

    # 预测数据点的异常状态，返回 1 表示正常，-1 表示异常
    predictions = model.predict(data_reshaped)

    # 输出异常值
    print("Anomalies detected at indices:")
    for i, val in enumerate(predictions):
        if val == -1:
            print(f"Index: {i}, Value: {data[i]}")


def detect_anomalies_with_rolling_std(data: list[float], window_size: int, threshold_factor: float = 3.0):
    # 将数据转换为Pandas Series以便使用滑动窗口
    data_series = pd.Series(data)
    
    # 计算滑动窗口的标准差
    rolling_std = data_series.rolling(window=window_size).std()
    
    # 计算全局标准差的均值，定义异常阈值为均值的指定倍数
    std_mean = rolling_std.mean()
    upper_threshold = std_mean + threshold_factor * std_mean
    lower_threshold = std_mean - threshold_factor * std_mean
    
    # 检测异常值
    anomalies = [(index, data[index]) for index, std_dev in enumerate(rolling_std)
                 if std_dev > upper_threshold or std_dev < lower_threshold]

    return anomalies

def main_TODO1():
    grouped_DataPoints: dict[str, list[DataPoint]] = load_DataPoints()
    for key,list_datapoint in grouped_DataPoints.items():
        east_coordinate_list = [data.east_coordinate for data in list_datapoint]
        north_coordinate_list = [data.north_coordinate for data in list_datapoint]
        distance_list = []
        for east, north in zip(east_coordinate_list, north_coordinate_list):
            distance = calculate_distance(0, 0, east, north)
            distance_list.append(distance)
        print(f"{key}:")
        detect_num = 10
        # for _ in range(detect_num):
        #     ipsodbscan_anomaly_detection(east_coordinate_list)
        # ListFloatDataMethod.calculate_and_print_static(east_coordinate_list)
        # ListFloatDataMethod.calculate_and_print_static(north_coordinate_list)
        # # 执行异常检测
        # outliers, labels = ipsodbscan_anomaly_detection(east_coordinate_list)
        # 运行检测


        outliers, anomaly_scores, threshold = knn_anomaly_detection(east_coordinate_list)
        print("Outliers:", outliers)

        # anomalies = detect_anomalies_with_rolling_std(distance_list, 4)
        # print("Anomalies detected at:")
        # for anomaly in anomalies:
        #     print(f"Index: {anomaly[0]}, Value: {anomaly[1]}")
        # outliers, lof_scores = lof_outlier_detect1(distance_list, k=3)
        # print("Outliers:", outliers)
        # print("LOF Scores:", lof_scores)


        # print("Anomaly scores:", scores)
        # print("Threshold:", threshold)


        # to_print_idx =[72, 81]
        to_print_idx =  [ 60,  72,  81]
        without_print_data_list = [data for idx, data in enumerate(east_coordinate_list) if idx not in to_print_idx]

        for idx in to_print_idx:
            print(list_datapoint[idx].start_time)
        # # # print(list_datapoint[72].start_time)
        LFDM.plot_ListFloat(east_coordinate_list, isShow=True)
        # ListFloatDataMethod.plot_ListFloat(without_print_data_list, isShow=True)
        LFDM.plot_ListFloat_with_markeridx(ListFloat=east_coordinate_list, isShow=True, markeridx=to_print_idx)
        for idx in outliers:
            print(list_datapoint[idx].start_time)
            #plot_float_with_mididx(east_coordinate_list, idx)
            
        # plot_float_with_idx(east_coordinate_list, 122, 130)



#TODO: 1.用双天线的数据来估计一个索道支架的位置，
#TODO: 2.用气象仪数据做一个相关性分析和可视化比较
#TODO: 3.做全年的
def main_TODO2():
    """
    坐标转换
    """
    x_DataPoints, y_DataPoints = load_DataPoints_in_ropeway()
    LFDM.plot_ListFloat(x_DataPoints['R051_1215'], isShow=True)
    LFDM.plot_ListFloat(y_DataPoints['R051_1215'], isShow=True)


if __name__ == "__main__":
    print("---------------------run-------------------")
    main_TODO2()
