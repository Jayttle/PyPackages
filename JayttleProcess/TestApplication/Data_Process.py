import os
import csv
import pandas as pd
import math
import chardet
from collections import defaultdict
from typing import List, Tuple, Dict
from collections import Counter
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering 
from sklearn.linear_model import Ridge , Lasso, LassoCV, ElasticNet, LinearRegression
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score, mean_squared_error, silhouette_score, davies_bouldin_score, calinski_harabasz_score, v_measure_score
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV, cross_val_score
from JayttleProcess import TimeSeriesDataMethod, TBCProcessCsv, CommonDecorator, ListFloatDataMethod
from JayttleProcess.ListFloatDataMethod import *
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid

from JayttleProcess.TimeSeriesDataMethod import TimeSeriesData
plt.rcParams['font.sans-serif'] = ['SimSun']  # 指定宋体为默认字体
plt.rcParams['font.sans-serif'] = 'SimSun'  # 使用指定的中文字体
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
warnings.filterwarnings("ignore")

class DataPoint:
    def __init__(self,csv_file: str, point_id: str, north_coordinate: float, east_coordinate: float, elevation: float,
                 latitude: float, longitude: float, ellipsoid_height: float, start_time: str, end_time: str,
                 duration: str, pdop: float, rms: float, horizontal_accuracy: float, vertical_accuracy: float,
                 north_coordinate_error: float, east_coordinate_error: float, elevation_error: float,
                 height_error: str):
        self.csv_file = csv_file # 读取的文件路径
        self.point_id = point_id  # 数据点的ID
        self.north_coordinate = north_coordinate  # 数据点的北坐标
        self.east_coordinate = east_coordinate  # 数据点的东坐标
        self.elevation = elevation  # 数据点的高程
        self.latitude = latitude  # 数据点的纬度
        self.longitude = longitude  # 数据点的经度
        self.ellipsoid_height = ellipsoid_height  # 数据点的椭球高度
        self.start_time = start_time  # 数据点的开始时间
        self.end_time = end_time  # 数据点的结束时间
        self.duration = duration  # 数据点的持续时间
        self.pdop = pdop  # 数据点的位置精度衰减因子（PDOP）
        self.rms = rms  # 数据点的均方根（RMS）
        self.horizontal_accuracy = horizontal_accuracy  # 数据点的水平精度
        self.vertical_accuracy = vertical_accuracy  # 数据点的垂直精度
        self.north_coordinate_error = north_coordinate_error  # 基线解算的北坐标差值
        self.east_coordinate_error = east_coordinate_error  # 基线解算的东坐标差    值
        self.elevation_error = elevation_error  # 数据点的高程误差
        self.height_error = height_error  # 数据点的高度误差



def find_data_by_date_and_id(data_points: list[DataPoint], point_id: str, date: str) -> list[DataPoint]:
    """
    根据给定的 point_id 和日期 date，找到对应的数据点列表

    参数:
    data_points (list[DataPoint]): DataPoint 对象列表
    point_id (str): 要搜索的数据点的ID
    date (str): 要匹配的日期字符串，格式为 'YYYY-MM-DD'

    返回:
    list[DataPoint]: 包含与指定 point_id 和日期相匹配的数据点列表
    """
    matched_data_points = []
    for data_point in data_points:
        # 将字符串日期转换为日期对象
        start_datetime = datetime.strptime(data_point.start_time, "%Y-%m-%d %H:%M:%S")
        # 提取日期部分进行匹配
        if data_point.point_id == point_id and start_datetime.date() == datetime.strptime(date, "%Y-%m-%d").date():
            matched_data_points.append(data_point)
    return matched_data_points


class TiltSensorData:
    def __init__(self, time: str, station_id: int, pitch: float, roll: float):
        try:
            self.datetime = datetime.strptime(time, "%Y-%m-%d %H:%M:%S.%f")
        except ValueError:
                # If parsing with microseconds fails, try parsing without microseconds
            self.datetime = datetime.strptime(time, "%Y-%m-%d %H:%M:%S")
        self.station_id = station_id
        self.pitch = pitch
        self.roll = roll
    
    def __str__(self) -> str:
        return f"Time: {self.time}, Station ID: {self.station_id}, Pitch: {self.pitch}, Roll: {self.roll}"

class TiltSensorDataReader:
    def __init__(self, file_path: str):
        self.file_path = file_path
    
    @CommonDecorator.log_function_call
    def read_data(self) -> List[TiltSensorData]:
        """
        读取10000 数据所需0.0626s
        """
        tilt_sensor_data: List[TiltSensorData] = []
        with open(self.file_path, 'r') as file:
            # Skip header
            next(file)
            for idx, line in enumerate(file):
                if idx >= 100000:  # Only read the first 10000 lines
                    break
                parts = line.strip().split('\t')
                if len(parts) == 4:  # Check if line has 4 columns
                    time, station_id, pitch, roll = parts
                    tilt_sensor_data.append(TiltSensorData(time, int(station_id), float(pitch), float(roll)))
        return tilt_sensor_data
    
    @CommonDecorator.log_function_call
    def check_data(self) -> None:
        """
        检查10000 数据所需31.4425s
        """
        lines_to_keep = []
        # Open the file and read lines to check data
        with open(self.file_path, 'r') as file:
            lines_to_keep.append(next(file))  # Keep the header
            for line in file:
                parts = line.strip().split('\t')
                if len(parts) == 4:  # Check if line has 4 columns
                    lines_to_keep.append(line)
        
        # Rewrite the file with only valid lines
        with open(self.file_path, 'w') as file:
            file.writelines(lines_to_keep)
    

def read_csv_to_datapoints(csv_file: str) -> list[DataPoint]:
    with open(csv_file, 'rb') as f:
        rawdata = f.read()
        global encoding
        encoding = chardet.detect(rawdata)['encoding']
    df = pd.read_csv(csv_file, sep='\t', nrows=1, parse_dates=['GNSS矢量观测.开始时间', 'GNSS矢量观测.结束时间'], encoding=encoding)
    datapoints = []
    for index, row in df.iterrows():
        datapoint = DataPoint(
            csv_file=csv_file,
            point_id=row['点ID'],
            north_coordinate=row['北坐标'],
            east_coordinate=row['东坐标'],
            elevation=row['高程'],
            latitude=row['纬度（全球）'],
            longitude=row['经度（全球）'],
            ellipsoid_height=row['GNSS矢量观测.起点ID'],
            start_time=row['GNSS矢量观测.开始时间'],
            end_time=row['GNSS矢量观测.结束时间'],
            duration=row['GNSS矢量观测.终点ID'],
            pdop=row['GNSS矢量观测.PDOP'],
            rms=row['GNSS矢量观测.均方根'],
            horizontal_accuracy=row['GNSS矢量观测.水平精度'],
            vertical_accuracy=row['GNSS矢量观测.垂直精度'],
            north_coordinate_error=row['GNSS矢量观测.X增量'],
            east_coordinate_error=row['GNSS矢量观测.Y增量'],
            elevation_error=row['GNSS矢量观测.Z增量'],
            height_error=row['GNSS矢量观测.解类型']
        )
        datapoints.append(datapoint)
    return datapoints


def read_csv_to_datapoints_version2(csv_file: str) -> list:
    with open(csv_file, 'rb') as f:
        rawdata = f.read()
        encoding = chardet.detect(rawdata)['encoding']
    df = pd.read_csv(csv_file, sep='\t', parse_dates=['GNSS矢量观测.开始时间', 'GNSS矢量观测.结束时间'], encoding=encoding)
    datapoints = []
    for index, row in df.iterrows():
        datapoint = DataPoint(
            csv_file=csv_file,
            point_id=row['点ID'],
            north_coordinate=row['北坐标'],
            east_coordinate=row['东坐标'],
            elevation=row['高程'],
            latitude=row['纬度（地方）'],  # 更新为“地方”
            longitude=row['经度（地方）'],  # 更新为“地方”
            ellipsoid_height=row['GNSS矢量观测.起点ID'],
            start_time=row['GNSS矢量观测.开始时间'],
            end_time=row['GNSS矢量观测.结束时间'],
            duration=row['GNSS矢量观测.持续时间'],  # 更新为“持续时间”
            pdop=row['GNSS矢量观测.PDOP'],
            rms=row['GNSS矢量观测.均方根'],
            horizontal_accuracy=row['GNSS矢量观测.水平精度'],
            vertical_accuracy=row['GNSS矢量观测.垂直精度'],
            north_coordinate_error=row['GNSS矢量观测.X增量'],
            east_coordinate_error=row['GNSS矢量观测.Y增量'],
            elevation_error=row['GNSS矢量观测.Z增量'],
            height_error=row['网平差.高度误差']  # 更新为“高度误差”
        )
        datapoints.append(datapoint)
    return datapoints

def read_csv_to_datapoints_with_all(csv_file: str) -> list[DataPoint]:
    with open(csv_file, 'rb') as f:
        rawdata = f.read()
        global encoding
        encoding = chardet.detect(rawdata)['encoding']
    df = pd.read_csv(csv_file, sep='\t', nrows=2, parse_dates=['GNSS矢量观测.开始时间', 'GNSS矢量观测.结束时间'], encoding=encoding)
    datapoints = []
    for index, row in df.iterrows():
        datapoint = DataPoint(
            csv_file=csv_file,
            point_id=row['点ID'],
            north_coordinate=row['北坐标'],
            east_coordinate=row['东坐标'],
            elevation=row['高程'],
            latitude=row['纬度（全球）'],
            longitude=row['经度（全球）'],
            ellipsoid_height=row['GNSS矢量观测.起点ID'],
            start_time=row['GNSS矢量观测.开始时间'],
            end_time=row['GNSS矢量观测.结束时间'],
            duration=row['GNSS矢量观测.终点ID'],
            pdop=row['GNSS矢量观测.PDOP'],
            rms=row['GNSS矢量观测.均方根'],
            horizontal_accuracy=row['GNSS矢量观测.水平精度'],
            vertical_accuracy=row['GNSS矢量观测.垂直精度'],
            north_coordinate_error=row['GNSS矢量观测.X增量'],
            east_coordinate_error=row['GNSS矢量观测.Y增量'],
            elevation_error=row['GNSS矢量观测.Z增量'],
            height_error=row['GNSS矢量观测.解类型']
        )
        datapoints.append(datapoint)
    return datapoints


def dataPoint_create_timeseries_data(datapoint: DataPoint, value_key: str) -> TimeSeriesData:
    """
    value_key = "north_coordinate"  # 用你想要提取的属性键替换这里
    timeseries_data = create_timeseries_data(datapoint, value_key)
    getattr() 函数用于获取对象的属性值。它接受对象和属性名作为参数，并返回指定属性的值。
    如果对象中不存在该属性，则可以提供一个默认值作为第三个参数（可选）。
    这个函数在需要动态地获取对象的属性时非常有用，特别是当属性名称在运行时确定时。
    """
    selected_value = getattr(datapoint, value_key, None)
    if selected_value is None:
        return None
    # 将 datapoint.start_time 转换为字符串
    start_time_str = str(datapoint.start_time)
    return TimeSeriesData(selected_value, start_time_str)


def export_datapoints_to_csv(datapoints: list[DataPoint], output_file: str) -> None:
    fieldnames = ['GNSS矢量观测.开始时间', 'GNSS矢量观测.结束时间', '点ID', '北坐标', '东坐标', '高程', '纬度（全球）', '经度（全球）',
                  'GNSS矢量观测.PDOP', 'GNSS矢量观测.均方根', 'GNSS矢量观测.水平精度', 'GNSS矢量观测.垂直精度',
                  'GNSS矢量观测.起点ID', 'GNSS矢量观测.终点ID', 'GNSS矢量观测.X增量', 'GNSS矢量观测.Y增量',
                  'GNSS矢量观测.Z增量', 'GNSS矢量观测.矢量长度', 'GNSS矢量观测.解类型', 'GNSS矢量观测.状态']
    global encoding
    with open(output_file, 'w', newline='', encoding=encoding) as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for datapoint in datapoints:
            writer.writerow({
                'GNSS矢量观测.开始时间': datapoint.start_time,
                'GNSS矢量观测.结束时间': datapoint.end_time,
                '点ID': datapoint.point_id,
                '北坐标': datapoint.north_coordinate,
                '东坐标': datapoint.east_coordinate,
                '高程': datapoint.elevation,
                '纬度（全球）': datapoint.latitude,
                '经度（全球）': datapoint.longitude,
                'GNSS矢量观测.PDOP': datapoint.pdop,
                'GNSS矢量观测.均方根': datapoint.rms,
                'GNSS矢量观测.水平精度': datapoint.horizontal_accuracy,
                'GNSS矢量观测.垂直精度': datapoint.vertical_accuracy,
                'GNSS矢量观测.起点ID': datapoint.ellipsoid_height,
                'GNSS矢量观测.终点ID': datapoint.duration,
                'GNSS矢量观测.X增量': datapoint.north_coordinate_error,
                'GNSS矢量观测.Y增量': datapoint.east_coordinate_error,
                'GNSS矢量观测.Z增量': datapoint.elevation_error,
                'GNSS矢量观测.矢量长度': datapoint.height_error,
                'GNSS矢量观测.解类型': datapoint.height_error,
                'GNSS矢量观测.状态': datapoint.height_error
            })

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


def calculate_distance_and_bearing(points: List[DataPoint]) -> List[Tuple[float, float]]:
    distances_and_bearings: List[Tuple[float, float]] = []
    for point in points:
        # 计算与原点 (0, 0) 的距离
        distance = calculate_distance(0, 0, point.east_coordinate, point.north_coordinate)
        # 计算方位角
        bearing = calculate_bearing(point.east_coordinate, point.north_coordinate)
        distances_and_bearings.append((distance, bearing))
    return distances_and_bearings


# 定义一个函数来绘制极坐标图
def plot_polar(distances_and_bearings: List[Tuple[float, float]], title = None):
    # 创建一个极坐标图
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    
    # 遍历距离和方位角，绘制每个点
    for distance, bearing in distances_and_bearings:
        # 将方位角转换为弧度
        theta = np.deg2rad(bearing)
        # 绘制点
        ax.plot(theta, distance, 'o', color='blue')
    
    # 设置极坐标图的标题
    ax.set_title(f'{title}极坐标')
    
    # 显示图形
    plt.show()


def plot_polar_with_rms(rms_list: List[float], distances_and_bearings: List[Tuple[float, float]], threshold: float, title=None):
    # 创建一个极坐标图
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    
    # 遍历距离、方位角和 RMS 值，绘制每个点
    for i, (distance, bearing) in enumerate(distances_and_bearings):
        # 将方位角转换为弧度
        theta = np.deg2rad(bearing)
        
        # 绘制点
        if rms_list[i] > threshold:
            ax.plot(theta, distance, 'o', color='red', label='RMS Exceeded Threshold')
        else:
            ax.plot(theta, distance, 'o', color='blue')
    
    # 设置极坐标图的标题
    ax.set_title(f'{title}极坐标')
    
    
    # 显示图形
    plt.show()

def plot_polar_with_rms_exceeded(rms_list: List[float], distances_and_bearings: List[Tuple[TimeSeriesData, TimeSeriesData]], threshold: float, title=None):
    # 创建一个极坐标图
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    
    # 遍历距离、方位角和 RMS 值，仅绘制超过阈值的点
    for i, (distance, bearing) in enumerate(distances_and_bearings):
        # 将方位角转换为弧度
        theta = np.deg2rad(bearing.value)
        
        # 绘制超过阈值的点
        if rms_list[i] > threshold:
            ax.plot(theta, distance.value, 'o', color='blue', label='RMS Exceeded Threshold')
    
    # 设置极坐标图的标题
    ax.set_title(f'{title}极坐标 (只显示超过阈值的点)')

    # 显示图形
    plt.show()

def plot_polar_without_rms_exceeded(rms_list: List[float], distances_and_bearings: List[Tuple[TimeSeriesData, TimeSeriesData]], threshold: float, title=None):
    # 创建一个极坐标图
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    
    # 遍历距离、方位角和 RMS 值，仅绘制超过阈值的点
    for i, (distance, bearing) in enumerate(distances_and_bearings):
        # 将方位角转换为弧度
        theta = np.deg2rad(bearing.value)
        
        # 绘制超过阈值的点
        if rms_list[i] < threshold:
            ax.plot(theta, distance.value, 'o', color='blue', label='RMS Exceeded Threshold')
    
    # 设置极坐标图的标题
    ax.set_title(f'{title}极坐标 (不显示超过阈值的点)')

    # 显示图形
    plt.show()

def plot_polar_in_month(distances_and_bearings: List[Tuple[TimeSeriesData, TimeSeriesData]], title = None):
    # 创建一个极坐标图
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    
    # 定义一个默认颜色
    default_color = 'k'
    
    # 创建一个颜色映射，将月份映射到颜色
    month_to_color = defaultdict(lambda: default_color)
    month_to_color.update({
        1: 'b', 2: 'b', 3: 'b',  # January, February, March: blue
        4: 'r', 5: 'r', 6: 'r',  # April, May, June: red
        7: 'g', 8: 'g', 9: 'g',  # July, August, September: green
        10: 'orange', 11: 'orange', 12: 'orange',  # October, November, December: yellow
    })

    # 遍历距离和方位角，绘制每个点
    for distance, bearing in distances_and_bearings:
        # 将方位角转换为弧度
        theta = np.deg2rad(bearing.value)
        # 获取月份
        month = distance.datetime.month
        color = month_to_color[month]
        # 绘制点
        ax.plot(theta, distance.value, 'o', color=color, label=f'Month {month}')

    # 添加假线条以创建图例
    for month, color in month_to_color.items():
        ax.plot([], [], 'o', color=color, label=f'Month {month}')

    # 绘制 bearing 角度为 176.91600324062733 的直线
    bearing_line_theta = np.deg2rad(176.91600324062733)
    ax.plot([bearing_line_theta, bearing_line_theta], [0, ax.get_ylim()[1]], color='k', linestyle='--', label='Bearing Line')
    
    # 设置极坐标图的标题
    ax.set_title(f'{title}极坐标')
    
    
    # 显示图形
    plt.show()

# 定义一个函数来绘制极坐标图
def plot_polar_in_float(distances_and_bearings: List[Tuple[float, float]], title = None):
    # 创建一个极坐标图
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    # 遍历距离和方位角，绘制每个点
    for distance, bearing in distances_and_bearings:
        # 将方位角转换为弧度
        theta = np.deg2rad(bearing)
        # 绘制点
        ax.plot(theta, distance, 'o', color='red')
        print(bearing)

    ax.plot(0, 0, 'o', color='red')
    # 设置极坐标图的标题
    ax.set_title(f'{title}极坐标')
    
    # 显示图形
    plt.show()



def calculate_polar_coordinates(points: dict) -> List[Tuple[float, float]]:
    polar_coordinates = []
    
    for point_name in ['R051', 'R071', 'R081']:
        # 计算点的 dx 和 dy 值
        dx_value = points[point_name][0] - points['R031'][0]
        dy_value = points[point_name][1] - points['R031'][1]

        # 计算点的线段距离和方位角
        line_distance = calculate_distance(0, 0, dx_value, dy_value)
        line_bearing = calculate_bearing(dx_value, dy_value)
        
        polar_coordinates.append((line_distance, line_bearing))
    
    return polar_coordinates

def plot_polar_compare(distances_and_bearings_list: List[List[Tuple[TimeSeriesData, TimeSeriesData]]], titles: List[str] = None) -> None:
    # 创建一个极坐标图
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    
    # 定义一组颜色
    colors = ['b', 'r', 'g', 'y', 'c', 'm']
    
    # 遍历每个数据集
    for i, distances_and_bearings in enumerate(distances_and_bearings_list):
        # 获取标题
        if titles is not None:
            title = titles[i]
        else:
            title = f'Dataset {i+1}'
        
        # 遍历距离和方位角，绘制每个点
        for distance, bearing in distances_and_bearings:
            # 将方位角转换为弧度
            theta = np.deg2rad(bearing.value)
            # 绘制点，并指定颜色
            ax.plot(theta, distance.value, 'o', color=colors[i], label=title)
    
    # 绘制 bearing 角度为 176.91600324062733 的直线
    bearing_line_theta = np.deg2rad(176.91600324062733)
    ax.plot([bearing_line_theta, bearing_line_theta], [0, ax.get_ylim()[1]], color='k', linestyle='--', label='Bearing Line')
    
    # 设置极坐标图的标题
    ax.set_title('Positions in Polar Coordinates')
    
    # 显示图形
    plt.show()


def plot_polar_compare_with_line(distances_and_bearings_list: List[List[Tuple[TimeSeriesData, TimeSeriesData]]], line_distance: float, line_bearing: float, titles: List[str] = None) -> None:
    # 创建一个极坐标图
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    
    # 定义一组颜色
    colors = ['b', 'r', 'g', 'y', 'c', 'm']
    
    # 遍历每个数据集
    for i, distances_and_bearings in enumerate(distances_and_bearings_list):
        # 获取标题
        if titles is not None:
            title = titles[i]
        else:
            title = f'Dataset {i+1}'
        
        # 遍历距离和方位角，绘制每个点
        for distance, bearing in distances_and_bearings:
            # 将方位角转换为弧度
            theta = np.deg2rad(bearing.value)
            # 绘制点，并指定颜色
            ax.plot(theta, distance.value, 'o', color=colors[i], label=title)
    
    # 设置极坐标图的标题
    ax.set_title('Positions in Polar Coordinates')
    
    # 将方位角转换为弧度
    line_bearing_rad = np.deg2rad(line_bearing)
    
    # 绘制以原点为起点的线段
    ax.plot([0, line_bearing_rad], [0, line_distance], color='k', linewidth=2, label='Line')

    # 显示图形
    plt.show()


def plot_points(coordinates):
    """
    Plot the positions of the points.

    Args:
    - coordinates (dict): A dictionary containing the coordinates of the points.
                          The keys are the names of the points, and the values are tuples
                          containing the east coordinate and the north coordinate.
    """
    # 提取点的东走标和北坐标
    east_coordinates = [coord[0] for coord in coordinates.values()]
    north_coordinates = [coord[1] for coord in coordinates.values()]

    # 绘制点
    plt.scatter(east_coordinates, north_coordinates, label='Points')

    # 添加点的标签
    for name, coord in coordinates.items():
        plt.text(coord[0], coord[1], name, fontsize=12, ha='right', va='bottom')

    # 设置图形标题和坐标轴标签
    plt.title('Position of Points')
    plt.xlabel('East Coordinate')
    plt.ylabel('North Coordinate')

    # 显示图例
    plt.legend()

    # 显示图形
    plt.grid(True)
    plt.show()


def load_csv_data(folder_path: str) -> List[DataPoint]:
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
            data_points.extend(read_csv_to_datapoints(csv_file_path))
    return data_points
def load_csv_data_version2(folder_path: str) -> List[DataPoint]:
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
            data_points.extend(read_csv_to_datapoints_version2(csv_file_path))
    return data_points


def load_csv_data_with_all(folder_path: str) -> List[DataPoint]:
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
            data_points.extend(read_csv_to_datapoints_with_all(csv_file_path))
    return data_points


def process_data_points(data_points: List[DataPoint], lat: float, lon: float, north_key: str, east_key: str) -> Tuple[List[TimeSeriesData], List[TimeSeriesData], float, float]:
    """
    Process data points and convert coordinates.
    """
    east, north = TBCProcessCsv.convert_coordinates(lat, lon)
    time_series_north: List[TimeSeriesData] = [dataPoint_create_timeseries_data(datapoint, north_key) for datapoint in data_points]
    time_series_east: List[TimeSeriesData] = [dataPoint_create_timeseries_data(datapoint, east_key) for datapoint in data_points]
    TimeSeriesDataMethod.remove_specific_value(time_series_north, north)
    TimeSeriesDataMethod.remove_specific_value(time_series_east, east)
    return time_series_north, time_series_east, east, north

def load_DataPoints() -> dict[str, list[DataPoint]]:
    locations = {
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

def load_DataPoints_mini() -> dict[str, list[DataPoint]]:
    locations = {
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
    Receiver_DataPoints = load_DataPoints_return_dict_mini()
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

def load_DataPoints_return_dict() -> dict[str, list[DataPoint]]:
    data_save_path = {
        'R031_1619': r'D:\Ropeway\R031_1619',
        'R031_0407': r'D:\Ropeway\FTPCsv2',
        'R051_1619': r'D:\Ropeway\R051_1619',
#        'R052_1619': r'D:\Ropeway\R052_1619', 因为R052_1619 有重复的数据
        'R081_1619': r'D:\Ropeway\FTPCsv3',
        'R082_1619': r'D:\Ropeway\R082_1619',
        'R031_1215': r'D:\Program Files (x86)\Software\OneDrive\PyPackages_DataSave\R031_1215',
        'R032_1215': r'D:\Program Files (x86)\Software\OneDrive\PyPackages_DataSave\R032_1215',
        'R051_1215': r'D:\Program Files (x86)\Software\OneDrive\PyPackages_DataSave\R051_1215',
        'R052_1215': r'D:\Program Files (x86)\Software\OneDrive\PyPackages_DataSave\R052_1215',
        'R071_1215': r'D:\Program Files (x86)\Software\OneDrive\PyPackages_DataSave\R071_1215',
        'R071_1619': r'D:\Program Files (x86)\Software\OneDrive\PyPackages_DataSave\R071_1619',
        'R072_1619': r'D:\Program Files (x86)\Software\OneDrive\PyPackages_DataSave\R072_1619',
    }
    sorted_keys = sorted(data_save_path.keys())
    Receiver_DataPoints = {}
    for key in sorted_keys:
        print(f"{key}: {data_save_path[key]}")
        Receiver_DataPoints[key] = load_csv_data(data_save_path[key])
    return Receiver_DataPoints

def load_DataPoints_return_dict_mini() -> dict[str, list[DataPoint]]:
    # 需要替换的组名
    groups = ['R031_0407','R051_0407', 'R071_0407', 'R081_0407', 'R031_1215','R051_1215', 'R071_1215', 'R081_1215']
    data_save_path = {}
    for item in groups:
        data_save_path[item] = rf'D:\Program Files (x86)\Software\OneDrive\PyPackages_DataSave\new_data\{item}'

    sorted_keys = sorted(data_save_path.keys())
    Receiver_DataPoints = {}
    for key in sorted_keys:
        print(f"{key}: {data_save_path[key]}")
        Receiver_DataPoints[key] = load_csv_data_version2(data_save_path[key])
    return Receiver_DataPoints




def calculate_daily_movements(combined_data: Dict[datetime, Tuple[Tuple[TimeSeriesData, TimeSeriesData], Tuple[TimeSeriesData, TimeSeriesData]]]) -> Dict[datetime, Tuple[float, float]]:
    daily_movements = {}
    
    for date, data in combined_data.items():
        north_east_1619, north_east_0407 = data
        north_movement = north_east_0407[0].value - north_east_1619[0].value
        east_movement = north_east_0407[1].value - north_east_1619[1].value
        daily_movements[date] = (north_movement, east_movement)
    
    return daily_movements


def pairwise_difference(data):
    # 添加第一个数据，差值设为0
    differences = [0]
    differences.extend([(second - first) * 1000 for first, second in zip(data, data[1:])])
    return differences

def main_TODO2():
    R031_1619_DataPoints, R031_0407_DataPoints, R051_1619_DataPoints, R052_1619_DataPoints, R081_1619_DataPoints, R082_1619_DataPoints = load_DataPoints()
    

    export_datapoints_to_csv(R031_1619_DataPoints,"temp.csv")
    # 获取相邻元素差值
    differences = pairwise_difference([point.north_coordinate for point in R031_1619_DataPoints])
    rms_list = [point.rms * 1000 for point in R031_1619_DataPoints]
    
    # 获取时间信息
    start_times = [point.start_time for point in R031_1619_DataPoints]
    
    # 将相邻元素差值写入文本文件
    with open('differences.txt', 'w') as file:
        for diff in differences:
            file.write(str(diff) + '\n')
    
    # 设置阈值
    threshold = 15  # 这里设定阈值为10，你可以根据需要进行调整
    
    # 绘制折线图
    for i in range(len(start_times) - 1):
        time_diff = start_times[i + 1] - start_times[i]
        if time_diff < timedelta(days=2):  # 如果时间间隔小于1天，则连接线段
            plt.plot([start_times[i], start_times[i+1]], [differences[i], differences[i+1]], marker='o', linestyle='-', color='blue')
            if rms_list[i] > threshold:
                plt.scatter(start_times[i], differences[i], color='red', s=100, label='RMS Exceeded Threshold')
            else:
                plt.scatter(start_times[i], differences[i], color='blue', s=50)
            if rms_list[i+1] > threshold:
                plt.scatter(start_times[i+1], differences[i+1], color='red', s=100, label='RMS Exceeded Threshold')
            else:
                plt.scatter(start_times[i+1], differences[i+1], color='blue', s=50)
        else:  # 否则不连接线段
            plt.plot(start_times[i:i+2], differences[i:i+2], marker='o', linestyle='', color='blue')
            if rms_list[i] > threshold:
                plt.scatter(start_times[i], differences[i], color='red', s=100, label='RMS Exceeded Threshold')
            else:
                plt.scatter(start_times[i], differences[i], color='blue', s=50)
            if rms_list[i+1] > threshold:
                plt.scatter(start_times[i+1], differences[i+1], color='red', s=100, label='RMS Exceeded Threshold')
            else:
                plt.scatter(start_times[i+1], differences[i+1], color='blue', s=50)

    plt.title('Pairwise Differences with RMS Exceeding Threshold (Multiplied by 1000)')
    plt.xlabel('Start Time')
    plt.ylabel('Difference')
    plt.xticks(rotation=45)
    plt.show()

def main_TODO4_rms():
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
    """
    Load data from multiple CSV files and process them.
    """
    R031_1619_DataPoints = load_csv_data_with_all(r'D:\Ropeway\R031_1619')
    R031_0407_DataPoints = load_csv_data_with_all(r'D:\Ropeway\FTPCsv2')
    R051_1619_DataPoints = load_csv_data_with_all(r'D:\Ropeway\R051_1619')
    R052_1619_DataPoints = load_csv_data_with_all(r'D:\Ropeway\R052_1619')
    R081_1619_DataPoints = load_csv_data_with_all(r'D:\Ropeway\FTPCsv3')
    R082_1619_DataPoints = load_csv_data_with_all(r'D:\Ropeway\R082_1619')

    # 创建一个字典来存储每个 start_time 对应的坐标误差数据
    error_data = {}
    accuracy_data = {}
    coordinate_data = {}
    # 处理 R031_1619_DataPoints 数据
    for datapoint in R031_1619_DataPoints:
        start_time = datapoint.start_time
        error_tuple = (datapoint.north_coordinate_error, datapoint.east_coordinate_error)
        accuracy_tuple = (datapoint.horizontal_accuracy, datapoint.vertical_accuracy)
        coordinate_tuple = (datapoint.north_coordinate, datapoint.east_coordinate)
        if start_time in error_data:
            error_data[start_time].append(error_tuple)
            accuracy_data[start_time].append(accuracy_tuple)
        else:
            error_data[start_time] = [error_tuple]
            accuracy_data[start_time] = [accuracy_tuple]
            coordinate_data[start_time] = [coordinate_tuple]


    # 打印结果
    for start_time, errors  in error_data.items():
        print(f"{start_time}")
        coordinates = coordinate_data[start_time]  # 获取所有坐标数据
        for idx, error_tuple in enumerate(errors, start=1):
            # 获取原始坐标和误差
            original_coordinate = coordinates[0]  # 注意这里的索引，确保与错误列表对应
            corrected_north = original_coordinate[0] - error_tuple[0]
            corrected_east = original_coordinate[1] - error_tuple[1]
            # 打印修正后的坐标
            print(f"Corrected Coordinate {idx}: North = {corrected_north}, East = {corrected_east}")

    print(east_north_coordinates["B011"])
    print(east_north_coordinates["B021"])


def statistics_data_each_month(datetimes):
    # 创建一个字典来存储每个月份中包含的天数
    days_per_month = Counter()

    # 遍历datetimes列表
    for dt in datetimes:
        # 获取日期的年份和月份
        year_month = (dt.year, dt.month)
        # 将日期添加到对应的月份计数中
        days_per_month[year_month] += 1

    # 打印结果
    for month, count in days_per_month.items():
        print(f"Year: {month[0]}, Month: {month[1]}, Days: {count}")


def statistics_continuous_data_each_month(datetimes):
    consecutive_days_threshold = 20  # 连续天数阈值

    start_date = None
    current_date = None
    consecutive_count = 0

    for dt in sorted(datetimes):
        if current_date is None:
            start_date = dt
            current_date = dt
            consecutive_count = 1
        elif (dt - current_date).days == 1:
            consecutive_count += 1
            current_date = dt
        else:
            if consecutive_count >= consecutive_days_threshold:
                end_date = current_date
                print(f"{consecutive_count}: {start_date} - {end_date}")
            start_date = dt
            current_date = dt
            consecutive_count = 1

    # 检查最后一组连续日期是否满足阈值
    if consecutive_count >= consecutive_days_threshold:
        end_date = current_date
        print(f"{consecutive_count}: {start_date} - {end_date}")


def plot_data_in_season(values: List[float], datetimes:List[datetime], marker_time: List[datetime] = None, title: str = None):
    # 设置颜色映射
    color_map = {1: (208, 227, 59),   # 1、2、3月份
                 2: (208, 227, 59),
                 3: (208, 227, 59),
                 4: (237, 120, 102),    # 4、5、6月份
                 5: (237, 120, 102),
                 6: (237, 120, 102),
                 7: (248, 186, 87),      # 7、8、9月份
                 8: (248, 186, 87),
                 9: (248, 186, 87),
                 10: (189, 231, 247),  # 10、11、12月份
                 11: (189, 231, 247),
                 12: (189, 231, 247)}
    # 修改标签和标题的文本为中文
    plt.figure(figsize=(14.4, 9.6)) # 单位是英寸
    plt.xlabel('日期')
    plt.ylabel('数值')
    plt.title(f'{title}季节时序图')

    # 设置日期格式化器和日期刻度定位器
    date_fmt = mdates.DateFormatter("%m-%d")  # 仅显示月-日
    date_locator = mdates.AutoDateLocator()  # 自动选择刻度间隔
    plt.gca().xaxis.set_major_formatter(date_fmt)
    plt.gca().xaxis.set_major_locator(date_locator)
    
    # 设置最大显示的刻度数
    plt.gcf().autofmt_xdate()  # 旋转日期标签以避免重叠
    plt.tight_layout()  # 自动调整子图间的间距和标签位置

    # 绘制折线图，根据阈值连接或不连接线段，并使用不同颜色
    prev_datetime = None
    prev_value = None
    prev_month = None
    for datetime, value in zip(datetimes, values):
        month = datetime.month
        if prev_datetime is not None:
            if datetime in marker_time:
                if value is not None:
                    plt.scatter(datetime, value, color='red', marker='o', label='Point')
            time_diff = datetime - prev_datetime
            if time_diff < timedelta(days=2):  # 如果时间间隔小于阈值，则连接线段
                color = tuple(c/255 for c in color_map[month])  # 将RGB转换为范围在0到1之间的值
                plt.plot([prev_datetime, datetime], [prev_value, value], linestyle='-', color=color)
            else:  # 否则不连接线段
                color = tuple(c/255 for c in color_map[prev_month])  # 将RGB转换为范围在0到1之间的值
                plt.plot([prev_datetime, datetime], [prev_value, value], linestyle='', color=color)
        prev_datetime = datetime
        prev_value = value
        prev_month = month

    plt.show()
    plt.close()



def plot_data_in_season_version2(values: List[float], datetimes: List[datetime], marker_time: List[datetime] = None, title: str = None):
    # 设置颜色映射
    color_map = {1: (208, 227, 59),   # 1、2、3月份
                 2: (208, 227, 59),
                 3: (208, 227, 59),
                 4: (237, 120, 102),    # 4、5、6月份
                 5: (237, 120, 102),
                 6: (237, 120, 102),
                 7: (248, 186, 87),      # 7、8、9月份
                 8: (248, 186, 87),
                 9: (248, 186, 87),
                 10: (189, 231, 247),  # 10、11、12月份
                 11: (189, 231, 247),
                 12: (189, 231, 247)}

    # 修改标签和标题的文本为中文
    plt.figure(figsize=(14.4, 9.6)) # 单位是英寸
    plt.xlabel('日期')
    plt.ylabel('数值')
    plt.title(f'{title}季节时序图')

    # 设置日期格式化器和日期刻度定位器
    date_fmt = mdates.DateFormatter("%Y-%m-%d")  # 日期格式为年-月-日
    date_locator = mdates.MonthLocator(interval=3)  # 每三个月显示一个刻度
    plt.gca().xaxis.set_major_formatter(date_fmt)
    plt.gca().xaxis.set_major_locator(date_locator)
    
    # 设置最大显示的刻度数
    plt.gcf().autofmt_xdate()  # 旋转日期标签以避免重叠
    plt.tight_layout()  # 自动调整子图间的间距和标签位置

    # 绘制折线图，根据季节连接或不连接线段，并使用不同颜色
    prev_datetime = None
    prev_value = None
    prev_season = None
    for datetime, value in zip(datetimes, values):
        season = (datetime.month - 1) // 3 + 1  # 计算季节标签
        if prev_datetime is not None:
            if datetime in marker_time:
                if value is not None:
                    plt.scatter(datetime, value, color='red', marker='o', label='Point')
            if season == prev_season:  # 如果属于同一季节，则连接线段
                color = tuple(c/255 for c in color_map[season])  # 将RGB转换为范围在0到1之间的值
                plt.plot([prev_datetime, datetime], [prev_value, value], linestyle='-', color=color)
            else:  # 否则不连接线段
                color = tuple(c/255 for c in color_map[prev_season])  # 将RGB转换为范围在0到1之间的值
                plt.plot([prev_datetime, datetime], [prev_value, value], linestyle='', color=color)
        prev_datetime = datetime
        prev_value = value
        prev_season = season

    plt.show()
    plt.close()

def plot_data_with_datetimes(value: List[float], datetimes:List[datetime], color='blue'):
    # 绘制折线图，根据阈值连接或不连接线段，并使用不同颜色
    prev_datetime = None
    prev_value = None
    prev_month = None
    for datetime, value in zip(datetimes, value):
        month = datetime.month
        if prev_datetime is not None:
            time_diff = datetime - prev_datetime
            if time_diff < timedelta(days=2):  # 如果时间间隔小于阈值，则连接线段
                plt.plot([prev_datetime, datetime], [prev_value, value], linestyle='-', color=color)
            else:  # 否则不连接线段
                plt.plot([prev_datetime, datetime], [prev_value, value], linestyle='', color=color)
        prev_datetime = datetime
        prev_value = value
        prev_month = month
    
    # 显示图形


def plot_list_DataPoint(DataPoints: list[DataPoint], title: str):
    time_list = [data.start_time.date() for data in DataPoints]
    north_list = [data.north_coordinate for data in DataPoints]
    east_list = [data.east_coordinate for data in DataPoints]
    # 修改标签和标题的文本为中文
    fig, axs = plt.subplots(2, figsize=(8, 6), sharex=True) # 创建包含两个子图的画布

    # 设置日期格式化器和日期刻度定位器
    date_fmt = mdates.DateFormatter("%m-%d")  # 仅显示月-日
    date_locator = mdates.AutoDateLocator()  # 自动选择刻度间隔
    for ax in axs:
        ax.xaxis.set_major_formatter(date_fmt)
        ax.xaxis.set_major_locator(date_locator)
        ax.grid(True) # 添加网格线

    # 设置最大显示的刻度数
    plt.gcf().autofmt_xdate()  # 旋转日期标签以避免重叠
    plt.tight_layout()  # 自动调整子图间的间距和标签位置
    axs[0].plot(time_list, north_list, label='north') # 第一个子图绘制north_list
    axs[1].plot(time_list, east_list, label='east') # 第二个子图绘制east_list

    # 添加子图标签和显示图例
    axs[0].set_ylabel('North') 
    axs[1].set_ylabel('East')
    axs[1].set_xlabel('日期')

        # 设置子图标题
    axs[0].set_title(f'{title}_North')
    axs[1].set_title(f'{title}_East')

    plt.show()

def plot_list_DataPoint_without_time(DataPoints: list[DataPoint], title: str):
    north_list = [data.north_coordinate for data in DataPoints]
    east_list = [data.east_coordinate for data in DataPoints]
    # 修改标签和标题的文本为中文
    fig, axs = plt.subplots(2, figsize=(8, 6), sharex=True) # 创建包含两个子图的画布

    date_locator = mdates.AutoDateLocator()  # 自动选择刻度间隔
    for ax in axs:
        ax.xaxis.set_major_locator(date_locator)
        ax.grid(True) # 添加网格线

    # 设置最大显示的刻度数
    plt.gcf().autofmt_xdate()  # 旋转日期标签以避免重叠
    plt.tight_layout()  # 自动调整子图间的间距和标签位置
    
    # 以个数索引序号为 x 轴
    x_values = range(len(north_list))
    
    axs[0].plot(x_values, north_list, label='north') # 第一个子图绘制north_list
    axs[1].plot(x_values, east_list, label='east') # 第二个子图绘制east_list

    # 添加子图标签和显示图例
    axs[0].set_ylabel('North') 
    axs[1].set_ylabel('East')
    axs[1].set_xlabel('日期')

    # 设置子图标题
    axs[0].set_title(f'{title}_North')
    axs[1].set_title(f'{title}_East')

    plt.show()

    
def analysis_accuracy_list_DataPoint(DataPoints: list[DataPoint]):
    accuracy_evaluate_list ={
        'rms': [data.rms*1000 for data in DataPoints],
        'vertical': [data.vertical_accuracy*1000 for data in DataPoints],
        'horizontal': [data.horizontal_accuracy*1000 for data in DataPoints],
    }
    #datetimes = [data.start_time for data in DataPoints]

    # 为每种数据类型选择不同的颜色
    # colors = [(223,122,94), (60,64,91), (130,178,154)]
    thresholds = [15, 20, 10]

    # 绘制rms_values、vertical_list和horizontal_list
    for i, (key,data) in enumerate(accuracy_evaluate_list.items()):
        # color = tuple(c/255 for c in colors[i])   # 将RGB转换为范围在0到1之间的值
        threshold = thresholds[i]
        data_above_threshold = sum(1 for value in data if value > threshold)
        data_percentage = (data_above_threshold / len(data)) * 100
        print(f"{key} above {threshold}: {data_percentage:.2f}%")
        # plot_data_with_datetimes(data, datetimes, color=color) 
    
    # plt.show()


def analysis_level_DataPoint(DataPoints: list[DataPoint]):

    vertical_threshold = 0.01 #垂直
    horizontal_threshold = 0.005 #水平

    # 计算满足条件的数据点数量
    B_below_threshold = sum(1 for item in DataPoints if  item.vertical_accuracy < vertical_threshold and item.horizontal_accuracy < horizontal_threshold) 
    
    vertical_threshold = 0.02
    horizontal_threshold = 0.01
    C_below_threshold = sum(1 for item in DataPoints if  item.vertical_accuracy < vertical_threshold and item.horizontal_accuracy < horizontal_threshold) 
    # 计算满足条件的数据点比例
    total_data_points = len(DataPoints)
    B_percentage = B_below_threshold / total_data_points * 100
    C_percentage = C_below_threshold / total_data_points * 100
    print(f"B级: {B_below_threshold}/{total_data_points} ({B_percentage:.2f}%)")
    print(f"C级: {C_below_threshold}/{total_data_points} ({C_percentage:.2f}%)")


def main_TODO6():
    grouped_DataPoints = load_DataPoints()
    
    value = [item.north_coordinate for item in grouped_DataPoints['R071_1619']]
    datetimes = [item.start_time for item in grouped_DataPoints['R071_1619']]
    statistics_data_each_month(datetimes)
    statistics_continuous_data_each_month(datetimes)
    plot_data_in_season(value, datetimes)


def filter_datapoints(data_points, start_date_str):
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d %H:%M:%S")  # Ensure timezone awareness
    end_date = start_date + timedelta(days=22)
    
    filtered_points = [dp for dp in data_points if start_date <= dp.start_time <= end_date]
    return filtered_points


def main_TODO8():
    grouped_DataPoints = load_DataPoints()
    distances_and_bearings = calculate_distance_and_bearing(grouped_DataPoints["R071_1619"])
    # plot_polar(distances_and_bearings, "R071_1619")


    color_map = {1: (208, 227, 59),
                2: (237, 120, 102),
                3: (248, 186, 87),
                4: (189, 231, 247)}

    dates = ["2023-03-06 23:59:42", "2023-08-03 23:59:42", "2023-12-03 23:59:42", "2024-02-03 23:59:42"]  # Selected dates
    filtered_results = {}

    # Filtering data points for "R071_1619"
    if "R071_1619" in grouped_DataPoints:
        for date in dates:
            filtered_data = filter_datapoints(grouped_DataPoints["R071_1619"], date)
            filtered_results[date] = filtered_data
            print(f"{date} to {datetime.strptime(date, '%Y-%m-%d %H:%M:%S') + timedelta(days=23)} data: {len(filtered_data)} data points")

    plt.figure(figsize=(14.4, 9.6))
    plt.xlabel('日期')
    plt.ylabel('数值')
    plt.title('时间序列数据')

    date_locator = mdates.AutoDateLocator()
    plt.gca().xaxis.set_major_locator(date_locator)

    # Plotting individual data points
    for i, date in enumerate(filtered_results):
        values = [item.north_coordinate for item in filtered_results[date]]
        line, = plt.plot(range(len(values)), values, label=date)
        color_index = i % len(color_map) + 1
        line.set_color(np.array(color_map[color_index]) / 255.0)

    # Plotting average values
    for i, (date, data) in enumerate(filtered_results.items()):
        average_value = np.mean([item.north_coordinate for item in data])
        color_index = i % len(color_map) + 1
        avg_color = np.array(color_map[color_index]) / 255.0
        plt.axhline(y=average_value, color=avg_color, linestyle='--', label=f'Avg. {date}')

    plt.legend()
    plt.show()
    plt.close()

def check_distance_return_time(list_datapoint: list[DataPoint]) -> list[datetime]:
    distances_and_bearings = calculate_distance_and_bearing(list_datapoint)
    # Extract distances from distances_and_bearings
    distances = [distance for distance, _ in distances_and_bearings]
    time_list = [data.start_time for data in list_datapoint]
    # plot_polar(distances_and_bearings, "R071_1619")

    outliers_times = []
    for idx, distance in enumerate(distances):
        if distance > 20:
            # print(f"时间点：{time_list[idx]}")
            outliers_times.append(time_list[idx])

    return outliers_times


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

    print(len(outliers))
    return outliers, anomaly_scores, threshold

def z_score_normalize(data):
    # 将数据转换为 NumPy 数组
    data_array = np.array(data)
    
    # 计算均值和标准差
    mean = np.mean(data_array)
    std_dev = np.std(data_array)
    
    # Z-score 标准化数据
    normalized_data = (data_array - mean) / std_dev
    
    return normalized_data.tolist()  # 将 NumPy 数组转换回列表并返回

def check_knn_return_time(list_datapoint: list[DataPoint]) -> list[datetime]:
    distances_and_bearings = calculate_distance_and_bearing(list_datapoint)
    # Extract distances from distances_and_bearings
    distances = [distance for distance, _ in distances_and_bearings]
    time_list = [data.start_time for data in list_datapoint]
    # normalized_data = z_score_normalize(distances)

    # for item in normalized_data:
    #     print(item)
    # print('-------------------------------')
    outliers_times = []
    outliers, anomaly_scores, threshold = knn_anomaly_detection(distances)
    for idx in outliers:
        # print(f"时间点：{time_list[idx]}")
        outliers_times.append(time_list[idx])
    return outliers_times

def find_consecutive_data(data_dict: dict[str, list[DataPoint]], target_key: str, target_date: str) -> int:
    # 找到目标日期的数据点
    target_data: list[DataPoint] = data_dict[target_key]
    data_time_list: list[datetime] = [datapoint.start_time.date() for datapoint in target_data]

    # 将目标日期字符串转换为 datetime 类型
    target_datetime = datetime.strptime(target_date, '%Y-%m-%d').date()

    if target_datetime not in data_time_list:
        return 0
    else:
        idx = data_time_list.index(target_datetime)
        count = 1
        
        # 向前搜索
        for i in range(idx, 0, -1):
            if data_time_list[i] - data_time_list[i-1] == timedelta(days=1):
                count += 1
            else:
                break
        
        # 向后搜索
        for i in range(idx, len(data_time_list) - 1):
            if data_time_list[i+1] - data_time_list[i] == timedelta(days=1):
                count += 1
            else:
                break

        # 检查前一天和后一天是否有数据
        if target_datetime - timedelta(days=1) in data_time_list and target_datetime + timedelta(days=1) in data_time_list:
            return count
        else:
            return 0

# def main_TODO11():
#     # Example usage:
#     # dates = ["2023-04-10", "2023-05-16","2023-06-06" ,"2023-07-07", "2023-07-09", "2023-08-11", "2023-10-31"]
#     dates_to_check = ["2023-08-08","2023-08-09","2023-08-10","2023-08-11","2023-08-12","2023-08-13","2023-08-14"]
#     receiver_to_check = ["R031_1215","R031_1619","R071_1215","R071_1619","R081_1619","R082_1619"]
#     grouped_DataPoints: dict[str, list[DataPoint]] = load_DataPoints()
#     outliers_time_dict: dict[str, list[str]] = {}
#     for key, list_datapoint in grouped_DataPoints.items():
#         print(f"{key}:{len(list_datapoint)}")
#         outliers_times = check_distance_return_time(list_datapoint)
#         for time_obj in outliers_times:
#             # 转换为字符串并进行切片操作
#             time_str = str(time_obj)
#             date_str = time_str[:10]
#             if date_str not in outliers_time_dict:
#                 outliers_time_dict[date_str] = []
#             outliers_time_dict[date_str].append(f"{key}")
#     for receiver in receiver_to_check:
#         data_points_to_check = grouped_DataPoints[receiver]
#         filter_list_datapoint = []
#         for data in data_points_to_check:
#             time_str = str(data.start_time)
#             date_str = time_str[:10]
#             if date_str in dates_to_check:
#                 filter_list_datapoint.append(data)

#         plot_list_DataPoint(filter_list_datapoint, receiver)

def data_point_to_dict(dp: DataPoint) -> dict:
    return {
        'Point ID': dp.point_id,
        'North Coordinate': dp.north_coordinate,
        'East Coordinate': dp.east_coordinate,
        'Elevation': dp.elevation,
        'Latitude': dp.latitude,
        'Longitude': dp.longitude,
        'Ellipsoid Height': dp.ellipsoid_height,
        'Start Time': dp.start_time,
        'End Time': dp.end_time,
        'Duration': dp.duration,
        'PDOP': dp.pdop,
        'RMS': dp.rms,
        'Horizontal Accuracy': dp.horizontal_accuracy,
        'Vertical Accuracy': dp.vertical_accuracy,
        'North Coordinate Error': dp.north_coordinate_error,
        'East Coordinate Error': dp.east_coordinate_error,
        'Elevation Error': dp.elevation_error,
        'Height Error': dp.height_error
    }

def convert_grouped_data_points_to_df(grouped_DataPoints: dict[str, list[DataPoint]]) -> pd.DataFrame:
    all_data = []
    for group_key, data_points in grouped_DataPoints.items():
        for dp in data_points:
            data_dict = data_point_to_dict(dp)
            data_dict['Group'] = group_key
            all_data.append(data_dict)
    
    return pd.DataFrame(all_data)

def save_df_to_txt(df: pd.DataFrame, filename: str):
    with open(filename, 'w') as file:
        file.write(df.to_string(index=False))

def main_TODO9():
    grouped_DataPoints: dict[str, list[DataPoint]] = load_DataPoints()
    outliers_time_dict: dict[str, list[str]] = {}
    
    for key, list_datapoint in grouped_DataPoints.items():
        print(f"{key}:{len(list_datapoint)}")
        # outliers_times = check_distance_return_time(list_datapoint)
        outliers_times = check_knn_return_time(list_datapoint)
        for time_obj in outliers_times:
            # 转换为字符串并进行切片操作
            time_str = str(time_obj)
            date_str = time_str[:10]
            if date_str not in outliers_time_dict:
                outliers_time_dict[date_str] = []
            outliers_time_dict[date_str].append(f"{key}")

    # 按照日期对 outliers_time_dict 进行排序
    sorted_outliers_time_dict = dict(sorted(outliers_time_dict.items()))

    for key, list_time in sorted_outliers_time_dict.items():
        print(f"{key}:{list_time}")

    # 将 grouped_DataPoints 转换为 DataFrame 并保存到文本文件
    df = convert_grouped_data_points_to_df(grouped_DataPoints)
    save_df_to_txt(df, 'grouped_data_points.txt')


def plot_df(filtered_df):
    # 确保数据类型正确
    filtered_df['start_time'] = pd.to_datetime(filtered_df['start_time'])
    filtered_df['north_coordinate'] = pd.to_numeric(filtered_df['north_coordinate'], errors='coerce')

    # 按时间排序
    filtered_df = filtered_df.sort_values('start_time')

    # 创建时间期数
    filtered_df['time_period'] = range(1, len(filtered_df) + 1)

    # 绘制时序图
    plt.figure(figsize=(12, 6))
    plt.plot(filtered_df['time_period'], filtered_df['north_coordinate'], marker='o', linestyle='-')

    plt.xlabel('数据期数')
    plt.ylabel('数值')
    plt.title('时间序列数据')

    # 隐藏右边框和上边框
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # 设置刻度朝向内部，并调整刻度与坐标轴的距离
    ax.tick_params(axis='x', direction='in', pad=10)
    ax.tick_params(axis='y', direction='in', pad=10)

    # 添加虚线到 Y 轴刻度线
    for y in ax.get_yticks():
        ax.axhline(y, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

    plt.tight_layout()  # 自动调整子图间的间距和标签位置
    plt.show()

    # 统计分析
    print(filtered_df['north_coordinate'].describe())


def ListDataPoint2pdDataFrame(grouped_DataPoints: dict[str, list[DataPoint]]) -> pd.DataFrame:
    # 创建一个空的列表来存储数据
    data_list = []
    
    # 遍历字典，提取数据
    for group, points in grouped_DataPoints.items():
        for point in points:
            data_list.append({
                'Group': group,
                'csv_file': point.csv_file,
                'point_id': point.point_id,
                'north_coordinate': point.north_coordinate,
                'east_coordinate': point.east_coordinate,
                'elevation': point.elevation,
                'latitude': point.latitude,
                'longitude': point.longitude,
                'ellipsoid_height': point.ellipsoid_height,
                'start_time': point.start_time,
                'end_time': point.end_time,
                'duration': point.duration,
                'pdop': point.pdop,
                'rms': point.rms,
                'horizontal_accuracy': point.horizontal_accuracy,
                'vertical_accuracy': point.vertical_accuracy,
                'north_coordinate_error': point.north_coordinate_error,
                'east_coordinate_error': point.east_coordinate_error,
                'elevation_error': point.elevation_error,
                'height_error': point.height_error
            })
    
    # 创建 DataFrame
    df = pd.DataFrame(data_list)
    # 对每组 group 中的 start_time 重复值只保留第一个
    df = df.sort_values('start_time').drop_duplicates(subset=['Group', 'start_time'], keep='first')
    return df

def plot_seasonal_statistics(result: pd.DataFrame) -> None:
    # 创建柱状图
    season_order = ['春季', '夏季', '秋季', '冬季']
    result['season'] = pd.Categorical(result['season'], categories=season_order, ordered=True)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=result, x='season', y='sem', hue='Group', ci=None)

    # 添加标题和标签
    plt.title('北坐标标准差按组和季节')
    plt.xlabel('季节')
    plt.ylabel('标准差')

    # 显示图例
    plt.legend(title='组')
    plt.tight_layout()

    # 显示图形
    plt.show()

def plot_seasonal_dispointGroup(result: pd.DataFrame) -> None:
    specific_group = 'R031_1215'
    season_order = ['春季', '夏季', '秋季', '冬季']
    result['season'] = pd.Categorical(result['season'], categories=season_order, ordered=True)
    specific_result = result[result['Group'] == specific_group]

    # 创建子图
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    metrics = ['count', 'mean', 'std', 'sem']
    titles = ['计数', '均值', '标准差', '标准误差']

    for ax, metric, title in zip(axs.flatten(), metrics, titles):
        sns.barplot(data=specific_result, x='season', y=metric, ax=ax, palette='Blues', ci=None)
        ax.set_title(f'{title}对比：{specific_group}')
        ax.set_xlabel('季节')
        ax.set_ylabel(title)

    plt.tight_layout()
    plt.show()

def main_TODO10():
    grouped_DataPoints: dict[str, list[DataPoint]] = load_DataPoints_mini()
    df = ListDataPoint2pdDataFrame(grouped_DataPoints)

    # 确保数据类型正确
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['north_coordinate'] = pd.to_numeric(df['north_coordinate'], errors='coerce')

    # 添加季节列
    def get_season(date):
        month = date.month
        if month in [12, 1, 2]:
            return '冬季'
        elif month in [3, 4, 5]:
            return '春季'
        elif month in [6, 7, 8]:
            return '夏季'
        else:
            return '秋季'

    df['season'] = df['start_time'].dt.to_period('M').dt.to_timestamp().apply(lambda x: get_season(x))

    # 统计各个组和季节的数据
    result = df.groupby(['Group', 'season'])['north_coordinate'].agg(
        count='count',
        mean='mean',
        std='std',
        min='min',
        max='max'
    ).reset_index()

    # 计算中误差
    result['sem'] = df.groupby(['Group', 'season'])['north_coordinate'].sem().values

    # 输出到 Excel
    result.to_excel("seasonal_statistics.xlsx", index=False)
    plot_seasonal_dispointGroup(result)
    print(result)


def read_and_prepare_data(df):
    # 确保数据类型正确
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['north_coordinate'] = pd.to_numeric(df['north_coordinate'], errors='coerce')

    # 将 start_time 设置为索引
    df.set_index('start_time', inplace=True)

    # 按日期排序
    df.sort_index(inplace=True)

    # 填补缺失日期的值（可以选择插值、前向填充等方法）
    df = df.asfreq('D')  # 使用每日频率
    df['north_coordinate'].fillna(method='ffill', inplace=True)  # 前向填充缺失值

    # 划分训练集与测试集
    df_train = df.iloc[:-24]
    df_test = df.iloc[-24:]

    print("训练集样本数：", len(df_train))
    print("测试集样本数：", len(df_test))
    return df_train, df_test

def evaluate_sarima(params, df_train, df_test):
    model = SARIMAX(df_train['north_coordinate'], 
                    order=params['order'], 
                    seasonal_order=params['seasonal_order'])
    model_fit = model.fit(disp=False)
    
    forecast = model_fit.forecast(steps=len(df_test))
    
    df_test['预测'] = forecast.values
    df_test['误差'] = (df_test['预测'] - df_test['north_coordinate']) / df_test['north_coordinate']
    
    mse = mean_squared_error(df_test['north_coordinate'], df_test['预测'])
    return mse

def grid_search_sarima(df_train, df_test):
    best_params = None
    best_mse = np.inf
    
    # 调整参数网格范围
    param_grid = {
        'order': [(p, d, q) for p in range(0, 3) for d in [0] for q in range(0, 3)],
        'seasonal_order': [
            (P, D, Q, S) for P in range(0, 2) 
            for D in [0, 1] 
            for Q in range(0, 2) 
            for S in [7, 12, 31, 365]  # 选择合适的季节性周期
        ]
    }

    for params in ParameterGrid(param_grid):
        mse = evaluate_sarima(params, df_train, df_test)
        print(f"测试参数 {params} 的均方误差: {mse}")
        
        if mse < best_mse:
            best_mse = mse
            best_params = params
    
    print(f"最佳参数组合: {best_params}")
    print(f"最佳均方误差: {best_mse}")
    
    return best_params

def train_and_predict_sarima(df_train, df_test):
    best_params = grid_search_sarima(df_train, df_test)
    
    model = SARIMAX(df_train['north_coordinate'], 
                    order=best_params['order'], 
                    seasonal_order=best_params['seasonal_order'])
    model_fit = model.fit(disp=False)
    
    forecast = model_fit.forecast(steps=len(df_test))
    
    df_test['预测'] = forecast.values
    df_test['误差'] = (df_test['预测'] - df_test['north_coordinate']) / df_test['north_coordinate']
    comparison_df = df_test[['north_coordinate', '预测', '误差']]
    
    print("\n实际值、预测值和误差对比:")
    print(comparison_df)
    
    mse = mean_squared_error(df_test['north_coordinate'], df_test['预测'])
    print(f"均方误差 (MSE): {mse}")

    plt.figure(figsize=(12, 6))
    plt.plot(df_train.index, df_train['north_coordinate'], label='训练集')
    plt.plot(df_test.index, df_test['north_coordinate'], label='实际值', color='blue')
    plt.plot(df_test.index, df_test['预测'], label='预测值', color='red')
    plt.legend()
    plt.title('SARIMA预测')
    plt.show()

def main_TODO11():
    grouped_DataPoints: dict[str, list[DataPoint]] = load_DataPoints_mini()
    df = ListDataPoint2pdDataFrame(grouped_DataPoints)
    specific_group = 'R031_1215'
    filtered_df = df[df['Group'] == specific_group]
    filtered_df['start_time'] = pd.to_datetime(filtered_df['start_time'])
    filtered_df['north_coordinate'] = pd.to_numeric(filtered_df['north_coordinate'], errors='coerce')
    filtered_df = filtered_df.sort_values('start_time')

    # 选择只有 start_time 和 north_coordinate 这两列
    filtered_df = filtered_df[['start_time', 'north_coordinate']]
    
    # 找出重复的 start_time
    duplicates = filtered_df[filtered_df.duplicated(subset='start_time', keep=False)]
    print("重复的 start_time 部分：")
    print(duplicates)

    # 继续后续处理（如果有）
    df_train, df_test = read_and_prepare_data(filtered_df)
    train_and_predict_sarima(df_train, df_test)


if __name__ == "__main__":
    print("---------------------run-------------------")
    main_TODO11()
    
#main_TODO2()
# endregion
#TODO: 计算distance 并且distance按照月份 挑选几周来进行对比
#TODO: 还得看看met中的风向
#TODO: 还得处理倾斜仪数据