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
from JayttleProcess import TimeSeriesDataMethod, TBCProcessCsv, CommonDecorator
from JayttleProcess.TimeSeriesDataMethod import TimeSeriesData

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
        self.east_coordinate_error = east_coordinate_error  # 基线解算的东坐标差值
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

    easting, northing = east_north_coordinates["R051"]

    east_test = 595169.880297 - easting
    east_test *= 1000
    north_test = 3927680.058262 - northing
    north_test *= 1000
    print(f"带网平差：east_test:{east_test}\t north_test={north_test}")
    
    east_test = 595169.873721 - easting
    east_test *= 1000
    north_test = 3927680.059320 - northing
    north_test *= 1000
    print(f"不带网平差：east_test:{east_test}\t north_test={north_test}")

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


def main_TODO3():
    locations = {
        "B011": (35.474852, 118.072091),
        "B011_11": (35.474853, 118.072092),
        "B021": (35.465000, 118.035312),
        "B021_11": (35.465001, 118.035313),
        "R031": (35.473642676796, 118.054358431073),
        "R032": (35.473666223407, 118.054360237421),
        "R051": (35.473944469154, 118.048584306326),
        "R052": (35.473974942138, 118.048586858521),
        "R071": (35.474177696631, 118.044201562812),
        "R072": (35.474204534806, 118.044203691212),
        "R081": (35.474245973695, 118.042930824340),
        "R082": (35.474269576552, 118.042932741649),
    }

    
    # 定义一个字典来存储每个位置对应的东北坐标
    east_north_coordinates = {}
    # 批量转换经纬度坐标为东坐标 北坐标
    for location, (lat, lon) in locations.items():
        easting, northing = TBCProcessCsv.convert_coordinates(lat, lon)
        east_north_coordinates[location] = (easting, northing)
    
    for location in east_north_coordinates:
        print(f"{location}: {east_north_coordinates[location]}")
    # # 计算以'R'开头的位置与'B011'和'B021'的距离
    # for location, (easting, northing) in east_north_coordinates.items():
    #     if location.startswith('R'):
    #         for b_location, (b_easting, b_northing) in east_north_coordinates.items():
    #             if b_location.startswith('B'):
    #                 distance = calculate_distance(easting, northing, b_easting, b_northing)
    #                 print(f"距离 {location} 到 {b_location} 的距离为: {distance:.2f} 米")


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


def main_TODO7():
    grouped_DataPoints = load_DataPoints()


    color_map = {1: (242, 204, 142),
                2: (223, 122, 94),
                3: (60, 64, 91),
                4: (130, 178, 154)}
    dates = ["2023-03-06 23:59:42", "2023-04-03 23:59:42", "2023-08-03 23:59:42", "2023-12-03 23:59:42", "2024-02-03 23:59:42"]
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
    for i, date in enumerate(filtered_results):
        values = [item.north_coordinate for item in filtered_results[date]]
        line, = plt.plot(range(len(values)), values, label=date)
        color_index = i % len(color_map) + 1
        line.set_color(np.array(color_map[color_index]) / 255.0)

    plt.legend()
    plt.show()
    plt.close()

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

def main_TODO9():
    grouped_DataPoints: dict[str, list[DataPoint]] = load_DataPoints()
    outliers_time_dict: dict[str, list[str]] = {}
    for key, list_datapoint in grouped_DataPoints.items():
        print(f"{key}:{len(list_datapoint)}")
        outliers_times = check_distance_return_time(list_datapoint)
        for time_obj in outliers_times:
            # 转换为字符串并进行切片操作
            time_str = str(time_obj)
            date_str = time_str[:10]
            if date_str not in outliers_time_dict:
                outliers_time_dict[date_str] = []
            outliers_time_dict[date_str].append(f"{key}")


    for key, list_time in outliers_time_dict.items():
        print(f"{key}:{list_time}")


    # 新建一个字典用于存储日期和对应的总权重
    date_weights = {}

    # 映射每个前缀与其对应的权重
    prefix_weights: dict[str, float] = {}
    for marker_name, list_datapoint in grouped_DataPoints.items():
        rms_list = [data.rms*1000 for data in list_datapoint]
        average_rms = sum(rms_list) / len(rms_list)
        prefix_weights[marker_name[:4]] = 1.0 / average_rms 
        #用rms平均值来作为权值


    # 遍历每个日期及其对应的字符串列表
    for date, time_list in outliers_time_dict.items():
        # 初始化总权重
        total_weight = 0
        # 统计每个前四个字符出现的次数，并分配权重
        for item in time_list:
            prefix = item[:4]
            if prefix in prefix_weights:
                total_weight += prefix_weights[prefix]

        # 存储日期和对应的总权重
        date_weights[date] = total_weight

    # 输出日期和其对应的总权重
    for date, weight in date_weights.items():
        if weight > 0.20:
            print(f"{date}: {weight:.2f}: {outliers_time_dict[date]}")
            for target_key in outliers_time_dict[date]:
                consecutive_count = find_consecutive_data(grouped_DataPoints, target_key, date)
                print(f"{target_key} around {date} is: {consecutive_count}")
    
    

    # consecutive_count = find_consecutive_data(grouped_DataPoints, target_key, date)
    # print(f"The number of consecutive occurrences of {target_key} around {date} is: {consecutive_count}")

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

#TODO:帮我对权值过大的日期前后多方面看看
    
    # plt.figure(figsize=(14.4, 9.6))
    # plt.xlabel('日期')
    # plt.ylabel('数值')
    # plt.title('时间序列数据')
    # date_locator = mdates.AutoDateLocator()
    # plt.gca().xaxis.set_major_locator(date_locator)

    # plt.plot(range(len(distances)), distances, label='距离')
    # plt.plot(range(len(rms_list)), rms_list, label='rms', alpha=0.3)
    # plt.legend()
    # plt.show()
    # plt.close()

def main_TODO10():
    # Example usage:
    file_path = r"D:\Program Files (x86)\Software\OneDrive\PyPackages_DataSave\tianmeng_tiltmeter.txt"
    tilt_sensor_reader = TiltSensorDataReader(file_path)
    tilt_sensor_data = tilt_sensor_reader.read_data()

    print(len(tilt_sensor_data))


def main_TODO11():
    # Example usage:
    # dates = ["2023-04-10", "2023-05-16","2023-06-06" ,"2023-07-07", "2023-07-09", "2023-08-11", "2023-10-31"]
    dates_to_check = ["2023-08-08","2023-08-09","2023-08-10","2023-08-11","2023-08-12","2023-08-13","2023-08-14"]
    receiver_to_check = ["R031_1215","R031_1619","R071_1215","R071_1619","R081_1619","R082_1619"]
    grouped_DataPoints: dict[str, list[DataPoint]] = load_DataPoints()
    outliers_time_dict: dict[str, list[str]] = {}
    for key, list_datapoint in grouped_DataPoints.items():
        print(f"{key}:{len(list_datapoint)}")
        outliers_times = check_distance_return_time(list_datapoint)
        for time_obj in outliers_times:
            # 转换为字符串并进行切片操作
            time_str = str(time_obj)
            date_str = time_str[:10]
            if date_str not in outliers_time_dict:
                outliers_time_dict[date_str] = []
            outliers_time_dict[date_str].append(f"{key}")
    for receiver in receiver_to_check:
        data_points_to_check = grouped_DataPoints[receiver]
        filter_list_datapoint = []
        for data in data_points_to_check:
            time_str = str(data.start_time)
            date_str = time_str[:10]
            if date_str in dates_to_check:
                filter_list_datapoint.append(data)

        plot_list_DataPoint(filter_list_datapoint, receiver)


def check_outliers_in_list_datapoint(list_datapoint: List[DataPoint]):
    # 使用列表推导式过滤出符合条件的 DataPoint 对象
    filtered_list = [datapoint for datapoint in list_datapoint if datapoint.east_coordinate <= 100]
    return filtered_list

def main_TODO12():
    grouped_DataPoints: dict[str, list[DataPoint]] = load_DataPoints()
    marker_name = "R071_1619"
    distances_and_bearings = calculate_distance_and_bearing(grouped_DataPoints[marker_name])
    datetime_list = [data.start_time for data in grouped_DataPoints[marker_name]]

    list_distance = []
    indices_to_remove = []  # 用于存储需要删除的索引
    for idx, (distance, bearing) in enumerate(distances_and_bearings):
        if distance >= 100:
            indices_to_remove.append(idx)
        else:
            list_distance.append(distance)

    print(f"错误数据有：{len(indices_to_remove)}")
    datetime_list = (data.start_time for idx, data in enumerate(grouped_DataPoints[marker_name]) if idx not in indices_to_remove)


    dates = ["2023-04-10", "2023-05-16", "2023-06-06", "2023-07-07", "2023-07-09", "2023-08-11", "2023-10-31"]

    # 将字符串日期转换为 datetime 对象
    datetime_dates = [datetime.strptime(date, "%Y-%m-%d") for date in dates]

    plot_data_in_season_version2(list_distance, datetime_list, marker_time=datetime_dates, title=marker_name)

def main_TODO13():
    grouped_DataPoints: dict[str, list[DataPoint]] = load_DataPoints()
    # marker_name = "R031_1619"
    # for marker_name, list_datapoint in grouped_DataPoints.items():
    #     print(f"{marker_name}:")
    #     analysis_level_DataPoint(list_datapoint)

if __name__ == "__main__":
    print("---------------------run-------------------")
    main_TODO13()

#main_TODO2()
# endregion
#TODO: 计算distance 并且distance按照月份 挑选几周来进行对比
#TODO: 还得看看met中的风向
#TODO: 还得处理倾斜仪数据

#TODO: 测量精度中的水平精度和垂直精度还需要 进行进一步的处理 因为我数据读取是一行，而基线解算的精度有两条
