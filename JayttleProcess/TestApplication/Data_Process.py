import os
import csv
import pandas as pd
import math
import chardet
from collections import defaultdict
from typing import List, Tuple, Dict
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
from JayttleProcess import TimeSeriesDataMethod, TBCProcessCsv, CommonDecorator
from JayttleProcess.TimeSeriesDataMethod import TimeSeriesData

class DataPoint:
    def __init__(self, point_id: str, north_coordinate: float, east_coordinate: float, elevation: float,
                 latitude: float, longitude: float, ellipsoid_height: float, start_time: str, end_time: str,
                 duration: str, pdop: float, rms: float, horizontal_accuracy: float, vertical_accuracy: float,
                 north_coordinate_error: float, east_coordinate_error: float, elevation_error: float,
                 height_error: str):
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


def read_csv_to_datapoints(csv_file: str) -> list[DataPoint]:
    with open(csv_file, 'rb') as f:
        rawdata = f.read()
        global encoding
        encoding = chardet.detect(rawdata)['encoding']
    df = pd.read_csv(csv_file, sep='\t', nrows=1, parse_dates=['GNSS矢量观测.开始时间', 'GNSS矢量观测.结束时间'], encoding=encoding)
    datapoints = []
    for index, row in df.iterrows():
        datapoint = DataPoint(
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

# 定义一个函数来计算距离和方位角
def calculate_distance_and_bearing(points: List[Tuple[TimeSeriesData, TimeSeriesData]]) -> List[Tuple[TimeSeriesData, TimeSeriesData]]:
    distances_and_bearings: List[Tuple[TimeSeriesData, TimeSeriesData]] = []
    for north_point, east_point in points:
        # 计算与原点 (0, 0) 的距离
        distance = calculate_distance(0, 0, east_point.value, north_point.value)
        # 计算方位角
        bearing = calculate_bearing(east_point.value, north_point.value)
        distances_and_bearings.append((TimeSeriesData(distance, north_point.datetime), TimeSeriesData(bearing, north_point.datetime)))
    return distances_and_bearings

# 定义一个函数来绘制极坐标图
def plot_polar(distances_and_bearings: List[Tuple[TimeSeriesData, TimeSeriesData]], title = None):
    # 创建一个极坐标图
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    
    # 遍历距离和方位角，绘制每个点
    for distance, bearing in distances_and_bearings:
        # 将方位角转换为弧度
        theta = np.deg2rad(bearing.value)
        # 绘制点
        ax.plot(theta, distance.value, 'o', color='blue')
    
    # 设置极坐标图的标题
    ax.set_title(f'{title}极坐标')
    
    # 显示图形
    plt.show()


def plot_polar_with_rms(rms_list: List[float], distances_and_bearings: List[Tuple[TimeSeriesData, TimeSeriesData]], threshold: float, title=None):
    # 创建一个极坐标图
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    
    # 遍历距离、方位角和 RMS 值，绘制每个点
    for i, (distance, bearing) in enumerate(distances_and_bearings):
        # 将方位角转换为弧度
        theta = np.deg2rad(bearing.value)
        
        # 绘制点
        if rms_list[i] > threshold:
            ax.plot(theta, distance.value, 'o', color='red', label='RMS Exceeded Threshold')
        else:
            ax.plot(theta, distance.value, 'o', color='blue')
    
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

def load_DataPoints():
    """
    Load data from multiple CSV files and process them.
    """
    R031_1619_DataPoints = load_csv_data(r'D:\Ropeway\R031_1619')
    R031_0407_DataPoints = load_csv_data(r'D:\Ropeway\FTPCsv2')
    R051_1619_DataPoints = load_csv_data(r'D:\Ropeway\R051_1619')
    R052_1619_DataPoints = load_csv_data(r'D:\Ropeway\R052_1619')
    R081_1619_DataPoints = load_csv_data(r'D:\Ropeway\FTPCsv3')
    R082_1619_DataPoints = load_csv_data(r'D:\Ropeway\R082_1619')
    return R031_1619_DataPoints, R031_0407_DataPoints, R051_1619_DataPoints, R052_1619_DataPoints, R081_1619_DataPoints, R082_1619_DataPoints

def load_all_data() -> Tuple[List[Tuple[TimeSeriesData, TimeSeriesData]], List[Tuple[TimeSeriesData, TimeSeriesData]], List[Tuple[TimeSeriesData, TimeSeriesData]], List[Tuple[TimeSeriesData, TimeSeriesData]], Dict[str, Tuple[float, float]]]:
    """
    Load data from multiple CSV files and process them.
    """
    R031_1619_DataPoints, R031_0407_DataPoints, R051_1619_DataPoints, R052_1619_DataPoints, R081_1619_DataPoints, R082_1619_DataPoints = load_DataPoints()

    R031_lat, R031_lon = 35.473642676796, 118.054358431073
    R032_lat, R032_lon = 35.473666223407, 118.054360237421
    R051_lat, R051_lon = 35.473944469154, 118.048584306326
    R052_lat, R052_lon = 35.473974942138, 118.048586858521
    R071_lat, R071_lon = 35.474177696631, 118.044201562812
    R072_lat, R072_lon = 35.474204534806, 118.044203691212
    R081_lat, R081_lon = 35.474245973695, 118.042930824340
    R082_lat, R082_lon = 35.474269576552, 118.042932741649


    points: Dict[str, Tuple[float, float]] = {
        'R031_1619': process_data_points(R031_1619_DataPoints, R031_lat, R031_lon, "north_coordinate", "east_coordinate"),
        'R031_0407': process_data_points(R031_0407_DataPoints, R031_lat, R031_lon, "north_coordinate", "east_coordinate"),
        'R051_1619': process_data_points(R051_1619_DataPoints, R051_lat, R051_lon, "north_coordinate", "east_coordinate"),
        'R052_1619': process_data_points(R052_1619_DataPoints, R052_lat, R052_lon, "north_coordinate", "east_coordinate"),
        'R081_1619': process_data_points(R081_1619_DataPoints, R081_lat, R081_lon, "north_coordinate", "east_coordinate"),
        'R082_1619': process_data_points(R082_1619_DataPoints, R082_lat, R082_lon, "north_coordinate", "east_coordinate"),
    }

    R031_1619_combined: List[Tuple[TimeSeriesData, TimeSeriesData]] = list(zip(points['R031_1619'][0], points['R031_1619'][1]))
    R031_0407_combined: List[Tuple[TimeSeriesData, TimeSeriesData]] = list(zip(points['R031_0407'][0], points['R031_0407'][1]))
    R051_1619_combined: List[Tuple[TimeSeriesData, TimeSeriesData]] = list(zip(points['R051_1619'][0], points['R051_1619'][1]))
    R052_1619_combined: List[Tuple[TimeSeriesData, TimeSeriesData]] = list(zip(points['R052_1619'][0], points['R052_1619'][1]))
    R081_1619_combined: List[Tuple[TimeSeriesData, TimeSeriesData]] = list(zip(points['R081_1619'][0], points['R081_1619'][1]))
    R082_1619_combined: List[Tuple[TimeSeriesData, TimeSeriesData]] = list(zip(points['R082_1619'][0], points['R082_1619'][1]))

    return R031_1619_combined, R031_0407_combined, R051_1619_combined, R052_1619_combined, R081_1619_combined, R082_1619_combined, points


def calculate_daily_movements(combined_data: Dict[datetime, Tuple[Tuple[TimeSeriesData, TimeSeriesData], Tuple[TimeSeriesData, TimeSeriesData]]]) -> Dict[datetime, Tuple[float, float]]:
    daily_movements = {}
    
    for date, data in combined_data.items():
        north_east_1619, north_east_0407 = data
        north_movement = north_east_0407[0].value - north_east_1619[0].value
        east_movement = north_east_0407[1].value - north_east_1619[1].value
        daily_movements[date] = (north_movement, east_movement)
    
    return daily_movements



@CommonDecorator.log_function_call
def main_TODO1():

    R031_1619_DataPoints, R031_0407_DataPoints, R051_1619_DataPoints, R052_1619_DataPoints, R081_1619_DataPoints, R082_1619_DataPoints = load_DataPoints()     
    R031_1619_combined, R031_0407_combined, R051_1619_combined, R052_1619_combined, R081_1619_combined, R082_1619_combined, points = load_all_data()
    R031_1619_combined = list(R031_1619_combined)
    R031_0407_combined = list(R031_0407_combined)
    R051_1619_combined = list(R051_1619_combined)
    R081_1619_combined = list(R081_1619_combined)
    # 调用函数计算每个点与原点 (0, 0) 的距离和方位角
    R031_1619_distances_and_bearings: List[Tuple[TimeSeriesData,TimeSeriesData]] = calculate_distance_and_bearing(R031_1619_combined)
    R031_0407_distances_and_bearings: List[Tuple[TimeSeriesData,TimeSeriesData]] = calculate_distance_and_bearing(R031_0407_combined)
    R051_1619_distances_and_bearings: List[Tuple[TimeSeriesData,TimeSeriesData]] = calculate_distance_and_bearing(R051_1619_combined)
    R052_1619_distances_and_bearings: List[Tuple[TimeSeriesData,TimeSeriesData]] = calculate_distance_and_bearing(R052_1619_combined)
    R081_1619_distances_and_bearings: List[Tuple[TimeSeriesData,TimeSeriesData]] = calculate_distance_and_bearing(R081_1619_combined)
    R082_1619_distances_and_bearings: List[Tuple[TimeSeriesData,TimeSeriesData]] = calculate_distance_and_bearing(R082_1619_combined)

     # plot_polar(R082_1619_DataPoints, "R031_1619")
    rms_list = [point.rms * 1000 for point in R031_1619_DataPoints]
    rms_list2 = [point.rms * 1000 for point in R031_0407_DataPoints]

    plot_polar_with_rms(rms_list, R031_1619_distances_and_bearings, 10, 'R031_1619 withRMS')
    plot_polar_with_rms_exceeded(rms_list, R031_1619_distances_and_bearings, 10, 'R031_1619 withRMS')
    plot_polar_without_rms_exceeded(rms_list, R031_1619_distances_and_bearings, 10, 'R031_1619 withRMS')
    plot_polar_without_rms_exceeded(rms_list2, R031_0407_distances_and_bearings, 10, 'R031_0407 withRMS')
    
    
    # plot_polar_in_month(R031_1619_distances_and_bearings, "R031_1619")
    # plot_polar_in_month(R031_0407_distances_and_bearings, "R031_0407")
    # plot_polar_in_month(R081_1619_distances_and_bearings, "R081_1619")
    # 调用 calculate_polar_coordinates 函数计算极坐标距离和方位角
    polar_coordinates = calculate_polar_coordinates(points)
    # 绘制极坐标图
    plot_polar_in_float(polar_coordinates, title="R051, R071, R081")

    # plot_polar_compare([R031_1619_distances_and_bearings, R031_0407_distances_and_bearings]) 

    # 定义一个字典来存储合并后的数据
    combined_data: Dict[datetime, Tuple[Tuple[TimeSeriesData, TimeSeriesData], Tuple[TimeSeriesData, TimeSeriesData]]] = defaultdict(tuple)

    # 初始化指针
    pointer_1619 = 0
    pointer_0407 = 0

    # 遍历两个列表，直到其中一个列表遍历完为止
    while pointer_1619 < len(R031_1619_combined) and pointer_0407 < len(R031_0407_combined):
        data_1619 = R031_1619_combined[pointer_1619]
        data_0407 = R031_0407_combined[pointer_0407]
        
        date_1619 = data_1619[0].datetime.date()
        date_0407 = data_0407[0].datetime.date()
        
        # 如果日期相同，则将数据组合并存入combined_data字典中
        if date_1619 == date_0407:
            combined_data[date_1619] = (data_1619, data_0407)
            # 移动指针以继续搜索下一对数据
            pointer_1619 += 1
            pointer_0407 += 1
        elif date_1619 < date_0407:
            # 如果1619的日期早于0407的日期，则只移动1619的指针
            pointer_1619 += 1
        else:
            # 如果0407的日期早于1619的日期，则只移动0407的指针
            pointer_0407 += 1
    
    # 定义一个字典来存储每天的坐标移动情况
    daily_movements: Dict[datetime, Tuple[float, float]] = calculate_daily_movements(combined_data)

    # 定义一个文件路径来保存数据
    file_path = "daily_movements.txt"

    with open(file_path, 'w') as file:
        for date, movements in daily_movements.items():
            north_movement, east_movement = movements
            file.write(f"Date: {date}, North Movement: {north_movement}, East Movement: {east_movement}\n")
    
    daily_movements_list: List[Tuple[TimeSeriesData, TimeSeriesData]] = []

    for date, movements in daily_movements.items():
        formatted_date = date.strftime("%Y-%m-%d 00:00:00")
        north_movement, east_movement = movements
        # 创建TimeSeriesData对象并添加到列表中
        north_time_series = TimeSeriesData(north_movement, formatted_date)
        east_time_series = TimeSeriesData(east_movement, formatted_date)
        daily_movements_list.append((north_time_series, east_time_series))

    daily_movements_distances_and_bearings: List[Tuple[TimeSeriesData,TimeSeriesData]] = calculate_distance_and_bearing(daily_movements_list)
    # plot_polar_in_month(daily_movements_distances_and_bearings,"1619->0407移动位置")

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
# 调用主函数
main_TODO1()
#main_TODO2()
# endregion
#TODO1：把东坐标 北坐标放入一个坐标系里看角度与分布  是否与风向有关
#TODO2：找出突变的原因 可以先计算他的前后差值如果大则查看他的计算精度

