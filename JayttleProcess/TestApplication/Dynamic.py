import csv
from datetime import datetime as dt
from datetime import timedelta
import os
import time
import chardet
import math
import pyproj
from collections import defaultdict
from JayttleProcess import TimeSeriesDataMethod, TBCProcessCsv, CommonDecorator
from JayttleProcess.TimeSeriesDataMethod import TimeSeriesData
import numpy as np
from matplotlib.colors import LinearSegmentedColormap,ListedColormap
from matplotlib.animation import FuncAnimation
import itertools
import matplotlib.pyplot as plt
from typing import Union, List, Tuple
from scipy import stats
# region 字体设置
plt.rcParams['font.sans-serif'] = ['SimSun']  # 指定宋体为默认字体
plt.rcParams['font.sans-serif'] = 'SimSun'  # 使用指定的中文字体

class GNSSData:
    def __init__(self, data_name: str, x_coord: float, y_coord: float, z_coord: float, timestamp: str):
        self.data_name = data_name
        self.x_coord = float(x_coord)
        self.y_coord = float(y_coord)
        self.z_coord = float(z_coord)
        self.timestamp = dt.strptime(timestamp.strip(), "%Y-%m-%d %H:%M:%S")

    def __str__(self) -> str:
        return f"Data Name: {self.data_name}\n" \
               f"X Coordinate: {self.x_coord}\n" \
               f"Y Coordinate: {self.y_coord}\n" \
               f"Z Coordinate: {self.z_coord}\n" \
               f"Timestamp: {self.timestamp}"

def load_file_dynamics_data(file_path: str) -> List[GNSSData]:
    start_time = time.time()
    gnss_data_array = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            data = line.split('\t')
            if len(data) == 5:
                gnss_data = GNSSData(data[0], data[1], data[2], data[3], data[4].strip())
                gnss_data_array.append(gnss_data)
    end_time = time.time()
    runtime = end_time - start_time
    print(f"程序运行时间：{runtime:.2f} 秒")
    return gnss_data_array

def save_data_to_file(data: Union[GNSSData, List[GNSSData], np.ndarray], file_path: str) -> None:
    with open(file_path, 'w') as file:
        if isinstance(data, np.ndarray):
            data = data.tolist()
        if not isinstance(data, list):
            data = [data]
        for item in data:
            file.write(str(item))
            file.write('\n')


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

def run_main():
    data_file = r"D:\python_proj2\daily_data_5.txt"
    gnss_data_array = load_file_dynamics_data(data_file)
    print(len(gnss_data_array))

    R031_lat, R031_lon = 35.473642676796, 118.054358431073
    east, north = TBCProcessCsv.convert_coordinates(R031_lat, R031_lon)

    # 初始化字典来存储不同时间戳的数据
    data_by_hour = {16: [], 20: [], 17: [], 21: [], 18: [], 22: [], 19: [], 23: []}

    # 分类每个 GNSSData 对象的数据
    for gnss_data in gnss_data_array:
        hour = gnss_data.timestamp.hour
        if hour in data_by_hour:
            gnss_data.x_coord -= east
            gnss_data.y_coord -= north
            gnss_data.x_coord *= 1000
            gnss_data.y_coord *= 1000
            distance = calculate_distance(0, 0, gnss_data.x_coord, gnss_data.y_coord)
            bearing = calculate_bearing(gnss_data.x_coord, gnss_data.y_coord)
            data_by_hour[hour].append((distance, bearing))

    # 创建子图网格
    fig, axs = plt.subplots(2, 4, subplot_kw={'projection': 'polar'}, figsize=(10, 20))

    # 遍历字典中的每个时间戳数据并绘制到相应的子图中
    for i, (hour, data) in enumerate(data_by_hour.items()):
        row = i % 2
        col = i // 2
        ax = axs[row, col]
        for distance, bearing in data:
            theta = np.deg2rad(bearing)
            ax.plot(theta, distance, 'o', color='blue')
        ax.set_title(f"{hour}点的极坐标")

    plt.tight_layout()
    plt.show()

def run_main1():
    data_files = [r"D:\python_proj2\daily_data_1.txt",
                  r"D:\python_proj2\daily_data_2.txt",
                  r"D:\python_proj2\daily_data_3.txt",
                  r"D:\python_proj2\daily_data_4.txt",
                  r"D:\python_proj2\daily_data_5.txt",
                  r"D:\python_proj2\daily_data_6.txt",
                  r"D:\python_proj2\daily_data_7.txt",
                  r"D:\python_proj2\daily_data_8.txt"]

    gnss_data_arrays = [load_file_dynamics_data(data_file) for data_file in data_files]

    R031_lat, R031_lon = 35.473642676796, 118.054358431073
    east, north = TBCProcessCsv.convert_coordinates(R031_lat, R031_lon)

    # 创建子图网格
    fig, axs = plt.subplots(2, 4, subplot_kw={'projection': 'polar'}, figsize=(10, 20))

    # 遍历每个 gnss_data_array，并绘制时间戳为1的数据到相应的子图中
    for i, gnss_data_array in enumerate(gnss_data_arrays):
        for gnss_data in gnss_data_array:
            hour = gnss_data.timestamp.hour
            if hour == 1:
                gnss_data.x_coord -= east
                gnss_data.y_coord -= north
                gnss_data.x_coord *= 1000
                gnss_data.y_coord *= 1000
                distance = calculate_distance(0, 0, gnss_data.x_coord, gnss_data.y_coord)
                bearing = calculate_bearing(gnss_data.x_coord, gnss_data.y_coord)

                row = i % 2
                col = i // 2
                ax = axs[row, col]
                theta = np.deg2rad(bearing)
                ax.plot(theta, distance, 'o', color='blue')
                ax.set_title(f"第{i}天的极坐标")

    plt.tight_layout()
    plt.show()

# def update(frame):
#     ax.clear()
#     distance, bearing = distance_list[frame], bearing_list[frame]
#     theta = np.deg2rad(bearing)
#     ax.plot(theta, distance, 'o', color='blue')
#     ax.set_title(f"极坐标 - 帧 {frame+1}/{len(distance_list)}")

if __name__ == "__main__":
    run_main1()

#TODO:用RTK和倾斜仪 数据来进行对比
#TODO:用4h数据来看是否变化；
#TODO:风速数据是否可用
#TODO: