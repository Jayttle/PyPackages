import os
import csv
import pandas as pd
import json
import chardet
import pyproj
import numpy as np
from datetime import datetime
from itertools import groupby
import matplotlib.pyplot as plt
from operator import attrgetter
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Dict
from JayttleProcess import CommonDecorator
from JayttleProcess import TimeSeriesDataMethod as TSD
from JayttleProcess.TimeSeriesDataMethod import TimeSeriesData



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
#TODO:检查下tianmeng_met中每天取到最大值最小值的时候的 时间段

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

    # Normalize data
    scaler_weather_max = MinMaxScaler()
    scaler_weather_min = MinMaxScaler()
    weather_max_temps_normalized = scaler_weather_max.fit_transform(np.array(weather_max_temps).reshape(-1, 1))
    weather_min_temps_normalized = scaler_weather_min.fit_transform(np.array(weather_min_temps).reshape(-1, 1))

    scaler_temperature_max = MinMaxScaler()
    scaler_temperature_min = MinMaxScaler()
    temperature_max_temps_normalized = scaler_temperature_max.fit_transform(np.array(temperature_max_temps).reshape(-1, 1))
    temperature_min_temps_normalized = scaler_temperature_min.fit_transform(np.array(temperature_min_temps).reshape(-1, 1))
    
    weather_max_temps_normalized_list = weather_max_temps_normalized.tolist()
    weather_min_temps_normalized_list = weather_min_temps_normalized.tolist()
    temperature_max_temps_normalized_list = temperature_max_temps_normalized.tolist()
    temperature_min_temps_normalized_list = temperature_min_temps_normalized.tolist()

    weather_max_temps_normalized_list = np.array(weather_max_temps_normalized_list).reshape(-1)
    temperature_max_temps_normalized_list = np.array(temperature_max_temps_normalized_list).reshape(-1)
    weather_min_temps_normalized_list = np.array(weather_min_temps_normalized_list).reshape(-1)
    temperature_min_temps_normalized_list = np.array(temperature_min_temps_normalized_list).reshape(-1)


    # Calculate correlation coefficients
    correlation_max_temp, _ = pearsonr(weather_max_temps_normalized_list, temperature_max_temps_normalized_list)
    correlation_min_temp, _ = pearsonr(weather_min_temps_normalized_list, temperature_min_temps_normalized_list)

    print(f"当天最高温度pearson相关系数: {correlation_max_temp}")
    print(f"当天最低温度pearson相关系数: {correlation_min_temp}")

    mse_max_temp = mean_squared_error(weather_max_temps_normalized_list, temperature_max_temps_normalized_list)
    mse_min_temp = mean_squared_error(weather_min_temps_normalized_list, temperature_min_temps_normalized_list)
    mae_max_temp = mean_absolute_error(weather_max_temps_normalized_list, temperature_max_temps_normalized_list)
    mae_min_temp = mean_absolute_error(weather_min_temps_normalized_list, temperature_min_temps_normalized_list)

    print(f"当天最高温度--均方误差: {mse_max_temp}")
    print(f"当天最低温度--均方误差: {mse_min_temp}")
    print(f"当天最高温度--平均绝对误差: {mae_max_temp}")
    print(f"当天最低温度--平均绝对误差: {mae_min_temp}")

    plt.figure(figsize=(10, 6))

    # Plot normalized weather_min_temps
    plt.plot(range(len(weather_max_temps_normalized_list)), weather_max_temps_normalized_list, label='微软天气', color='blue')

    # Plot normalized temperature_min_temps
    plt.plot(range(len(temperature_max_temps_normalized_list)), temperature_max_temps_normalized_list, label='气象仪', color='red')

    # Adding labels and title
    plt.xlabel('Days')
    plt.ylabel('Normalized 当天最低温度')
    plt.title('微软天气与气象仪数据对比--当天最高温度（归一化）')
    plt.legend()

    # Show plot
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Calculate differences
    # temp_differences = [weather_temp - met_temp for weather_temp, met_temp in zip(weather_min_temps, temperature_min_temps)]
    #     # Calculate standard deviation
    # std_deviation = np.std(temp_differences)

    # # Calculate mean
    # mean_temp_diff = np.mean(temp_differences)

    # # Calculate coefficient of variation
    # coefficient_variation = (std_deviation / mean_temp_diff) * 100
    
    # print(f"温度差值的平均值为: {mean_temp_diff}")
    # print(f"温度差值的标准差为: {std_deviation}")
    # print(f"温度差值的变异系数为: {coefficient_variation}%")
    
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

    plt.figure(figsize=(9,6))
    # plt.plot(range(len(humidity_max_temps)), humidity_max_temps, label='传感器最低湿度', color='green')
    plt.plot(range(len(humidity_mean_temps)), humidity_mean_temps, label='传感器平均湿度', color='green')
    # plt.plot(range(len(humidity_min_temps)), humidity_min_temps, label='传感器最高湿度', color='red')
    plt.plot(range(len(weather_humidity_data_list)), weather_humidity_data_list, label='微软天气湿度', color='blue')
    # plt.bar(range(len(weather_humidity_data_list)), weather_humidity_data_list, label='微软天气湿度', color='blue', alpha=0.1)
    plt.grid(color='#95a5a6',linestyle='--',linewidth=1,axis='y',alpha=0.6)

    plt.xlabel('Days')
    plt.ylabel('湿度')
    plt.title('湿度数据对比')
 
    plt.tight_layout()
    plt.show()

    mse_max_temp = mean_squared_error(weather_humidity_data_list, humidity_mean_temps)
    mae_max_temp = mean_absolute_error(weather_humidity_data_list, humidity_mean_temps)

    print(f"均方误差: {mse_max_temp}")
    # print(f"当天最低温度--均方误差: {mse_min_temp}")
    print(f"平均绝对误差: {mae_max_temp}")
    # print(f"当天最低温度--平均绝对误差: {mae_min_temp}")
    

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

    # Iterate over each group
    for date, group in grouped_data:
        # Extract temperatures with corresponding timestamps from the group
        wind_timestamps = [(met.wind_speed, met.datetime_obj.time()) for met in group]
        
        # Calculate maximum and minimum temperatures
        max_temp, max_temp_time = max(wind_timestamps, key=lambda x: x[0])
        min_temp, min_temp_time = min(wind_timestamps, key=lambda x: x[0])

        wind_max_temps.append(max_temp)
        wind_min_temps.append(min_temp)
        
        weather_wind_data_list.append(weather_wind_data[str(date)])
        # Store max and min temperatures along with their times in the dictionaries
        wind_data[str(date)] = (max_temp, min_temp)

        
        # Calculate average humidity
        avg_humidity = sum(val[0] for val in wind_timestamps) / len(wind_timestamps) *3.6
        wind_mean_temps.append(avg_humidity)



    correlation_max_temp, _ = pearsonr(wind_max_temps, weather_wind_data_list)
    correlation_min_temp, _ = pearsonr(wind_min_temps, weather_wind_data_list)
    correlation_mean_temp, _ = pearsonr(wind_mean_temps, weather_wind_data_list)

    print(f"传感器最高湿度与微软天气湿度pearson相关系数: {correlation_max_temp}")
    print(f"传感器最低湿度与微软天气湿度pearson相关系数: {correlation_min_temp}")
    print(f"传感器平均湿度与微软天气湿度pearson相关系数: {correlation_mean_temp}")

    plt.figure(figsize=(9,6))
    # plt.plot(range(len(humidity_max_temps)), humidity_max_temps, label='传感器最低湿度', color='green')
    plt.plot(range(len(wind_mean_temps)), wind_mean_temps, label='传感器平均风速', color='green')
    # plt.plot(range(len(humidity_min_temps)), humidity_min_temps, label='传感器最高湿度', color='red')
    plt.plot(range(len(weather_wind_data_list)), weather_wind_data_list, label='微软天气风速', color='blue')
    # plt.bar(range(len(weather_humidity_data_list)), weather_humidity_data_list, label='微软天气湿度', color='blue', alpha=0.1)
    plt.grid(color='#95a5a6',linestyle='--',linewidth=1,axis='y',alpha=0.6)

    plt.xlabel('Days')
    plt.ylabel('风速')
    plt.title('风速数据对比')
 
    plt.tight_layout()
    plt.show()

    mse_max_temp = mean_squared_error(weather_wind_data_list, wind_mean_temps)
    mae_max_temp = mean_absolute_error(weather_wind_data_list, wind_mean_temps)

    print(f"均方误差: {mse_max_temp}")
    # print(f"当天最低温度--均方误差: {mse_min_temp}")
    print(f"平均绝对误差: {mae_max_temp}")
    # print(f"当天最低温度--平均绝对误差: {mae_min_temp}")


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

run_compare_wind()