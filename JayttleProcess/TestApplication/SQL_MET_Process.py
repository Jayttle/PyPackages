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

    temperature_get_maxmin_time: Tuple[str, str] = []
    # Iterate over each group
    for date, group in grouped_data:
        # Extract temperatures from the group
        temperatures = [met.temperature for met in group]
        
        # Calculate maximum and minimum temperatures
        max_temp = max(temperatures)
        min_temp = min(temperatures)
        
        # Store the results in the dictionary
        temperature_data[str(date)] = (max_temp, min_temp)
    
    return temperature_data

#TODO:检查下tianmeng_met中每天取到最大值最小值的时候的 时间段

def run_compare():
    # File path
    file_path = r"D:\Program Files (x86)\Software\OneDrive\PyPackages_DataSave\weather_temperature.txt"   
    # Read weather data
    weather_data: Dict[str, Tuple[float, float]] = read_weather_data(file_path)
    temperature_data: Dict[str, Tuple[float, float]] = read_tianmeng_met()

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

run_compare()