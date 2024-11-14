from avr import Avr
from ggkx import Ggkx, GgkxExtended
import ggkx 
from tiltmeter import TiltmeterData
from datetime import datetime, timedelta
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import List, Optional, Tuple, Union, Set, Dict
from scipy.stats import pearsonr

# region 字体设置
plt.rcParams['font.sans-serif'] = ['SimSun']  # 指定宋体为默认字体
plt.rcParams['font.sans-serif'] = 'SimSun'  # 使用指定的中文字体
plt.rcParams['axes.unicode_minus']=False#用来正常显示负号

class Avr_process:
    def __init__(self):
        self._load_data()


    def _load_data(self):
        self.avr_file_path = r'C:\Users\Jayttle\Desktop\avr_20230704.txt'
        self.tiltmeter_file_path = r'D:\Program Files (x86)\Software\OneDrive\PyPackages_DataSave\tiltmeter_20230704.txt'

        self.ggkxR071_file_path = r"C:\Users\Jayttle\Desktop\ggkx_0704_R071.txt"
        self.ggkxR072_file_path = r"C:\Users\Jayttle\Desktop\ggkx_0704_R072.txt"

        self.ggkxR071 = Ggkx._from_file(self.ggkxR071_file_path)
        self.ggkxR072 = Ggkx._from_file(self.ggkxR072_file_path)
        # list_height_diff, list_angle = ggkx.calculate_height_differences(self.ggkxR071, self.ggkxR072, "height_diff.txt")


        # time_list = [data.time for data in self.ggkxR071]
        # plot_time_data(time_list, list_angle)

        # avg_east, avg_north = ggkx.get_average_coords(self.ggkxR071)
        # avg_east, avg_north = ggkx.get_average_coords(self.ggkxR072)

        self.avr_data = self._read_avr_data()
        self.avr_data_R07: list['Avr'] = []
        for avr in self.avr_data:
            if avr.station_id == 7 and avr.fix_mode == 3:
                self.avr_data_R07.append(avr)

        self.tiltmeter_data_list: List['TiltmeterData'] = TiltmeterData._from_file(self.tiltmeter_file_path)

        matched_pitch_roll_yaw_tilt = []
        avr_index = 0
        tilt_index = 0

        while avr_index < len(self.avr_data_R07) and tilt_index < len(self.tiltmeter_data_list):
            avr_time_exact_second = self.avr_data_R07[avr_index].time.replace(microsecond=0)
            tilt_time_exact_second = self.tiltmeter_data_list[tilt_index].time.replace(microsecond=0)
            
            if avr_time_exact_second < tilt_time_exact_second:
                avr_index += 1
            elif avr_time_exact_second > tilt_time_exact_second:
                tilt_index += 1
            else:  # avr_time_exact_second == tilt_time_exact_second
                matched_pitch_roll_yaw_tilt.append((
                    self.tiltmeter_data_list[tilt_index].pitch,
                    self.tiltmeter_data_list[tilt_index].roll,
                    self.avr_data_R07[avr_index].yaw,
                    self.avr_data_R07[avr_index].tilt
                ))
                tilt_index += 1
                avr_index += 1

        # Print or use matched_pitch_roll_yaw_tilt as needed
        print(f"Number of matches found: {len(matched_pitch_roll_yaw_tilt)}")
        # Convert to numpy array for easier computation
        data_array = np.array(matched_pitch_roll_yaw_tilt)

        # Calculate correlation coefficients
        correlation_matrix = np.corrcoef(data_array, rowvar=False)

        # Print the correlation matrix
        print("Correlation matrix:")
        print(correlation_matrix)

        # Alternatively, calculate pairwise correlations manually
        for i in range(data_array.shape[1]):
            for j in range(i + 1, data_array.shape[1]):
                corr, _ = pearsonr(data_array[:, i], data_array[:, j])
                print(f"Correlation between column {i} and column {j}: {corr:.4f}")
        
        # Initialize an empty list to store tuples of (roll, tilt)
        roll_tilt_pairs = []

        # Populate roll_tilt_pairs with (roll, tilt) from matched_pitch_roll_yaw_tilt
        for entry in matched_pitch_roll_yaw_tilt:
            roll_tilt_pairs.append((entry[1], entry[3]))  # entry[1] is roll, entry[3] is tilt

        # Convert roll_tilt_pairs to numpy array for correlation calculation
        data_array = np.array(roll_tilt_pairs)

        # Calculate Pearson correlation coefficient between roll and tilt
        corr, _ = pearsonr(data_array[:, 0], data_array[:, 1])

        # Print the correlation coefficient
        print(f"Pearson correlation coefficient between roll and tilt: {corr:.4f}")

        correlations = {}
        for i in range(len(matched_pitch_roll_yaw_tilt[0])):
            for j in range(i + 1, len(matched_pitch_roll_yaw_tilt[0])):
                var1_name = ['pitch', 'roll', 'yaw', 'tilt'][i]
                var2_name = ['pitch', 'roll', 'yaw', 'tilt'][j]
                var1 = [entry[i] for entry in matched_pitch_roll_yaw_tilt]
                var2 = [entry[j] for entry in matched_pitch_roll_yaw_tilt]
                corr, _ = pearsonr(var1, var2)
                correlations[f"{var1_name} vs {var2_name}"] = corr

        # Print correlations
        print("Pearson correlations:")
        for key, value in correlations.items():
            print(f"{key}: {value:.4f}")
        
    def _read_avr_data(self) -> list['Avr']:
        avr_data = []
        with open(self.avr_file_path, 'r') as file:
            next(file)  # Skip the header line
            for line in file:
                if line.strip():  # Ensure the line is not empty
                    avr_instance = Avr._from_string(line)
                    avr_data.append(avr_instance)
        return avr_data
    
    def _print_avr_data_R07_statistics(self):
        if not self.avr_data_R07:
            print("avr_data_R07 is empty.")
            return
        
        # 计算平均值
        num_records = len(self.avr_data_R07)
        yaw_total = sum(avr.yaw for avr in self.avr_data_R07)
        tilt_total = sum(avr.tilt for avr in self.avr_data_R07)
        range_val_total = sum(avr.range_val for avr in self.avr_data_R07)

        yaw_avg = yaw_total / num_records
        tilt_avg = tilt_total / num_records
        range_val_avg = range_val_total / num_records

        # 打印结果
        print(f"Average Yaw: {yaw_avg}")
        print(f"Average Tilt: {tilt_avg}")
        print(f"Average Range Value: {range_val_avg}")


def plot_tiltmeter_data(tiltmeter_data_list: List['TiltmeterData']):
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

def plot_avr_data(avr_data_list: List[Avr]):
    # 提取时间、yaw、tilt和range_val数据
    time_list = [data.time for data in avr_data_list]
    yaw_data = [data.yaw for data in avr_data_list]
    tilt_data = [data.tilt for data in avr_data_list]
    range_val_data = [data.range_val for data in avr_data_list]

    # 创建图像，并设置日期格式化器
    plt.figure(figsize=(14.4, 12))
    date_fmt = mdates.DateFormatter("%m-%d:%H")
    date_locator = mdates.AutoDateLocator()

    # 绘制yaw数据子图
    plt.subplot(3, 1, 1)
    plt.plot(time_list, yaw_data, color='blue')
    plt.title('Yaw角度随时间变化')
    plt.xlabel('日期')
    plt.ylabel('Yaw角度')
    plt.gca().xaxis.set_major_formatter(date_fmt)
    plt.gca().xaxis.set_major_locator(date_locator)
    plt.yticks(plt.yticks()[0][::len(plt.yticks()[0]) // 5])

    # 绘制tilt数据子图
    plt.subplot(3, 1, 2)
    plt.plot(time_list, tilt_data, color='green')
    plt.title('Tilt角度随时间变化')
    plt.xlabel('日期')
    plt.ylabel('Tilt角度')
    plt.gca().xaxis.set_major_formatter(date_fmt)
    plt.gca().xaxis.set_major_locator(date_locator)
    plt.yticks(plt.yticks()[0][::len(plt.yticks()[0]) // 5])

    # 绘制range_val数据子图
    plt.subplot(3, 1, 3)
    plt.plot(time_list, range_val_data, color='red')
    plt.title('Range数值随时间变化')
    plt.xlabel('日期')
    plt.ylabel('Range数值')
    plt.gca().xaxis.set_major_formatter(date_fmt)
    plt.gca().xaxis.set_major_locator(date_locator)
    plt.yticks(plt.yticks()[0][::len(plt.yticks()[0]) // 5])

    plt.tight_layout(pad=3.0)
    plt.show()


def plot_time_data(time_list, val_data):
    # 假设你有一个阈值，以秒为单位
    threshold_seconds = 60  # 例如阈值为60秒

    # 将阈值转换为datetime.timedelta对象
    threshold = dt.timedelta(seconds=threshold_seconds)

    # 创建图像，并设置日期格式化器
    plt.figure(figsize=(14.4, 12))
    date_fmt = mdates.DateFormatter("%m-%d:%H")
    date_locator = mdates.AutoDateLocator()
    
    prev_datetime = None
    prev_value = None
    for datetime, value in zip(time_list, val_data):
        if prev_datetime is not None:
            time_diff = datetime - prev_datetime
            if time_diff < threshold:  # 如果时间间隔小于阈值，则连接线段
                plt.plot([prev_datetime, datetime], [prev_value, value], linestyle='-', color='blue')
            else:  # 否则不连接线段
                plt.plot([prev_datetime, datetime], [prev_value, value], linestyle='', color='blue')
        prev_datetime = datetime
        prev_value = value
    
    plt.title('Yaw角度随时间变化')
    plt.xlabel('日期')
    plt.ylabel('Yaw角度')
    plt.gca().xaxis.set_major_formatter(date_fmt)
    plt.gca().xaxis.set_major_locator(date_locator)
    
    # 自定义坐标轴范围示例
    # plt.xlim(start_date, end_date)  # 设置x轴的日期范围
    plt.ylim(-1, 1)  # 设置y轴的数值范围
    
    plt.show()


def plot_avr_data(avr_data_list: List[Avr]):

    # 假设你有一个阈值，以秒为单位
    threshold_seconds = 60  # 例如阈值为60秒

    # 将阈值转换为datetime.timedelta对象
    threshold = dt.timedelta(seconds=threshold_seconds)
    # 提取时间、yaw、tilt和range_val数据
    time_list = [data.time for data in avr_data_list]
    yaw_data = [data.yaw for data in avr_data_list]
    tilt_data = [data.tilt for data in avr_data_list]
    range_val_data = [data.range_val for data in avr_data_list]

    # 创建图像，并设置日期格式化器
    plt.figure(figsize=(14.4, 12))
    date_fmt = mdates.DateFormatter("%m-%d:%H")
    date_locator = mdates.AutoDateLocator()

    # 绘制yaw数据子图
    plt.subplot(3, 1, 1)
    prev_datetime = None
    prev_value = None
    for datetime, value in zip(time_list, yaw_data):
        if prev_datetime is not None:
            time_diff = datetime - prev_datetime
            if time_diff < threshold:  # 如果时间间隔小于阈值，则连接线段
                plt.plot([prev_datetime, datetime], [prev_value, value], linestyle='-', color='blue')
            else:  # 否则不连接线段
                plt.plot([prev_datetime, datetime], [prev_value, value], linestyle='', color='blue')
        prev_datetime = datetime
        prev_value = value

    plt.title('Yaw角度随时间变化')
    plt.xlabel('日期')
    plt.ylabel('Yaw角度')
    plt.gca().xaxis.set_major_formatter(date_fmt)
    plt.gca().xaxis.set_major_locator(date_locator)
    plt.yticks(plt.yticks()[0][::len(plt.yticks()[0]) // 5])
    plt.legend()

    # 绘制tilt数据子图
    plt.subplot(3, 1, 2)
    prev_datetime = None
    prev_value = None
    for datetime, value in zip(time_list, tilt_data):
        if prev_datetime is not None:
            time_diff = datetime - prev_datetime
            if time_diff < threshold:  # 如果时间间隔小于阈值，则连接线段
                plt.plot([prev_datetime, datetime], [prev_value, value], linestyle='-', color='green')
            else:  # 否则不连接线段
                plt.plot([prev_datetime, datetime], [prev_value, value], linestyle='', color='green')
        prev_datetime = datetime
        prev_value = value

    plt.title('Tilt角度随时间变化')
    plt.xlabel('日期')
    plt.ylabel('Tilt角度')
    plt.gca().xaxis.set_major_formatter(date_fmt)
    plt.gca().xaxis.set_major_locator(date_locator)
    plt.yticks(plt.yticks()[0][::len(plt.yticks()[0]) // 5])
    plt.legend()

    # 绘制range_val数据子图
    plt.subplot(3, 1, 3)
    prev_datetime = None
    prev_value = None
    for datetime, value in zip(time_list, range_val_data):
        if prev_datetime is not None:
            time_diff = datetime - prev_datetime
            if time_diff < threshold:  # 如果时间间隔小于阈值，则连接线段
                plt.plot([prev_datetime, datetime], [prev_value, value], linestyle='-', color='red')
            else:  # 否则不连接线段
                plt.plot([prev_datetime, datetime], [prev_value, value], linestyle='', color='red')
        prev_datetime = datetime
        prev_value = value

    plt.title('Range数值随时间变化')
    plt.xlabel('日期')
    plt.ylabel('Range数值')
    plt.gca().xaxis.set_major_formatter(date_fmt)
    plt.gca().xaxis.set_major_locator(date_locator)
    plt.yticks(plt.yticks()[0][::len(plt.yticks()[0]) // 5])
    plt.legend()

    plt.tight_layout(pad=3.0)
    plt.show()



# Example usage:
if __name__ == '__main__':
    ap = Avr_process()
    # ap._print_avr_data_R07_statistics()
    # plot_avr_data(ap.avr_data_R07)
    
