from typing import List, Optional, Tuple, Union, Set, Dict
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import io

# region 字体设置
plt.rcParams['font.sans-serif'] = ['SimSun']  # 指定宋体为默认字体
plt.rcParams['font.sans-serif'] = 'SimSun'  # 使用指定的中文字体
plt.rcParams['axes.unicode_minus']=False#用来正常显示负
class Avr:
    def __init__(self, time: str, station_id: int, fix_mode: int, yaw: float, tilt: float, range_val: float, pdop: float, sat_num: int):
        try:
            self.time = datetime.strptime(time, '%Y-%m-%d %H:%M:%S.%f')  # 尝试包含毫秒部分的格式
        except ValueError:
            self.time = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')  # 失败时尝试不包含毫秒部分的格式
        self.station_id = station_id
        self.fix_mode = fix_mode
        self.yaw = yaw
        self.tilt = tilt
        self.range_val = range_val
        self.pdop = pdop
        self.sat_num = sat_num
    
    @classmethod
    def _from_string(cls, line: str) -> 'Avr':
        parts = line.strip().split(',')
        time = parts[0]
        station_id = int(parts[1])
        fix_mode = int(parts[2])
        yaw = float(parts[3])
        tilt = float(parts[4])
        range_val = float(parts[5])
        pdop = float(parts[6])
        sat_num = int(parts[7])
        return cls(time, station_id, fix_mode, yaw, tilt, range_val, pdop, sat_num)

    @classmethod
    def _from_file(cls, file_path: str) -> pd.DataFrame:
        avr_data: list[Avr] = []
        with open(file_path, 'r') as file:
            next(file)  # Skip the header line
            for line in file:
                if line.strip():  # Ensure the line is not empty
                    avr_instance = cls._from_string(line)
                    avr_data.append(avr_instance)
        # Convert to DataFrame
        data = [{
            'Time': a.time,
            'StationID': a.station_id,
            'Fix Mode': a.fix_mode,
            'Yaw': a.yaw,
            'Tilt': a.tilt,
            'Range': a.range_val,
            'PDOP': a.pdop,
            'Sat Num': a.sat_num
        } for a in avr_data]
        return pd.DataFrame(data) if avr_data else pd.DataFrame()

    @classmethod
    def _split_avr_columns(cls, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        # 选择需要的列并创建新的 DataFrame
        df_yaw = df[['Time', 'Yaw']].copy()
        df_tilt = df[['Time', 'Tilt']].copy()
        df_range = df[['Time', 'Range']].copy()
        
        # 返回字典，其中包含三个 DataFrame
        return {
            'Yaw': df_yaw,
            'Tilt': df_tilt,
            'Range': df_range
        }
    
    
    @classmethod
    def _split_station_data(cls, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        # Split the DataFrame based on StationID values 3 and 8
        df_station_3 = df[df['StationID'] == 3]
        df_station_8 = df[df['StationID'] == 8]
        return df_station_3, df_station_8

    
class Ggkx:
    def __init__(self, time: str, station_id: int, receiver_id: int, lat: float, lon: float, geo_height: float,
                 fix_mode: int, sate_num: int, pdop: float, sigma_e: float, sigma_n: float, sigma_u: float,
                 prop_age: int):
        try:
            self.time = datetime.strptime(time, '%Y-%m-%d %H:%M:%S.%f')  # 尝试包含毫秒部分的格式
        except ValueError:
            self.time = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')  # 失败时尝试不包含毫秒部分的格式
        self.station_id = station_id
        self.receiver_id = receiver_id
        self.lat = lat
        self.lon = lon
        self.geo_height = geo_height
        self.fix_mode = fix_mode
        self.sate_num = sate_num
        self.pdop = pdop
        self.sigma_e = sigma_e
        self.sigma_n = sigma_n
        self.sigma_u = sigma_u
        self.prop_age = prop_age

    @classmethod
    def _from_file(cls, file_path: str) -> pd.DataFrame:
        ggkx_data_list: List['Ggkx'] = []
        with open(file_path, 'r') as file:
            for line in file:
                data = line.strip().split(',')
                if len(data) == 13:
                    # Convert appropriate data types
                    time = data[0]
                    station_id = int(data[1])
                    receiver_id = int(data[2])
                    lat = float(data[3])
                    lon = float(data[4])
                    geo_height = float(data[5])
                    fix_mode = int(data[6])
                    sate_num = int(data[7])
                    pdop = float(data[8])
                    sigma_e = float(data[9])
                    sigma_n = float(data[10])
                    sigma_u = float(data[11])
                    prop_age = int(data[12])
                    
                    if fix_mode == 3:
                        ggkx_data_list.append(cls(time, station_id, receiver_id, lat, lon, geo_height,
                                                  fix_mode, sate_num, pdop, sigma_e, sigma_n, sigma_u, prop_age))
        # Convert to DataFrame
        data = [{
            'Time': g.time,
            'StationID': g.station_id,
            'Receiver ID': g.receiver_id,
            'Lat': g.lat,
            'Lon': g.lon,
            'Geo Height': g.geo_height,
            'Fix Mode': g.fix_mode,
            'Sate Num': g.sate_num,
            'PDOP': g.pdop,
            'Sigma E': g.sigma_e,
            'Sigma N': g.sigma_n,
            'Sigma U': g.sigma_u,
            'Prop Age': g.prop_age
        } for g in ggkx_data_list]
        return pd.DataFrame(data) if ggkx_data_list else pd.DataFrame()

    @classmethod
    def _split_ggkx_columns(cls, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        # 选择需要的列并创建新的 DataFrame
        df_lat = df[['Time', 'Lat']].copy()
        df_lon = df[['Time', 'Lon']].copy()
        df_geoheight = df[['Time', 'GeoHeight']].copy()
        
        # 返回字典，其中包含三个 DataFrame
        return {
            'Lat': df_lat,
            'Lon': df_lon,
            'GeoHeight': df_geoheight
        }
    
class Met:
    def __init__(self, time: str, station_id: int, temperature: float, humidness: float, pressure: float, windSpeed: float, windDirection: float) -> None:
        try:
            self.time = datetime.strptime(time, '%Y-%m-%d %H:%M:%S.%f')  # 尝试包含毫秒部分的格式
        except ValueError:
            self.time = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')  # 失败时尝试不包含毫秒部分的格式
        self.station_id = station_id
        self.temperature = temperature
        self.humidness = humidness
        self.pressure = pressure
        self.windSpeed = windSpeed
        self.windDirection = windDirection

    @classmethod
    def _from_file(cls, file_path: str) -> pd.DataFrame:
        met_data_list: List['Met'] = []
        with open(file_path, 'r') as file:
            for line in file:
                data = line.strip().split(',')
                if len(data) == 7:
                    # Convert appropriate data types
                    time = data[0]
                    station_id = int(data[1])
                    temperature = float(data[2])
                    humidness = float(data[3])
                    pressure = float(data[4])
                    windSpeed = float(data[5])
                    windDirection = float(data[6])
                    
                    met_data_list.append(cls(time, station_id, temperature, humidness, pressure, windSpeed, windDirection))
        # Convert to DataFrame
        data = [{
            'Time': t.time,
            'StationID': t.station_id,
            'Temperature': t.temperature,
            'Humidness': t.humidness,
            'Pressure': t.pressure,
            'WindSpeed': t.windSpeed,
            'WindDirection': t.windDirection
        } for t in met_data_list]
        return pd.DataFrame(data) if met_data_list else pd.DataFrame()
    
    @classmethod
    def _split_met_columns(cls, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        # 选择需要的列并创建新的 DataFrame
        df_Temperature = df[['Time', 'Temperature']].copy()
        df_Humidness = df[['Time', 'Humidness']].copy()
        df_Pressure = df[['Time', 'Pressure']].copy()
        df_WindSpeed = df[['Time', 'WindSpeed']].copy()
        df_WindDirection = df[['Time', 'WindDirection']].copy()
        
        # 返回字典，其中包含五个 DataFrame
        return {
            'Temperature': df_Temperature,
            'Humidness': df_Humidness,
            'Pressure': df_Pressure,
            'WindSpeed': df_WindSpeed,
            'WindDirection': df_WindDirection
        }
    

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
    def _from_file(cls, file_path: str) -> pd.DataFrame:
        tiltmeter_data_list: List['TiltmeterData'] = []
        with open(file_path, 'r') as file:
            for line in file:
                data = line.strip().split(',')
                if len(data) == 4:
                    # Convert appropriate data types
                    time = data[0]
                    station_id = int(data[1])
                    pitch = float(data[2])
                    roll = float(data[3])
                    
                    tiltmeter_data_list.append(cls(time, station_id, pitch, roll))
        # Convert to DataFrame
        data = [{
            'Time': t.time,
            'StationID': t.station_id,
            'Pitch': t.pitch,
            'Roll': t.roll
        } for t in tiltmeter_data_list]
        return pd.DataFrame(data) if tiltmeter_data_list else pd.DataFrame()

    @classmethod
    def _split_tiltmeter_columns(cls, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        # 选择需要的列并创建新的 DataFrame
        df_pitch = df[['Time', 'Pitch']].copy()
        df_roll = df[['Time', 'Roll']].copy()
        
        # 返回字典，其中包含三个 DataFrame
        return {
            'Pitch': df_pitch,
            'Roll': df_roll,
        }


class FileConfig:
    def __init__(self):
        self.avr_yesterday_file_path = r"D:\Program Files (x86)\Software\OneDrive\PyPackages\avr_yesterday_data.txt"                     
        self.ggkx_yesterday_file_path = r"D:\Program Files (x86)\Software\OneDrive\PyPackages\ggkx_yesterday_data.txt"
        self.met_yesterday_file_path = r"D:\Program Files (x86)\Software\OneDrive\PyPackages\met_yesterday_data.txt"
        self.tiltmeter_yesterday_file_path = r"D:\Program Files (x86)\Software\OneDrive\PyPackages\tiltmeter_yesterday_data.txt"        

class pdDataFrameProcess:
    @classmethod
    def _pdData_clean(cls, data: pd.DataFrame) -> pd.DataFrame:
        # 计算每列的缺失值数量
        missing_values = data.isnull().sum()
        print("Missing values per column:\n", missing_values)

        # 计算并删除重复行
        duplicate_count = data.duplicated().sum()
        print(f"Duplicate rows count: {duplicate_count}")
        data = data.drop_duplicates()
        
        return data

    @classmethod
    def _clean_time_data(cls, data: pd.DataFrame, time_column: str) -> pd.DataFrame:
        # 处理时间数据
        if time_column not in data.columns:
            raise ValueError(f"Column '{time_column}' not found in DataFrame.")

        # 将时间列转换为 datetime 类型，处理格式错误
        data[time_column] = pd.to_datetime(data[time_column], errors='coerce')

        # 打印转换后的时间列
        print(f"After conversion, unique time values:\n{data[time_column].unique()}")

        # 检查转换后的缺失值
        missing_time_values = data[time_column].isnull().sum()
        print(f"Missing time values count: {missing_time_values}")

        # 删除时间列中无效的时间数据
        data = data.dropna(subset=[time_column])
        
        return data


    @classmethod
    def _save_pdDataFrame(self, df: pd.DataFrame, save_file_path: str) -> None:
        df.to_csv(save_file_path, sep='\t', index=False)

        
    @classmethod
    def _create_summary_dataframe(self, datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
        summary = []
        for name, df in datasets.items():
            if df.empty:
                # 处理空的数据框
                row_count = 0
                col_count = 0
                missing_values = 0
                duplicate_count = 0
            else:
                # 计算缺失值总数
                missing_values = df.isnull().sum().sum()
                # 计算重复行数
                duplicate_count = df.duplicated().sum()
                # 统计行数和列数
                row_count = df.shape[0]
                col_count = df.shape[1]

            summary_entry = {
                'Dataset': name,
                'Rows': row_count,
                'Columns': col_count,
                'Missing Values': missing_values,
                'Duplicate Rows': duplicate_count,
            }
            summary.append(summary_entry)
        
        return pd.DataFrame(summary)

    # 将 DataFrame 的打印输出捕获到一个字符串中
    @classmethod
    def _capture_dataframe_output(cls, df: pd.DataFrame) -> str:
        output = io.StringIO()
        df.to_string(buf=output)
        return output.getvalue()
    
class TimeDataFrameMethod:
    @classmethod
    def _save_pdDataFrame(self, df: pd.DataFrame, save_file_path: str) -> None:
        df.to_csv(save_file_path, sep='\t', index=False)

    @classmethod
    def _plot_timeDF(cls, df: pd.DataFrame, colum_key: str, title: str, xlabel: str, ylabel: str, save_path: str) -> None:
        # 时间序列分析
        df['Time'] = pd.to_datetime(df['Time'])
        df = df.sort_values('Time')

        plt.figure(figsize=(9, 6))
        ax = plt.gca()

        # 设置日期格式化器和日期刻度定位器
        date_fmt = mdates.DateFormatter("%H:%M")  # 显示小时:分钟
        date_locator = mdates.AutoDateLocator()  # 自动选择刻度间隔

        # 隐藏右边框和上边框
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # 绘制折线图
        plt.plot(df['Time'], df[colum_key], marker='', linestyle='-', color='b')

        # 设置x轴日期格式和刻度定位
        ax.xaxis.set_major_formatter(date_fmt)
        ax.xaxis.set_major_locator(date_locator)

        # 设置刻度朝向内部，并调整刻度与坐标轴的距离
        ax.tick_params(axis='x', direction='in', pad=10)
        ax.tick_params(axis='y', direction='in', pad=10)

        plt.xlabel(xlabel, fontproperties='SimSun', fontsize=10)
        plt.ylabel(ylabel, fontproperties='SimSun', fontsize=10)
        plt.title(title, fontproperties='SimSun', fontsize=12)
        plt.grid(True)

        # 调整底部边界向上移动一点
        plt.subplots_adjust(bottom=0.15)
        
        # 保存图表到本地文件
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


    @staticmethod
    def _analyze_yaw(df: pd.DataFrame) -> None:
        # 1. 数据检查
        print("数据检查:")
        print(df.info())
        # 2. 基本统计
        print("\n基本统计:")
        mean_yaw = df['Yaw'].mean()
        std_yaw = df['Yaw'].std()
        min_yaw = df['Yaw'].min()
        max_yaw = df['Yaw'].max()
        print(f"均值: {mean_yaw}")
        print(f"标准差: {std_yaw}")
        print(f"最小值: {min_yaw}")
        print(f"最大值: {max_yaw}")
        
        # 3. 时间序列分析
        df['Time'] = pd.to_datetime(df['Time'])
        df = df.sort_values('Time')
        plt.figure(figsize=(10, 5))
        plt.plot(df['Time'], df['Yaw'], marker='o', linestyle='-', color='b')
        plt.xlabel('Time')
        plt.ylabel('Yaw')
        plt.title('Yaw Over Time')
        plt.grid(True)
        plt.show()
        
        # 4. 趋势分析
        df['Yaw_MA'] = df['Yaw'].rolling(window=3).mean()  # 计算3点移动平均
        plt.figure(figsize=(10, 5))
        plt.plot(df['Time'], df['Yaw'], marker='o', linestyle='-', color='b', label='Yaw')
        plt.plot(df['Time'], df['Yaw_MA'], color='r', label='Moving Average (3)')
        plt.xlabel('Time')
        plt.ylabel('Yaw')
        plt.title('Yaw with Moving Average')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # 5. 异常检测
        df['Yaw_Deviation'] = df['Yaw'] - df['Yaw'].mean()
        plt.figure(figsize=(10, 5))
        plt.plot(df['Time'], df['Yaw_Deviation'], marker='o', linestyle='-', color='g')
        plt.xlabel('Time')
        plt.ylabel('Deviation from Mean')
        plt.title('Yaw Deviation from Mean')
        plt.grid(True)
        plt.show()

if __name__ == '__main__':
    dataprocess_exe = FileConfig()
    avr_df: pd.DataFrame = Avr._from_file(dataprocess_exe.avr_yesterday_file_path)
    ggkx_df: pd.DataFrame = Ggkx._from_file(dataprocess_exe.ggkx_yesterday_file_path)
    met_df: pd.DataFrame = Met._from_file(dataprocess_exe.met_yesterday_file_path)
    tiltmeter_df: pd.DataFrame = TiltmeterData._from_file(dataprocess_exe.tiltmeter_yesterday_file_path)

    # 创建汇总数据框
    datasets: dict[str, pd.DataFrame] = {
        'AVR': avr_df,
        'GGKX': ggkx_df,
        'MET': met_df,
        'Tiltmeter': tiltmeter_df
    }

    summary_df = pdDataFrameProcess._create_summary_dataframe(datasets)
    summary_str = pdDataFrameProcess._capture_dataframe_output(summary_df)
    print(summary_str)
    avr_df_3, avr_df_8 = Avr._split_station_data(avr_df)

    # 使用 pdDataFrameProcess 类中的方法
    avr3_split_dict: dict[str, pd.DataFrame] = Avr._split_avr_columns(avr_df_3)
    avr8_split_dict: dict[str, pd.DataFrame] = Avr._split_avr_columns(avr_df_8)
    met_split_dict: dict[str, pd.DataFrame] = Met._split_met_columns(met_df)
    tiltmeter_split_dict: dict[str, pd.DataFrame] = TiltmeterData._split_tiltmeter_columns(tiltmeter_df)

    # 确保目标文件夹存在
    output_folder = 'output_images'
    os.makedirs(output_folder, exist_ok=True)   
    # 使用文件夹路径来保存图片
    for key, df in avr3_split_dict.items():
        save_path = os.path.join(output_folder, f'avr_03_{key}.png')
        TimeDataFrameMethod._plot_timeDF(df, colum_key=key, title=f'{key}时序图', xlabel='时间', ylabel=key, save_path=save_path)
        
    for key, df in avr8_split_dict.items():
        save_path = os.path.join(output_folder, f'avr_08_{key}.png')
        TimeDataFrameMethod._plot_timeDF(df, colum_key=key, title=f'{key}时序图', xlabel='时间', ylabel=key, save_path=save_path)
        
    for key, df in met_split_dict.items():
        save_path = os.path.join(output_folder, f'met_{key}.png')
        TimeDataFrameMethod._plot_timeDF(df, colum_key=key, title=f'{key}时序图', xlabel='时间', ylabel=key, save_path=save_path)
        
    for key, df in tiltmeter_split_dict.items():
        save_path = os.path.join(output_folder, f'tiltmeter_{key}.png')
        TimeDataFrameMethod._plot_timeDF(df, colum_key=key, title=f'{key}时序图', xlabel='时间', ylabel=key, save_path=save_path)

