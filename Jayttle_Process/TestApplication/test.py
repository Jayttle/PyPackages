import os
import shutil
import threading
import concurrent.futures
import subprocess
from enum import Enum
from collections import defaultdict, OrderedDict
from openpyxl import Workbook
from openpyxl.styles import Alignment
from datetime import datetime, timedelta
from pathlib import Path
from time import sleep
from typing import Union, Optional
from JayttleProcess import CommonDecorator
from JayttleProcess import RinexCommonManage

class RinexFileType(Enum):
    O = 1
    N = 2
    Unkonw  = 3


class FileFormat(Enum):
    CRX = 1
    RNX = 2
    Unkonw  = 3


class StationType(Enum):
    RoverStation = 1
    BaseStation = 2
    Unkonw = 3


class RinexFileInfo:
    def __init__(self, input_path: str, ftp_file: bool):
        self.ftp_file: bool = ftp_file  # 是否来自FTP
        self.input_path: str = input_path  # 输入路径
        self.file_info: Path = Path(input_path)  # 文件信息对象
        self.file_name: str = self.file_info.name  # 文件名
        self.station_name: str = self.file_name[:3]  # 站点名称，前三个字符
        self.marker_name: str = self.file_name[:4]  # 标记名称，前四个字符
        self.station_id: int = int(self.station_name[2])  # 站点ID，从站点名称中第三个字符转换而来
        self.station_type: Union[StationType, str] = {  # 站点类型，根据站点名称的第一个字符确定，如果未知则为 Unknown
            'R': StationType.RoverStation,
            'B': StationType.BaseStation
        }.get(self.station_name[0], StationType.Unkonw)
        self.receiver_id: int = int(self.file_name[3])  # 接收器ID，从文件名第四个字符转换而来
        split = self.file_name.split('_')
        self.start_gps_time_str: str = split[2]  # 起始GPS时间字符串，从文件名中获取
        self.start_gps_time: datetime = self.get_time_from_string(self.start_gps_time_str)  # 起始GPS时间，使用自定义函数从字符串中获取
        self.duration_str: str = split[3]  # 持续时间字符串，从文件名中获取
        self.duration: timedelta = self.get_duration_from_string(self.duration_str)  # 持续时间，使用自定义函数从字符串中获取
        self.time_str: str = f"{self.start_gps_time_str}_{self.duration_str}"  # 时间字符串，包含起始GPS时间和持续时间
        file_type = split[-1][1]
        self.file_type: Union[RinexFileType, str] = {  # 文件类型，根据文件名中的第二个字符确定，如果未知则为 Unknown
            'O': RinexFileType.O,
            'N': RinexFileType.N
        }.get(file_type, RinexFileType.Unkonw)
        split1 = self.file_name.split('.')
        self.info_str: str = split1[0]  # 信息字符串，从文件名中获取
        compressed = split1[-1].lower()
        self.compressed: bool = compressed == "zip" or compressed == "z"  # 是否为压缩文件
        self.format: Union[FileFormat, str] = {    # 文件格式，根据文件扩展名确定，如果未知则为 Unknown
            "crx": FileFormat.CRX,
            "rnx": FileFormat.RNX
        }.get(split1[1].lower(), FileFormat.Unkonw)


    @staticmethod
    def get_time_from_string(time_str: str) -> datetime:
        """
        将字符串形式的时间转换为 datetime 对象。

        参数：
            time_str (str): 表示时间的字符串，格式为 'YYYYDDDHHMM'，其中：
                YYYY: 年份
                DDD: 年份中的第几天
                HH: 小时
                MM: 分钟

        返回：
            datetime: 表示转换后的时间的 datetime 对象。
        """
        year = int(time_str[:4])  # 年份
        day_of_year = int(time_str[4:7])  # 年份中的第几天
        hour = int(time_str[7:9])  # 小时
        minute = int(time_str[9:11])  # 分钟
        return datetime(year, 1, 1) + timedelta(days=day_of_year - 1, hours=hour, minutes=minute)


    @staticmethod
    def get_duration_from_string(duration_str: str) -> timedelta:
        """
        将字符串形式的时长转换为 timedelta 对象。

        参数：
            duration_str (str): 表示时长的字符串，格式为 'X[D/H/M/S]'，其中：
                X: 数字，表示时长的数量
                D: 天
                H: 小时
                M: 分钟
                S: 秒

        返回：
            timedelta: 表示转换后的时长的 timedelta 对象。
        """
        days = int(duration_str[:-1])
        unit = duration_str[-1]
        if unit == 'D':
            return timedelta(days=days)
        elif unit == 'H':
            return timedelta(hours=days)
        elif unit == 'M':
            return timedelta(minutes=days)
        elif unit == 'S':
            return timedelta(seconds=days)

    @staticmethod
    def get_string_from_duration(duration: timedelta) -> str:
        """
        将 timedelta 对象转换为字符串形式的时长。

        参数：
            duration (timedelta): 表示时长的 timedelta 对象。

        返回：
            str: 表示转换后的字符串形式的时长，格式为 '[X]D/H/M/S'，其中：
                X: 数字，表示时长的数量
                D: 天
                H: 小时
                M: 分钟
                S: 秒
        """
        if duration < timedelta(minutes=1):
            return f"{duration.total_seconds():.0f}S"
        elif duration < timedelta(hours=1):
            return f"{duration.total_seconds() / 60:.0f}M"
        elif duration < timedelta(days=1):
            return f"{duration.total_seconds() / 3600:.0f}H"
        else:
            return f"{duration.days}D"

def Process_Part1():
    """
    1.完成  _toDownload.txt中存放需下载文件
    """
    # 指定文件夹
    directory_path = r"D:\Ropeway\GNSS\FTP_File_Situation"
    # 要读取得所有FTP文件的情况
    file_list_path = os.path.join(directory_path, "all_files.txt")
    # 读取rnx文件信息 list[RinexFileInfo]
    rinex_files_info: list[RinexFileInfo] = RinexCommonManage.read_rinex_files_info(file_list_path)
    R031_dict = defaultdict(int)
    R032_dict = defaultdict(int)
    R051_dict = defaultdict(int)
    R052_dict = defaultdict(int)
    R071_dict = defaultdict(int)
    R072_dict = defaultdict(int)
    R081_dict = defaultdict(int)
    R082_dict = defaultdict(int)
    B011_dict = defaultdict(int)
    B021_dict = defaultdict(int)

    # 确定日期范围：2023年2月28日到2024年7月1日
    start_date = datetime(2023, 2, 28).date()
    end_date = datetime(2024, 7, 1).date()
    
    # 生成日期范围内的所有日期
    date_range = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]
    
    count = 0
    # 遍历日期范围，确保每个日期在对应的 defaultdict 中有条目，如果没有则设为0
    for date in date_range:
        date_str = date.strftime('%Y-%m-%d')
        for count_dict in [R031_dict, R032_dict, R051_dict, R052_dict, R071_dict, R072_dict, R081_dict, R082_dict, B011_dict, B021_dict]:
            if date_str not in count_dict:
                count_dict[date_str] = 0
        count += 48       
    print(f"总应有文件数量：{count}")
    # 遍历rinex_files_info，对每个marker进行计数
    for rnx in rinex_files_info:
        if rnx.marker_name == 'R031' and rnx.duration_str == '01H':
            R031_dict[rnx.start_gps_time.date().strftime('%Y-%m-%d')] += 1
        elif rnx.marker_name == 'R032' and rnx.duration_str == '01H':
            R032_dict[rnx.start_gps_time.date().strftime('%Y-%m-%d')] += 1
        elif rnx.marker_name == 'R051' and rnx.duration_str == '01H':
            R051_dict[rnx.start_gps_time.date().strftime('%Y-%m-%d')] += 1
        elif rnx.marker_name == 'R052' and rnx.duration_str == '01H':
            R052_dict[rnx.start_gps_time.date().strftime('%Y-%m-%d')] += 1
        elif rnx.marker_name == 'R071' and rnx.duration_str == '01H':
            R071_dict[rnx.start_gps_time.date().strftime('%Y-%m-%d')] += 1
        elif rnx.marker_name == 'R072' and rnx.duration_str == '01H':
            R072_dict[rnx.start_gps_time.date().strftime('%Y-%m-%d')] += 1
        elif rnx.marker_name == 'R081' and rnx.duration_str == '01H':
            R081_dict[rnx.start_gps_time.date().strftime('%Y-%m-%d')] += 1
        elif rnx.marker_name == 'R082' and rnx.duration_str == '01H':
            R082_dict[rnx.start_gps_time.date().strftime('%Y-%m-%d')] += 1
        elif rnx.marker_name == 'B011' and rnx.duration_str == '01H':
            B011_dict[rnx.start_gps_time.date().strftime('%Y-%m-%d')] += 1
        elif rnx.marker_name == 'B021' and rnx.duration_str == '01H':
            B021_dict[rnx.start_gps_time.date().strftime('%Y-%m-%d')] += 1
    # 输出每天对应的 marker_name 的个数到各自的txt文件
    output_dicts = {
        'R031': R031_dict,
        'R032': R032_dict,
        'R051': R051_dict,
        'R052': R052_dict,
        'R071': R071_dict,
        'R072': R072_dict,
        'R081': R081_dict,
        'R082': R082_dict,
        'B011': B011_dict,
        'B021': B021_dict,
    }
    
    for marker_name, count_dict in output_dicts.items():
        output_file = os.path.join(directory_path, f"{marker_name}_output.txt")
        with open(output_file, 'w') as f:
            for date_key, count in sorted(count_dict.items()):
                f.write(f"{date_key}\t{count}\n")
        print(f"Output for {marker_name} written to {marker_name}_output.txt")
    print(len(rinex_files_info))
if __name__ == '__main__':
    print("----------程序运行----------")
    Process_Part1()