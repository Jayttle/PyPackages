import os
import subprocess
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
from time import sleep
from JayttleProcess import CommonDecorator

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
        self.ftp_file = ftp_file  # 是否来自FTP
        self.input_path = input_path  # 输入路径
        self.file_info = Path(input_path)  # 文件信息对象
        self.file_name = self.file_info.name  # 文件名
        self.station_name = self.file_name[:3]  # 站点名称，前三个字符
        self.marker_name = self.file_name[:4]  # 标记名称，前四个字符
        self.station_id = int(self.station_name[2])  # 站点ID，从站点名称中第三个字符转换而来
        self.station_type = {  # 站点类型，根据站点名称的第一个字符确定，如果未知则为 Unknown
            'R': StationType.RoverStation,
            'B': StationType.BaseStation
        }.get(self.station_name[0], StationType.Unkonw)
        self.receiver_id = int(self.file_name[3])  # 接收器ID，从文件名第四个字符转换而来
        split = self.file_name.split('_')
        self.start_gps_time_str = split[2]  # 起始GPS时间字符串，从文件名中获取
        self.start_gps_time = self.get_time_from_string(self.start_gps_time_str)  # 起始GPS时间，使用自定义函数从字符串中获取
        self.duration_str = split[3]  # 持续时间字符串，从文件名中获取
        self.duration = self.get_duration_from_string(self.duration_str)  # 持续时间，使用自定义函数从字符串中获取
        self.time_str = f"{self.start_gps_time_str}_{self.duration_str}"  # 时间字符串，包含起始GPS时间和持续时间
        file_type = split[-1][1]
        self.file_type = {  # 文件类型，根据文件名中的第二个字符确定，如果未知则为 Unknown
            'O': RinexFileType.O,
            'N': RinexFileType.N
        }.get(file_type, RinexFileType.Unkonw)
        split1 = self.file_name.split('.')
        self.info_str = split1[0]  # 信息字符串，从文件名中获取
        compressed = split1[-1].lower()
        self.compressed = compressed == "zip" or compressed == "z"  # 是否为压缩文件
        self.format = {    # 文件格式，根据文件扩展名确定，如果未知则为 Unknown
            "crx": FileFormat.CRX,
            "rnx": FileFormat.RNX
        }.get(split1[1].lower(), FileFormat.Unkonw)

    @staticmethod
    def get_time_from_string(time_str: str) -> datetime:
        year = int(time_str[:4])  # 年份
        day_of_year = int(time_str[4:7])  # 年份中的第几天
        hour = int(time_str[7:9])  # 小时
        minute = int(time_str[9:11])  # 分钟
        return datetime(year, 1, 1) + timedelta(days=day_of_year - 1, hours=hour, minutes=minute)

    @staticmethod
    def get_duration_from_string(duration_str: str) -> timedelta:
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
        if duration < timedelta(minutes=1):
            return f"{duration.total_seconds():.0f}S"
        elif duration < timedelta(hours=1):
            return f"{duration.total_seconds() / 60:.0f}M"
        elif duration < timedelta(days=1):
            return f"{duration.total_seconds() / 3600:.0f}H"
        else:
            return f"{duration.days}D"
        
def get_rnx_files(directory_path: str) -> list[RinexFileInfo]:
    rnx_files = []
    for file_name in os.listdir(directory_path):
        if file_name.endswith(".rnx"):
            file_path = os.path.join(directory_path, file_name)
            rnx_file = RinexFileInfo(file_path, ftp_file=False)
            rnx_files.append(rnx_file)
    return rnx_files


class MergeFiles:
    @staticmethod
    @CommonDecorator.log_function_call
    def merge_files(files: list[RinexFileInfo], merge_path: str, rewrite: bool = False) -> bool:
        if files is None or len(files) <= 1:
            return False
        if not os.path.exists(merge_path):
            os.makedirs(merge_path)

        first = files[0]
        last = files[-1]

        split = first.file_name.split('_')
        start_time = first.start_gps_time
        start_time_str = first.start_gps_time_str
        duration = (last.start_gps_time - first.start_gps_time) + last.duration
        duration_str = RinexFileInfo.get_string_from_duration(duration)
        split[2] = start_time_str
        split[3] = duration_str
        merge_name = '_'.join(split)
        merge_full_name = os.path.join(merge_path, merge_name)
        if os.path.exists(merge_full_name):
            if not rewrite:
                return True
            else:
                os.remove(merge_full_name)

        file_names = ' '.join([str(f.file_info) for f in files])
        arguments = f"-finp {file_names} -kv -fout {merge_full_name} -vo 3.02"
        process = subprocess.Popen(
                    ["D:\Program Files (x86)\Software\OneDrive\C#\windows_C#\Cableway.Net7\Cableway.Download\Rinex\gfzrnx_2.1.0_win64.exe", arguments], 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE)
        process.communicate()
        sleep(0.1)
        print(merge_full_name)
        return os.path.exists(merge_full_name)


# 使用示例
directory_path = r"D:\Ropeway\GNSS\B011"
merge_path = r"D:\Ropeway\GNSS\Merge"
rnx_files = get_rnx_files(directory_path)
o_files = [rnx_file for rnx_file in rnx_files if rnx_file.file_type == RinexFileType.O]
n_files = [rnx_file for rnx_file in rnx_files if rnx_file.file_type == RinexFileType.N]

for o_file in o_files:
    print(o_file.__dict__)
    print()

for n_file in n_files:
    print(n_file.__dict__)
    print()
MergeFiles.merge_files(o_files, merge_path)