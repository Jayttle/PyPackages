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

        # 构建正确的参数列表
        file_names = [f.file_info for f in files]
        command = [r"D:\Program Files (x86)\Software\OneDrive\C#\WPF\Cableway.Download\Rinex\gfzrnx_2.1.0_win64.exe"]
        options = ["-finp"] + file_names + ["-kv", "-fout", merge_full_name, "-vo", "3.02"]

        # 执行外部程序
        process = subprocess.Popen(
            command + options,
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate()

        # # 打印外部程序的输出
        # print(stdout.decode())
        # print(stderr.decode())

        # # 检查外部程序是否执行成功
        # if process.returncode == 0:
        #     print("External program executed successfully.")
        # else:
        #     print(f"External program encountered an error: {stderr.decode()}")

        sleep(0.1)
        return os.path.exists(merge_full_name)

def proj_merge_rnx():
    # 使用示例
    directory_path = r"D:\Ropeway\GNSS\B011"
    merge_path = r"D:\Ropeway\GNSS\Merge"
    rnx_files = get_rnx_files(directory_path)
    o_files = [rnx_file for rnx_file in rnx_files if rnx_file.file_type == RinexFileType.O]


    # 逐行打印 RinexFileInfo 实例的属性及其对应的值
    for file_info in o_files:
        for key, value in file_info.__dict__.items():
            print(f"{key}: {value}")
        print()

        
    n_files = [rnx_file for rnx_file in rnx_files if rnx_file.file_type == RinexFileType.N]
    MergeFiles.merge_files(n_files, merge_path)

def get_special_datetimes(directory: str) -> list:
    special_datetimes = []
    for folder_name in os.listdir(directory):
        try:
            dt = RinexFileInfo.get_time_from_string(folder_name)
            # 检查是否在指定时间范围内
            if dt.time() >= datetime.strptime("12:00:00", "%H:%M:%S").time() and dt.time() <= datetime.strptime("23:00:00", "%H:%M:%S").time():
                special_datetimes.append(dt)
            elif dt.time() == datetime.strptime("00:00:00", "%H:%M:%S").time():
                special_datetimes.append(dt)
        except ValueError:
            pass  # 文件夹名称格式不正确，跳过
    return special_datetimes


def filter_folders(directory: str) -> list:
    valid_folders = []
    for folder_name in os.listdir(directory):
        if folder_name[-2:].isdigit():
            last_two_digits = int(folder_name[-2:])
            if (last_two_digits >= 12 and last_two_digits <= 23) or last_two_digits == 0:
                valid_folders.append(int(folder_name))  # 将文件夹名称转换为数字
    valid_folders.sort()  # 按数字大小进行排序
    return valid_folders

# 测试
directory = r"D:\Ropeway\MyFtpSavePath"
valid_folders = filter_folders(directory)
print(valid_folders)




# # # 测试
# directory = r"D:\Ropeway\AllRFtpSave"
# special_datetimes = get_special_datetimes(directory)
# print(special_datetimes)

