import os
import subprocess
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
from time import sleep

class RinexFileType(Enum):
    O = 1
    N = 2
    Unknown  = 3

class FileFormat(Enum):
    CRX = 1
    RNX = 2
    Unknown  = 3

class StationType(Enum):
    测站 = 1
    基站 = 2
    Unknown  = 3

class RinexFileInfo:
    def __init__(self, input_path: str, ftp_file: bool):
        self.ftp_file = ftp_file
        self.input_path = input_path
        self.file_info = Path(input_path)
        self.file_name = self.file_info.name
        self.station_name = self.file_name[:3]
        self.marker_name = self.file_name[:4]
        self.station_id = int(self.station_name[2])
        self.station_type = {
            'R': StationType.RoverStation,
            'B': StationType.BaseStation
        }.get(self.station_name[0], StationType.Unkonw)
        self.receiver_id = int(self.file_name[3])
        split = self.file_name.split('_')
        self.start_gps_time_str = split[2]
        self.start_gps_time = self.get_time_from_string(self.start_gps_time_str)
        self.duration_str = split[3]
        self.duration = self.get_duration_from_string(self.duration_str)
        self.time_str = f"{self.start_gps_time_str}_{self.duration_str}"
        file_type = split[-1][1]
        self.file_type = {
            'O': RinexFileType.O,
            'N': RinexFileType.N
        }.get(file_type, RinexFileType.Unkonw)
        split1 = self.file_name.split('.')
        self.info_str = split1[0]
        compressed = split1[-1].lower()
        self.compressed = compressed == "zip" or compressed == "z"
        self.format = {
            "crx": FileFormat.CRX,
            "rnx": FileFormat.RNX
        }.get(split1[1].lower(), FileFormat.Unkonw)

    @staticmethod
    def get_time_from_string(time_str: str) -> datetime:
        return datetime.strptime(time_str, "%Y%m%d%H%M%S")

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
    def merge_files(files: list[RinexFileInfo], merge_path: str, rewrite: bool = False) -> bool:
        if files is None or len(files) <= 1:
            return False
        if not os.path.exists(merge_path):
            os.makedirs(merge_path)

        first = files[0]
        last = files[-1]

        split = first.filename.split('_')
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

        file_names = ' '.join([f.file_info.absolute() for f in files])
        arguments = f"-finp {file_names} -kv -fout {merge_full_name} -vo 3.02"
        process = subprocess.Popen(
                    ["D:\Program Files (x86)\Software\OneDrive\C#\windows_C#\Cableway.Net7\Cableway.Download\Rinex\gfzrnx_2.1.0_win64.exe", arguments], 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE)
        process.communicate()
        sleep(0.1)
        return os.path.exists(merge_full_name)


# 使用示例
directory_path = r"D:\Ropeway\GNSS\B011"
rnx_files = get_rnx_files(directory_path)
for rnx_file in rnx_files:
    print(rnx_file.file_name)
    