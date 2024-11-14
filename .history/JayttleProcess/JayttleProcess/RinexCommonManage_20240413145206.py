import os
from enum import Enum
from datetime import datetime, timedelta

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
    def __init__(self, input_path, ftp_file):
        self.FtpFile = ftp_file  # 是否来自FTP
        self.InputPath = input_path  # 文件路径
        self.FileInfo = None  # 文件信息对象
        self.FileName = None  # 文件名
        self.StationName = None  # 测站名
        self.MarkerName = None  # 测站标记名
        self.StationId = None  # 测站ID
        self.StationType = None  # 测站类型
        self.ReceiverId = None  # 接收器ID
        self.StartGpsTimeStr = None  # 起始GPS时间字符串
        self.StartGpsTime = None  # 起始GPS时间
        self.DurationStr = None  # 持续时间字符串
        self.Duration = None  # 持续时间
        self.TimeStr = None  # 时间字符串
        self.FileType = None  # 文件类型
        self.InfoStr = None  # 信息字符串
        self.Compressed = None  # 是否压缩
        self.Format = None  # 文件格式

        self.parse_file_name()

    def parse_file_name(self):
        # 解析文件名信息
        self.FileInfo = os.path.abspath(self.InputPath)
        self.FileName = os.path.basename(self.InputPath)
        self.StationName = self.FileName[:3]
        self.MarkerName = self.FileName[:4]
        self.StationId = int(self.StationName[2])
        self.StationType = self.get_station_type(self.StationName[0])
        self.ReceiverId = int(self.FileName[3])
        split = self.FileName.split('_')
        self.StartGpsTimeStr = split[2]
        self.StartGpsTime = self.get_time_from_string(self.StartGpsTimeStr)
        self.DurationStr = split[3]
        self.Duration = self.get_duration_from_string(self.DurationStr)
        self.TimeStr = f"{self.StartGpsTimeStr}_{self.DurationStr}"
        type_char = split[-1][1]
        self.FileType = self.get_rinex_file_type(type_char)
        split1 = self.FileName.split('.')
        self.InfoStr = split1[0]
        compressed = split1[-1].lower()
        self.Compressed = compressed == "zip" or compressed == "z"
        self.Format = self.get_file_format(split1[1].lower())

    @staticmethod
    def get_station_type(station_char):
        # 获取测站类型
        return {
            'R': StationType.测站,
            'B': StationType.基站,
        }.get(station_char, StationType.未知)

    @staticmethod
    def get_time_from_string(time_str):
        # 从字符串中获取时间
        return datetime.strptime(time_str, "%y%m%d%H%M%S")

    @staticmethod
    def get_duration_from_string(duration_str):
        # 从字符串中获取持续时间
        hours = int(duration_str[:2])
        minutes = int(duration_str[2:4])
        seconds = int(duration_str[4:6])
        return timedelta(hours=hours, minutes=minutes, seconds=seconds)

    @staticmethod
    def get_rinex_file_type(type_char):
        # 获取RINEX文件类型
        return {
            'O': RinexFileType.O,
            'N': RinexFileType.N,
        }.get(type_char, RinexFileType.未知)

    @staticmethod
    def get_file_format(format_str):
        # 获取文件格式
        return {
            "crx": FileFormat.CRX,
            "rnx": FileFormat.RNX,
        }.get(format_str, FileFormat.未知)
