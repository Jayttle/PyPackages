import os
from enum import Enum
from datetime import datetime, timedelta
import zipfile

class RinexFileType(Enum):
    O = 1
    N = 2
    Unknown = 3

class FileFormat(Enum):
    CRX = 1
    RNX = 2
    Unknown = 3

class StationType(Enum):
    RoverStation = 1
    BaseStation = 2
    Unknown = 3

class RinexFileInfo:
    def __init__(self, input_path, ftp_file):
        self.FtpFile = ftp_file
        self.InputPath = input_path
        self.FileInfo = None  # Assuming you have FileInfo object
        self.FileName = None
        self.StationName = None
        self.MarkerName = None
        self.StationId = None
        self.StationType = None
        self.ReceiverId = None
        self.StartGpsTimeStr = None
        self.StartGpsTime = None
        self.DurationStr = None
        self.Duration = None
        self.TimeStr = None
        self.FileType = None
        self.InfoStr = None
        self.Compressed = None
        self.Format = None

        self.parse_file_name()

    def parse_file_name(self):
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
        return {
            'R': StationType.RoverStation,
            'B': StationType.BaseStation,
        }.get(station_char, StationType.Unknown)

    @staticmethod
    def get_time_from_string(time_str):
        return datetime.strptime(time_str, "%y%m%d%H%M%S")

    @staticmethod
    def get_duration_from_string(duration_str):
        hours = int(duration_str[:2])
        minutes = int(duration_str[2:4])
        seconds = int(duration_str[4:6])
        return timedelta(hours=hours, minutes=minutes, seconds=seconds)

    @staticmethod
    def get_rinex_file_type(type_char):
        return {
            'O': RinexFileType.O,
            'N': RinexFileType.N,
        }.get(type_char, RinexFileType.Unknown)

    @staticmethod
    def get_file_format(format_str):
        return {
            "crx": FileFormat.CRX,
            "rnx": FileFormat.RNX,
        }.get(format_str, FileFormat.Unknown)
