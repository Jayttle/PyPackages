from typing import List, Optional, Tuple, Union, Set, Dict
from datetime import datetime, timedelta

class Avr:
    def __init__(self, time, station_id, fix_mode, yaw, tilt, range_val, pdop, sat_num):
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
    def _from_string(cls, line):
        # Assuming the input line format is consistent and tab-separated
        parts = line.strip().split('\t')
        time = parts[0]
        station_id = int(parts[1])
        fix_mode = int(parts[2])
        yaw = float(parts[3])
        tilt = float(parts[4])
        range_val = float(parts[5])
        pdop = float(parts[6])
        sat_num = int(parts[7])
        return cls(time, station_id, fix_mode, yaw, tilt, range_val, pdop, sat_num)

    def __str__(self):
        return f"Time: {self.time}, Station ID: {self.station_id}, Fix Mode: {self.fix_mode}, Yaw: {self.yaw}, Tilt: {self.tilt}, Range: {self.range_val}, PDOP: {self.pdop}, Sat Num: {self.sat_num}"