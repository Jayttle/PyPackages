from typing import List, Optional, Tuple, Union, Set, Dict
from datetime import datetime, timedelta

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
    def _from_file(cls, file_path: str) -> List['TiltmeterData']:
        """
        数据读取 129904 24.2381s
        效率 5361个/s
        8757913 356.5892s (有判断的)
        """
        tiltmeter_data_list: List['TiltmeterData'] = []
        with open(file_path, 'r') as file:
            next(file)  # Skip the header row
            for line in file:
                data = line.strip().split('\t')
                if len(data) == 4:
                    data[0] = str(data[0])
                    data[1] = int(data[1])  
                    data[2] = float(data[2])  
                    data[3] = float(data[3]) 
                    tiltmeter_data_list.append(TiltmeterData(*data))

        return tiltmeter_data_list