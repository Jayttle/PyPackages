from typing import List, Optional, Tuple, Union, Set, Dict
from datetime import datetime, timedelta
import pyproj
import math
from JayttleProcess import TBCProcessCsv

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
    def _from_file(cls, file_path: str) -> List['Ggkx']:
        ggkx_data_list: List['Ggkx'] = []
        with open(file_path, 'r') as file:
            next(file)  # Skip the header row
            for line in file:
                data = line.strip().split('\t')
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
                        ggkx_data_list.append(Ggkx(time, station_id, receiver_id, lat, lon, geo_height,
                                                fix_mode, sate_num, pdop, sigma_e, sigma_n, sigma_u, prop_age))

        return ggkx_data_list
    
class GgkxExtended(Ggkx):
    def __init__(self, time: str, station_id: int, receiver_id: int, lat: float, lon: float, geo_height: float,
                 fix_mode: int, sate_num: int, pdop: float, sigma_e: float, sigma_n: float, sigma_u: float,
                 prop_age: int):
        super().__init__(time, station_id, receiver_id, lat, lon, geo_height, fix_mode, sate_num, pdop, sigma_e,
                         sigma_n, sigma_u, prop_age)
        self.east_coord, self.north_coord, self.adjusted_elev = convert_coordinates(lat, lon, geo_height)

    @classmethod
    def _from_file(cls, file_path: str) -> List['GgkxExtended']:
        ggkx_data_list: List[GgkxExtended] = []
        ggkx_list = super()._from_file(file_path)  # Assume _from_file method returns List[Ggkx]
        
        # Iterate through ggkx_list, taking every 10th element
        for i in range(0, len(ggkx_list), 10):
            ggkx = ggkx_list[i]
            ggkx_extended = GgkxExtended(ggkx.time.strftime('%Y-%m-%d %H:%M:%S.%f'), ggkx.station_id,
                                         ggkx.receiver_id, ggkx.lat, ggkx.lon, ggkx.geo_height, ggkx.fix_mode,
                                         ggkx.sate_num, ggkx.pdop, ggkx.sigma_e, ggkx.sigma_n, ggkx.sigma_u,
                                         ggkx.prop_age)
            ggkx_data_list.append(ggkx_extended)

        return ggkx_data_list
    

def calculate_distances(ggkx_list1: List[GgkxExtended], ggkx_list2: List[GgkxExtended]) -> List[float]:
    """
    计算两个 GgkxExtended 对象列表中前1000条数据对应元素之间的距离
    
    Args:
        ggkx_list1 (List[GgkxExtended]): 第一个 GgkxExtended 对象列表
        ggkx_list2 (List[GgkxExtended]): 第二个 GgkxExtended 对象列表
        
    Returns:
        List[float]: 距离列表，每个元素对应两个列表中对应元素的距离（单位：千米）
    """
    distances = []

    # 遍历两个列表的前1000个元素
    for ggkx1, ggkx2 in zip(ggkx_list1[:1000], ggkx_list2[:1000]):
        # 假设每个对象的 east_coord 和 north_coord 单位是米
        east1, north1 = ggkx1.east_coord, ggkx1.north_coord
        east2, north2 = ggkx2.east_coord, ggkx2.north_coord

        # 计算 coord1 和 coord2 之间的距离（单位：千米）
        distance_in_meters = calculate_distance_between_points(east1, north1, east2, north2)
        distances.append(distance_in_meters)

    return distances

def convert_coordinates(lat: float, lon: float, geo_height: float) -> tuple[float, float]:
    # 定义原始坐标系（WGS84）
    from_proj = pyproj.Proj(proj='latlong', datum='WGS84')

    # 定义目标投影坐标系（例如，Transverse Mercator投影）
    to_proj_params = {
        'proj': 'tmerc',
        'lat_0': 0,
        'lon_0': 117,
        'k': 1,
        'x_0': 500000,
        'y_0': 0,
        'ellps': 'WGS84',
        'units': 'm'
    }
    to_proj = pyproj.Proj(to_proj_params)

    # 执行坐标转换
    easting, northing = pyproj.transform(from_proj, to_proj, lon, lat)


    # 使用 pyproj.Geod 对象计算椭球高度的影响
    geod = pyproj.Geod(ellps='WGS84')
    _, _, _, adjusted_elev = geod.npts(lon, lat, lon, lat, npts=1, radians=False, heights=[geo_height])

    return easting, northing, adjusted_elev
def calculate_distance_between_points(east1: float, north1: float, east2: float, north2: float) -> float:
    """
    计算平面直角坐标系中两点之间的直线距离（欧几里得距离）
    
    Args:
        east1 (float): 第一个点的东方向坐标
        north1 (float): 第一个点的北方向坐标
        east2 (float): 第二个点的东方向坐标
        north2 (float): 第二个点的北方向坐标
        
    Returns:
        float: 两点之间的直线距离，单位是米
    """
    distance_squared = (east2 - east1)**2 + (north2 - north1)**2
    distance = distance_squared**0.5
    return distance

def get_average_coords(ggkx_extended_list: List['GgkxExtended']) -> tuple:
    if not ggkx_extended_list:
        return None, None

    total_east = 0.0
    total_north = 0.0
    num_records = len(ggkx_extended_list)

    for ggkx_extended in ggkx_extended_list:
        total_east += ggkx_extended.east_coord
        total_north += ggkx_extended.north_coord

    avg_east = total_east / num_records
    avg_north = total_north / num_records

    return avg_east, avg_north

def calculate_height_differences(ggkxR071: list['Ggkx'], ggkxR072: list['Ggkx'], output_file_path):
    range = 2.984
    list_height_diff = []
    list_angle = []
    # Open a file for writing the results
    with open(output_file_path, 'w') as output_file:
        # Initialize pointers for both lists
        i, j = 0, 0
        
        # Iterate until we reach the end of either list
        while i < len(ggkxR071) and j < len(ggkxR072):
            ggkx1 = ggkxR071[i]
            ggkx2 = ggkxR072[j]
            
            if ggkx1.time == ggkx2.time:
                # Calculate height difference
                height_difference = ggkx1.geo_height - ggkx2.geo_height
                list_height_diff.append(height_difference)
                angle = math.asin(height_difference / range)
                angle = math.degrees(angle)
                
                # Write the result to the file
                output_file.write(f"{ggkx1.time.strftime('%Y-%m-%d %H:%M:%S.%f')}\t{height_difference}\t{angle}\n")
                list_angle.append(angle)
                # Move both pointers forward
                i += 1
                j += 1
            elif ggkx1.time < ggkx2.time:
                # Move pointer of ggkxR071 forward
                i += 1
            else:
                # Move pointer of ggkxR072 forward
                j += 1
    return list_height_diff, list_angle