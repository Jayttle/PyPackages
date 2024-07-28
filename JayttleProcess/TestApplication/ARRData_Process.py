# 标准库导入
import math
import random
from datetime import datetime, timedelta
import time
import warnings
import os
# 相关第三方库导入
import numpy as np
from math import floor
import chardet
import pandas as pd
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from matplotlib.font_manager import FontProperties
import statsmodels.api as sm
from collections import defaultdict
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.linear_model import Ridge, Lasso, LassoCV, ElasticNet, LinearRegression
from sklearn.metrics import (root_mean_squared_error, mean_absolute_error, r2_score,
                             mean_squared_error, silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, v_measure_score)
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.mixture import GaussianMixture
from sklearn.svm import OneClassSVM
from sklearn.model_selection import GridSearchCV, cross_val_score
from scipy.spatial.distance import euclidean
from scipy import signal, fft, interpolate, stats
from scipy.cluster import hierarchy
from scipy.stats import t, shapiro, pearsonr, f_oneway, gaussian_kde, chi2_contingency
from scipy.signal import hilbert, find_peaks
from PyEMD import EMD, EEMD, CEEMDAN
from typing import List, Optional, Tuple, Union, Set, Dict
import pywt
from itertools import groupby
from operator import attrgetter
from dataclasses import dataclass

# 自定义库导入
from pyswarm import pso
from JayttleProcess import ListFloatDataMethod as LFDM
from JayttleProcess import TimeSeriesDataMethod as TSM, TBCProcessCsv, CommonDecorator
from JayttleProcess.TimeSeriesDataMethod import TimeSeriesData
from scipy.interpolate import interp1d


class ARRData:
    def __init__(self, time: str, station_id: int, fix_mode: int, vector_num: int, sate_num: int,
                 delta1: float, delta2: float, delta3: float, std1: float, std2: float, std3: float,
                 corr12: float, corr13: float, corr23: float, ref_id: int, vector_coor: int,
                 vector_operation: int, clock_assum: int):
        self.time = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
        self.station_id = station_id
        self.fix_mode = fix_mode
        self.vector_num = vector_num
        self.sate_num = sate_num
        self.delta1 = delta1
        self.delta2 = delta2
        self.delta3 = delta3
        self.std1 = std1
        self.std2 = std2
        self.std3 = std3
        self.corr12 = corr12
        self.corr13 = corr13
        self.corr23 = corr23
        self.ref_id = ref_id
        self.vector_coor = vector_coor
        self.vector_operation = vector_operation
        self.clock_assum = clock_assum

    def __str__(self):
        return f"Time: {self.time}, StationID: {self.station_id}, " \
               f"FixMode: {self.fix_mode}, VectorNum: {self.vector_num}, " \
               f"SateNum: {self.sate_num}, Delta1: {self.delta1}, " \
               f"Delta2: {self.delta2}, Delta3: {self.delta3}, " \
               f"STD1: {self.std1}, STD2: {self.std2}, STD3: {self.std3}, " \
               f"Corr12: {self.corr12}, Corr13: {self.corr13}, Corr23: {self.corr23}, " \
               f"RefID: {self.ref_id}, VectorCoor: {self.vector_coor}, " \
               f"VectorOperation: {self.vector_operation}, ClockAssum: {self.clock_assum}"

def read_arr_data(file_path: str) -> List[ARRData]:
    arr_data_list: List[ARRData] = []
    with open(file_path, 'r') as file:
        next(file)  # Skip the header row
        for line in file:
            data = line.strip().split('\t')
            # 数据类型转换
            time = str(data[0])
            station_id = int(data[1])
            fix_mode = int(data[2])
            vector_num = int(data[3])
            sate_num = int(data[4])
            delta1 = float(data[5])
            delta2 = float(data[6])
            delta3 = float(data[7])
            std1 = float(data[8])
            std2 = float(data[9])
            std3 = float(data[10])
            corr12 = float(data[11])
            corr13 = float(data[12])
            corr23 = float(data[13])
            ref_id = int(data[14])
            vector_coor = int(data[15])
            vector_operation = int(data[16])
            clock_assum = int(data[17])
            
            arr_obj = ARRData(time, station_id, fix_mode, vector_num, sate_num,
                              delta1, delta2, delta3, std1, std2, std3,
                              corr12, corr13, corr23, ref_id, vector_coor,
                              vector_operation, clock_assum)
            arr_data_list.append(arr_obj)
    
    return arr_data_list


def split_by_station_id(arr_data_list: List[ARRData]) -> Dict[int, List[ARRData]]:
    station_id_dict: Dict[int, List[ARRData]] = {}
    
    for data_obj in arr_data_list:
        station_id = data_obj.station_id
        
        if station_id not in station_id_dict:
            station_id_dict[station_id] = []
        
        station_id_dict[station_id].append(data_obj)
    
    return station_id_dict

def run_arr():
    file_path = r"D:\Program Files (x86)\Software\OneDrive\PyPackages_DataSave\arr0701.txt"

    arr_data_list = read_arr_data(file_path)
    station_id_dict = split_by_station_id(arr_data_list)

    for key in station_id_dict:
        print(f'{key}:{len(station_id_dict[key])}')

    
if __name__ == "__main__":
    run_arr()
