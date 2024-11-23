import pandas as pd
from typing import List, Dict
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import seaborn as sns
from scipy import stats
import json
import os
from sklearn.metrics import mean_squared_error
from scipy.stats import linregress

# region 字体设置
plt.rcParams['font.sans-serif'] = ['SimSun']  # 指定宋体为默认字体
plt.rcParams['font.sans-serif'] = 'SimSun'  # 使用指定的中文字体
plt.rcParams['axes.unicode_minus']=False#用来正常显示负

def 读取时间段的存储(output_path: str) -> List[pd.DataFrame]:
    """
    从 Excel 文件中读取按时间分段的数据。

    :param output_path: 存储的 Excel 文件路径
    :return: 读取后的按时间分段的 DataFrame 列表
    """
    # 使用 pandas 读取 Excel 文件
    with pd.ExcelFile(output_path) as xls:
        # 获取所有sheet的名称
        sheet_names = xls.sheet_names
        
        # 用来存储每个分段的 DataFrame
        segments = []
        
        # 遍历每个sheet，读取每个分段的 DataFrame
        for sheet in sheet_names:
            # 读取每个sheet中的数据
            df = pd.read_excel(xls, sheet_name=sheet)
            segments.append(df)
    
    return segments

if __name__ == '__main__':
    file_path = r"C:\Users\juntaox\Desktop\工作1\3.蒸汽设备能耗监测\xjt-处理\每段时段第一行数据舍去\不同recipename\时段存储.xlsx"
    segments = 读取时间段的存储(file_path)
    for item in segments:
        print(item.head())