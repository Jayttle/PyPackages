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
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# region 字体设置
plt.rcParams['font.sans-serif'] = ['SimSun']  # 指定宋体为默认字体
plt.rcParams['font.sans-serif'] = 'SimSun'  # 使用指定的中文字体
plt.rcParams['axes.unicode_minus']=False#用来正常显示负


if __name__ == '__main__':
    # 指定 Excel 文件路径和工作表名称
    file_path = r"C:\Users\juntaox\Desktop\工作\28.产值聚类和预测\电量明细表.xlsx"
    sheet_name1 = r'Sheet1'
    sheet_name2 = r'Sheet2'


    # 使用 pandas 读取指定的工作表
    df = pd.read_excel(file_path, sheet_name=sheet_name1)
    df_2 = pd.read_excel(file_path, sheet_name=sheet_name2)

    # 定义目标行业名称列表
    target_industries = [
        "化学药品原料药制造",
        "中药饮片加工",
        "化学药品制剂制造",
        "生物药品制造",
        "卫生材料及医药用品制造",
        "药用辅料及包装材料",
        "基因工程药物和疫苗制造",
        "兽用药品制造",
        "中成药生产"
    ]
    # 使用 .isin() 方法筛选行业名称列中的目标行业
    filtered_df = df_2[df_2['行业名称'].isin(target_industries)]

    # 创建一个空字典，用于存储行业名称和对应的组织机构代码
    organization_codes = {}

    # 遍历过滤后的 DataFrame，按行业名称逐个存入字典
    for _, row in filtered_df.iterrows():
        industry = row['行业名称']
        org_code = row['组织机构代码']
        
        # 如果该行业名称还没有在字典中，则创建一个新列表
        if industry not in organization_codes:
            organization_codes[industry] = []
        
        # 将当前的组织机构代码添加到该行业的列表中
        organization_codes[industry].append(org_code)

    # 打印筛选结果
    print(organization_codes)
    # 创建一个空字典，用于存储每个行业的过滤后的 DataFrame
    # 假设 organization_codes 是一个字典，格式类似于 {行业: [组织机构代码1, 组织机构代码2, ...]}
    filter_df = {}

    # 根据 organization_codes 筛选 df 中对应的行，并存入 filter_df 字典
    for industry, org_codes in organization_codes.items():
        # 只筛选组织机构代码在 org_codes 列表中的行
        filter_df[industry] = df[df['组织机构代码'].isin(org_codes)]

    # 打印筛选后的 DataFrame（根据字典存储的结果）
    print("\nFiltered DataFrames:")
    for industry, filtered_data in filter_df.items():
        print(f"\nIndustry: {industry}")
        print(filtered_data.head())  # 只打印前几行以节省空间

    
    # 输出到新的 Excel 文件，其中每个行业对应一个工作表
    with pd.ExcelWriter(r"C:\Users\juntaox\Desktop\工作\28.产值聚类和预测\filtered_output.xlsx") as writer:
        for industry, filtered_data in filter_df.items():
            filtered_data.to_excel(writer, sheet_name=industry[:31], index=False)  # Sheet name can be max 31 chars