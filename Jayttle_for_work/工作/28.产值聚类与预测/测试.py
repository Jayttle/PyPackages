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

def 读取文件夹中的所有Excel文件(folder_path: str, marker_name:str) -> List[pd.DataFrame]:
    """
    读取指定文件夹中的所有 Excel 文件，并返回一个 DataFrame 列表。

    :param folder_path: Excel 文件所在的文件夹路径
    :return: 包含所有 Excel 文件内容的 DataFrame 列表
    """
    # 获取文件夹中的所有文件
    files = os.listdir(folder_path)
    
    # 过滤出所有 Excel 文件（.xls 或 .xlsx）
    excel_files = [file for file in files if file.endswith('.xlsx') or file.endswith('.xls')]
    
    # 用来存储每个 Excel 文件的数据
    data_frames = []
    
    # 遍历每个 Excel 文件，读取并保存到列表
    for file in excel_files:
        file_path = os.path.join(folder_path, file)
        
        # 读取 Excel 文件，返回 DataFrame
        df = pd.read_excel(file_path)
        
        # 只筛选第一列以271到278开头的行数据
        df_filtered = df[df.iloc[:, 0].astype(str).str.startswith(('271', '272', '273', '274', '275', '276', '277', '278'))]
        
        # 确保 DataFrame 中有 "现价产值" 列，且保留第一列和"现价产值"列
        if marker_name in df.columns:
            df_filtered = df_filtered[[marker_name, df.columns[0]]]  # 第一列和"现价产值"列
            
            # 调整列的顺序，使得第一列在前
            df_filtered = df_filtered[[df.columns[0], marker_name]]
        
        # 为每一行根据其第一个列值设置索引
        df_filtered.index = df_filtered[df.columns[0]].astype(str).apply(lambda x: x[:3])  # 提取前三位作为索引
        
        # 将筛选后的 DataFrame 添加到列表中
        data_frames.append(df_filtered)
    
    return data_frames

def 合并数据frames(data_frames: List[pd.DataFrame]) -> pd.DataFrame:
    """
    合并所有 DataFrame，将每个 DataFrame 的第二列添加一列，值为索引。
    
    :param data_frames: 包含多个 DataFrame 的列表
    :return: 合并后的 DataFrame
    """
    merged_data = {}  # 使用字典存储数据
    marker_year = 2023
    for idx, df in enumerate(data_frames):
        # 将 '现价产值' 列添加到字典中，键为 '数据{idx}'
        marker_month = idx + 2
        if marker_month > 12:
            marker_month = marker_month % 12 + 1
            marker_year = 2024
        merged_data[f'{marker_year}{marker_month:02}'] = df['现价产值']
        
    # 将字典合并成 DataFrame
    result_df = pd.concat(merged_data, axis=1, ignore_index=False)
    # 重命名指定的列
    result_df.rename(columns={'202407': '202409', '202408': '202410'}, inplace=True)
    return result_df


def 绘制散点图和折线图(df: pd.DataFrame):
    """
    遍历 DataFrame 每一行数据，绘制散点图和折线图，忽略第一行。
    :param df: 要绘制的 DataFrame
    """
    # 创建一个图形对象（所有行都在同一个图中）
    plt.figure(figsize=(10, 6))
    
    # 定义颜色集合，用于区分不同行的数据
    colors = plt.cm.get_cmap('tab10', len(df))  # 使用颜色地图，根据行数选择颜色
    i = 0
    # 跳过第一行，从第二行开始绘制
    for idx, row in df.iterrows():  # 从第二行开始
        # 行的索引作为 x 轴，行的值作为 y 轴
        x = row.index  # 列名作为 x 轴
        y = row.values  # 行的值作为 y 轴
        
        # 将非数值转换为 NaN，并确保 y 是数值型
        y = pd.to_numeric(y, errors='coerce')  # 将非数字转换为 NaN
        
        # 筛选掉 NaN 值，确保散点图只绘制有效的点
        valid_x = x[~np.isnan(y)]  # 取出不为 NaN 的 x 值
        valid_y = y[~np.isnan(y)]  # 取出不为 NaN 的 y 值
        
        # 绘制散点图，只绘制有效的点
        plt.scatter(valid_x, valid_y, color=colors(i), zorder=5)
        
        # 绘制折线图，跳过 NaN 值自动不连线
        plt.plot(x, y, color=colors(i), label=f'{idx}', linestyle='-', zorder=4)
        i += 1
    # 添加图例
    plt.xlabel('日期')
    plt.ylabel('现价产值')
    plt.title('行业数据示意图')
    plt.legend()
    
    # 显示图表
    plt.show()

def 可视化聚类结果(final_df, kmeans, labels):
    """
    可视化聚类结果。
    :param final_df: 原始的 DataFrame
    :param kmeans: 聚类模型
    :param labels: 每个样本的聚类标签
    """
    # 添加聚类标签
    final_df['Cluster'] = labels
    
    # 画出聚类后的折线图
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=final_df.T, palette="tab10", linewidth=2)
    plt.title('行业时序数据聚类结果')
    plt.xlabel('时间')
    plt.ylabel('现价产值')
    plt.legend(title='Cluster', labels=[f'Cluster {i+1}' for i in range(kmeans.n_clusters)])
    plt.show()

    # 可视化聚类的热图
    plt.figure(figsize=(10, 6))
    sns.heatmap(final_df.drop(columns='Cluster').T, cmap="coolwarm", cbar_kws={'label': '现价产值'}, annot=True)
    plt.title('行业时序数据聚类热图')
    plt.show()

def 输出聚类分类结果(final_df, new_index):
    """
    输出每个类别的聚类标签及对应的行业名称。
    :param final_df: 包含聚类结果的 DataFrame
    """
    # 按照聚类标签进行分组，并输出每个聚类对应的行业名称
    for cluster_num in final_df['Cluster'].unique():
        print(f"Cluster {cluster_num + 1}:")
        cluster_items = final_df[final_df['Cluster'] == cluster_num].index.tolist()
        print(", ".join([new_index[item] for item in cluster_items]))
        print("\n")


if __name__ == '__main__':
    # 调用函数读取指定文件夹中的所有 Excel 文件
    folder_path = r'C:\Users\juntaox\Desktop\工作\28.产值聚类和预测\2023年至今分行业数据'
    marker_names = ['现价产值']
    data_frames = 读取文件夹中的所有Excel文件(folder_path)
    
    # 合并所有 DataFrame
    final_df = 合并数据frames(data_frames)
    # 新的索引列表
    new_index = [
        "271、化学药品原料药制造",
        "272、化学药品制剂制造",
        "273、中药饮片加工",
        "274、中成药生产",
        "275、兽用药品制造",
        "276、生物药品制品制造",
        "277、卫生材料及医药用品制造",
        "278、药用辅料及包装材料"
    ]
    # 确保新的索引长度和 final_df 的行数相同
    if len(new_index) == len(final_df):
        final_df.index = new_index
    else:
        print("新的索引列表长度与 DataFrame 的行数不匹配。")
    绘制散点图和折线图(final_df)
