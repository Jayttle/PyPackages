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

def 得到sum值df(file_path: str) -> pd.DataFrame:
    # 读取 Excel 文件
    excel_file = pd.ExcelFile(file_path)

    # 存储每个sheet的计算结果
    sheet_sum_dict = {}

    # 遍历每个工作表
    for sheet_name in excel_file.sheet_names:
        # 读取每个工作表
        df = excel_file.parse(sheet_name)
   
        # 获取以 '电量' 结尾且以 '2023' 或 '2024' 开头的列
        electric_columns = [col for col in df.columns if '电量' in str(col) and (str(col).startswith('2023') or str(col).startswith('2024'))]
        
        # 例如：将202301电量加到202302电量，202401电量加到202402电量
        for i in range(len(electric_columns)):
            col = electric_columns[i]
            # 判断如果是202301列，则加到202302列
            if '202301' in str(col) and f"202302" in str(electric_columns):
                df['202302电量'] = df.get('202302电量', 0) + df.get('202301电量', 0)
            # 判断如果是202401列，则加到202402列
            elif '202401' in str(col) and f"202402" in str(electric_columns):
                df['202402电量'] = df.get('202402电量', 0) + df.get('202401电量', 0)


        # 计算这些列的总和
        sum_values = df[electric_columns].sum()
        
        # 将结果保存到字典中，sheet_name作为键
        sheet_sum_dict[sheet_name] = sum_values.tolist()  # 转换为列表保存

    # 将结果转化为 DataFrame
    # 将字典转换为 DataFrame, 每个sheet的名字为name列，每个电量列的sum值为后面的列
    rows = []
    for sheet_name, sum_values in sheet_sum_dict.items():
        # 将每个 sheet_name 作为一行的 name，并且后面添加电量列的 sum
        row = [sheet_name] + sum_values
        rows.append(row)

    # 创建 DataFrame，第一列为name，后面为电量列的sum值
    df_result = pd.DataFrame(rows, columns=['name'] + electric_columns)
    # 设置 'name' 列为索引
    df_result.set_index('name', inplace=True)
    df_result.drop(columns=['202301电量', '202401电量', '2023年总电量'], inplace=True, errors='ignore')
    # 打印最终的 DataFrame
    return df_result

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
    plt.ylabel('售电量')
    plt.title('行业数据示意图')
    plt.legend()
    
    # 显示图表
    plt.show()

def 聚类分析(df, k=3):
    # 数据标准化
    scaler = StandardScaler()
    standardized_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    
    # KMeans 聚类
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(standardized_df)  # 使用标准化后的数据进行聚类
    labels = kmeans.labels_
    
    # 确保 labels 和 standardized_df 的长度一致
    if len(labels) == len(standardized_df):
        standardized_df['Cluster'] = labels
    else:
        raise ValueError(f"labels 长度 ({len(labels)}) 与 standardized_df 行数 ({len(standardized_df)}) 不匹配")
    
    
    return standardized_df, kmeans

def 可视化聚类结果(final_df:pd.DataFrame, kmeans, labels):
    """
    可视化聚类结果。
    :param final_df: 原始的 DataFrame
    :param kmeans: 聚类模型
    :param labels: 每个样本的聚类标签
    """
    # 添加聚类标签
    final_df['Cluster'] = labels

    # 可视化聚类的热图
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        final_df.drop(columns='Cluster').T,  # 转置后删除 'Cluster' 列
        cmap="coolwarm", 
        cbar_kws={'label': '售电量'}, 
        annot=True, 
        xticklabels=final_df.index,  # 使用 final_df 的 index 作为热图的 x 轴标签
        yticklabels=True  # 显示 y 轴标签
    )
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
    # 指定 Excel 文件路径和工作表名称
    file_path = r"C:\Users\juntaox\Desktop\工作\28.产值聚类和预测\filtered_output.xlsx"
    final_df = 得到sum值df(file_path)
    new_index = [
    "271.化学药品原料药制造",
    "273.中药饮片加工",
    "272.化学药品制剂制造",
    "276.生物药品制造",
    "277.卫生材料及医药用品制造",
    "278.药用辅料及包装材料",
    "基因工程药物和疫苗制造",
    "275.兽用药品制造",
    "274.中成药生产"
    ]
    # 确保新的索引长度和 final_df 的行数相同
    if len(new_index) == len(final_df):
        final_df.index = new_index
    # 绘制散点图和折线图(final_df)
    # 聚类分析
    final_df_standardized, kmeans = 聚类分析(final_df, k=3)  # 假设聚为2类
    # # 调用函数时传入 final_df 和 new_index
    输出聚类分类结果(final_df_standardized, new_index)

    new_index = [
    "271.化学药品原料药制造",
    "273.中药饮片加工",
    "272.化学药品制剂制造",
    "276.生物药品制造",
    "277.卫生材料及医药用品制造",
    "278.药用辅料及包装材料",
    "基因工程药物和疫苗制造",
    "275.兽用药品制造",
    "274.中成药生产"
    ]
    # 确保新的索引长度和 final_df 的行数相同
    if len(new_index) == len(final_df):
        final_df_standardized.index = new_index
    # # 可视化聚类结果
    # 可视化聚类结果(final_df_standardized, kmeans, final_df_standardized['Cluster'])

        # 保存 final_df 和 final_df_standardized 到 Excel
    with pd.ExcelWriter(r"C:\Users\juntaox\Desktop\工作\28.产值聚类和预测\药业-售电量.xlsx") as writer:
        final_df.to_excel(writer, sheet_name='汇总数据')
        final_df_standardized.to_excel(writer, sheet_name='标准化数据')
    
    print("数据已成功保存到 final_output.xlsx 文件中")