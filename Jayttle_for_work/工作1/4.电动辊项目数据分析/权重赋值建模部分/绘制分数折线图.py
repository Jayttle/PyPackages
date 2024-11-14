import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import seaborn as sns
from scipy import stats
import json
import os
from scipy.signal import savgol_filter

# # region 字体设置
# plt.rcParams['font.sans-serif'] = ['SimSun']  # 指定宋体为默认字体
# plt.rcParams['font.sans-serif'] = 'SimSun'  # 使用指定的中文字体
# plt.rcParams['axes.unicode_minus']=False#用来正常显示负

def plt_score(file_path: str):
    # 读取 Excel 文件中的前120行数据，只读取需要的列
    df = pd.read_excel(file_path, usecols=['hour', '综合评价得分', 'type'])

    # 将 'hour' 列转换为 datetime 类型
    df['hour'] = pd.to_datetime(df['hour'], errors='coerce')

    # 确保数据没有缺失
    df = df.dropna(subset=['hour', '综合评价得分', 'type'])

    # 对 '综合评价得分' 列进行归一化处理，使其范围从0到100
    min_score = df['综合评价得分'].min()
    max_score = df['综合评价得分'].max()
    
    df['normalized_score'] = 100 * (df['综合评价得分'] - min_score) / (max_score - min_score)

    # 计算 'hour' 列相邻两行的时间差（以小时为单位）
    df['time_diff'] = df['hour'].diff().dt.total_seconds() / 3600  # 转换为小时
    
    # 对于时间差大于1小时的行，将其对应的 'normalized_score' 设置为 NaN
    df.loc[df['time_diff'] > 1, 'normalized_score'] = None

    # 打印归一化后的数据查看
    print(df[['hour', '综合评价得分', 'normalized_score', 'type']].head())

    # 按 'type' 列分组
    grouped = df.groupby('type')

    # 如果分组的个数大于10，选择等间隔的10个进行绘图
    group_keys = list(grouped.groups.keys())
    
    if len(group_keys) > 18:
        # 使用 numpy.linspace 选择10个等间隔的分组
        selected_groups = np.linspace(0, len(group_keys) - 1, 18, dtype=int)
        selected_group_keys = [group_keys[i] for i in selected_groups]
    else:
        # 如果分组少于或等于10个，选择所有分组
        selected_group_keys = group_keys

    # 绘图
    plt.figure(figsize=(15, 9))
    for label in selected_group_keys:
        group = grouped.get_group(label)
        plt.plot(group['hour'], group['normalized_score'], linestyle='-', label=f"motor_id_{label}")

    # 设置图表标题和标签
    plt.title('Equipment Health Over Time')
    plt.xlabel('Time')
    plt.ylabel('Health Score(%)')
    plt.xticks(rotation=45)  # 让时间标签旋转，防止重叠
    plt.grid(True)

    # 设置y轴的范围
    plt.ylim(0, 100)

    # 在图例中显示每个 'type' 的标签，并设置字体大小
    plt.legend(fontsize=8, loc='upper left')  # 这里可以调节 fontsize 和 loc

    # 显示图表
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # 文件路径
    file_path = r"C:\Users\juntaox\Desktop\熵权法-TOPSIS\结果-not15.xlsx"
    
    # 调用函数
    plt_score(file_path)