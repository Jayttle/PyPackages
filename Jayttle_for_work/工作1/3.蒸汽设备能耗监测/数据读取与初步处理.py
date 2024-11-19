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

def 运行01_过滤设备A的运行(file_path: str) -> pd.DataFrame:
    """
    读取 CSV 文件，筛选指定列并进行清洗处理，返回处理后的 DataFrame。

    :param file_path: CSV 文件的路径
    :return: 处理后的 DataFrame
    """

    # 显式指定列的数据类型
    dtype_dict = {
        'recipename': 'str',  # 假设第1列应该是浮动类型
        'module': 'str',    # 假设第2列应该是整数类型
        'module_task': 'str'       # 假设第3列应该是字符串类型
    }

    df = pd.read_csv(file_path, dtype=dtype_dict)

    # 提取相关列
    columns_to_select = [
        'datetime', 'recipename', 'module', 'module_task', 'shift', 'team', 
        'no_gap', 'A_HT_phase', 'A_HT_steam_total', 'A_HT_steam_total_blend',
        'B_HT_steam_total', 'B_HT_steam_total_blend', 'TT1142_steam_total', 'TT1142_steam_total_blend'
    ]
    
    # 选择所需的列
    df_selected = df[columns_to_select]
    
    # 筛选掉 A_HT_steam_total_blend 为 0 的行
    df_filtered = df_selected[df_selected['A_HT_steam_total_blend'] != 0]

    df_filtered_cleaned = df_filtered.copy()

    # 删除包含 NaN 值的行
    df_filtered_cleaned = df_filtered.dropna()

    # 使用 .loc 进行赋值操作，避免 SettingWithCopyWarning
    df_filtered_cleaned.loc[:, 'datetime'] = pd.to_datetime(df_filtered_cleaned['datetime'])
    
    # 找到连续相等的 'A_HT_steam_total_blend' 数据段
    df_filtered_cleaned.loc[:, 'A_HT_steam_total_blend_shifted'] = df_filtered_cleaned['A_HT_steam_total_blend'].shift(1)
    
    # 标记出数据相等的行
    df_filtered_cleaned.loc[:, 'is_duplicate'] = df_filtered_cleaned['A_HT_steam_total_blend'] == df_filtered_cleaned['A_HT_steam_total_blend_shifted']
    
    # 只保留数据发生变化的行
    df_final = df_filtered_cleaned[~df_filtered_cleaned['is_duplicate']]
    
    # 删除辅助列
    df_final = df_final.drop(columns=['A_HT_steam_total_blend_shifted', 'is_duplicate'])
    
    return df_final

def 运行02_分割连续时间段(df: pd.DataFrame, time_threshold: str = '2min') -> List[pd.DataFrame]:
    """
    根据时间间隔将 DataFrame 按指定时间阈值分段。

    :param df: 要分段的 DataFrame，必须包含 'datetime' 列
    :param time_threshold: 时间间隔阈值，默认 '2min'
    :return: 按时间分段后的 DataFrame 列表
    """
    # 用来存储每个时段的分段
    segments = []
    
    # 初始化时段编号
    period_number = 1
    
    # 新的时段的起始行
    segment_start = df.iloc[0]
    
    # 遍历数据框中的行，计算时间间隔
    for i in range(1, len(df)):
        # 计算当前行与上一行的时间间隔
        time_diff = df.iloc[i]['datetime'] - df.iloc[i - 1]['datetime']
        
        # 如果时间间隔大于规定阈值，分段
        if time_diff > pd.Timedelta(time_threshold):
            # 将前一个时段的部分保存
            segments.append(df[(df['datetime'] >= segment_start['datetime']) & (df['datetime'] < df.iloc[i - 1]['datetime'])])
            # 更新下一个时段的起始行
            segment_start = df.iloc[i]
            period_number += 1
    
    # 最后一个时段的数据添加
    segments.append(df[(df['datetime'] >= segment_start['datetime'])])
    
    return segments


def 运行02_分割按不同类型(df: pd.DataFrame, item: str) -> List[pd.DataFrame]:
    """
    根据'recipename'列的不同数值将 DataFrame 按照 'recipename' 分段。

    :param df: 要分段的 DataFrame，必须包含 'recipename' 列
    :return: 按'recipename'分段后的 DataFrame 列表
    """
    # 用来存储每个分段
    segments = []
    
    # 根据 'recipename' 列进行分组
    group = df.groupby(item)
    
    # 遍历每个分组，将每个分组的数据保存到 segments 列表
    for recipename, group_df in group:
        segments.append(group_df)
    
    return segments


def 连续时间段_绘图(segments: List[pd.DataFrame], output_img_path: str):
    """
    绘制所有 segment 中 'A_HT_steam_total_blend' 的折线图。
    x 轴为数据的索引（按顺序），y 轴为 'A_HT_steam_total_blend' 数值。
    所有折线图将绘制在同一个图上，用不同颜色表示不同的时段。

    :param segments: 按时间分段后的 DataFrame 列表
    """
    plt.figure(figsize=(12, 8))  # 设置一个大的图形以容纳多个折线图

    # 遍历每个分段
    for i, segment in enumerate(segments):
        # 使用 range(len(segment)) 作为 x 轴（即索引顺序）
        plt.plot(range(len(segment)), segment['A_HT_steam_total_blend'], label=f'Segment {i+1}')
    
    # 添加图标标题、坐标轴标签等
    plt.title('All Segments - A_HT_steam_total_blend')
    plt.xlabel('Index')  # x轴为数据的索引
    plt.ylabel('A_HT_steam_total_blend')
    plt.legend()  # 显示图例，区分不同的时段
    plt.grid(True)  # 显示网格
    plt.tight_layout()  # 自动调整布局以防标签重叠
    if output_img_path is not None:
        plt.savefig(output_img_path)
    else:
        plt.show()
    plt.close()


def 连续时间段01_特征统计(segments: List[pd.DataFrame]) -> List[Dict]:
    """
    处理多个 DataFrame，统计每个 DataFrame 的行数、指定列的 unique 值、datetime 范围和运行时间，
    以及 'A_HT_steam_total_blend' 的统计信息，并根据差值为负的地方切割数据，
    拟合每段数据的斜率，并评估拟合效果。

    :param segments: List[pd.DataFrame] - 一个 DataFrame 列表
    :return: List[Dict] - 包含统计信息和每段拟合斜率的字典列表
    """
    result_dict = []
    
    # 遍历每个 DataFrame
    for idx, df in enumerate(segments, 1):
        # 统计数据个数（行数）
        num_rows = len(df)
        
        # 获取 'recipename', 'module', 'module_task' 列的 unique 值
        unique_recipename = df['recipename'].unique().tolist() if 'recipename' in df.columns else []
        unique_module = df['module'].unique().tolist() if 'module' in df.columns else []
        unique_module_task = df['module_task'].unique().tolist() if 'module_task' in df.columns else []
        
        # 计算 'datetime' 列的最小和最大值（假设 'datetime' 列是 datetime 类型）
        if 'datetime' in df.columns:
            # 确保 'datetime' 列是 datetime 类型
            df.loc[:, 'datetime'] = pd.to_datetime(df['datetime'], errors='coerce')  # 使用 .loc 防止 SettingWithCopyWarning
            min_datetime = df['datetime'].min()
            max_datetime = df['datetime'].max()
            
            # 计算时间差
            if pd.notna(min_datetime) and pd.notna(max_datetime):
                run_time = max_datetime - min_datetime
                # 将时间差转换为 hh:mm:ss 格式
                run_time_str = str(run_time).split()[2]  # 只取时间部分 (小时:分钟:秒)
            else:
                run_time_str = "N/A"
            
            datetime_range = f"{min_datetime} ~ {max_datetime}"
        else:
            datetime_range = "N/A"
            run_time_str = "N/A"
        
        # 统计 'A_HT_steam_total_blend' 的最大值
        if 'A_HT_steam_total_blend' in df.columns:
            max_blend = df['A_HT_steam_total_blend'].max()
            
            # 按分钟分组，并计算每分钟的变化平均值和标准差
            # 使用 .loc 显式指定行和列索引来修改 DataFrame
            # 确保 'datetime' 列是 datetime 类型
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
            df.loc[:, 'minute'] = df['datetime'].dt.floor('min')  # 通过 .loc 修改 'minute' 列
            group_by_minute = df.groupby('minute')['A_HT_steam_total_blend']
            
            # 每分钟变化的平均值和标准差
            mean_change_per_minute = group_by_minute.diff().mean() if len(group_by_minute) > 1 else 0
            std_change_per_minute = group_by_minute.diff().std() if len(group_by_minute) > 1 else 0
            
            # 每分钟的最大值和最小值
            max_per_minute = group_by_minute.max().max()  # 最大值
            min_per_minute = group_by_minute.min().min()  # 最小值
        else:
            max_blend = mean_change_per_minute = std_change_per_minute = max_per_minute = min_per_minute = "N/A"
        
        # 切割数据段并拟合每段数据的斜率
        if 'A_HT_steam_total_blend' in df.columns:
            # 计算相邻数据点的差值时也使用 .loc 来避免警告
            df.loc[:, 'diff'] = df['A_HT_steam_total_blend'].diff()  # 计算相邻数据点的差值
            cut_segments = []  # 存储切割后的子段
            current_segment = []  # 当前正在处理的子段
            slopes = []
            r2_scores = []  # 存储每段的 R² 值
            mse_values = []  # 存储每段的 MSE 值
            rmse_values = []  # 存储每段的 RMSE 值
            
            # 遍历数据，根据差值为负的地方切割数据
            for i in range(1, len(df)):
                if df['diff'].iloc[i] < 0:  # 当差值为负，说明数据下降
                    if current_segment:
                        cut_segments.append(current_segment)
                    current_segment = [df.iloc[i-1]]  # 新的段从当前点开始
                else:
                    current_segment.append(df.iloc[i])  # 否则继续添加到当前段

            # 处理最后一个段
            if current_segment:
                cut_segments.append(current_segment)
            
            # 对每个数据段拟合斜率并计算评估指标
            for segment in cut_segments:
                segment_df = pd.DataFrame(segment)
                x = np.arange(len(segment_df))  # 用数据点的索引作为自变量
                y = segment_df['A_HT_steam_total_blend'].values  # 目标变量是 'A_HT_steam_total_blend'
                
                # 进行线性回归拟合
                if len(segment_df) > 1:  # 确保段内数据有足够的点
                    slope, intercept, r_value, p_value, std_err = linregress(x, y)
                    slopes.append(slope)
                    r2_scores.append(r_value ** 2)  # R²
                    y_pred = slope * x + intercept
                    mse = mean_squared_error(y, y_pred)  # 计算 MSE
                    mse_values.append(mse)
                    rmse = np.sqrt(mse)  # 计算 RMSE
                    rmse_values.append(rmse)
                else:
                    slopes.append(np.nan)  # 如果段内数据点数不足，则设置为 NaN
                    r2_scores.append(np.nan)
                    mse_values.append(np.nan)
                    rmse_values.append(np.nan)
        else:
            cut_segments = []
            slopes = []
            r2_scores = []
            mse_values = []
            rmse_values = []

        # 将统计结果保存到字典中
        result = {
            '时段序号': idx,
            '数据行数': num_rows,
            '批次': unique_recipename,
            '模块号': unique_module,
            '模块任务号': unique_module_task,
            '时间段': datetime_range,
            '时长': run_time_str,
            '最大蒸汽累积量': max_blend,
            '一分钟蒸汽累积量_平均': mean_change_per_minute,
            '一分钟蒸汽累积量_标准差': std_change_per_minute,
            '一分钟蒸汽累计量_最大值': max_per_minute,
            '一分钟蒸汽累计量_最小值': min_per_minute,
            '斜率': slopes,  # 每段的斜率
            'R方值': r2_scores,  # 每段的 R² 值
            'MSE': mse_values,  # 每段的 MSE 值
            'RMSE': rmse_values  # 每段的 RMSE 值
        }
        
        # 将字典添加到结果列表中
        result_dict.append(result)

    return result_dict

def 连续时间段的存储(segments: List[pd.DataFrame], output_path: str) -> None:
    """
    将按时间分段的数据保存到 Excel 文件。

    :param segments: 按时间分段后的 DataFrame 列表
    :param output_path: 保存的 Excel 文件路径
    :return: None
    """
    # 将每个时段的数据保存到不同的sheet中
    with pd.ExcelWriter(output_path) as writer:
        for idx, segment in enumerate(segments):
            # 保存时段数据到Excel的对应sheet
            segment.to_excel(writer, sheet_name=f'时段{idx + 1}', index=False)

def save_to_json(result_dict: List[Dict], file_path: str) -> None:
    """
    将统计结果保存到 JSON 文件。
    
    :param result_dict: 统计结果的字典列表
    :param file_path: 要保存的文件路径
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        # 使用 json.dump 将数据写入 JSON 文件，确保中文字符正确编码
        json.dump(result_dict, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    file_path = r"C:\Users\juntaox\Desktop\工作1\3.蒸汽设备能耗监测\原始材料\energy数据\energy\NengYuan_20240706_20240712.csv"

    # 读取和处理数据
    df_filtered = 运行01_过滤设备A的运行(file_path)
    
    # 按时间间隔超过2分钟进行分段
    # time_threshold = '2min'  # 设置时间间隔阈值
    # segments = 运行02_分割连续时间段(df_filtered, time_threshold)
    item_lists = ['datetime', 'recipename', 'module', 'module_task']
    for item in item_lists:
        out_folder_path = rf"C:\Users\juntaox\Desktop\工作1\3.蒸汽设备能耗监测\xjt-处理\不同{item}"
        # 判断文件夹是否存在
        if not os.path.exists(out_folder_path):
            # 如果文件夹不存在，则创建该文件夹
            os.makedirs(out_folder_path)
        if item == 'datetime':
            time_threshold = '2min'  # 设置时间间隔阈值
            segments = 运行02_分割连续时间段(df_filtered, time_threshold)
        else:
            segments = 运行02_分割按不同类型(df_filtered, item)
        # # 将分段后的数据保存到Excel
        连续时间段的存储(segments, os.path.join(out_folder_path, r'时段存储.xlsx'))
        连续时间段_绘图(segments,os.path.join(out_folder_path, r'不同时段绘图.png'))
        result_dict = 连续时间段01_特征统计(segments)
        result_df = pd.DataFrame(result_dict)
        result_df.to_excel(os.path.join(out_folder_path, r'特征统计.xlsx'))
        print("数据已成功保存到 Excel 文件中！")
