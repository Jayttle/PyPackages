import pandas as pd
from typing import List, Dict
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import seaborn as sns
from scipy import stats
import json
import glob
import os
from sklearn.metrics import mean_squared_error
from scipy.stats import linregress

# region 字体设置
plt.rcParams['font.sans-serif'] = ['SimSun']  # 指定宋体为默认字体
plt.rcParams['font.sans-serif'] = 'SimSun'  # 使用指定的中文字体
plt.rcParams['axes.unicode_minus']=False#用来正常显示负

def NengYuan_read_csv(file_path: str) -> pd.DataFrame:
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

    # 筛选掉 'module' 为 0 的行
    df_filtered = df[df['module'] != '0']

    df_filtered_cleaned = df_filtered.copy()

    # 删除包含 NaN 值的行
    df_filtered_cleaned = df_filtered_cleaned.dropna()

    # 将 'datetime' 列转换为 pandas 的 datetime 类型
    df_filtered_cleaned['datetime'] = pd.to_datetime(df_filtered_cleaned['datetime'], errors='coerce')

    # 检查是否存在无效的 'datetime' 数据，如果有的话，删除这些行
    df_filtered_cleaned = df_filtered_cleaned.dropna(subset=['datetime'])

    return df_filtered_cleaned

def segments_in_recipename(df: pd.DataFrame) -> List[pd.DataFrame]:
    """
    根据 'recipename' 列的不同数值将 DataFrame 按照 'recipename' 分段，并计算列的差值diff。

    :param df: 要分段的 DataFrame，必须包含 'recipename' 列
    :return: 按 'recipename' 分段后的 DataFrame 列表
    """
    # 用来存储每个分段
    segments = []
    
    # 根据 'recipename' 列进行分组
    group = df.groupby('recipename')
    
    # 遍历每个分组，将每个分组的数据保存到 segments 列表
    for recipename, group_df in group:
        # 对每个segment进行处理，计算差值
        for col in ['A_HT_steam_total_blend', 'B_HT_steam_total_blend', 'TT1142_steam_total_blend']:
            # 计算当前列与上一行的差值 (即 diff)
            group_df[f'{col}_diff'] = group_df[col].diff()
            
            # 如果差值小于 0，用当前行的值代替
            group_df[f'{col}_diff'] = group_df.apply(
                lambda row: 0 if pd.isna(row[f'{col}_diff']) or row[f'{col}_diff'] < 0 else row[f'{col}_diff'],
                axis=1
            )
        
        # 将处理后的分段数据加入到 segments 列表
        segments.append(group_df)
    
    return segments

def spc_criteria_check(group_df: pd.DataFrame, item: str, mean: float, std: float, SPC_upper: float, SPC_lower: float) -> dict:
    """
    根据 SPC 判异准则进行检测，返回判异结果。
    
    :param group_df: 需要分析的分段 DataFrame
    :param item: 需要进行统计分析的列名
    :param mean: 计算得到的均值
    :param std: 计算得到的标准差
    :param SPC_upper: 控制线的上限
    :param SPC_lower: 控制线的下限
    :return: 判异结果的字典
    """
    results = {}

    # 1. 1个点落在A区以外
    if (group_df[item] > SPC_upper).any() or (group_df[item] < SPC_lower).any():
        results['A_Zone_Violation'] = True
    else:
        results['A_Zone_Violation'] = False

    # 2. 连续9个点落在中心线的同一侧
    center_line = mean
    consecutive_same_side = ((group_df[item] > center_line).rolling(window=9).sum() == 9) | ((group_df[item] < center_line).rolling(window=9).sum() == 9)
    results['Consecutive_9_Same_Side'] = consecutive_same_side.any()

    # 3. 连续6个点递增或递减
    diff = group_df[item].diff()
    consecutive_increasing_or_decreasing = ((diff > 0).rolling(window=6).sum() == 6) | ((diff < 0).rolling(window=6).sum() == 6)
    results['Consecutive_6_Increasing_Decreasing'] = consecutive_increasing_or_decreasing.any()

    # 4. 连续14个点中相邻点交替上下
    consecutive_alternate = (group_df[item].diff().apply(np.sign) != 0).rolling(window=14).sum() == 14
    results['Consecutive_14_Alternating'] = consecutive_alternate.any()

    # 5. 连续3个点中有2个点落在中心线同一侧的B区以外
    B_zone_upper = mean + 2 * std
    B_zone_lower = mean - 2 * std
    consecutive_b_zone_violation = ((group_df[item] > B_zone_upper) | (group_df[item] < B_zone_lower)).rolling(window=3).sum() >= 2
    results['Consecutive_3_B_Zone_Violation'] = consecutive_b_zone_violation.any()

    # 6. 连续5个点中有4个点落在中心线同一侧的C区以外
    C_zone_upper = mean + 3 * std
    C_zone_lower = mean - 3 * std
    consecutive_c_zone_violation = ((group_df[item] > C_zone_upper) | (group_df[item] < C_zone_lower)).rolling(window=5).sum() >= 4
    results['Consecutive_5_C_Zone_Violation'] = consecutive_c_zone_violation.any()

    # 7. 连续15个点落在中心线两侧的C区以内
    consecutive_c_zone = (group_df[item] > C_zone_lower) & (group_df[item] < C_zone_upper)
    results['Consecutive_15_C_Zone'] = (consecutive_c_zone.rolling(window=15).sum() == 15).any()

    # 8. 连续8个点中没有点落在C区内
    condition = (group_df[item] > C_zone_lower) & (group_df[item] < C_zone_upper)
    # 改进这里的计算，直接检查是否有8个连续点都不在C区内
    results['Consecutive_8_No_C_Zone'] = (~condition.rolling(window=8).sum().gt(0)).any()

    return results

def segments_analysis(df_segments: List[pd.DataFrame], item: str, USL: float, LSL: float) -> pd.DataFrame:
    """
    对按 'recipename' 分段后的 DataFrame 列表进行统计分析，特别是对指定列进行分析。
    计算 SPC 控制线 (mean ± 3 * std)，Cpk 和 Ppk 指标，并根据 SPC判异准则进行检查。
    
    :param df_segments: 按 'recipename' 分段后的 DataFrame 列表
    :param item: 需要进行统计分析的列名
    :param USL: 上规格限 (Upper Specification Limit)
    :param LSL: 下规格限 (Lower Specification Limit)
    :return: 一个包含每个 'recipename' 的指定列统计信息和能力指标的 DataFrame
    """
    stats = []

    # 遍历每个分段 (DataFrame)
    for group_df in df_segments:
        recipename = group_df['recipename'].iloc[0]
        
        if item in group_df.columns:
            group_df_filtered = group_df[group_df[item] != 0].dropna()

            if not group_df_filtered.empty:
                # 获取描述性统计信息
                stats_for_group = group_df_filtered[item].describe().to_frame().T
                mean = stats_for_group['mean'].iloc[0]
                std = stats_for_group['std'].iloc[0]

                sum_val = group_df_filtered[item].sum()
                stats_for_group['sum'] = sum_val
                stats_for_group['std'] = std

                # 计算 SPC 控制线
                SPC_upper = mean + 3 * std
                SPC_lower = mean - 3 * std
                stats_for_group['SPC_upper'] = SPC_upper
                stats_for_group['SPC_lower'] = SPC_lower

                # 计算 Cpk 和 Ppk
                Cpk = min((USL - mean) / (3 * std), (mean - LSL) / (3 * std))
                stats_for_group['Cpk'] = Cpk

                total_mean = np.mean(group_df_filtered[item])
                total_std = np.std(group_df_filtered[item])
                Ppk = min((USL - total_mean) / (3 * total_std), (total_mean - LSL) / (3 * total_std))
                stats_for_group['Ppk'] = Ppk

                # 使用 SPC 判异准则
                spc_results = spc_criteria_check(group_df_filtered, item, mean, std, SPC_upper, SPC_lower)
                for rule, violation in spc_results.items():
                    stats_for_group[rule] = violation

                # 添加 'recipename' 列
                stats_for_group['recipename'] = recipename
                stats.append(stats_for_group)
    
    if stats:
        stats_df = pd.concat(stats)
    else:
        print("No valid data to concatenate.")
        stats_df = pd.DataFrame()  

    return stats_df

def process_files_in_directory(directory: str, output_file: str):
    # 获取目录下所有 CSV 文件
    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    
    # 创建一个空的 DataFrame 用来存储最终结果
    final_result_df = pd.DataFrame()
    
    # 遍历所有 CSV 文件
    for file_path in csv_files:
        # 读取 CSV 文件
        df = NengYuan_read_csv(file_path)
        
        # 获取分段数据
        df_segments_in_recipe = segments_in_recipename(df)
        
        # 分析并得到结果
        result_df = segments_analysis(df_segments_in_recipe, 'A_HT_steam_total_blend_diff', 0.4, 0.1)
        
        # 将结果追加到 final_result_df 中
        final_result_df = pd.concat([final_result_df, result_df], ignore_index=True)
        
        # 如果需要，可以将每个文件的分段数据保存为 Excel 进行检查
        df_segments_in_recipe[0].to_excel(file_path.replace('.csv', '_check_data.xlsx'))
    
    # 输出最终的合并结果到 Excel
    final_result_df.to_excel(output_file, index=False)
    print(f"Processed results saved to {output_file}")

if __name__ == '__main__':
    directory = r"C:\Users\juntaox\Desktop\工作1\3.蒸汽设备能耗监测\原始材料\energy数据\energy"
    csv_files = glob.glob(os.path.join(directory, "*.csv"))
       # 创建一个空的 DataFrame 用来存储最终结果
    final_result_df = pd.DataFrame()
    for csv_file in csv_files:
        df = NengYuan_read_csv(csv_file)
        df_segments_in_recipe = segments_in_recipename(df)
        result_df = segments_analysis(df_segments_in_recipe, 'A_HT_steam_total_blend_diff',0.4 ,0.1)
        # 将每个 result_df 追加到 final_result_df 中
        final_result_df = pd.concat([final_result_df, result_df], ignore_index=True)
    final_result_df.to_excel(r"C:\Users\juntaox\Desktop\工作1\3.蒸汽设备能耗监测\result_df.xlsx", index=False)



