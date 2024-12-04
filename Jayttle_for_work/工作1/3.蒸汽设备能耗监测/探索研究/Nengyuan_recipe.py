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
    并计算三列差值的总和 'steam_total_blend_diff'。
    为每个分组添加一个新的字段 'recipename_label'，表示按顺序排列的 'recipename' 序号。

    :param df: 要分段的 DataFrame，必须包含 'recipename' 列
    :return: 按 'recipename' 分段后的 DataFrame 列表
    """
    # 用来存储每个分段
    segments = []
    
    # 根据 'recipename' 列进行分组
    group = df.groupby('recipename')
    
    # 遍历每个分组，将每个分组的数据保存到 segments 列表
    for recipename, group_df in group:
        # 给每个分组添加 'recipename_label'，表示是按顺序排列的递增数字
        group_df['recipename_label'] = group.ngroup().loc[group_df.index]
        
        # 对每个segment进行处理，计算差值
        for col in ['A_HT_steam_total_blend', 'B_HT_steam_total_blend', 'TT1142_steam_total_blend']:
            # 计算当前列与上一行的差值 (即 diff)
            group_df[f'{col}_diff'] = group_df[col].diff()
            
            # 如果差值小于 0，用当前行的值代替
            group_df[f'{col}_diff'] = group_df.apply(
                lambda row: 0 if pd.isna(row[f'{col}_diff']) or row[f'{col}_diff'] < 0 else row[f'{col}_diff'],
                axis=1
            )
        
        # 计算 'steam_total_blend_diff'，将三个差值列相加
        group_df['steam_total_blend_diff'] = (
            group_df['A_HT_steam_total_blend_diff'] +
            group_df['B_HT_steam_total_blend_diff'] +
            group_df['TT1142_steam_total_blend_diff']
        )
        
        # 将处理后的分段数据加入到 segments 列表
        segments.append(group_df)
    
    return segments

def spc_criteria_check(group_df: pd.DataFrame, item: str, mean: float, std: float) -> dict:
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

    A_zone_upper = mean + 3 * std
    A_zone_lower = mean - 3 * std
    # 1. 1个点落在A区以外
    if (group_df[item] > A_zone_upper).any() or (group_df[item] < A_zone_lower).any():
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
    C_zone_upper = mean + std  
    C_zone_lower = mean - std  
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

def segments_analysis(df_segments: List[pd.DataFrame], item: str) -> pd.DataFrame:
    """
    对按 'recipename' 分段后的 DataFrame 列表进行统计分析，特别是对指定列进行分析。
    计算 SPC 控制线 (mean ± 3 * std)，Cpk, CPU, CPL 和 Ppk 指标，并根据 SPC判异准则进行检查。
    
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
                q1 = stats_for_group['25%'].iloc[0]
                q2 = stats_for_group['50%'].iloc[0]
                q3 = stats_for_group['75%'].iloc[0]

                sum_val = group_df_filtered[item].sum()
                stats_for_group['sum'] = sum_val
                stats_for_group['std'] = std

                # 计算变异系数 (CV)
                CV = std / mean if mean != 0 else np.nan
                stats_for_group['CV'] = CV

                # 计算 SPC 控制线
                SPC_upper = mean + 3 * std
                SPC_lower = mean - 3 * std
                stats_for_group['SPC_upper'] = SPC_upper
                stats_for_group['SPC_lower'] = SPC_lower


                # 计算四分位数间距 (IQR)
                IQR = q3 - q1
                stats_for_group['IQR'] = IQR

                USL = q3 + 1.5 * IQR
                LSL = q1 - 1.5 * IQR
                # 计算 Cp (过程能力指数)
                Cp = (USL - LSL) / (6 * std) if std != 0 else np.nan
                stats_for_group['Cp'] = Cp

                # 计算 CPU 和 CPL
                CPU = (USL - mean) / (3 * std) if std != 0 else np.nan
                CPL = (mean - LSL) / (3 * std) if std != 0 else np.nan
                stats_for_group['CPU'] = CPU
                stats_for_group['CPL'] = CPL
                # 计算 Cpk
                Cpk = min(CPU, CPL)
                stats_for_group['Cpk'] = Cpk
                # 计算 Ppk
                total_mean = np.mean(group_df_filtered[item])
                total_std = np.std(group_df_filtered[item])
                Pp = (USL - LSL) / (6 * total_std) if total_std != 0 else np.nan
                stats_for_group['Pp'] = Pp

                Ppk = min((USL - total_mean) / (3 * total_std), (total_mean - LSL) / (3 * total_std))
                stats_for_group['Ppk'] = Ppk



                # 使用 SPC 判异准则
                spc_results = spc_criteria_check(group_df_filtered, item, mean, std)
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

def analysis_df_item(df: pd.DataFrame, item: str) -> pd.DataFrame:
    """
    对按 'recipename' 分段后的 DataFrame 列表进行统计分析，特别是对指定列进行分析。
    计算 SPC 控制线 (mean ± 3 * std)，Cpk, CPU, CPL 和 Ppk 指标，并根据 SPC判异准则进行检查。
    
    :param df_segments: 按 'recipename' 分段后的 DataFrame 列表
    :param item: 需要进行统计分析的列名
    :param USL: 上规格限 (Upper Specification Limit)
    :param LSL: 下规格限 (Lower Specification Limit)
    :return: 一个包含每个 'recipename' 的指定列统计信息和能力指标的 DataFrame
    """
    stats = []
    stats_for_group = df[item].describe().to_frame().T
    mean = stats_for_group['mean'].iloc[0]
    std = stats_for_group['std'].iloc[0]
    q1 = stats_for_group['25%'].iloc[0]
    q2 = stats_for_group['50%'].iloc[0]
    q3 = stats_for_group['75%'].iloc[0]

    sum_val = df[item].sum()
    stats_for_group['sum'] = sum_val
    stats_for_group['std'] = std

    # 计算变异系数 (CV)
    CV = std / mean if mean != 0 else np.nan
    stats_for_group['CV'] = CV

    # 计算 SPC 控制线
    SPC_upper = mean + 3 * std
    SPC_lower = mean - 3 * std
    stats_for_group['SPC_upper'] = SPC_upper
    stats_for_group['SPC_lower'] = SPC_lower


    # 计算四分位数间距 (IQR)
    IQR = q3 - q1
    stats_for_group['IQR'] = IQR

    USL = q2 + 1.5 * IQR
    LSL = q2 - 1.5 * IQR
    # 计算 Cp (过程能力指数)
    Cp = (USL - LSL) / (6 * std) if std != 0 else np.nan
    stats_for_group['Cp'] = Cp

    # 计算 CPU 和 CPL
    CPU = (USL - mean) / (3 * std) if std != 0 else np.nan
    CPL = (mean - LSL) / (3 * std) if std != 0 else np.nan
    stats_for_group['CPU'] = CPU
    stats_for_group['CPL'] = CPL
    # 计算 Cpk
    Cpk = min(CPU, CPL)
    stats_for_group['Cpk'] = Cpk
    # 计算 Ppk
    total_mean = np.mean(df[item])
    total_std = np.std(df[item])
    Pp = (USL - LSL) / (6 * total_std) if total_std != 0 else np.nan
    stats_for_group['Pp'] = Pp

    Ppk = min((USL - total_mean) / (3 * total_std), (total_mean - LSL) / (3 * total_std))
    stats_for_group['Ppk'] = Ppk
    # 使用 SPC 判异准则
    spc_results = spc_criteria_check(df, item, mean, std)
    for rule, violation in spc_results.items():
        stats_for_group[rule] = violation
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
    
def check_segments(segments: List[pd.DataFrame]) -> List[dict]:
    """
    分析每个分段中的 'recipename'、'module'、'module_task'、'no_gap'、'A_HT_phase'、'B_HT_phase'、
    'C_HT_phase'、'shift' 和 'team' 列的唯一值，并添加 datetime 的最小值、最大值，以及它们之间的持续时间。
    仅分析 'steam_total_blend_diff' 不等于 0 的记录。

    :param segments: 由 segments_in_recipename 函数返回的分段列表
    :return: 一个列表，包含每个分段的分析结果，每个元素是一个字典，字典中包含各列的唯一值以及时间分析结果
    """
    analysis_results = []

    # 遍历每个分段
    for segment_df in segments:
        # 筛选出 'steam_total_blend_diff' 不等于 0 的记录
        segment_df = segment_df[segment_df['steam_total_blend_diff'] != 0]
        
        # 如果筛选后的 DataFrame 为空，则跳过该分段
        if segment_df.empty:
            continue
        
        # 提取每个分段的 'recipename'、'module'、'module_task'、'no_gap'、'A_HT_phase'、'B_HT_phase'、'C_HT_phase'、'shift' 和 'team' 的唯一值
        recipename_unique = segment_df['recipename'].unique().tolist()
        module_unique = segment_df['module'].unique().tolist()
        module_task_unique = segment_df['module_task'].unique().tolist()
        no_gap_unique = segment_df['no_gap'].unique().tolist()
        
        # 新增对 'A_HT_phase'、'B_HT_phase'、'C_HT_phase'、'shift' 和 'team' 列的唯一值提取
        # 对 'A_HT_phase'、'B_HT_phase'、'TT1142_HT_phase' 列进行类型转换为 int
        A_HT_phase_unique = segment_df['A_HT_phase'].astype(int).unique().tolist()
        B_HT_phase_unique = segment_df['B_HT_phase'].astype(int).unique().tolist()
        TT1142_phase_unique = segment_df['TT1142_phase'].astype(int).unique().tolist()
        shift_unique = segment_df['shift'].astype(int).unique().tolist()
        team_unique = segment_df['team'].astype(int).unique().tolist()

        # 假设 'datetime' 列存在，并且已经是 datetime 类型，如果不是需要转化
        if 'datetime' in segment_df.columns:
            segment_df.loc[:, 'datetime'] = pd.to_datetime(segment_df['datetime'])
            
            # 计算 datetime 的最小值和最大值
            datetime_min = segment_df['datetime'].min()
            datetime_max = segment_df['datetime'].max()
            
            # 计算时间差并转换为 hh:mm:ss 格式
            time_difference = datetime_max - datetime_min
            time_duration = str(time_difference)  # timedelta 转字符串，格式为 "days hh:mm:ss"
            
            # 提取出小时、分钟和秒
            hours, remainder = divmod(time_difference.seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            time_duration_hms = f"{hours:02}:{minutes:02}:{seconds:02}"
        else:
            # 如果没有 'datetime' 列，返回 None 或其他默认值
            datetime_min = None
            datetime_max = None
            time_duration_hms = None

        # 将结果保存到字典中
        analysis_results.append({
            'recipename_unique': recipename_unique,
            'module_unique': module_unique,
            'module_task_unique': module_task_unique,
            'no_gap_unique': no_gap_unique,
            'A_HT_phase_unique': A_HT_phase_unique,
            'B_HT_phase_unique': B_HT_phase_unique,
            'TT1142_phase_unique': TT1142_phase_unique,
            'shift_unique': shift_unique,
            'team_unique': team_unique,
            'datetime_min': datetime_min,
            'datetime_max': datetime_max,
            'time_duration_hms': time_duration_hms
        })

    return analysis_results

def remove_brackets(value):
    if isinstance(value, list):  # 如果是列表
        return ', '.join(map(str, value))  # 将列表转换为字符串，元素之间用逗号分隔
    else:
        return value  # 如果不是列表，直接返回原值
    

def calculate_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    对每个分段的数据进行统计分析，计算所需的指标。

    :param df: 每个分段的 DataFrame，包含 'recipename', 'start_time', 'end_time', 和一些数值列
    :return: 计算后的统计信息，作为一个 DataFrame
    """
    # 确保 'datetime' 列为时间戳类型
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # 计算持续时间
    duration = (df['datetime'].max() - df['datetime'].min()).total_seconds()
    
    # 对数值列计算统计量：mean, std, max, min, total
    summary_stats = df[['A_HT_steam_total_blend', 'B_HT_steam_total_blend', 'TT1142_steam_total_blend']].agg(
        ['mean', 'std', 'max', 'min', 'sum']
    ).T
    
    # 计算 CV (变异系数) = std / mean
    summary_stats['CV'] = summary_stats['std'] / summary_stats['mean']
    
    # 假设 'cpk' 和 'ppk' 计算需要特定公式，这里仅做示意
    summary_stats['cpk'] = np.nan  # 这里假设你已有 cpk 计算公式
    summary_stats['ppk'] = np.nan  # 这里假设你已有 ppk 计算公式
    
    # 将分段的 'recipename', 'module', 'phase' 等字段提取出来
    # 这些字段可能是数据的一部分，或者可以通过数据推断出（例如 module 可能与 recipe 相关）
    result = {
        'recipename': df['recipename'].iloc[0],
        'module': df['module'].iloc[0],  # 假设这些列已存在
        'target': df['target'].iloc[0],  # 假设 target 列已存在
        'start_time': df['datetime'].min(),
        'end_time': df['datetime'].max(),
        'duration': duration
    }
    
    # 将所有统计数据拼接成一个新的 DataFrame
    result.update(summary_stats.iloc[0].to_dict())
    
    return pd.DataFrame([result])

def run_main_0():
    """
    对原数据NY整理成result_df.xlsx 供Nengyuan_result_analysis.py处理
    """
    directory = r"C:\Users\juntaox\Desktop\工作1\3.蒸汽设备能耗监测\原始材料\energy数据\energy"
    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    
    # 创建一个空的 DataFrame 用来存储最终结果
    final_result_df = pd.DataFrame()
    
    for csv_file in csv_files:
        # 读取数据
        df = NengYuan_read_csv(csv_file)
        
        # 获取配方分段数据
        df_segments_in_recipe = segments_in_recipename(df)
        
        # 分析配方数据
        result_df = segments_analysis(df_segments_in_recipe, 'steam_total_blend_diff')
        
        # 检查配方数据中的元素
        analysis_results = check_segments(df_segments_in_recipe)
        
        # 清理数据
        df_check = pd.DataFrame(analysis_results)
        df_check_cleaned = df_check.map(remove_brackets)
        
        # 试图合并 df_check_cleaned 和 result_df，缺少 recipename 列时跳过当前循环
        try:
            merged_df = pd.merge(df_check_cleaned, result_df, left_on='recipename_unique', right_on='recipename', how='left')
            # 合并成功，追加到最终结果
            final_result_df = pd.concat([final_result_df, merged_df], ignore_index=True)
        except KeyError:
            # 如果找不到 recipename 列，跳过当前文件
            print(f"Warning: Missing 'recipename' column in {csv_file}. Skipping this file.")
            continue
    
    # 保存最终结果到 Excel
    final_result_df.to_excel(r"C:\Users\juntaox\Desktop\工作1\3.蒸汽设备能耗监测\xjt-处理\result_df_total.xlsx", index=False)
    stats_df = analysis_df_item(final_result_df, 'sum')
    stats_df.to_excel(r"C:\Users\juntaox\Desktop\工作1\3.蒸汽设备能耗监测\xjt-处理\stats_df.xlsx", index=False)

def run_main_1():
    """
    对原数据NY分割成NYrecipe
    """
    directory = r"C:\Users\juntaox\Desktop\工作1\3.蒸汽设备能耗监测\原始材料\energy数据\energy"
    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    # 创建一个空的 DataFrame 用来存储最终结果
    final_result_df = pd.DataFrame()
    for csv_file in csv_files:
        # print(csv_file)
        # print(os.path.dirname(csv_file))
        df = NengYuan_read_csv(csv_file)
        df_segments_in_recipe = segments_in_recipename(df)
        # 提取文件名（不包括路径和扩展名）
        filename = os.path.splitext(os.path.basename(csv_file))[0]
        # 提取日期部分
        date_part = "_".join(filename.split("_")[1:3])  # 获取第二和第三部分，形成日期
        excel_file_path = os.path.join(r'C:\Users\juntaox\Desktop\工作1\3.蒸汽设备能耗监测\原始材料\energy数据\NYrecipe', f"NYrecipe_{date_part}.xlsx")
        with pd.ExcelWriter(excel_file_path) as writer:
            for df_seg in df_segments_in_recipe:
                recipename = df_seg['recipename'].iloc[0] # 获取对应的 recipename
                df_seg.to_excel(writer, sheet_name=recipename, index=False)

def run_main_2():
    """
    对原数据 NY 分割成 recipe, module, phase 等列，并统计相关指标。
    """
    directory = r"C:\Users\juntaox\Desktop\工作1\3.蒸汽设备能耗监测\原始材料\energy数据\energy"
    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    
    # 创建一个空的 DataFrame 用来存储最终结果
    final_result_df = pd.DataFrame()
    
    for csv_file in csv_files:
        df = NengYuan_read_csv(csv_file)
        df_segments_in_recipe = segments_in_recipename(df)
        
        # 计算每个分段的统计数据
        for segment in df_segments_in_recipe:
            stats_df = calculate_statistics(segment)
            final_result_df = pd.concat([final_result_df, stats_df], ignore_index=True)
    
    # 将最终的统计结果保存到文件
    final_result_df.to_csv("final_result.csv", index=False)

if __name__ == '__main__':
    print('-------------------------run-------------------------')
    run_main_2()
