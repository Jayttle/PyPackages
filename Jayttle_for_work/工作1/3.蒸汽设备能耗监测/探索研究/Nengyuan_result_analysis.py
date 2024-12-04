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

def plt_SPC_UCL_LCL(result_df: pd.DataFrame, item: str, save_path: str = None) -> None:
    # 检查 item 列是否存在
    if item in result_df.columns:
        # 计算 item 列的均值和标准偏差
        mean = result_df[item].mean()
        std_dev = result_df[item].std()
        
        # 计算 SPC 上下限
        UCL = mean + 3 * std_dev  # 上控制限
        LCL = mean - 3 * std_dev  # 下控制限

        print(f"Mean: {mean}, Standard Deviation: {std_dev}")
        print(f"UCL: {UCL}, LCL: {LCL}")

        # 绘制 item 列的折线图
        plt.figure(figsize=(10, 6))  # 设置图形大小
        plt.plot(result_df[item], linestyle='-', color='b', label=item)  # 绘制折线图

        # 绘制上下控制限
        plt.axhline(UCL, color='r', linestyle='--', label=f'UCL ({UCL:.2f})')  # 上控制限
        plt.axhline(LCL, color='r', linestyle='--', label=f'LCL ({LCL:.2f})')  # 下控制限

        # 添加图表标题、标签和图例
        plt.title('Sum Column Line Chart with SPC Control Limits')  # 设置图表标题
        plt.xlabel('Index')  # 设置 x 轴标签
        plt.ylabel('Sum Value')  # 设置 y 轴标签
        plt.grid(True)  # 添加网格
        plt.legend()  # 显示图例
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    else:
        print("item column not found in the DataFrame.")

def plt_SPC_rolling_window(result_df: pd.DataFrame, item: str, window_size: int = 20, save_path: str = None) -> None:
    # 检查 item 列是否存在
    if item in result_df.columns:
        # 使用滚动窗口计算均值和标准偏差
        rolling_mean = result_df[item].rolling(window=window_size).mean()
        rolling_std = result_df[item].rolling(window=window_size).std()
        
        # 计算全局均值和标准偏差
        global_mean = result_df[item].mean()
        global_std = result_df[item].std()
        
        # 计算上下控制限
        UCL = global_mean + 3 * global_std  # 上控制限
        LCL = global_mean - 3 * global_std  # 下控制限

        print(f"Global Mean: {global_mean}, Global Standard Deviation: {global_std}")
        print(f"Global UCL: {UCL}, Global LCL: {LCL}")

        # 绘制滚动窗口均值和标准偏差图
        plt.figure(figsize=(12, 6))  # 设置图形大小

        # 绘制原始数据
        plt.plot(result_df[item], linestyle='-', color='b', label=f'{item} Data')
        
        # 绘制滚动窗口的均值
        plt.plot(rolling_mean, linestyle='-', color='g', label=f'{item} Rolling Mean')
        
        # 绘制滚动窗口的标准偏差
        plt.plot(rolling_mean + 3 * rolling_std, linestyle='--', color='orange', label=f'{item} Rolling UCL')
        plt.plot(rolling_mean - 3 * rolling_std, linestyle='--', color='orange', label=f'{item} Rolling LCL')

        # 绘制全局UCL和LCL
        plt.axhline(UCL, color='r', linestyle='--', label=f'Global UCL ({UCL:.2f})')
        plt.axhline(LCL, color='r', linestyle='--', label=f'Global LCL ({LCL:.2f})')

        # 添加图表标题、标签和图例
        plt.title(f'{item} Rolling Mean, UCL & LCL (Window Size = {window_size})')  # 设置图表标题
        plt.xlabel('Index')  # 设置 x 轴标签
        plt.ylabel(f'{item} Value')  # 设置 y 轴标签
        plt.grid(True)  # 添加网格
        plt.legend()  # 显示图例
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    else:
        print(f"'{item}' column not found in the DataFrame.")

def plot_line_chart(df: pd.DataFrame, item: str, save_path: str = None) -> None:
    """
    绘制给定 DataFrame 中某一列的折线图。

    :param df: pandas DataFrame，包含要绘制的数据。
    :param item: 字符串，指定要绘制的列名。
    """
    if item not in df.columns:
        print(f"Error: The item '{item}' is not found in the DataFrame.")
        return
    # 绘制折线图
    plt.figure(figsize=(10, 6))
    plt.plot(df[item], linestyle='-', color='b', label=item)
    plt.title(f"Line Plot of {item}")
    plt.xlabel("Index")
    plt.ylabel(item)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def analysis_shift_team(df: pd.DataFrame):
    # 确保 shift_unique 和 team_unique 是整数类型
    df['shift_unique'] = df['shift_unique'].astype(int)
    df['team_unique'] = df['team_unique'].astype(int)
    # 创建新的标签列，格式为 'shift_unique-team_unique'
    df['shift-team'] = df['shift_unique'].astype(str) + '-' + df['team_unique'].astype(str)
    # 打印结果查看前几行
    print(df[['shift_unique', 'team_unique', 'shift-team']].head())
    # 进行聚合计算
    grouped = df.groupby('shift-team')

    result = {}
    for module_name, group in grouped:
        # 调用 analysis_df_item 函数对每个分组进行分析，并将结果保存到 result 字典中
        analysis_result = analysis_df_item(group, item='sum')  # 假设你需要分析的列是 'sum'
        
        # 将结果保存到字典，以 module_unique 为键
        result[module_name] = analysis_result
    # 用于存储所有的分析结果
    all_results = []

    # 遍历 result 字典
    for module_name, analysis_result in result.items():
        # 将 analysis_result 的索引设置为对应的 module_name
        analysis_result.index = [module_name]  # 如果是 Series，直接设置索引
        # 将每个修改后的 analysis_result 添加到 all_results 列表中
        all_results.append(analysis_result)

    # 合并所有结果为一个 DataFrame
    final_df = pd.concat(all_results)
    # 打印合并后的 DataFrame
    print("\nMerged DataFrame:")
    print(final_df)
    return final_df

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

def analysis_module(df: pd.DataFrame):
    # 确保 module_unique 是整数类型
    df['module_unique'] = df['module_unique'].astype(str)

    # 筛选出只包含 D19, D22, D28, D38, D98, D54 的数据
    valid_modules = ['D19', 'D22', 'D28', 'D38', 'D98', 'D54']
    df_filtered = df[df['module_unique'].isin(valid_modules)]
    
    # 创建新的标签列，格式为 'module_unique'
    df_filtered['module'] = df_filtered['module_unique'].astype(str)
    
    # 打印结果查看前几行
    print(df_filtered[['module_unique', 'module']].head())
    
    # 初始化一个空字典，用于存储每个 module 的分析结果
    result = {}
    
    # 按 'module_unique' 分组并分别处理每个组
    grouped = df_filtered.groupby('module_unique')
    
    for module_name, group in grouped:
        # 调用 analysis_df_item 函数对每个分组进行分析，并将结果保存到 result 字典中
        analysis_result = analysis_df_item(group, item='sum')  # 假设你需要分析的列是 'sum'
        
        # 将结果保存到字典，以 module_unique 为键
        result[module_name] = analysis_result
    # 用于存储所有的分析结果
    all_results = []

    # 遍历 result 字典
    for module_name, analysis_result in result.items():
        # 将 analysis_result 的索引设置为对应的 module_name
        analysis_result.index = [module_name]  # 如果是 Series，直接设置索引
        # 将每个修改后的 analysis_result 添加到 all_results 列表中
        all_results.append(analysis_result)

    # 合并所有结果为一个 DataFrame
    final_df = pd.concat(all_results)
    # 打印合并后的 DataFrame
    print("\nMerged DataFrame:")
    print(final_df)
    return final_df

def plt_shift_team(df: pd.DataFrame, value_column: str, save_path: str):
    # 确保 shift_unique 和 team_unique 是整数类型
    df['shift_unique'] = df['shift_unique'].astype(int)
    df['team_unique'] = df['team_unique'].astype(int)
    
    # 创建新的标签列，格式为 'shift_unique-team_unique'
    df['shift-team'] = df['shift_unique'].astype(str) + '-' + df['team_unique'].astype(str)
    
    # 打印结果查看前几行
    print(df[['shift_unique', 'team_unique', 'shift-team']].head())
    
    # 获取所有 unique 的 shift-team 值，并按照数字顺序排序
    shift_teams = sorted(df['shift-team'].unique(), key=lambda x: (int(x.split('-')[0]), int(x.split('-')[1])))
    
    # 设置画布大小和 DPI（调整为适合3x3布局的尺寸）
    fig, axes = plt.subplots(3, 3, figsize=(18, 12), dpi=150)
    axes = axes.flatten()  # 将 3x3 的二维数组展平为一维数组，方便迭代
    
    # 设置绘图样式
    sns.set(style="whitegrid")
    
    # 获取 y 轴的统一范围
    y_min, y_max = df[value_column].min(), df[value_column].max()
    
    # 按 shift-team 分组并绘制箱形图
    for i, shift_team in enumerate(shift_teams[:9]):  # 这里只选择前 9 个 shift-team
        # 获取当前 shift-team 对应的子图
        ax = axes[i]
        
        # 过滤出当前 shift-team 的数据
        team_data = df[df['shift-team'] == shift_team]
        
        # 绘制箱形图
        sns.boxplot(x='shift-team', y=value_column, data=team_data, ax=ax, width=0.2, palette='Set2')
        
        # 设置标题和标签
        ax.set_title(f'Boxplot of {value_column} - {shift_team}', loc='center', fontsize=12, fontweight='bold')
        ax.set_xlabel('Shift-Team', fontsize=12, fontweight='bold')
        ax.set_ylabel(value_column, fontsize=12, fontweight='bold')
        
        # 去除 x 轴上的标签和刻度
        ax.set_xticks([])  # 这会去除 x 轴的刻度线和标签
        
        # 统一 y 轴范围
        ax.set_ylim(y_min, y_max)
    
    # 调整布局，避免重叠
    plt.tight_layout()
    
    # 保存图像到指定路径
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plt_module(df: pd.DataFrame, value_column: str, save_path: str):
    # 确保 module_unique 列是字符串类型
    df['module_unique'] = df['module_unique'].astype(str)
    # 定义需要绘制的 module_unique 值
    target_modules = ['D19', 'D22', 'D28', 'D38', 'D98', 'D54']
    
    # 筛选出需要的 module_unique 数据
    df_filtered = df[df['module_unique'].isin(target_modules)]
    
    # 创建画布和 6 个子图（2x3 格局）
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=150)
    axes = axes.flatten()  # 将 2x3 的二维数组展平为一维数组，方便迭代
    
    # 设置绘图样式
    sns.set(style="whitegrid")
    
    # 计算全数据的 y 轴范围
    min_value = df_filtered[value_column].min()
    max_value = df_filtered[value_column].max()

    # 按 module_unique 分组并绘制箱形图
    for i, module in enumerate(target_modules):  # 遍历每个指定的 module_unique
        # 获取当前 module_unique 对应的子图
        ax = axes[i]
        
        # 过滤出当前 module_unique 的数据
        team_data = df_filtered[df_filtered['module_unique'] == module]
        
        # 绘制箱形图
        sns.boxplot(x='module_unique', y=value_column, data=team_data, ax=ax, width=0.2, hue='module_unique', palette='Set2')

        
        # 设置标题和标签
        ax.set_title(f'Boxplot of {module}', loc='center', fontsize=12, fontweight='bold')
        ax.set_xlabel('module', fontsize=12, fontweight='bold')
        ax.set_ylabel(value_column, fontsize=12, fontweight='bold')
        
        # 去除 x 轴上的标签和刻度
        ax.set_xticks([])  # 这会去除 x 轴的刻度线和标签
        
        # 设置统一的 y 轴范围
        ax.set_ylim(min_value, max_value)

    # 调整布局，避免重叠
    plt.tight_layout()
    
    # 保存图像到指定路径
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plt_shift_team_line(df: pd.DataFrame, value_column: str, save_path: str):
    """
    根据 'shift-team' 分组，绘制折线图，按分组的顺序绘制，不按时间排序。
    
    参数:
    df : pd.DataFrame : 输入的 DataFrame，应该包含 'shift-team', 'value_column' 列。
    save_path : str : 保存图像的路径。
    value_column : str : 需要绘制折线的值列名。
    """
    
    # 确保 shift_unique 和 team_unique 是整数类型
    df['shift_unique'] = df['shift_unique'].astype(int)
    df['team_unique'] = df['team_unique'].astype(int)
    
    # 创建新的标签列，格式为 'shift_unique-team_unique'
    df['shift-team'] = df['shift_unique'].astype(str) + '-' + df['team_unique'].astype(str)
    
    # 打印结果查看前几行
    print(df[['shift_unique', 'team_unique', 'shift-team']].head())
    
    # 按照 'shift-team' 列进行分组
    grouped = df.groupby('shift-team')
    
    # 计算所有分组中的 y 值的最小值和最大值
    min_value = df[value_column].min()
    max_value = df[value_column].max()
    
    # 设置画布大小和 DPI
    fig, axes = plt.subplots(3, 3, figsize=(18, 12), dpi=150)
    axes = axes.flatten()  # 将 3x3 的二维数组展平为一维数组，方便迭代
    
    # 设置绘图样式
    sns.set(style="whitegrid")
    
    # 按 shift-team 分组并绘制折线图
    for i, (shift_team, group) in enumerate(grouped):  # 遍历每个分组
        if i >= 9:  # 只绘制前 9 个分组
            break
        
        # 处理当前分组数据，reset_index()
        group_reset = group.reset_index(drop=True)
        
        # 获取当前子图
        ax = axes[i]
        
        # 绘制当前分组的折线图
        sns.lineplot(x=group_reset.index, y=value_column, data=group_reset, ax=ax, marker='o', palette='Set2')
        
        # 设置标题和标签
        ax.set_title(f'Lineplot of {value_column} - {shift_team}', loc='center', fontsize=12, fontweight='bold')
        ax.set_xlabel('Batch Order', fontsize=12, fontweight='bold')
        ax.set_ylabel(value_column, fontsize=12, fontweight='bold')
        
        # 统一 y 轴范围
        ax.set_ylim(min_value, max_value)
    
    # 调整布局，避免重叠
    plt.tight_layout()
    
    # 保存图像到指定路径
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plt_module_line(df: pd.DataFrame,  value_column: str, save_path: str):
    """
    根据指定的 module_unique 列值（例如 D19, D22, D28, D38, D98, D54），
    绘制折线图，并且每个模块对应一个子图，共 6 个子图。
    
    参数:
    df : pd.DataFrame : 输入的 DataFrame，应该包含 'module_unique', 'value_column' 列。
    save_path : str : 保存图像的路径。
    value_column : str : 需要绘制折线的值列名。
    """
    
    # 确保 module_unique 列是字符串类型
    df['module_unique'] = df['module_unique'].astype(str)
    
    # 定义需要绘制的 module_unique 值
    target_modules = ['D19', 'D22', 'D28', 'D38', 'D98', 'D54']
    
    # 筛选出需要的 module_unique 数据
    df_filtered = df[df['module_unique'].isin(target_modules)]
    
    # 创建画布和 6 个子图（2x3 格局）
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=150)
    axes = axes.flatten()  # 将 2x3 的二维数组展平为一维数组，方便迭代
    
    # 设置绘图样式
    sns.set(style="whitegrid")
    
    # 计算所有模块的 y 值的最小值和最大值
    y_min = df_filtered[value_column].min()
    y_max = df_filtered[value_column].max()

    # 按 module_unique 分组并绘制折线图
    for i, module in enumerate(target_modules):  # 遍历每个指定的 module_unique
        group = df_filtered[df_filtered['module_unique'] == module]  # 当前模块的分组数据
                
        # 确保数据框的索引被正确重置
        group_reset = group.reset_index()

        # 获取当前子图
        ax = axes[i]
        
        # 绘制当前模块的折线图
        sns.lineplot(x=group_reset.index, y=group_reset[value_column], data=group_reset, ax=ax, marker='o', palette='Set2')
            
        # 设置标题和标签
        ax.set_title(f'Lineplot of {value_column} - {module}', loc='center', fontsize=12, fontweight='bold')
        ax.set_xlabel('Batch Order', fontsize=12, fontweight='bold')
        ax.set_ylabel(value_column, fontsize=12, fontweight='bold')
        
        # 设置统一的 y 轴范围
        ax.set_ylim(y_min, y_max)
    
    # 调整布局，避免重叠
    plt.tight_layout()
    
    # 保存图像到指定路径
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plt_module_corr_heatmap(df: pd.DataFrame, value_column: str, save_path: str):
    """
    根据 'module_unique' 分组，计算每个分组内的 Pearson 相关性，并绘制相关性热图。
    
    参数:
    df : pd.DataFrame : 输入的 DataFrame，应该包含 'module_unique', 'value_column' 列。
    value_column : str : 需要计算相关性的列名。
    save_path : str : 保存图像的路径。
    """
    
    # 确保 'module_unique' 列是字符串类型
    df['module_unique'] = df['module_unique'].astype(str)
    
    # 定义需要绘制的 module_unique 值
    target_modules = ['D19', 'D22', 'D28', 'D38', 'D98', 'D54']
    
    # 筛选出 module_unique 在 target_modules 中的行
    df_filtered = df[df['module_unique'].isin(target_modules)]
    
    # 按照 'module_unique' 列进行分组
    grouped = df_filtered.groupby('module_unique')[value_column].apply(list)
    
    # 创建一个空的 DataFrame 用于存储相关性矩阵
    corr_matrix = pd.DataFrame(index=grouped.index, columns=grouped.index)
    
    # 计算 Pearson 相关性并填充到矩阵中
    for module_1 in grouped.index:
        for module_2 in grouped.index:
            # 计算每一对 'module_unique' 之间的 Pearson 相关性
            corr_matrix.loc[module_1, module_2] = pd.Series(grouped[module_1]).corr(pd.Series(grouped[module_2]), method='pearson')
    
    # 将相关性矩阵的值转换为数值类型
    corr_matrix = corr_matrix.astype(float)
    
    # 设置画布大小
    plt.figure(figsize=(10, 8))
    
    # 使用 seaborn 绘制相关性热图
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt=".2f", linewidths=0.5, cbar_kws={"shrink": 0.8})
    
    # 设置标题
    plt.title(f'Pearson Correlation Heatmap of {value_column} by module_unique', fontsize=16)
    
    # 保存图像到指定路径
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def segment_shift_team(df: pd.DataFrame, save_path: str):
    # 确保 shift_unique 和 team_unique 是整数类型
    df['shift_unique'] = df['shift_unique'].astype(int)
    df['team_unique'] = df['team_unique'].astype(int)
    
    # 创建新的标签列，格式为 'shift_unique-team_unique'
    df['shift-team'] = df['shift_unique'].astype(str) + '-' + df['team_unique'].astype(str)
    
    # 打印结果查看前几行
    print(df[['shift_unique', 'team_unique', 'shift-team']].head())
    
    # 使用 ExcelWriter 写入多个 sheet
    with pd.ExcelWriter(save_path) as writer:
        # 按照 'shift-team' 分组
        for shift_team, group in df.groupby('shift-team'):
            # 每个分组写入到不同的 sheet，sheet 名称为 shift-team
            group.drop('shift-team', axis=1).to_excel(writer, sheet_name=shift_team, index=False)

def plt_shift_team_corr_heatmap(df: pd.DataFrame, value_column: str, save_path: str):
    """
    根据 'shift-team' 分组，计算每个分组内的 Pearson 相关性，并绘制相关性热图。
    
    参数:
    df : pd.DataFrame : 输入的 DataFrame，应该包含 'shift-team', 'value_column' 列。
    value_column : str : 需要计算相关性的列名。
    save_path : str : 保存图像的路径。
    """
    
    # 确保 'shift-team' 列的类型是字符串，并且该列是由 'shift_unique' 和 'team_unique' 列组合而来
    df['shift-team'] = df['shift_unique'].astype(str) + '-' + df['team_unique'].astype(str)
    
    # 按照 'shift-team' 列进行分组
    grouped = df.groupby('shift-team')[value_column].apply(list)
    
    # 创建一个空的 DataFrame 用于存储相关性矩阵
    corr_matrix = pd.DataFrame(index=grouped.index, columns=grouped.index)
    
    # 计算 Pearson 相关性并填充到矩阵中
    for shift_team_1 in grouped.index:
        for shift_team_2 in grouped.index:
            # 计算每一对 'shift-team' 之间的 Pearson 相关性
            corr_matrix.loc[shift_team_1, shift_team_2] = pd.Series(grouped[shift_team_1]).corr(pd.Series(grouped[shift_team_2]), method='pearson')
    
    # 将相关性矩阵的值转换为数值类型
    corr_matrix = corr_matrix.astype(float)
    
    # 设置画布大小
    plt.figure(figsize=(10, 8))
    
    # 使用 seaborn 绘制相关性热图
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt=".2f", linewidths=0.5, cbar_kws={"shrink": 0.8})
    
    # 设置标题
    plt.title(f'Pearson Correlation Heatmap of {value_column} by shift-team', fontsize=16)
    
    # 保存图像到指定路径
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def run_main_plt(file_path, save_path, item):
    # 读取 Excel 文件
    result_df = pd.read_excel(file_path)
    plt_SPC_UCL_LCL(result_df, 'sum', os.path.join(save_path, f'plt_SPC_UCL_LCL.png'))
    plot_line_chart(result_df, 'sum', os.path.join(save_path, f'plot_line_chart.png'))
    plt_SPC_rolling_window(result_df, 'sum', 20, os.path.join(save_path, f'plt_SPC_rolling_window.png'))
    plt_shift_team(result_df, 'sum', os.path.join(save_path, f'plt_shift_team.png')) 
    plt_shift_team_line(result_df, 'sum', os.path.join(save_path, f'plt_shift_team_line.png'))
    plt_module(result_df,  'sum',  os.path.join(save_path, f'plt_module.png')) 
    plt_module_line(result_df, 'sum',  os.path.join(save_path, f'plt_module_line.png'))
    plt_shift_team_corr_heatmap(result_df, 'sum', os.path.join(save_path, f'plt_shift_team_corr_heatmap.png'))
    plt_module_corr_heatmap(result_df, 'sum', os.path.join(save_path, f'plt_module_corr_heatmap.png'))

    shift_team_analysis_df = analysis_shift_team(result_df)
    module_analysis_df = analysis_module(result_df)
    shift_team_analysis_df.to_excel(os.path.join(save_path, f'shift_team_analysis_df.xlsx'))
    module_analysis_df.to_excel(os.path.join(save_path, f'module_analysis_df.xlsx'))

if __name__ == '__main__':
    # 读取 Excel 文件
    item_list = ['result_df_A', 'result_df_B', 'result_df_TT1142', 'result_df_total']
    for item in item_list:
        # 通过替换字符串中的 'result_df_' 部分，得到 'result_A' 之类的字符串
        result_name = item.replace('result_df_', 'result_')
        file_path = rf"C:\Users\juntaox\Desktop\工作1\3.蒸汽设备能耗监测\xjt-处理\{item}.xlsx"
        save_path = rf"C:\Users\juntaox\Desktop\工作1\3.蒸汽设备能耗监测\xjt-处理\{result_name}"
        run_main_plt(file_path, save_path, 'sum')