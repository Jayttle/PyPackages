import os
import pandas as pd
import numpy as np
import json
import warnings
# 忽略警告
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in log")

def 读取数据(folder_path):
    # 获取所有 Excel 文件
    # 获取最后一部分路径名称
    folder_name = os.path.basename(folder_path)

    excel_files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx') or f.endswith('.xls')]

    # 创建一个空的 DataFrame 用于合并所有文件的数据
    all_data = pd.DataFrame()

    # 创建一个字典用于保存每个列的 min 和 max
    min_max_values = {}

    # 遍历每一个 Excel 文件
    for file in excel_files:
        # 构造完整的文件路径
        file_path = os.path.join(folder_path, file)
        file_name = os.path.basename(file_path)
        # 去掉 'hour' 前缀
        file_name = file_name.lstrip('hour_df')
        # 去掉 '.xlsx' 后缀
        file_name = file_name.rstrip('.xlsx')
    
        # 读取 Excel 文件
        df = pd.read_excel(file_path)

        # 转换 'hour' 列为日期时间格式，并处理错误
        df['hour'] = pd.to_datetime(df['hour'], errors='coerce')

        # 过滤掉 'hour' 列小于 2024-10-11 00:00:00 的数据
        df = df[df['hour'] > pd.to_datetime('2024-10-11 00:00:00')]

        # 只保留特定的列
        df = df[['hour', 'motor_speed_mean_true', 'motor_temperature_mean_true', 'motor_torque_mean_true']]
        df['type'] = file_name
        columns = ['motor_speed_mean_true', 'motor_temperature_mean_true', 'motor_torque_mean_true']
        
        # 记录每个列的 min 和 max 值
        for col in columns:
            col_min = df[col].min()
            col_max = df[col].max()
            min_max_values[col] = {'min': col_min, 'max': col_max}

        # 进行标准化操作（min-max 标准化）
        df_standardized = df[columns].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

        df_standardized['hour'] = df['hour']
        df_standardized = df_standardized[['hour'] + [col for col in df_standardized.columns if col != 'hour']]

        df_standardized['type'] = df['type']
        df_standardized = df_standardized[['type'] + [col for col in df_standardized.columns if col != 'type']]
        # 去掉索引列
        df_standardized.reset_index(drop=True, inplace=True)

        # 合并当前 DataFrame 到 all_data 中
        all_data = pd.concat([all_data, df_standardized], ignore_index=True)


    # 返回合并后的所有数据和 min-max 信息
    return all_data, min_max_values


def 熵权法(df):
    # 选择需要计算熵权的列
    columns = ['motor_speed_mean_true', 'motor_temperature_mean_true', 'motor_torque_mean_true']
    df_standardized = df[columns]
    # 步骤 2: 计算每列的熵值
    n = df_standardized.shape[0]  # 样本数
    m = df_standardized.shape[1]  # 指标数
    k = 1 / np.log(n)  # 常数
    
    # 计算 p_ij (每个数据点的占比)
    p = df_standardized / df_standardized.sum(axis=0)
    
    # 计算熵值
    entropy = -k * (p * np.log(p)).sum(axis=0)
    
    # 步骤 3: 计算权重
    entropy_weight = (1 - entropy) / (1 - entropy).sum()
    
    # 返回计算的熵权
    return entropy_weight

def TOPSIS(df, weights):
    # 1. 基于均值和标准差标准化数据 (Z-Score 标准化)
    # 2. 加权标准化矩阵
    columns = ['motor_speed_mean_true', 'motor_temperature_mean_true', 'motor_torque_mean_true']
    df_standardized = df[columns]
    weighted_df = df_standardized * weights
    
    # 3. 计算理想解和负理想解
    ideal_solution = weighted_df.max(axis=0)
    negative_ideal_solution = weighted_df.min(axis=0)
    
    # 4. 计算各个方案到理想解和负理想解的距离
    distance_to_ideal = np.sqrt(((weighted_df - ideal_solution) ** 2).sum(axis=1))
    distance_to_negative_ideal = np.sqrt(((weighted_df - negative_ideal_solution) ** 2).sum(axis=1))
    
    # 5. 计算相对接近度
    closeness = distance_to_negative_ideal / (distance_to_ideal + distance_to_negative_ideal)
    
    # 6. 综合评价结果
    df['综合评价得分'] = closeness
    return df, ideal_solution, negative_ideal_solution# 返回按相对接近度排序的结果

if __name__ == '__main__':
    # 文件夹路径
    folder_paths = [r'C:\Users\juntaox\Desktop\excel241113\curve1',
                    r'C:\Users\juntaox\Desktop\excel241113\curve2',
                    r'C:\Users\juntaox\Desktop\excel241113\not15',
                    r'C:\Users\juntaox\Desktop\excel241113\15',
                    ]
    for folder_path in folder_paths:
        df, min_max_values = 读取数据(folder_path)
        # 获取最后一部分路径名称
        folder_name = os.path.basename(folder_path)
        # 示例：获取熵权
        weights = 熵权法(df)

        # 使用计算的权重对数据进行加权
        df_weighted = df[['motor_speed_mean_true', 'motor_temperature_mean_true', 'motor_torque_mean_true']].mul(weights, axis=1)

        # 使用TOPSIS方法进行综合评价
        result, ideal_solution, negative_ideal_solution = TOPSIS(df, weights)
        result.to_excel(rf"C:\Users\juntaox\Desktop\熵权法-TOPSIS\结果-{folder_name}.xlsx")


        # 创建字典来保存数据
        data_to_save = {
            '标准化参数': min_max_values,
            '权值': weights.to_dict(),  # 将 Series 转换为字典格式
            '理想解': ideal_solution.iloc[0],  # 通过位置获取元素
            '负理想解': negative_ideal_solution.iloc[0]  # 通过位置获取元素
        }

        # 输出路径
        output_path = rf"C:\Users\juntaox\Desktop\熵权法-TOPSIS\模型参数{folder_name}.json"

        # 将字典保存为 JSON 文件
        with open(output_path, 'w', encoding='utf-8') as json_file:
            json.dump(data_to_save, json_file, ensure_ascii=False, indent=4)
        # 在输出时保留 8 位小数
        print(f"熵权:[{weights.iloc[0]:.8f}, {weights.iloc[2]:.8f}, {weights.iloc[1]:.8f}] 转速、扭矩、温度")
        print(f"理想解与负理想解：[{ideal_solution.iloc[0]:.8f}, {negative_ideal_solution.iloc[0]:.8f}]")
            
        print(f"结果已保存为 JSON 文件: {output_path}")

