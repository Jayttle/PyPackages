import os
import pandas as pd

def merge_csv_in_folder(folder_path):
    """
    读取指定文件夹中的所有CSV文件，并将它们合并，
    最后根据 'datetime' 列进行排序，并保存为一个新的CSV文件。

    参数：
    folder_path (str): 目标文件夹的路径。
    """
    # 获取文件夹中所有CSV文件
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    if len(csv_files) == 0:
        print(f"文件夹 '{folder_path}' 中没有CSV文件，跳过。")
        return

    # 读取所有CSV文件并合并
    dfs = []
    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        df = pd.read_csv(file_path)
        dfs.append(df)
    
    # 合并所有CSV文件
    merged_df = pd.concat(dfs, ignore_index=True)

    # 假设'datetime'列存在，确保其是日期时间格式
    merged_df['datetime'] = pd.to_datetime(merged_df['datetime'], errors='coerce')

    # 按照'datetime'列排序
    merged_df = merged_df.sort_values(by='datetime')

    # 生成合并后的文件名
    merged_file_path = os.path.join(folder_path, "4778-1_0901-1204.csv")

    # 将合并后的数据保存到新的CSV文件
    merged_df.to_csv(merged_file_path, index=False)

    print(f"所有CSV文件已合并并根据 'datetime' 列排序，结果保存为 'merged_sorted.csv'")

# 使用示例
folder_path = r'C:\Users\Jayttle\Desktop\电动辊原始数据\4778-1'  # 替换为你的文件夹路径
merge_csv_in_folder(folder_path)
