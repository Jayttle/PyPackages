import os
import pandas as pd

def merge_csv_in_subfolders(base_dir):
    """
    遍历指定文件夹（base_dir）中的所有子文件夹，
    将每个子文件夹内的两个CSV文件合并，并将合并后的结果保存在每个子文件夹内。

    参数：
    base_dir (str): 存储子文件夹的根目录路径。
    """
    # 遍历指定文件夹下的所有子文件夹
    for subfolder in os.listdir(base_dir):
        subfolder_path = os.path.join(base_dir, subfolder)

        # 确保是一个文件夹
        if os.path.isdir(subfolder_path):
            # 获取子文件夹中的所有CSV文件
            csv_files = [f for f in os.listdir(subfolder_path) if f.endswith('.csv')]

            if len(csv_files) == 2:
                # 假设子文件夹中只有两个CSV文件
                file1_path = os.path.join(subfolder_path, csv_files[0])
                file2_path = os.path.join(subfolder_path, csv_files[1])

                # 读取两个CSV文件
                df1 = pd.read_csv(file1_path)
                df2 = pd.read_csv(file2_path)

                # 合并两个CSV文件（按行连接）
                merged_df = pd.concat([df1, df2], ignore_index=True)

                # 生成合并后的文件名
                merged_file_path = os.path.join(subfolder_path, "merged.csv")

                # 将合并后的数据保存到子文件夹中的新CSV文件
                merged_df.to_csv(merged_file_path, index=False)

                print(f"已在 '{subfolder_path}' 中合并并保存为 'merged.csv'")
            else:
                print(f"子文件夹 '{subfolder}' 中没有找到两个CSV文件，跳过该文件夹。")
        else:
            print(f"'{subfolder}' 不是一个文件夹，跳过。")

# 使用示例
base_dir = r'C:\Users\Jayttle\Desktop\电动辊原始数据'  # 替换为你的根目录路径
merge_csv_in_subfolders(base_dir)