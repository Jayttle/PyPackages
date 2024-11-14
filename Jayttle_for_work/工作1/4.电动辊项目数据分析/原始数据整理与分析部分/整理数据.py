import os
import shutil
import json
import pandas as pd

def copy_files_without_csv(src_folder, dest_folder):
    # 遍历源文件夹中的所有子文件夹
    for subfolder_name in os.listdir(src_folder):
        subfolder_path = os.path.join(src_folder, subfolder_name)

        # 确保该路径是一个文件夹
        if os.path.isdir(subfolder_path):
            # 创建新的父文件夹
            new_parent_folder = os.path.join(dest_folder, subfolder_name)
            os.makedirs(new_parent_folder, exist_ok=True)  # 创建新文件夹，已存在则跳过

            # 遍历子文件夹中的文件
            for file_name in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, file_name)

                # 只复制非 .csv 文件
                if os.path.isfile(file_path) and not file_name.endswith('.csv'):
                    # 构建目标路径
                    dest_file_path = os.path.join(new_parent_folder, file_name)

                    # 复制文件
                    shutil.copy(file_path, dest_file_path)
                    print(f"复制 {file_path} 到 {dest_file_path}")

def load_json_data_from_subfolders(src_folder):
    """
    遍历源文件夹中的所有子文件夹，如果子文件夹中有json文件，
    则将json文件的内容存储到字典中，字典的key为子文件夹的名字，值为json数据。

    :param src_folder: 源文件夹路径
    :return: dict, 存储子文件夹名称和json数据的字典
    """
    json_data_dict = {}

    # 遍历源文件夹中的所有子文件夹
    for subfolder_name in os.listdir(src_folder):
        subfolder_path = os.path.join(src_folder, subfolder_name)

        # 确保该路径是一个文件夹
        if os.path.isdir(subfolder_path):
            # 遍历子文件夹中的文件
            for file_name in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, file_name)

                # 如果是json文件
                if os.path.isfile(file_path) and file_name.endswith('.json'):
                    # 打开并加载json文件
                    try:
                        with open(file_path, 'r', encoding='utf-8') as json_file:
                            json_data = json.load(json_file)
                            # 将json数据存入字典，key为子文件夹名称
                            json_data_dict[subfolder_name] = json_data
                            print(f"加载 {file_path} 到字典")
                    except Exception as e:
                        print(f"无法加载 {file_path}: {e}")

    return json_data_dict

def json_to_dataframe(json_data_dict):
    """
    将json_data_dict转换为pandas DataFrame
    :param json_data_dict: 字典，key为文件夹名称，value为json数据
    :return: pandas DataFrame
    """
    # 存储整理的数据
    data = []

    # 遍历字典并提取所需的数据
    for folder_name, json_data in json_data_dict.items():
        row = {
            'folder_name': folder_name,
            'speed_lower_bound': json_data.get('speed_lower_bound', None),
            'speed_upper_bound': json_data.get('speed_upper_bound', None),
            'temperature_lower_bound': json_data.get('temperature_lower_bound', None),
            'temperature_upper_bound': json_data.get('temperature_upper_bound', None),
            'torque_lower_bound': json_data.get('torque_lower_bound', None),
            'torque_upper_bound': json_data.get('torque_upper_bound', None),
            'motor_speed_median': json_data.get('motor_speed_median', None),
            'motor_speed_max': json_data.get('motor_speed_max', None),
            'motor_speed_mean': json_data.get('motor_speed_mean', None),
            'motor_speed_std': json_data.get('motor_speed_std', None),
            'motor_temperature_median': json_data.get('motor_temperature_median', None),
            'motor_temperature_max': json_data.get('motor_temperature_max', None),
            'motor_temperature_mean': json_data.get('motor_temperature_mean', None),
            'motor_temperature_std': json_data.get('motor_temperature_std', None),
            'motor_torque_median': json_data.get('motor_torque_median', None),
            'motor_torque_max': json_data.get('motor_torque_max', None),
            'motor_torque_mean': json_data.get('motor_torque_mean', None),
            'motor_torque_std': json_data.get('motor_torque_std', None),
        }
        data.append(row)

    # 转换为 DataFrame
    df = pd.DataFrame(data)

    return df

def save_dataframe_to_excel(df, output_path):
    """
    将 DataFrame 保存为 Excel 文件
    :param df: pandas DataFrame
    :param output_path: 输出文件的路径
    """
    df.to_excel(output_path, index=False)
    print(f"数据已保存到 {output_path}")

def delete_empty_subfolders(parent_folder):
    # 遍历指定目录的所有子文件夹
    for root, dirs, files in os.walk(parent_folder, topdown=False):  # topdown=False 保证从子文件夹开始遍历
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            # 如果子文件夹为空，则删除它
            if not os.listdir(dir_path):  # os.listdir() 返回目录下所有文件和文件夹的列表，空目录返回空列表
                print(f"删除空文件夹: {dir_path}")
                os.rmdir(dir_path)  # 删除空文件夹

def 删除非csv文件(root_folder: str):
    # 遍历指定文件夹下的每个子文件夹
    for subfolder in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, subfolder)
        
        # 确保是文件夹
        if os.path.isdir(subfolder_path):
            # 获取子文件夹中的所有文件
            for file_name in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, file_name)
                
                # 只保留 .csv 文件，其他文件删除
                if os.path.isfile(file_path) and not file_name.endswith('.csv'):
                    try:
                        os.remove(file_path)
                        print(f"已删除：{file_path}")
                    except Exception as e:
                        print(f"无法删除 {file_path}: {e}")
    
    print("处理完成！")

if __name__ == "__main__":
    # 示例调用
    src_folder = r'C:\Users\Jayttle\Desktop\电动辊原始数据'  # 源文件夹路径
    dest_folder = r"C:\Users\Jayttle\Desktop\无原始数据电动辊_1112"  # 目标父文件夹路径
    # 执行复制操作
    copy_files_without_csv(src_folder, dest_folder)

    # 调用函数，获取字典
    json_data_dict = load_json_data_from_subfolders(src_folder)
    # 打印字典内容
  # 将字典转换为 DataFrame
    df = json_to_dataframe(json_data_dict)

    # 输出到 Excel 文件
    output_excel_path = os.path.join(dest_folder, '界定范围汇总.xlsx')
    save_dataframe_to_excel(df, output_excel_path)

    # 打印 DataFrame
    print("整理后的数据：")
    print(df)
    delete_empty_subfolders(dest_folder)