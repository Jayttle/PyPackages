import pandas as pd
import os

def read_and_save_excel(file_path, output_path):
    """
    读取指定的 Excel 文件并将内容保存到一个新的 Excel 文件
    :param file_path: 输入 Excel 文件的路径
    :param output_path: 输出 Excel 文件的路径
    """
    # 读取 Excel 文件
    df = pd.read_excel(file_path, sheet_name=None)  # sheet_name=None 会读取所有的 sheet

    # 输出每个 sheet 的内容（也可以根据需求输出特定 sheet）
    for sheet_name, sheet_data in df.items():
        print(f"Sheet name: {sheet_name}")
        print(sheet_data.head())  # 只显示前五行数据

    # 将读取到的数据保存到新的 Excel 文件
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for sheet_name, sheet_data in df.items():
            sheet_data.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"文件已保存到：{output_path}")


if __name__ == '__main__':
    # 获取当前工作目录
    current_directory = os.getcwd()

    # 定义文件路径为当前目录下的 Excel 文件
    file_path = os.path.join(current_directory, r"C:\Users\juntaox\Desktop\工作\23.浦东四大重点区域负荷分析\原始材料\浦东新区重点经济圈用电需求专题研究10.28-修改.xlsx")

    # 定义输出的 Excel 文件路径
    output_path = os.path.join(current_directory, 'output_excel.xlsx')

    # 读取 Excel 文件并保存其内容
    read_and_save_excel(file_path, output_path)