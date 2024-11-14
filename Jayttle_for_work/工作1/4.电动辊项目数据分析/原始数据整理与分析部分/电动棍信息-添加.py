import os
import shutil
import json
import pandas as pd


if __name__ == "__main__":
    # 示例调用
    file1 = r"C:\Users\Jayttle\Desktop\无原始数据电动辊_1112\界定范围汇总.xlsx"
    file2 = r"C:\Users\Jayttle\Documents\WeChat Files\wxid_uzs67jx3j0a322\FileStorage\File\2024-11\电辊筒信息-供读取.xlsx"
    
    # 读取Excel文件
    df1 = pd.read_excel(file1)
    df2 = pd.read_excel(file2)

    # 确保folder_name和motor_id列存在于df1和df2中
    if 'folder_name' in df1.columns and 'motor_id' in df2.columns:
        # 合并df1和df2，匹配folder_name和motor_id
        df_merged = pd.merge(df1, df2[['motor_id', '类型', '带的无动力辊数量']], 
                             left_on='folder_name', right_on='motor_id', how='left')

        # 对于没有匹配到的行，填充缺失值
        df_merged['类型'].fillna('直道', inplace=True)
        df_merged['带的无动力辊数量'].fillna(15, inplace=True)

        # 删除'motor_id'列
        df_merged.drop('motor_id', axis=1, inplace=True)

        # 显示合并后的结果
        print(df_merged.head())
    else:
        print("df1或df2中缺少必要的列")
    
    # 打印完整的df_merged（如果需要）
    print(df_merged)
    df_merged.to_excel(r"C:\Users\Jayttle\Desktop\无原始数据电动辊_1112\所有辊统计特征汇总.xlsx")