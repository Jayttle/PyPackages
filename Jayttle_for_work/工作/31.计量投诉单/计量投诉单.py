import pandas as pd
def 月份(df: pd.DataFrame, output_file: str, sheet_name: str):
    # 确保 '受理时间' 列为 datetime 类型
    df['受理时间'] = pd.to_datetime(df['受理时间'], errors='coerce')  # 如果转换失败，使用 NaT (缺失值)
    
    # 提取月份信息
    df['month'] = df['受理时间'].dt.month
    
    # 按月份统计个数
    month_counts = df['month'].value_counts().sort_index()  # 统计各个月份的个数，并按月份排序
    
    # 计算总行数
    total_count = len(df)
    
    # 计算每个月的占比
    month_percentage = round(month_counts / total_count * 100, 2)
    
    # 创建一个新的 DataFrame 来保存统计结果
    result_df = pd.DataFrame({
        '月份': month_counts.index,
        '个数': month_counts.values,
        '占比（%）': month_percentage
    })
    
    # 输出结果到指定的 Excel 文件和工作表
    with pd.ExcelWriter(output_file, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
        result_df.to_excel(writer, sheet_name=sheet_name, index=False)
    
def 单位名称(df: pd.DataFrame, output_file: str, sheet_name: str):
    # 按月份统计个数
    dw_counts = df['单位名称'].value_counts().sort_index()  # 统计各个月份的个数，并按月份排序
    
    # 计算总行数
    total_count = len(df)
    
    # 计算每个月的占比
    dw_percentage = round(dw_counts / total_count * 100, 2)
    
    # 创建一个新的 DataFrame 来保存统计结果
    result_df = pd.DataFrame({
        '索引': dw_counts.index,
        '个数': dw_counts.values,
        '占比（%）': dw_percentage
    })
    
    # 输出结果到指定的 Excel 文件和工作表
    with pd.ExcelWriter(output_file, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
        result_df.to_excel(writer, sheet_name=sheet_name, index=False)
    

def 指定列(df: pd.DataFrame, output_file: str, column: str):
    # 按月份统计个数
    dw_counts = df[column].value_counts().sort_index()  # 统计各个月份的个数，并按月份排序
    
    # 计算总行数
    total_count = len(df)
    
    # 计算每个月的占比
    dw_percentage = round(dw_counts / total_count * 100, 2)
    
    # 创建一个新的 DataFrame 来保存统计结果
    result_df = pd.DataFrame({
        '索引': dw_counts.index,
        '个数': dw_counts.values,
        '占比（%）': dw_percentage
    })
    
    # 输出结果到指定的 Excel 文件和工作表
    with pd.ExcelWriter(output_file, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
        result_df.to_excel(writer, sheet_name=column, index=False)


def 二级分类(df: pd.DataFrame, output_file: str, item: str):
    # 筛选出 '二级分类' 为 '计量装置安装' 的数据
    df_filtered = df[df['二级分类'] == item]
    
    # 计算 '三级分类' 的种类和个数
    category_counts = df_filtered['三级分类'].value_counts()  # 统计各个三级分类的个数
    
    # 计算总行数
    total_count = len(df_filtered)
    
    # 计算每个三级分类的占比
    category_percentage = round(category_counts / total_count * 100, 2)
    
    # 创建一个新的 DataFrame 来保存统计结果
    result_df = pd.DataFrame({
        '三级分类': category_counts.index,
        '个数': category_counts.values,
        '占比（%）': category_percentage.values
    })
    
    # 输出结果到指定的 Excel 文件和工作表
    with pd.ExcelWriter(output_file, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
        result_df.to_excel(writer, sheet_name=item, index=False)

    # 打印确认信息
    print(f"\n结果已保存到 {output_file} 的 '{item}' 工作表。")
    print("\n三级分类个数：")
    print(category_counts)  # 打印三级分类的个数
    
    print("\n三级分类占比（%）：")
    print(category_percentage)  # 打印三级分类的占比
    print("\n筛选后数据总行数：", total_count)

if __name__ == '__main__':
    print('-----------------------running:-----------------------')
    
    # 读取Excel文件
    df = pd.read_excel(r"C:\Users\juntaox\Desktop\工作\32.电能计量诉求意见\国网95598-电能计量诉求意见类0101-1124.xlsx")
    out_file = r"C:\Users\juntaox\Desktop\工作\32.电能计量诉求意见\电能计量诉求分析结果.xlsx"
    月份(df, out_file, '月份')
    指定列(df, out_file, '单位名称')
    指定列(df, out_file, '三级分类')
    指定列(df, out_file, '二级分类')
    unique_values = df['二级分类'].unique()

    for item in unique_values:
        二级分类(df, out_file, item)
    

