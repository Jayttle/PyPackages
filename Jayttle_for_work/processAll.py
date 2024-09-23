import pandas as pd
import logging

# 配置日志
log_file_path = r'E:\vscode_proj\data_analysis.log'
logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', encoding='utf-8')

def read_and_preprocess_excel(file_path: str) -> pd.DataFrame:
    """
    读取 Excel 文件并进行初步预处理。
    """
    df = pd.read_excel(file_path)
    columns_to_drop = ['户名', '地址']
    df = df.drop(columns=columns_to_drop, errors='ignore')
    return df

def clean_data(df: pd.DataFrame, date_threshold_str: str) -> pd.DataFrame:
    """
    清洗数据，确保 "立户日期" 列是日期类型，并删除 "立户日期" 大于指定阈值的数据行。
    """
    df['立户日期'] = pd.to_datetime(df['立户日期'], format='%Y/%m/%d', errors='coerce')
    date_threshold = pd.Timestamp(date_threshold_str)

        
    # 统计大于指定日期的数据量
    count_greater_than_threshold = df[df['立户日期'] > date_threshold].shape[0]
    print(f"大于 {date_threshold_str} 的数据量: {count_greater_than_threshold}")

    df = df[df['立户日期'] <= date_threshold]
    return df

def filter_user_type(df: pd.DataFrame, valid_types: list) -> pd.DataFrame:
    """
    根据有效的用电类别过滤数据。
    """
    return df[df['用户类型'].isin(valid_types)]


def get_unique_values(df: pd.DataFrame, column_name: str) -> pd.Series:
    """
    统计指定列中的唯一值及其频次。
    """
    if column_name in df.columns:
        return df[column_name].value_counts()
    else:
        return pd.Series([f"列 '{column_name}' 不存在"], name=column_name)

def count_records_by_month(df: pd.DataFrame, year: int, month: int) -> pd.DataFrame:
    """
    统计 "立户日期" 列中指定年份和月份的数据量，并返回符合条件的新的 DataFrame。
    """
    if '立户日期' not in df.columns:
        raise ValueError("数据中不包含 '立户日期' 列")
    
    df['立户日期'] = pd.to_datetime(df['立户日期'], format='%Y/%m/%d', errors='coerce')
    mask = (df['立户日期'].dt.year == year) & (df['立户日期'].dt.month == month)
    filtered_df = df[mask]
    print(f"{year}年{month}月的数据量: {filtered_df.shape[0]}")
    return filtered_df

def log_unique_values(df: pd.DataFrame, column_name: str) -> None:
    """
    记录指定列的唯一值及其频次到日志文件中。
    """
    unique_values = get_unique_values(df, column_name)
    logging.info(f"唯一值统计 - {column_name}:")
    for value, count in unique_values.items():
        logging.info(f"{value}: {count}")

def sum_column(df: pd.DataFrame, column_name: str) -> float:
    """
    计算指定列的数值总和。
    """
    if column_name not in df.columns:
        raise ValueError(f"数据中不包含 '{column_name}' 列")
    
    # 使用 .loc 进行赋值操作
    df.loc[:, column_name] = pd.to_numeric(df[column_name], errors='coerce')
    
    # 计算列的总和
    total_sum = df[column_name].sum()
    return total_sum

def sum_column_for_user_type(df: pd.DataFrame, user_type_column: str, user_type_value: str, column_name: str) -> float:
    """
    筛选出 '用户类型' 列中值为指定类型的行，然后计算指定列的数值总和。
    
    :param df: DataFrame 包含数据
    :param user_type_column: '用户类型' 列名
    :param user_type_value: 需要筛选的用户类型值
    :param column_name: 要计算总和的列名
    :return: 计算后的列总和
    """
    if user_type_column not in df.columns:
        raise ValueError(f"数据中不包含 '{user_type_column}' 列")
    if column_name not in df.columns:
        raise ValueError(f"数据中不包含 '{column_name}' 列")
    
    # 筛选出 '用户类型' 列中值为指定类型的行
    filtered_df = df[df[user_type_column] == user_type_value]
    
    # 使用 .loc 进行赋值操作以确保没有 SettingWithCopyWarning
    filtered_df.loc[:, column_name] = pd.to_numeric(filtered_df[column_name], errors='coerce')
    
    # 计算列的总和
    total_sum = filtered_df[column_name].sum()
    return total_sum


def clear_log(log_file_path: str) -> None:
    """
    清空指定日志文件的内容。
    """
    open(log_file_path, 'w').close()

def calculate_year_over_year(current_value: float, previous_value: float) -> float:
    """
    计算同比增长率。
    
    :param current_value: 当前值
    :param previous_value: 去年同月值
    :return: 同比增长率
    """
    if previous_value == 0:
        return float('inf')  # 避免除以零
    return (current_value - previous_value) / previous_value * 100


def main() -> None:
    # clear_log(log_file_path)
    print("------------------------------run------------------------------")
    
    file_path = r"C:\Users\juntaox\Desktop\浦东月报-数据需求清单 - 202407.xlsx"  # 替换为你的 Excel 文件路径
    
    # 读取和预处理数据
    df = read_and_preprocess_excel(file_path)

    # 清洗数据
    date_threshold_str = '2024-07-31'
    df = clean_data(df, date_threshold_str)

    # 显示数据的前几行
    print(f"数据行数: {df.shape[0]}")
    # 示例用法
    user_type_column = '用户类型'
    user_type_value = '居民'

    # 获取 202404电量 和 202304电量 的总和
    column_name_current = '202404电量'
    column_name_previous = '202304电量'
    total_sum_current = sum_column_for_user_type(df, user_type_column, user_type_value, column_name_current)
    total_sum_previous = sum_column_for_user_type(df, user_type_column, user_type_value, column_name_previous)
    yoy_current_vs_previous = calculate_year_over_year(total_sum_current, total_sum_previous)
    print(f"至今充电桩的数据：'{user_type_value}' 类型下 '{column_name_current}' 列的总和: {total_sum_current}")
    print(f"至今充电桩的数据：'{user_type_value}' 类型下 '{column_name_previous}' 列的总和: {total_sum_previous}")
    print(f"同比增长率（{column_name_current} vs {column_name_previous}）: {yoy_current_vs_previous:.2f}%")

    # 获取 202401-05电量 和 202301-05电量 的总和
    column_name_current = '202401-05电量'
    column_name_previous = '202301-05电量'
    total_sum_current = sum_column_for_user_type(df, user_type_column, user_type_value, column_name_current)
    total_sum_previous = sum_column_for_user_type(df, user_type_column, user_type_value, column_name_previous)
    yoy_current_vs_previous = calculate_year_over_year(total_sum_current, total_sum_previous)
    print(f"至今充电桩的数据：'{user_type_value}' 类型下 '{column_name_current}' 列的总和: {total_sum_current}")
    print(f"至今充电桩的数据：'{user_type_value}' 类型下 '{column_name_previous}' 列的总和: {total_sum_previous}")
    print(f"同比增长率（{column_name_current} vs {column_name_previous}）: {yoy_current_vs_previous:.2f}%")

    print("------------------------------只针对2024年7月数据---------------------------------")
    # 示例用法
    year = 2024
    month = 7
    df_filter_202407: pd.DataFrame = count_records_by_month(df, year, month)
    # log_unique_values(df_filter_202407, '用户类型')
    # 示例用法
    column_name = '202404电量'
    total_sum = sum_column(df_filter_202407, column_name)
    print(f"{year}年{month}月： '{column_name}' 列的总和: {total_sum}")

    # 示例用法
    user_type_column = '用户类型'
    user_type_value = '居民'
    column_name = '202404电量'
    total_sum = sum_column_for_user_type(df_filter_202407, user_type_column, user_type_value, column_name)
    print(f"{year}年{month}月：'{user_type_value}' 类型下 '{column_name}' 列的总和: {total_sum}")


    print("------------------------------对非居民用电分析---------------------------------")
    # 过滤用户类型
    valid_types = ['商业用电', '非工业']
    df = filter_user_type(df, valid_types)
    

if __name__ == '__main__':
    main()
