import pandas as pd
import logging

# 配置日志
log_file_path = r'E:\vscode_proj\data_analysis.log'
logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', encoding='utf-8')

def read_and_process_excel(file_path: str, date_threshold_str: str) -> pd.DataFrame:
    """
    读取 Excel 文件，进行初步预处理，清洗数据并返回处理后的 DataFrame。
    
    参数:
    - file_path: Excel 文件的路径
    - date_threshold_str: 日期阈值，格式为 'YYYY-MM-DD'
    
    返回:
    - 处理后的 DataFrame
    """
    # 读取 Excel 文件
    df = pd.read_excel(file_path)
    
    # 清洗数据
    df['立户日期'] = pd.to_datetime(df['立户日期'], format='%Y/%m/%d', errors='coerce')
    date_threshold = pd.Timestamp(date_threshold_str)
    
    # 统计大于指定日期的数据量
    count_greater_than_threshold = df[df['立户日期'] > date_threshold].shape[0]
    print(f"大于 {date_threshold_str} 的数据量: {count_greater_than_threshold}")
    
    # 删除 "立户日期" 大于指定阈值的数据行
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
    
    # 计算列的总和并除以10000，保留三位小数
    total_sum = filtered_df[column_name].sum() / 10000
    return round(total_sum, 3)


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

class MonthlyReportolution:
    def __init__(self, file_path: str, analysis_month: str, arg_1tox_newAdd: float, arg_x_newAdd: float, arg_x_monUser_kwh: float, arg_1tox_monUser_kwh: float) -> None:
        self.analysis_month = analysis_month  # 第x月的月报
        self.start_date = f"{analysis_month}-01" # 第x月开始的日期
        self.end_date = f"{analysis_month}-31" #  默认是31号 是该月最后一天
        self.df = read_and_process_excel(file_path, self.end_date) 
        self.arg_1tox_newAdd = arg_1tox_newAdd  # 上海市截止 所有充电桩数量
        self.arg_x_newAdd = arg_x_newAdd # 上海市第x月 新增充电桩
        self.arg_x_monUser_kwh = arg_x_monUser_kwh  # 上海市第x月居民用电量总量
        self.arg_1tox_monUser_kwh = arg_1tox_monUser_kwh # 上海市1—x月居民用电量总量
        self.x_monUser_pct = None # 第x月居民用电量总量 占上海市占比
        self.tox_monUser_pct = None # 1—x月居民用电量总量 占上海市占比
        self.total_sum_current_1 = None # 202404电量 总和 （x月 交流 总和）
        self.total_sum_previous_1 = None  # 202304电量 总和 （去年x月 交流 总和）
        self.yoy_current_vs_previous_1 = None # 202404电量 与 202304 电量同比
        self.total_sum_current_2 = None # 1-x月 202404电量 总和 （1-x月 交流 总和）
        self.total_sum_previous_2 = None # 去年1-x月 202404电量 总和 （去年1-x月 交流 总和）
        self.yoy_current_vs_previous_2 = None # 1-x月  与 去年1-x月 电量同比
        self.x_newAdd = None #  新增充电桩
        self.x_newAdd_pct = None # 新增充电桩 全市占比
        self.tox_newAdd = None # 截至到现在 充电桩数量
        self.tox_newAdd_pct = None # 所有充电桩 全市占比
        # 计算数据
        self.total_Resident_data()
        self.total_notResident_data()
    
    def total_Resident_data(self) -> None:
        '''统计居民的数据'''
        user_type_column = '用户类型'
        user_type_value = '居民'
        
        # 获取 202404电量 和 202304电量 的总和
        column_name_current_1 = '202404电量'
        column_name_previous_1 = '202304电量'
        self.total_sum_current_1 = sum_column_for_user_type(self.df, user_type_column, user_type_value, column_name_current_1)
        self.total_sum_previous_1 = sum_column_for_user_type(self.df, user_type_column, user_type_value, column_name_previous_1)
        self.yoy_current_vs_previous_1 = calculate_year_over_year(self.total_sum_current_1, self.total_sum_previous_1)

        # 获取 202401-05电量 和 202301-05电量 的总和
        column_name_current_2 = '202401-05电量'
        column_name_previous_2 = '202301-05电量'
        self.total_sum_current_2 = sum_column_for_user_type(self.df, user_type_column, user_type_value, column_name_current_2)
        self.total_sum_previous_2 = sum_column_for_user_type(self.df, user_type_column, user_type_value, column_name_previous_2)
        self.yoy_current_vs_previous_2 = calculate_year_over_year(self.total_sum_current_2, self.total_sum_previous_2)
        self.x_monUser_pct = (self.total_sum_current_1 / self.arg_x_monUser_kwh) * 100
        self.tox_monUser_pct = (self.total_sum_current_2 / self.arg_1tox_monUser_kwh) * 100

        mask = (self.df[user_type_column] == user_type_value) & (self.df['立户日期'] >= self.start_date) & (self.df['立户日期'] <= self.end_date)
        self.x_newAdd = self.df.loc[mask].shape[0]
        self.x_newAdd_pct = (self.x_newAdd / self.arg_x_newAdd) * 100
        mask = (self.df[user_type_column] == user_type_value) & (self.df['立户日期'] <= self.end_date)
        self.tox_newAdd = self.df.loc[mask].shape[0]
        self.tox_newAdd_pct = (self.tox_newAdd / self.arg_1tox_newAdd) * 100

    def total_notResident_data(self) -> None:
        """统计非居民的数据"""
        self.num_gongong = 0
        self.num_gongjiao = 0
        self.num_andian = 0

        # 初始化合同容量累积变量
        self.total_gongjiao_capacity = 0
        self.total_andian_capacity = 0
        self.total_gongong_capacity = 0

        # 创建一个筛选条件
        mask = (self.df['用户类型'] != '居民') & \
            ~self.df['用电类别'].str.contains('居民') & \
            (self.df['立户日期'] >= self.start_date) & \
            (self.df['立户日期'] <= self.end_date)
        
        # 筛选数据
        filtered_df = self.df[mask]
        
        # 遍历筛选后的数据
        for index, row in filtered_df.iterrows():
            # 获取合同容量
            capacity = row['合同容量']
            
            # 检查户名是否包含"公共交通"
            if '公共交通' in row['户名']:
                self.num_gongjiao += 1
                self.total_gongjiao_capacity += capacity
            # 如果户名不包含"公共交通"，则检查地址是否包含"岸电"
            elif '岸电' in row['地址']:
                self.num_andian += 1
                self.total_andian_capacity += capacity
            # 如果都不包含，则num_gongong加1
            else:
                self.num_gongong += 1
                self.total_gongong_capacity += capacity
        # 转换单位
        self.total_gongjiao_capacity /= 10000
        self.total_andian_capacity /= 10000
        self.total_gongong_capacity /= 10000
        self.total_num = self.num_gongjiao + self.num_andian + self.num_gongong
        self.total_capacity = self.total_gongjiao_capacity + self.total_andian_capacity + self.total_gongong_capacity
        
        # 第二组统计数据
        self.num_gongong_2 = 0
        self.num_gongjiao_2 = 0
        self.num_andian_2 = 0
        self.total_gongjiao_capacity_2 = 0
        self.total_andian_capacity_2 = 0
        self.total_gongong_capacity_2 = 0

        # 创建一个新的筛选条件
        mask2 = (self.df['用户类型'] != '居民') & \
                ~self.df['用电类别'].str.contains('居民') & \
                ~self.df['电价名称'].str.contains('居民') & \
                (self.df['立户日期'] <= self.end_date)

        # 筛选数据
        filtered_df2 = self.df[mask2]

        # 遍历筛选后的数据
        for index, row in filtered_df2.iterrows():
            # 获取合同容量
            capacity = row['合同容量']
            
            # 检查户名是否包含"公共交通"
            if '公共交通' in row['户名']:
                self.num_gongjiao_2 += 1
                self.total_gongjiao_capacity_2 += capacity
            # 如果户名不包含"公共交通"，则检查地址是否包含"岸电"
            elif '岸电' in row['地址']:
                self.num_andian_2 += 1
                self.total_andian_capacity_2 += capacity
            # 如果都不包含，则num_gongong加1
            else:
                self.num_gongong_2 += 1
                self.total_gongong_capacity_2 += capacity

        # 转换单位
        self.total_gongjiao_capacity_2 /= 10000
        self.total_andian_capacity_2 /= 10000
        self.total_gongong_capacity_2 /= 10000
        self.total_num_2 = self.num_gongjiao_2 + self.num_andian_2 + self.num_gongong_2
        self.total_capacity_2 = self.total_gongjiao_capacity_2 + self.total_andian_capacity_2 + self.total_gongong_capacity_2

    def __str__(self) -> str:
        column_name_current_1 = '202404电量'
        column_name_previous_1 = '202304电量'
        column_name_current_2 = '202401-05电量'
        column_name_previous_2 = '202301-05电量'
        
        result = (
            f"-----------------------------居民数据-----------------------------\n"
            f"------------表 2 浦东新区充换电设施数量情况（单位：个）  ----------- \n"
            f"个人 {self.analysis_month[-2:]}月 新增: {self.x_newAdd}\n"
            f"占全市比重: {self.x_newAdd_pct:.2f}%\n"
            f"个人截至 {self.analysis_month[-2:]}月 共有: {self.tox_newAdd}\n"
            f"占全市比重: {self.tox_newAdd_pct:.2f}%\n"

            f"--------表 3 浦东新区充换电设施的充电量情况（单位：万千瓦时）-------- \n"
            f"个人 {self.analysis_month[-2:]}月 交流: {self.total_sum_current_1}\n"
            f"同比：（{column_name_current_1} vs {column_name_previous_1}）: {self.yoy_current_vs_previous_1:.2f}%\n"
            f"占全市比重： {self.x_monUser_pct:.2f}%\n"
            f"个人 1-{self.analysis_month[-2:]} 交流：{self.total_sum_current_2}\n"
            f"同比：（{column_name_current_2} vs {column_name_previous_2}）: {self.yoy_current_vs_previous_2:.2f}%\n"
            f"占全市比重： {self.tox_monUser_pct:.2f}%\n"

            f"--------表 4 浦东公司单独立户充电站情况（单位：万千瓦）-------- \n"
            f"公共{self.analysis_month[-2:]}月新增 户数:{self.num_gongong}\t容量:{self.total_gongong_capacity}\n"
            f"公交{self.analysis_month[-2:]}月新增 户数:{self.num_gongjiao}\t容量:{self.total_gongjiao_capacity}\n"
            f"岸电{self.analysis_month[-2:]}月新增 户数:{self.num_andian}\t容量:{self.total_andian_capacity}\n"
            f"合计{self.analysis_month[-2:]}月新增 户数:{self.total_num}\t容量:{self.total_capacity}\n"

            f"公共截止{self.analysis_month[-2:]}月 户数:{self.num_gongong_2}\t容量:{self.total_gongong_capacity_2}\n"
            f"公交截止{self.analysis_month[-2:]}月 户数:{self.num_gongjiao_2}\t容量:{self.total_gongjiao_capacity_2}\n"
            f"岸电截止{self.analysis_month[-2:]}月 户数:{self.num_andian_2}\t容量:{self.total_andian_capacity_2}\n"
            f"合计截止{self.analysis_month[-2:]}月 户数:{self.total_num_2}\t容量:{self.total_capacity_2}\n"
        )
        return result
    
if __name__ == '__main__':
    # 统计 7月内居民用户个数
    analysis_month = '2024-07' # 7月份月报
    file_path = r"C:\Users\juntaox\Desktop\浦东月报-数据需求清单 - 202407.xlsx"  # 替换为你的 Excel 文件路径
    arg_1tox_newAdd = 569992  # 上海市 截止充电桩总量
    arg_x_newAdd = 8692  # 上海市 充电桩某月新增
    arg_x_monUser_kwh = 12725.4055  # 上海市第x月居民用电量总量 万kwh
    arg_1tox_monUser_kwh = 80686.2642   # 上海市1-x月居民用电量总量 万kwh
    # 创建 MonthlyReportolution 实例
    monthly_report = MonthlyReportolution(file_path, analysis_month, arg_1tox_newAdd, arg_x_newAdd, arg_x_monUser_kwh, arg_1tox_monUser_kwh)
    
    # 打印结果
    print(monthly_report)
