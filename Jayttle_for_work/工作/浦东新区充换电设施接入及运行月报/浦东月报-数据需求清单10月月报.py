from docx import Document
import pandas as pd
from datetime import datetime, timedelta
import calendar

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
    def __init__(self, file_path: str, file_path2: str = None, docx_path: str = None, analysis_month: str = None) -> None:
        self.analysis_month = analysis_month  # 第x月的月报
        self.start_date = f"{analysis_month}-01" # 第x月开始的日期
        # 获取该月的最后一天
        # 解析年月
        year, month = map(int, analysis_month.split('-'))
        last_day = calendar.monthrange(year, month)[1]
        self.end_date = f"{analysis_month}-{last_day}"  # 该月最后一天
        self.df = self.read_and_process_excel(file_path, self.end_date) 
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
        self.arg_1tox_newAdd, self.arg_x_newAdd, self.arg_1tox_monUser_kwh, self.arg_x_monUser_kwh = self.extract_values_from_sheet2(file_path)
        self.total_Resident_data()
        self.total_notResident_data()
        self.MCP = MonthlyChargePile(file_path2, analysis_month)
        self.df1, self.df2 = self.read_two_tables_from_word(docx_path)
        self.df_covert2_list_calculate(self.df1)
        self.df_covert2_list_calculate2(self.df2)
        
    def read_and_process_excel(self, file_path: str, date_threshold_str: str) -> pd.DataFrame:
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
    
    def sum_column_for_user_type(self, df: pd.DataFrame, user_type_column: str, user_type_value: str, column_name: str) -> float:
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
    

    def total_Resident_data(self) -> None:
        '''统计居民的数据'''
        user_type_column = '用户类型'
        user_type_value = '居民'
        last_four_headers = self.df.columns[-4:]
        # 获取 202404电量 和 202304电量 的总和
        column_name_current_1 = last_four_headers[0]
        column_name_previous_1 = last_four_headers[1]
        # 获取 202401-05电量 和 202301-05电量 的总和
        column_name_current_2 = last_four_headers[2]
        column_name_previous_2 = last_four_headers[3]

        self.total_sum_current_1 = self.sum_column_for_user_type(self.df, user_type_column, user_type_value, column_name_current_1)
        self.total_sum_previous_1 = self.sum_column_for_user_type(self.df, user_type_column, user_type_value, column_name_previous_1)
        self.yoy_current_vs_previous_1 = calculate_year_over_year(self.total_sum_current_1, self.total_sum_previous_1)


        self.total_sum_current_2 = self.sum_column_for_user_type(self.df, user_type_column, user_type_value, column_name_current_2)
        self.total_sum_previous_2 = self.sum_column_for_user_type(self.df, user_type_column, user_type_value, column_name_previous_2)
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
            ~self.df['电价名称'].str.contains('居民') & \
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
        self.total_gongjiao_capacity = round(self.total_gongjiao_capacity / 10000, 2)
        self.total_andian_capacity = round(self.total_andian_capacity / 10000, 2)
        self.total_gongong_capacity = round(self.total_gongong_capacity / 10000, 2)

        self.total_num = round(self.num_gongjiao + self.num_andian + self.num_gongong, 2)
        self.total_capacity = round(self.total_gongjiao_capacity + self.total_andian_capacity + self.total_gongong_capacity, 2)
        
        # 第二组统计数据
        self.num_gongong_2 = 0
        self.num_gongjiao_2 = 0
        self.num_andian_2 = 0
        self.total_gongjiao_capacity_2 = 0
        self.total_andian_capacity_2 = 0
        self.total_gongong_capacity_2 = 0

        # 创建一个新的筛选条件
        mask2 = (self.df['用户类型'] != '居民') & \
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
        self.total_gongjiao_capacity_2 = round(self.total_gongjiao_capacity_2 / 10000, 2)
        self.total_andian_capacity_2 = round(self.total_andian_capacity_2 / 10000, 2)
        self.total_gongong_capacity_2 = round(self.total_gongong_capacity_2 / 10000, 2)

        self.total_num_2 = round(self.num_gongjiao_2 + self.num_andian_2 + self.num_gongong_2, 2)
        self.total_capacity_2 = round(self.total_gongjiao_capacity_2 + self.total_andian_capacity_2 + self.total_gongong_capacity_2, 2)

    def __str__(self) -> str:
        last_four_headers = self.df.columns[-4:]
        # 获取 202404电量 和 202304电量 的总和
        column_name_current_1 = last_four_headers[0]
        column_name_previous_1 = last_four_headers[1]
        # 获取 202401-05电量 和 202301-05电量 的总和
        column_name_current_2 = last_four_headers[2]
        column_name_previous_2 = last_four_headers[3]
        str_month = self.analysis_month[-1] if self.analysis_month[-2] == '0' else self.analysis_month[-2:]
        
        result = (
            f"\n2024年{str_month}月新增单独立户充电站{self.total_num}个，新增容量{self.total_capacity}万千瓦。截至2024年{str_month}月，浦东单独立户充电站{self.total_num_2}个，容量{self.total_capacity_2}万千瓦。\n"
            f"\n--------表 4 浦东公司单独立户充电站情况（单位：万千瓦）-------- \n"
            f"公共{self.analysis_month[-2:]}月新增 户数:{self.num_gongong}\t容量:{self.total_gongong_capacity}\t截止 户数:{self.num_gongong_2}\t容量:{self.total_gongong_capacity_2}\n"
            f"公交{self.analysis_month[-2:]}月新增 户数:{self.num_gongjiao}\t容量:{self.total_gongjiao_capacity}\t截止 户数:{self.num_gongjiao_2}\t容量:{self.total_gongjiao_capacity_2}\n"
            f"岸电{self.analysis_month[-2:]}月新增 户数:{self.num_andian}\t容量:{self.total_andian_capacity}\t截止 户数:{self.num_andian_2}\t容量:{self.total_andian_capacity_2}\n"
            f"合计{self.analysis_month[-2:]}月新增 户数:{self.total_num}\t容量:{self.total_capacity}\t截止 户数:{self.total_num_2}\t容量:{self.total_capacity_2}\n"
        )
        return result

    def read_two_tables_from_word(self, docx_path):
        # 打开 Word 文档
        doc = Document(docx_path)
        
        # 提取所有表格
        tables = doc.tables
        
        if len(tables) < 2:
            raise ValueError("文档中没有足够的表格。")

        # 读取第一个表格
        table1 = tables[0]
        data1 = []
        for row in table1.rows:
            data1.append([cell.text for cell in row.cells])
        df1 = pd.DataFrame(data1[1:], columns=data1[0])
        
        # 读取第二个表格
        table2 = tables[1]
        data2 = []
        for row in table2.rows:
            data2.append([cell.text for cell in row.cells])
        df2 = pd.DataFrame(data2[1:], columns=data2[0])
        
        return df1, df2
    
    @classmethod
    def extract_values_from_sheet2(self, file_path):
        # 读取所有工作表的名称
        all_sheets = pd.ExcelFile(file_path).sheet_names
        
        # 获取第二个工作表的名称
        second_sheet_name = all_sheets[1]  # 索引从0开始，第二个工作表的索引是1
        
        # 读取第二个工作表的数据
        df = pd.read_excel(file_path, sheet_name=second_sheet_name)
                
        # 查找“总数/总量”的位置
        target_value = '总数/总量'
        target_row, target_col = None, None

        # 查找目标值的位置
        for row_idx, row in df.iterrows():
            if target_value in row.values:
                target_row = row_idx
                target_col = row.index[row.values == target_value].tolist()[0]
                break
        if target_row is not None and target_col is not None:
            # 提取目标单元格下方的四个数值
            # 如果 target_col 是列名而非整数索引
            if isinstance(target_col, str):
                target_col = df.columns.get_loc(target_col)
            
            values = df.iloc[target_row + 1:target_row + 5, target_col]
            
            # 将值存储到变量中
            value1, value2, value3, value4 = values.tolist()
            value3 /= 10000
            value4 /= 10000
            result = (
                f"\n-----------------------------月报抽数-----------------------------\n"
                f"居民充电桩总数 截止至2024.x月底:  {value1}\n"
                f"2024年x月居民新增充电桩总数:      {value2}\n"
                f"24年1-12月居民用户电量:          {value3}\n"
                f"24年x月居民用户电量:             {value4}\n"
            )
            print(result)
        else:
            print(f"'{target_value}' not found in the DataFrame.")

        return value1, value2, value3, value4
    
    def df_covert2_list_calculate(self, df1: pd.DataFrame):
        # 将第一行作为新的列名
        new_header = df1.iloc[0]  # 获取第一行数据
        df1 = df1[1:]  # 取从第二行开始的数据
        df1.columns = new_header  # 更新列名
        # 确保列名没有多余的空格
        df1.columns = df1.columns.str.strip()
        # 将“类型”列设置为索引
        df1.set_index('类型', inplace=True)
        
        # 转换为列表
        df_list: list = df1.reset_index().values.tolist()
        df_list.append(['个人', '0', f'{self.tox_newAdd}', f'{self.tox_newAdd_pct:.2f}%', '0', '0', f'{self.x_newAdd}', f'{self.x_newAdd_pct:.2f}%', '0'])

        # 计算 public_use 和 special_use 的年度和月度数据
        def calculate_totals(data_list, index):
            year_values = (float(data_list[index][1]) + float(data_list[index][2]))
            year_percent = percent_to_decimal(data_list[index][3])
            last_year_percent = percent_to_decimal(data_list[index][4])
            
            month_values = (float(data_list[index][5]) + float(data_list[index][6]))
            month_percent = percent_to_decimal(data_list[index][7])
            last_month_percent = percent_to_decimal(data_list[index][8])

            total_year = round(year_values / year_percent, 2) if year_percent != 0 else 0
            last_total_year = round(year_values / (last_year_percent + 1), 2) if last_year_percent != 0 else 0

            total_month = round(month_values / month_percent, 2) if month_percent != 0 else 0
            last_total_month = round(month_values / (last_month_percent + 1), 2) if last_month_percent != 0 else 0
            
            return total_year, last_total_year, total_month, last_total_month

        # 计算 total_public_use
        total_public_use_year, last_total_public_use_year, total_public_use_month, last_total_public_use_month = calculate_totals(df_list, 0)

        # 计算 total_special_use
        total_special_use_year, last_total_special_use_year, total_special_use_month, last_total_special_use_month = calculate_totals(df_list, 1)
        
        # 计算 private_use
        if len(df_list) > 2:
            total_private_use_year, last_total_private_use_year, total_private_use_month, last_total_private_use_month = calculate_totals(df_list, 2)
        else:
            total_private_use_year = last_total_private_use_year = total_private_use_month = last_total_private_use_month = 0
        year_values, month_values = calculate_sum_totals(df_list)
        sum_total_year = total_public_use_year + total_special_use_year + total_private_use_year
        sum_total_month = total_public_use_month + total_special_use_month + total_private_use_month
        ratio_sum_year = round(year_values / sum_total_year, 4) * 100
        ratio_sum_month = round(month_values / sum_total_month, 4) * 100

        # 设置固定宽度
        width = 15
        result = (
            f'\n2024年8月浦东新区充换电设施新增{month_values}个，占全市充换电设施新增总数{ratio_sum_month:.2f}%。截至2024年8月，浦东新区充换电设施总计{round(year_values / 10000, 2)}万个，占全市充换电设施总数{ratio_sum_year:.2f}%。\n'
            f'------------------------表 2 浦东新区充换电设施数量情况（单位：个）--------------------\n'
        )
        # 格式化输出
        result += f'公用：{df_list[0][5].ljust(width)}{df_list[0][6].ljust(width)}{df_list[0][7].ljust(width)}{df_list[0][1].ljust(width)}{df_list[0][2].ljust(width)}{df_list[0][3].ljust(width)}\n'
        result += f'专用：{df_list[1][5].ljust(width)}{df_list[1][6].ljust(width)}{df_list[1][7].ljust(width)}{df_list[1][1].ljust(width)}{df_list[1][2].ljust(width)}{df_list[1][3].ljust(width)}\n'
        result += f'个人：{df_list[2][5].ljust(width)}{df_list[2][6].ljust(width)}{df_list[2][7].ljust(width)}{df_list[2][1].ljust(width)}{df_list[2][2].ljust(width)}{df_list[2][3].ljust(width)}\n'
        result += f'合计：{month_values}\t占比：{ratio_sum_month:.2f}%\t合计：{year_values}\t占比：{ratio_sum_year:.2f}%\n'
        print(result)

    def df_covert2_list_calculate2(self, df1: pd.DataFrame):
        # 将第一行作为新的列名
        new_header = df1.iloc[0]  # 获取第一行数据
        df1 = df1[1:]  # 取从第二行开始的数据
        df1.columns = new_header  # 更新列名
        # 确保列名没有多余的空格
        df1.columns = df1.columns.str.strip()
        # 将“类型”列设置为索引
        df1.set_index('类型', inplace=True)
        
        # 转换为列表
        df_list: list = df1.reset_index().values.tolist()
        df_list.append(['个人', '0', f'{self.total_sum_current_2}', f'{self.tox_monUser_pct:.2f}%', f'{self.yoy_current_vs_previous_2:.2f}%', '0', f'{self.total_sum_current_1:.2f}', f'{self.x_monUser_pct:.2f}%', f'{self.yoy_current_vs_previous_1:.2f}%'])
        print(df_list)


        # 计算 public_use 和 special_use 的年度和月度数据
        def calculate_totals(data_list, index):
            year_values = (float(data_list[index][1]) + float(data_list[index][2]))
            year_percent = percent_to_decimal(data_list[index][3])
            last_year_percent = percent_to_decimal(data_list[index][4])
            
            month_values = (float(data_list[index][5]) + float(data_list[index][6]))
            month_percent = percent_to_decimal(data_list[index][7])
            last_month_percent = percent_to_decimal(data_list[index][8])

            total_year = round(year_values / year_percent, 2) if year_percent != 0 else 0
            last_total_year = round(year_values / (last_year_percent + 1), 2) if last_year_percent != 0 else 0

            total_month = round(month_values / month_percent, 2) if month_percent != 0 else 0
            last_total_month = round(month_values / (last_month_percent + 1), 2) if last_month_percent != 0 else 0
            
            return total_year, last_total_year, total_month, last_total_month

        # 计算 total_public_use
        total_public_use_year, last_total_public_use_year, total_public_use_month, last_total_public_use_month = calculate_totals(df_list, 0)

        # 计算 total_special_use
        total_special_use_year, last_total_special_use_year, total_special_use_month, last_total_special_use_month = calculate_totals(df_list, 1)
 
        total_private_month = float(df_list[2][6]) / self.x_monUser_pct * 100
        total_private_year = float(df_list[2][2]) / self.tox_monUser_pct * 100

        sum_total_month = float(df_list[0][5]) + float(df_list[0][6]) + float(df_list[1][5]) + float(df_list[1][6]) + float(df_list[2][6])
        sum_total_year = float(df_list[0][1]) + float(df_list[0][2]) + float(df_list[1][1]) + float(df_list[1][2]) + float(df_list[2][2])
        sum_total_month = round(sum_total_month, 2)
        sum_total_year = round(sum_total_year, 2)

        ratio_sum_total_month = round(sum_total_month / (total_public_use_month + total_special_use_month + total_private_month), 4) * 100
        ratio_sum_total_year = round(sum_total_year / (total_public_use_year + total_special_use_year + total_private_year), 4) * 100

        # 计算同比
        last_total_private_use_year = float(df_list[2][2]) /(1 + self.yoy_current_vs_previous_2 / 100)
        last_total_private_use_month = float(df_list[2][6]) /(1 + self.yoy_current_vs_previous_1 / 100)
        print('\n')
        print(f'last_total_private_use_year=  {last_total_private_use_year}')
        print(f'last_total_private_use_month = {last_total_private_use_month}')
        last_sum_total_year = last_total_public_use_year + last_total_special_use_year + last_total_private_use_year
        last_sum_total_month = last_total_public_use_month + last_total_special_use_month + last_total_private_use_month

        yay_sum_year = round((sum_total_year - last_sum_total_year) / last_sum_total_year, 4) * 100
        yay_sum_month = round((sum_total_month - last_sum_total_month) / last_sum_total_month, 4) * 100
        # 设置固定宽度
        width = 15
        result = (
            # f'\n2024年8月浦东新区充换电设施新增{month_values}个，占全市充换电设施新增总数{ratio_sum_month:.2f}%。截至2024年8月，浦东新区充换电设施总计{round(year_values / 10000, 2)}万个，占全市充换电设施总数{ratio_sum_year:.2f}%。\n'
            f'------------------------表 3 浦东新区充换电设施的充电量情况（单位：万千瓦时）--------------------\n'
        )
        # 格式化输出
        result += f'公用：{df_list[0][5].ljust(width)}{df_list[0][6].ljust(width)}{df_list[0][8].ljust(width)}{df_list[0][7].ljust(width)}{df_list[0][1].ljust(width)}{df_list[0][2].ljust(width)}{df_list[0][4].ljust(width)}{df_list[0][3].ljust(width)}\n'
        result += f'专用：{df_list[1][5].ljust(width)}{df_list[1][6].ljust(width)}{df_list[1][8].ljust(width)}{df_list[1][7].ljust(width)}{df_list[1][1].ljust(width)}{df_list[1][2].ljust(width)}{df_list[1][4].ljust(width)}{df_list[1][3].ljust(width)}\n'
        result += f'个人：{df_list[2][5].ljust(width)}{df_list[2][6].ljust(width)}{df_list[2][8].ljust(width)}{df_list[2][7].ljust(width)}{df_list[2][1].ljust(width)}{df_list[2][2].ljust(width)}{df_list[2][4].ljust(width)}{df_list[2][3].ljust(width)}\n'
        result += f'合计：{sum_total_month}\t{yay_sum_month}%\t{ratio_sum_total_month:.2f}%\t{sum_total_year}\t{yay_sum_year}%\t{ratio_sum_total_year:.2f}%\n'
        print(result)

class MonthlyChargePile:
    def __init__(self, file_path: str, analysis_month: str) -> None:
        self.current_month = analysis_month.replace('-', '') #输出: 202408
        self.first_month_of_year, self.first_month_of_last_year, self.last_year_same_month = get_special_months(self.current_month)

        self.df = pd.read_excel(file_path)
        self.analysis_MCP(self.df)
        
    def analysis_MCP(self, df: pd.DataFrame):
        # 为第一列添加表头
        df.columns = ['月份'] + df.columns[1:].tolist()
        
        # 确保 '月份' 列为字符串类型
        df['月份'] = df['月份'].astype(str).str.strip()
        
        str_month = self.current_month[-1] if self.current_month[-2] == '0' else self.current_month[-2:]
        self.result_str = f'\n截至2024年{str_month}月，浦东公司自营站点xxx个，直流桩xxxx个，交流桩xxx个。'
        print(f"\n--------表 5 浦东公司自营站点的充电量（单位：万千瓦时）-------- \n")
        # 过滤月份范围202408
        df_filtered_2024 = df[(df['月份'] == self.current_month)]
        
        # 过滤月份范围202308
        df_filtered_2023 = df[(df['月份'] == self.last_year_same_month)]

        self.calculate_total_df_filtered(df_filtered_2024, df_filtered_2023)
        self.result_str += f'2024年{str_month}月充电{self.total_sum_2024}万千瓦时，同比增长{self.yoy_sum_total:.2f}%。'

        # 过滤月份范围 202401 到 202408
        df_filtered_2024 = df[(df['月份'] >= self.first_month_of_year) & (df['月份'] <= self.current_month)]
        
        # 过滤月份范围 202301 到 202308
        df_filtered_2023 = df[(df['月份'] >= self.first_month_of_last_year) & (df['月份'] <= self.last_year_same_month)]

        self.calculate_total_df_filtered(df_filtered_2024, df_filtered_2023)
        self.result_str += f'2024年1-{str_month}月充电量合计{self.total_sum_2024}万千瓦时，同比增长{self.yoy_sum_total:.2f}%。\n'
        print(self.result_str)

    def calculate_total_df_filtered(self, df_filtered_2024, df_filtered_2023):
        # 计算所需列的总和（2024年数据）
        self.total_dc_public_2024 = round(df_filtered_2024['直流城市公共'].sum() / 10000, 2)
        self.total_ac_public_2024 = round(df_filtered_2024['交流城市公共'].sum() / 10000, 2)
        self.total_dc_bus_2024 = round(df_filtered_2024['直流公交'].sum() / 10000, 2)
        self.total_dc_fast_2024 = round(df_filtered_2024['直流高速'].sum() / 10000, 2)
        self.total_ac_residential_2024 = round(df_filtered_2024['交流小区'].sum() / 10000, 2)
        self.total_dc_internal_2024 = round(df_filtered_2024['直流单位内部'].sum() / 10000, 2)
        self.total_ac_internal_2024 = round(df_filtered_2024['交流单位内部'].sum() / 10000, 2)
        
        # 计算所需列的总和（2023年数据）
        self.total_dc_public_2023 = round(df_filtered_2023['直流城市公共'].sum() / 10000, 2)
        self.total_ac_public_2023 = round(df_filtered_2023['交流城市公共'].sum() / 10000, 2)
        self.total_dc_bus_2023 = round(df_filtered_2023['直流公交'].sum() / 10000, 2)
        self.total_dc_fast_2023 = round(df_filtered_2023['直流高速'].sum() / 10000, 2)
        self.total_ac_residential_2023 = round(df_filtered_2023['交流小区'].sum() / 10000, 2)
        self.total_dc_internal_2023 = round(df_filtered_2023['直流单位内部'].sum() / 10000, 2)
        self.total_ac_internal_2023 = round(df_filtered_2023['交流单位内部'].sum() / 10000, 2)
        
        # 计算城市公共总和
        self.total_city_public_2024 = round(self.total_dc_public_2024 + self.total_ac_public_2024, 2)
        self.total_city_public_2023 = round(self.total_dc_public_2023 + self.total_ac_public_2023, 2)
        
        self.total_internal_2024 = round(self.total_dc_internal_2024 + self.total_ac_internal_2024, 2)
        self.total_internal_2023 = round(self.total_dc_internal_2023 + self.total_ac_internal_2023, 2)
        
        self.sum_total_2024 = round(self.total_internal_2024 + self.total_city_public_2024 + self.total_dc_bus_2024 + self.total_dc_fast_2024 + self.total_ac_residential_2024, 2)
        self.sum_total_2023 = round(self.total_internal_2023 + self.total_city_public_2023 + self.total_dc_bus_2023 + self.total_dc_fast_2023 + self.total_ac_residential_2023, 2)
        
        # 计算同比
        self.yoy_city_public = round((self.total_city_public_2024 - self.total_city_public_2023) / self.total_city_public_2023 * 100, 2) if self.total_city_public_2023 != 0 else float('inf')
        self.yoy_internal = round((self.total_internal_2024 - self.total_internal_2023) / self.total_internal_2023 * 100, 2) if self.total_internal_2023 != 0 else float('inf')
        self.yoy_dc_bus = round((self.total_dc_bus_2024 - self.total_dc_bus_2023) / self.total_dc_bus_2023 * 100, 2) if self.total_dc_bus_2023 != 0 else float('inf')
        self.yoy_dc_fast = round((self.total_dc_fast_2024 - self.total_dc_fast_2023) / self.total_dc_fast_2023 * 100, 2) if self.total_dc_fast_2023 != 0 else float('inf')
        self.yoy_ac_residential = round((self.total_ac_residential_2024 - self.total_ac_residential_2023) / self.total_ac_residential_2023 * 100, 2) if self.total_ac_residential_2023 != 0 else float('inf')
        self.yoy_sum_total = round((self.sum_total_2024 - self.sum_total_2023) / self.sum_total_2023 * 100, 2) if self.sum_total_2023 != 0 else float('inf')

        self.total_sum_2024 = self.total_city_public_2024 + self.total_dc_bus_2024 + self.total_dc_fast_2024 + self.total_ac_residential_2024 + self.total_dc_internal_2024 + self.total_ac_internal_2024

        self.pct_city_public = (self.total_city_public_2024 / self.total_sum_2024 * 100) if self.total_sum_2024 != 0 else 0
        self.pct_dc_bus = (self.total_dc_bus_2024 / self.total_sum_2024 * 100) if self.total_sum_2024 != 0 else 0
        self.pct_dc_fast = (self.total_dc_fast_2024 / self.total_sum_2024 * 100) if self.total_sum_2024 != 0 else 0
        self.pct_ac_residential = (self.total_ac_residential_2024 / self.total_sum_2024 * 100) if self.total_sum_2024 != 0 else 0
        self.pct_dc_internal = (self.total_dc_internal_2024 / self.total_sum_2024 * 100) if self.total_sum_2024 != 0 else 0

        self.total_sum_2024 = self.total_city_public_2024 + self.total_dc_bus_2024 + self.total_dc_fast_2024 + self.total_ac_residential_2024 + self.total_dc_internal_2024 + self.total_ac_internal_2024

        self.pct_city_public = (self.total_city_public_2024 / self.total_sum_2024 * 100) if self.total_sum_2024 != 0 else 0
        self.pct_dc_bus = (self.total_dc_bus_2024 / self.total_sum_2024 * 100) if self.total_sum_2024 != 0 else 0
        self.pct_dc_fast = (self.total_dc_fast_2024 / self.total_sum_2024 * 100) if self.total_sum_2024 != 0 else 0
        self.pct_ac_residential = (self.total_ac_residential_2024 / self.total_sum_2024 * 100) if self.total_sum_2024 != 0 else 0
        self.pct_dc_internal = (self.total_dc_internal_2024 / self.total_sum_2024 * 100) if self.total_sum_2024 != 0 else 0
        self.pct_ac_internal = (self.total_ac_internal_2024 / self.total_sum_2024 * 100) if self.total_sum_2024 != 0 else 0
        self.pct_internal = self.pct_dc_internal + self.pct_ac_internal
        self.pct_total = self.pct_internal + self.pct_ac_residential + self.pct_dc_fast + self.pct_dc_bus + self.pct_city_public

        # 打印结果
        print(f"2024年 公共: {self.total_dc_public_2024}\t公共: {self.total_ac_public_2024}\t同比 城市公共: {self.yoy_city_public:.2f}%\t占比: {self.pct_city_public:.2f}%")
        print(f"2024年 公交: {self.total_dc_bus_2024}\t同比 公交: {self.yoy_dc_bus:.2f}%\t 占比: {self.pct_dc_bus:.2f}%")
        print(f"2024年 高速: {self.total_dc_fast_2024}\t同比 高速: {self.yoy_dc_fast:.2f}%\t占比: {self.pct_dc_fast:.2f}%")
        print(f"2024年 单位内部: {self.total_dc_internal_2024}\t单位内部: {self.total_ac_internal_2024}\t同比:{self.yoy_internal:.2f}%\t占比:{self.pct_internal:.2f}%")
        print(f"2024年 小区共享: {self.total_ac_residential_2024}\t同比 小区: {self.yoy_ac_residential:.2f}%\t占比: {self.pct_ac_residential:.2f}%")
        print(f"合计:   {self.total_sum_2024}\t同比: {self.yoy_sum_total:.2f}%\t占比: {self.pct_total:.2f}%")

def get_special_months(current_month):
    # 解析当前月份
    year = int(current_month[:4])
    # 获取当年的第一个月
    first_month_of_year = f"{year}01"  
    # 获取去年的第一个月
    last_year = year - 1
    first_month_of_last_year = f"{last_year}01"
    # 获取去年的当月
    last_year_same_month = f"{last_year}{current_month[4:]}"
    return first_month_of_year, first_month_of_last_year, last_year_same_month

def report_car(current_data_plug_in_hybrid, current_data_pure_electric, last_month_data_plug_in_hybrid, last_month_data_pure_electric):
    # 计算当月新增数据
    monthly_increase_plug_in_hybrid = current_data_plug_in_hybrid - last_month_data_plug_in_hybrid
    monthly_increase_pure_electric = current_data_pure_electric - last_month_data_pure_electric

    total_increase = monthly_increase_plug_in_hybrid + monthly_increase_pure_electric
    # 计算当月新增占比
    monthly_increase_ratio_plug_in_hybrid = round((monthly_increase_plug_in_hybrid / total_increase) * 100, 2)
    monthly_increase_ratio_pure_electric = round((monthly_increase_pure_electric / total_increase) * 100, 2)

    total_increase_ratio = monthly_increase_ratio_plug_in_hybrid + monthly_increase_ratio_pure_electric
    # 截至累计数据（假设当前数据即为截至累计数据）
    cumulative_data_plug_in_hybrid = current_data_plug_in_hybrid
    cumulative_data_pure_electric = current_data_pure_electric
    total_cumulative = cumulative_data_plug_in_hybrid + cumulative_data_pure_electric

    # 截至累计数据占比
    cumulative_data_ratio_plug_in_hybrid = round((cumulative_data_plug_in_hybrid / total_cumulative) * 100, 2)
    cumulative_data_ratio_pure_electric = round((cumulative_data_pure_electric / total_cumulative) * 100, 2)

    total_cumulative_ratio = cumulative_data_ratio_plug_in_hybrid + cumulative_data_ratio_pure_electric

    total_cumulative_w = round(total_cumulative / 10000, 2)
    cumulative_data_ratio_pure_electric_w = round (cumulative_data_pure_electric / 10000, 2)
    cumulative_data_ratio_plug_in_hybrid_w = round (cumulative_data_plug_in_hybrid / 10000, 2)

    ratio_pure_and_hybrid = round(cumulative_data_pure_electric / cumulative_data_plug_in_hybrid, 2)
    result = (

        f"\n截至2024年x月，浦东新区新能源汽车累计推广{total_cumulative_w}万辆。其中，纯电动汽车{cumulative_data_ratio_pure_electric_w}万辆，插电式混动汽车{cumulative_data_ratio_plug_in_hybrid_w}万辆，纯电动汽车与插电式混动汽车比例{ratio_pure_and_hybrid}:1。\n"
        f"\n------------表 1 浦东新区新能源汽车分类统计（单位：辆）  ----------- \n"
        f"纯电: 新增 {monthly_increase_pure_electric}\t占比: {monthly_increase_ratio_pure_electric}%\t截至: {cumulative_data_pure_electric}\t占比: {cumulative_data_ratio_pure_electric}%\n"
        f"纯电: 新增 {monthly_increase_plug_in_hybrid}\t占比: {monthly_increase_ratio_plug_in_hybrid}%\t截至: {cumulative_data_plug_in_hybrid}\t占比: {cumulative_data_ratio_plug_in_hybrid}%\n"
        f"合计: 新增 {total_increase}\t占比: {total_increase_ratio:.2f}%\t截至: {total_cumulative}\t占比: {total_cumulative_ratio:.2f}%\n"


    )
    return result


def percent_to_decimal(percent_str):
    """Convert a percentage string (e.g., '27.33%') to a decimal (e.g., 0.2733)."""
    try:
        return float(percent_str.strip('%')) / 100
    except ValueError:
        return 0

    # 计算 public_use 和 special_use 的年度和月度数据
def calculate_sum_totals(data_list):
    year_values = 0
    month_values = 0
    for index in range(len(data_list)):
        year_values += (float(data_list[index][1]) + float(data_list[index][2]))
        month_values += (float(data_list[index][5]) + float(data_list[index][6]))
    return year_values, month_values


if __name__ == '__main__':
    # 统计 7月内居民用户个数
    analysis_month = '2024-09' # 7月份月报
    file_path = rf"C:\Users\juntaox\Desktop\工作\9.浦东新区充换电设施_10月\浦东月报-数据需求清单 - 202409 - 副本.xlsx"  # 数据需求清单✔
    file_path2 = rf"C:\Users\juntaox\Desktop\工作\9.浦东新区充换电设施_10月\浦东自营充电站充电量统计-202409.xlsx" #充电站充电量统计 
    docx_path = rf"C:\Users\juntaox\Desktop\工作\9.浦东新区充换电设施_10月\联联数据需求2024年9月.docx" # 联联数据
    # 创建 MonthlyReportolution 实例
    monthly_report = MonthlyReportolution(file_path, file_path2, docx_path, analysis_month)
    # 打印结果
    print(monthly_report)

    # 当月数据
    current_data_plug_in_hybrid = 96000
    current_data_pure_electric = 209600

    # 上一月数据
    last_month_data_plug_in_hybrid = 96000
    last_month_data_pure_electric = 206200
    print(report_car(current_data_plug_in_hybrid, current_data_pure_electric, last_month_data_plug_in_hybrid, last_month_data_pure_electric))

# 截至2024年9月，浦东公司自营站点114个，直流桩1312个，交流桩564个。

