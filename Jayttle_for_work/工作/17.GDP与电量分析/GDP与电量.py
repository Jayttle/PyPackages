import pandas as pd

class Result:
    def __init__(self):
        # 初始化所有属性为字典
        self.data = {
            '电力消费弹性系数': {},  # 电力消费弹性系数 ✔
            'GDP增长率': {},         # GDP增长率 ✔
            '售电量增长率': {},      # 售电量增长率 ✔
            '季度售电量': {},        # 季度售电量
            '季度GDP': {},           # 季度GDP
            '季度售电量同比': {},     # 季度售电量同比
            '季度GDP同比': {},       # 季度GDP同比
            '季度电力消费弹性系数': {},  # 季度电力消费弹性系数
            '二产GDP占比': {},        # 二产GDP占比 ✔
            '三产GDP占比': {},       # 三产GDP占比 ✔
            '二产售电量占比': {},     # 二产售电量占比 ✔
            '三产售电量占比': {},     # 三产售电量占比 ✔
            '二产GDP增长率': {},      # 二产GDP增长率 ✔
            '三产GDP增长率': {},      # 三产GDP增长率 ✔
            '二产电力消费弹性系数': {},  # 二产电力消费弹性系数 ✔
            '三产电力消费弹性系数': {},  # 三产电力消费弹性系数 ✔
            '二产售电量增长率': {},    # 二产售电量增长率 ✔
            '三产售电量增长率': {},    # 三产售电量增长率 ✔
            '工业电力消费弹性系数': {},  # 工业电力消费弹性系数 ✔
            '建筑业电力消费弹性系数': {}, # 建筑业电力消费弹性系数 ✔
            '交通运输业电力消费弹性系数': {}, # 交通运输业电力消费弹性系数 ✔
            '批发和零售业电力消费弹性系数': {}, # 批发和零售业电力消费弹性系数 ✔
            '住宿和餐饮业电力消费弹性系数': {}, # 住宿和餐饮业电力消费弹性系数 ✔
            '金融业电力消费弹性系数': {}, # 金融业电力消费弹性系数 ✔
            '房地产业电力消费弹性系数': {}, # 房地产业电力消费弹性系数
            '租赁和商务服务业电力消费弹性系数': {}, # 租赁和商务服务业电力消费弹性系数
            '信息传输业电力消费弹性系数': {}, # 信息传输业电力消费弹性系数
        }

    def add_data(self, key, sub_key, value):
        """向指定属性的字典中添加数据"""
        if key in self.data:
            self.data[key][sub_key] = value
        else:
            raise KeyError(f"{key} 不是有效的属性名称")

    def get_data(self, key):
        """获取指定属性的字典数据"""
        return self.data.get(key, None)
    
    def calculate_gdp_growth(self, current_gdp, old_gdp):
        if old_gdp == 0:
            return None  # 避免除以零
        return round((current_gdp - old_gdp) / old_gdp, 4)

    def calculate_sales_growth(self, current_sales, old_sales):
        if old_sales == 0:
            return None  # 避免除以零
        return round((current_sales - old_sales) / old_sales, 4)

    def calculate_elasticity(self, sales_growth, gdp_growth):
        if gdp_growth == 0:
            return None  # 避免除以零
        return round(sales_growth / gdp_growth, 4)

def test1():
    # 示例使用
    result = Result()
    result.quarterly_sales = [1000, 1200, 1100]  # 示例数据
    result.quarterly_gdp = [2000, 2200, 2100]  # 示例数据

    # 计算并存储增长率
    for i in range(1, len(result.quarterly_sales)):
        sales_growth = result.calculate_sales_growth(result.quarterly_sales[i], result.quarterly_sales[i-1])
        gdp_growth = result.calculate_gdp_growth(result.quarterly_gdp[i], result.quarterly_gdp[i-1])
        elasticity = result.calculate_elasticity(sales_growth, gdp_growth)

        # 存储结果
        result.sales_growth.append(sales_growth)
        result.gdp_growth.append(gdp_growth)
        result.electricity_elasticity.append(elasticity)

    # 打印结果
    print(f"电力消费弹性系数: {result.electricity_elasticity}")
    print(f"GDP增长率: {result.gdp_growth}")
    print(f"售电量增长率: {result.sales_growth}")
    
def run_main_data(file_path: str, isSave: bool = False) -> pd.DataFrame:
    # 读取 Excel 文件
    df = pd.read_excel(file_path)

    # 将年份列转换为数值型
    df['年份'] = pd.to_numeric(df['年份'], errors='coerce')

    # 确保各个产业列都为数值类型并去除空格
    columns_to_process = ['新区', '一产', '二产', '三产', '工业',
                          '建筑', '批发和零售', '交通运输', 
                          '住宿和餐饮', '金融', '房地产', 
                          '信息传输', '租赁和商务', '科学研究', 
                          '其他服务业']

    for column in columns_to_process:
        df[column] = df[column].astype(str).str.replace(" ", "")  # 去除所有空格
        df[column] = pd.to_numeric(df[column], errors='coerce')  # 转换为数值类型

    # 筛选年份在1995到2023之间的数据
    filtered_df = df[(df['年份'] >= 1995) & (df['年份'] <= 2023)].copy()  # 使用 .copy() 方法

    # 计算增长率函数
    def calculate_growth_rate(column):
        return round(column.pct_change(fill_method=None) * 100, 2)  # 使用 fill_method=None 以避免警告

    # 计算各个产业的增长率
    for column in columns_to_process:
        filtered_df.loc[:, f'{column}增长率'] = calculate_growth_rate(filtered_df[column])  # 使用 .loc[] 赋值
    if isSave:
        filtered_df.to_excel(r"C:\Users\juntaox\Desktop\工作\17.浦东新区售电量与GDP关联分析报告\数据表\数据表-结果.xlsx")
    return filtered_df

def run_main_elc(file_path: str, isSave:bool = False) -> pd.DataFrame:
    # 读取 Excel 文件
    df = pd.read_excel(file_path)

    # 将年份列转换为数值型
    df['年份'] = pd.to_numeric(df['年份'], errors='coerce')

    # 确保各个产业列都为数值类型并去除空格
    columns_to_process = ['新区', '一产', '二产', '三产', '工业',
                          '建筑', '批发和零售', '交通运输', 
                          '住宿和餐饮', '金融', '房地产', 
                          '信息传输', '租赁和商务', '科学研究', 
                          '其他服务业', '公共服务']

    for column in columns_to_process:
        df[column] = df[column].astype(str).str.replace(" ", "")  # 去除所有空格
        df[column] = pd.to_numeric(df[column], errors='coerce')  # 转换为数值类型

    # 筛选年份在1995到2023之间的数据
    filtered_df = df[(df['年份'] >= 1995) & (df['年份'] <= 2023)].copy()  # 使用 .copy() 方法

    # 计算增长率函数
    def calculate_growth_rate(column):
        return round(column.pct_change(fill_method=None) * 100, 2)  # 使用 fill_method=None 以避免警告

    # 计算各个产业的增长率
    for column in columns_to_process:
        filtered_df.loc[:, f'{column}增长率'] = calculate_growth_rate(filtered_df[column])  # 使用 .loc[] 赋值
    if isSave:
        filtered_df.to_excel(r"C:\Users\juntaox\Desktop\工作\17.浦东新区售电量与GDP关联分析报告\数据表\电量表-结果.xlsx")
    return filtered_df

def filtered_df(df: pd.DataFrame) -> pd.DataFrame:
    # 转换年份列为数值型
    df['年份'] = pd.to_numeric(df['年份'], errors='coerce')

    # 只保留年份在2018到2023之间的数据
    df = df[(df['年份'] >= 2018) & (df['年份'] <= 2023)]

    # 确保各个产业列都为数值类型并去除空格
    columns_to_process = ['新区', '一产', '二产', '三产', '工业',
                          '建筑', '批发和零售', '交通运输', 
                          '住宿和餐饮', '金融', '房地产', 
                          '信息传输', '租赁和商务', '科学研究', 
                          '其他服务业', '公共服务']
    
    # 过滤出以 columns_to_process 中前两个字符串开头的列
    filtered_columns = [col for col in df.columns if any(col.startswith(prefix) for prefix in columns_to_process)]
    
    # 仅保留 '年份' 列和过滤后的产业列
    filtered_df = df[['年份'] + filtered_columns]
    return filtered_df
def run_main_result(file_path, file_path_elc):
    data_df = run_main_data(file_path)
    elc_df = run_main_elc(file_path_elc)

    data_df = filtered_df(data_df)
    elc_df = filtered_df(elc_df)

    # 假设 data_df 和 elc_df 都有 '年份' 和相关的增长率列
    for index, row in data_df.iterrows():
        year = row['年份']  # 从 data_df 中获取年份

        # 使用 .loc 提取 elc_df 的值，更加简洁
        if year in elc_df['年份'].values:
            elc_row = elc_df.loc[elc_df['年份'] == year].iloc[0]  # 获取对应年份的行

            # 直接从 elc_row 中提取需要的值
            elc_growth_rate = elc_row['新区增长率']
            elc_2c = elc_row['二产']
            elc_3c = elc_row['三产']
            elc_2c_growth = elc_row['二产增长率']
            elc_3c_growth = elc_row['三产增长率']
            elc_xinqu = elc_row['新区']
            data_growth_rate = row['新区增长率']
            elc_industry = elc_row['工业增长率']
            elc_construction = elc_row['建筑增长率']
            elc_transportation = elc_row['交通运输增长率']
            elc_wholesale = elc_row['批发和零售增长率']
            elc_accommodation = elc_row['住宿和餐饮增长率']
            elc_financial = elc_row['金融增长率']
            elc_fdc = elc_row['房地产增长率']
            elc_xxcs = elc_row['信息传输增长率']
            elc_zlhsw = elc_row['租赁和商务增长率']


            # 计算弹性系数和比率
            if data_growth_rate != 0:  # 防止除以零
                electricity_elasticity = round(elc_growth_rate / data_growth_rate, 2)  # 修改此行
                secondary_elasticity = round(elc_2c_growth / row['二产增长率'], 2) if row['二产增长率'] != 0 else 0  # 修改此行
                tertiary_elasticity = round(elc_3c_growth / row['三产增长率'], 2) if row['三产增长率'] != 0 else 0  # 修改此行
                # 计算其他行业的弹性系数
                industry_elasticity = round(elc_industry / row['工业增长率'], 2) if row['工业增长率'] != 0 else 0  # 修改此行
                transportation_elasticity = round(elc_transportation / row['交通运输增长率'], 2) if row['交通运输增长率'] != 0 else 0  # 修改此行
                construction_elasticity = round(elc_construction / row['建筑增长率'], 2) if row['建筑增长率'] != 0 else 0  # 修改此行
                wholesale_elasticity = round(elc_wholesale / row['批发和零售增长率'], 2) if row['批发和零售增长率'] != 0 else 0  # 修改此行
                accommodation_elasticity = round(elc_accommodation / row['住宿和餐饮增长率'], 2) if row['住宿和餐饮增长率'] != 0 else 0  # 修改此行
                financial_elasticity = round(elc_financial / row['金融增长率'], 2) if row['金融增长率'] != 0 else 0  # 修改此行
                fdc_elasticity = round(elc_fdc / row['房地产增长率'], 2) if row['房地产增长率'] != 0 else 0  # 修改此行
                xxcs_elasticity = round(elc_xxcs / row['信息传输增长率'], 2) if row['信息传输增长率'] != 0 else 0  # 修改此行
                zlhsw_elasticity = round(elc_zlhsw / row['租赁和商务增长率'], 2) if row['租赁和商务增长率'] != 0 else 0  # 修改此行

                # 创建一个字典存储结果
                result_data = {
                    '电力消费弹性系数': electricity_elasticity,
                    'GDP增长率': data_growth_rate,
                    '售电量增长率': elc_growth_rate,
                    '二产GDP占比': round(row['二产'] / row['新区'] * 100, 2),
                    '三产GDP占比': round(row['三产'] / row['新区'] * 100, 2),
                    '二产售电量占比': round(elc_2c / elc_xinqu * 100, 2),
                    '三产售电量占比': round(elc_3c / elc_xinqu * 100, 2),
                    '二产GDP增长率': row['二产增长率'],
                    '三产GDP增长率': row['三产增长率'],
                    '二产售电量增长率': elc_2c_growth,
                    '三产售电量增长率': elc_3c_growth,
                    '二产电力消费弹性系数': secondary_elasticity,
                    '三产电力消费弹性系数': tertiary_elasticity,
                    '工业电力消费弹性系数': industry_elasticity,
                    '交通运输业电力消费弹性系数': transportation_elasticity,
                    '建筑业电力消费弹性系数': construction_elasticity,
                    '批发和零售业电力消费弹性系数': wholesale_elasticity,
                    '住宿和餐饮业电力消费弹性系数': accommodation_elasticity,
                    '金融业电力消费弹性系数': financial_elasticity,
                    '房地产业电力消费弹性系数': fdc_elasticity,
                    '信息传输业电力消费弹性系数': xxcs_elasticity,
                    '租赁和商务服务业电力消费弹性系数': zlhsw_elasticity,
                }

                # 使用循环将数据添加到报告中
                for key, value in result_data.items():
                    result_report.add_data(key, year, value)
    for key in result_report.data:
        print(f"{key}:\t{result_report.data[key]}")

def write_report(result_report: Result):
    df = pd.DataFrame(result_report.data)
    print(df)
    df.to_excel("result_report2.xlsx")
if __name__ == '__main__':
    file_path = r"Jayttle_for_work\17.GDP与电量分析\data_excel\数据表-浦东新区.xlsx"
    file_path_elc = r"Jayttle_for_work\17.GDP与电量分析\data_excel\电量表-产业行业.xlsx"

    result_report = Result()
    run_main_result(file_path, file_path_elc)
    write_report(result_report)
