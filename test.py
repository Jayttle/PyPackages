import pandas as pd
def read_data(file_path: str) -> pd.DataFrame:
    # 读取 Excel 文件，并返回一个 DataFrame
    df = pd.read_excel(file_path)
    return df  # 返回 DataFrame

def select_columns(columns: list, time_ranges: list) -> list:
    selected = []
    for col in columns:
        hour = int(col[1:]) 
        if hour == 2400:
            hour = 0
        if any(start <= hour < end for start, end in time_ranges):
            selected.append(col)
    return selected

def df_sum_time_range(df: pd.DataFrame, time_range: dict) -> dict:
    t_columns = [col for col in df.columns if col.startswith('T')]
    selected_columns = {key: select_columns(t_columns, ranges) for key, ranges in time_range.items()}
    sum_results = {}

    for key, cols in selected_columns.items():
        sum_results[key] = df[cols].sum().sum()  # Sum the selected columns

    # Print the results
    print("\n各时间段数据总和：")
    for key, total in sum_results.items():
        print(f"{key}: {total}")
    return sum_results

def df_peak_time_stats(df: pd.DataFrame, time_ranges: list) -> pd.DataFrame:
    # 确保日期列存在，并将其设置为索引
    df['日期'] = pd.to_datetime(df['日期'])
    df.set_index('日期', inplace=True)

    # 选择以 'T' 开头的列
    t_columns = [col for col in df.columns if col.startswith('T')]
    selected_columns = select_columns(t_columns, time_ranges)

    # 初始化结果列表
    results = []

    # 遍历 DataFrame 中的每一行
    for index, row in df.iterrows():
        # 选取当前行中符合条件的列
        selected_values = row[selected_columns]

        # 计算最大值和平均值
        if not selected_values.empty:
            max_value = selected_values.max()
            avg_value = selected_values.mean()
        else:
            max_value = None
            avg_value = None
        
        # 将结果添加到结果列表中
        results.append({
            'time': str(index),  # 将索引转换为字符串格式
            'max': max_value,
            'average': avg_value
        })
    results_df = pd.DataFrame(results)
    return results_df


def df_all_time_stats(df: pd.DataFrame, Feng_time_ranges: list = None, Gu_time_ranges: list = None) -> dict:
    # 选择以 'T' 开头的列
    t_columns = [col for col in df.columns if col.startswith('T')]
    Feng_selected_columns = select_columns(t_columns, Feng_time_ranges)
    Gu_selected_columns = select_columns(t_columns, Gu_time_ranges)
    # 初始化结果列表
    results = []

    # 遍历 DataFrame 中的每一行
    for index, row in df.iterrows():
        selected_values = row[t_columns]
        Feng_values = row[Feng_selected_columns]
        Gu_values = row[Gu_selected_columns]

        # 计算最大值和最小值
        max_value = selected_values.max()
        min_value = selected_values.min()
        all_sum = selected_values.sum()

        # 获取最大值和最小值的列名
        max_column = selected_values.idxmax()  # 获取最大值的列名
        min_column = selected_values.idxmin()  # 获取最小值的列名

        max_feng = Feng_values.max()
        feng_mean = Feng_values.mean()
        sum_feng = Feng_values.sum()
        feng_ratio = round(sum_feng / all_sum, 4)

        max_gu = Gu_values.min()
        gu_mean = Gu_values.mean()
        sum_gu = Gu_values.sum()
        gu_ratio = round(sum_gu / all_sum, 4)
        # 计算最大最小的差值
        difference = max_value - min_value
        
        # 计算最大最小的差值与最大值的比例
        ratio = difference / max_value if max_value != 0 else None  # 避免除以零的情况
        # 获取日期
        date_value = row['日期']  # 假设您的日期列命名为 '日期'
        # 将结果添加到结果列表中
        results.append({
            '时间': str(date_value),  # 将索引转换为字符串格式
            '负荷最大值': max_value,
            '负荷最大值所在列': max_column,  
            '负荷最小值': min_value,
            '负荷最小值所在列': min_column, 
            '峰谷差': difference,
            '峰谷差率': ratio,
            '峰时段最大': max_feng,
            '峰时段平均': feng_mean,
            '峰时段占比': feng_ratio,
            '谷时段最低': max_gu,
            '谷时段平均': gu_mean,
            '谷时段占比': gu_ratio,
            '平时段占比': 1-gu_ratio-feng_ratio,
        })

    return results



def appoint_date_df(file_path, date: str):
    df = read_data(file_path)
    # 转换‘日期’列
    df['日期'] = pd.to_datetime(df['日期'].astype(str), format='%Y%m%d')

    # 筛选出指定日期2024年7月22日的数据
    filtered_df = df[df['日期'] == date]

    # 1. 筛选以 'T' 开头的列名
    t_columns = [col for col in filtered_df.columns if col.startswith('T')]
    GFtime_ranges = {
        '峰': [(800, 1100), (1700, 1800), (2000, 2300)],
        '高峰': [(1800, 2000)],
        '谷': [(0, 700), (1100, 1300)],
        '平': [(700, 800), (1300, 1700), (2300, 2400)]
    }
    NotGFtime_ranges = {
        '峰': [(800, 1100), (1700, 2300)],
        '谷': [(0, 700), (1100, 1300)],
        '平': [(700, 800), (1300, 1700), (2300, 2400)]
    }

    results_dict = df_all_time_stats(filtered_df, NotGFtime_ranges['峰'], NotGFtime_ranges['谷'])
    return results_dict

def appoint_date_df_with_filepath(appoint_file_path:str, date: str):
    df = read_data(appoint_file_path)
    # 转换‘日期’列
    df['日期'] = pd.to_datetime(df['日期'].astype(str), format='%Y%m%d')

    # 筛选出指定日期2024年7月22日的数据
    filtered_df = df[df['日期'] == date]

    # 1. 筛选以 'T' 开头的列名
    t_columns = [col for col in filtered_df.columns if col.startswith('T')]
    GFtime_ranges = {
        '峰': [(800, 1100), (1700, 1800), (2000, 2300)],
        '高峰': [(1800, 2000)],
        '谷': [(0, 700), (1100, 1300)],
        '平': [(700, 800), (1300, 1700), (2300, 2400)]
    }
    NotGFtime_ranges = {
        '峰': [(800, 1100), (1700, 2300)],
        '谷': [(0, 700), (1100, 1300)],
        '平': [(700, 800), (1300, 1700), (2300, 2400)]
    }

    results_dict = df_all_time_stats(filtered_df, NotGFtime_ranges['峰'], NotGFtime_ranges['谷'])
    return results_dict

# def base_and_eva_date_compare(base_date: str = '2021-04-10', eva_date: str = '2024-04-06'):
def base_and_eva_date_compare(base_date: str = '2021-07-09', eva_date: str = '2024-07-22'):
    base_results = appoint_date_df(base_date)
    eva_results = appoint_date_df(eva_date)

    # 确保访问列表中的第一个元素（假设列表只包含一个字典）
    base_results_dict = base_results[0] if isinstance(base_results, list) else base_results
    eva_results_dict = eva_results[0] if isinstance(eva_results, list) else eva_results

    # 计算变化并保留四位有效数字
    comparison_results = {
        '高峰时段最大负荷上升/下降': round(eva_results_dict['峰时段最大'] - base_results_dict['峰时段最大'], 2),
        '峰时段平均负荷上升/下降': round(eva_results_dict['峰时段平均'] - base_results_dict['峰时段平均'], 2),
        '低谷时段最小负荷上升/下降': round(eva_results_dict['谷时段最低'] - base_results_dict['谷时段最低'], 2),
        '平均负荷上升/下降': round(eva_results_dict['谷时段平均'] - base_results_dict['谷时段平均'], 2),
        '峰时段占比变化': round(eva_results_dict['峰时段占比'] - base_results_dict['峰时段占比'], 4),
        '谷时段占比变化': round(eva_results_dict['谷时段占比'] - base_results_dict['谷时段占比'], 4),
        '峰谷差变化': round(eva_results_dict['峰谷差'] - base_results_dict['峰谷差'], 4),
        '峰谷差率变化': round(eva_results_dict['峰谷差率'] - base_results_dict['峰谷差率'], 4)
    }


    # 返回比较结果
    return base_results, eva_results, comparison_results

def df_monthly_stats(df: pd.DataFrame, time_ranges: dict) -> dict:
    # 选择以 'T' 开头的列
    t_columns = [col for col in df.columns if col.startswith('T')]
    columns_dict = {}
    for key in time_ranges:
        columns_dict[key] = select_columns(t_columns, time_ranges[key])

    sum_results = {}
    for key, cols in columns_dict.items():
        sum_results[key] = df[cols].sum().sum()  # Sum the selected columns
    return sum_results

def df_monthly_analysis_GF(file_path: str, year: int = None, isWinter: bool = False):
    df = read_data(file_path)
    
    # 确保 '日期' 列是字符串
    df['日期'] = df['日期'].astype(str)

    # 过滤掉不符合格式的日期
    valid_dates = df['日期'].str.match(r'^\d{6}$')  # 确保是8位数字的字符串

    # 使用有效日期进行转换
    df['日期'] = pd.to_datetime(df.loc[valid_dates, '日期'], format='%Y%m', errors='coerce')


    if not isWinter:
    # 筛选出2024年6月的数据
        filtered_df = df[(df['日期'].dt.year == year) & (df['日期'].dt.month.isin([12]))]
    else:
        filtered_df = df[
            (df['日期'].dt.year == year - 1) & (df['日期'].dt.month == 12)
        ]


    t_columns = [col for col in filtered_df.columns if col.startswith('T')]
    GFtime_ranges = {
        '峰': [(800, 1100), (1700, 1800), (2000, 2300)],
        '高峰': [(1800, 2000)],
        '谷': [(0, 700), (1100, 1300)],
        '平': [(700, 800), (1300, 1700), (2300, 2400)]
    }
    return df_monthly_stats(filtered_df, GFtime_ranges)

def df_monthly_analysis_NOGF(file_path: str, year: int = None, isWinter: bool = False):
    df = read_data(file_path)
    
    # 确保 '日期' 列是字符串
    df['日期'] = df['日期'].astype(str)

    # 过滤掉不符合格式的日期
    valid_dates = df['日期'].str.match(r'^\d{6}$')  # 确保是8位数字的字符串

    # 使用有效日期进行转换
    df['日期'] = pd.to_datetime(df.loc[valid_dates, '日期'], format='%Y%m', errors='coerce')

    if not isWinter:
    # 筛选出2024年6月的数据
        filtered_df = df[(df['日期'].dt.year == year) & (df['日期'].dt.month.isin([5]))]
    else:
        filtered_df = df[
            (df['日期'].dt.year == year) & (df['日期'].dt.month.isin([2])) 
        ]


    # 1. 筛选以 'T' 开头的列名
    t_columns = [col for col in filtered_df.columns if col.startswith('T')]
    NotGFtime_ranges = {
        '峰': [(800, 1100), (1700, 2300)],
        '谷': [(0, 700), (1100, 1300)],
        '平': [(700, 800), (1300, 1700), (2300, 2400)]
    }
    return df_monthly_stats(filtered_df, NotGFtime_ranges)

def run_main_2021(file_path: str, isWinter: bool = False) -> None:
    # find_eva_date()
    sum_results_GF = df_monthly_analysis_NOGF(file_path, 2021, isWinter)
    sum_results_GF['峰'] = round(sum_results_GF['峰'] / 10000 / 10000,2)
    sum_results_GF['谷'] = round(sum_results_GF['谷'] / 10000 / 10000,2)
    sum_results_GF['平'] = round(sum_results_GF['平'] / 10000/ 10000,2)
    # sum_results_GF['高峰'] = round(sum_results_GF['高峰'] / 10000 / 10000 ,2)
    total_sum = round(sum(sum_results_GF.values()) ,2)

    # 计算比例
    proportions = {key: value / total_sum for key, value in sum_results_GF.items()}

    # 可选：转换为百分比
    percentage = {key: round(value * 100, 4) for key, value in proportions.items()}
    percentage['总和'] = total_sum
    return sum_results_GF, percentage

def run_main_2024(file_path: str,isWinter: bool = False) -> None:
    # find_eva_date()
    sum_results_GF = df_monthly_analysis_NOGF(file_path, 2024, isWinter)

    sum_results_GF['峰'] = round(sum_results_GF['峰'] / 10000 / 10000 ,2)
    sum_results_GF['谷'] = round(sum_results_GF['谷'] / 10000 / 10000,2)
    sum_results_GF['平'] = round(sum_results_GF['平'] / 10000 / 10000,2)
    # sum_results_GF['高峰'] = round(sum_results_GF['高峰'] / 10000 / 10000 ,2)
    total_sum = round(sum(sum_results_GF.values()), 4)
    # 计算比例
    proportions = {key: value / total_sum for key, value in sum_results_GF.items()}

    # 可选：转换为百分比
    percentage = {key: round(value * 100, 2) for key, value in proportions.items()}
    percentage['总和'] = total_sum

    return sum_results_GF, percentage

def run_main():
    marker_name = ['总和', '工业', '一般工商业', '居民电量', '居民电动汽车', '主要行业']
    for item in marker_name:
        # print(f"--------------------{item}-----------------------")
        file_path = rf"C:\Users\juntaox\Desktop\工作\15.山西分时电价成效分析\数据表\电量表-{item}.xlsx"
        sum_results_2021, percen_2021 = run_main_2021(file_path, isWinter=False)
        sum_results_2024, percen_2024 = run_main_2024(file_path, isWinter=False)
        # 计算比较百分比
        compare_percent = {key: round(percen_2024[key] - percen_2021[key], 2) for key in percen_2021.keys()}

        compare_percent['总体'] = round((percen_2024['总和'] - percen_2021['总和']) / percen_2021['总和'] * 100, 2)
        print(rf"{item}用户峰、平、谷电量分别为{sum_results_2024['峰']}、{sum_results_2024['平']}、{sum_results_2024['谷']}亿千瓦时，占比分别为{percen_2024['峰']}%、{percen_2024['平']}%、{percen_2024['谷']}%，较分时电价调整前提高{compare_percent['峰']}%、{compare_percent['平']}%、{compare_percent['谷']}%。整体用电量了{compare_percent['总和']}%")
        # print(rf"{item}用户尖峰、峰、平、谷电量分别为{sum_results_2024['高峰']}、{sum_results_2024['峰']}、{sum_results_2024['平']}、{sum_results_2024['谷']}亿千瓦时，占比分别为{percen_2024['高峰']}%、{percen_2024['峰']}%、{percen_2024['平']}%、{percen_2024['谷']}%，较分时电价调整前提高{compare_percent['高峰']}%、{compare_percent['峰']}%、{compare_percent['平']}%、{compare_percent['谷']}%。整体用电量了{compare_percent['总和']}%")

# 运行主程序
if __name__ == "__main__":
    run_main()