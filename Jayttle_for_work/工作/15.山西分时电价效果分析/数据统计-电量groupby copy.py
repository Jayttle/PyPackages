import pandas as pd

file_path = r"C:\Users\juntaox\Desktop\工作\15.山西分时电价成效分析\典型月分时电量_20231222.xlsx"

def select_columns(columns: list, time_ranges: list) -> list:
    selected = []
    for col in columns:
        hour = int(col[1:]) 
        if hour == 2400:
            hour = 0
        if any(start <= hour < end for start, end in time_ranges):
            selected.append(col)
    return selected

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

def appoint_date_df(date: str):
    df = pd.read_excel(file_path)
    # 将 '日期' 列转换为 datetime 格式
    df['日期'] = pd.to_datetime(df['日期'], format='%Y%m%d')

    # 将数值列转换为 float 类型，处理缺失值
    numeric_columns = df.columns[1:]  # 选择从 0:15 到 2:45 的列
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # 对相同日期和主要行业的数据进行分组并求和
    df = df.groupby(['日期'])[numeric_columns].sum().reset_index()
    # df.to_excel(r"C:\Users\juntaox\Desktop\工作\15.山西分时电价成效分析\数据表\数据表-主要行业总和.xlsx")
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

# def base_and_eva_date_compare(base_date: str = '2020-12-29', eva_date: str = '2023-12-22'):
def base_and_eva_date_compare(base_date: str = '2021-04-10', eva_date: str = '2024-04-06'):
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

def subtract_15_minutes(time_str: str) -> str:
    # 从字符串中提取小时和分钟
    hour = int(time_str[1:3])  # 提取小时
    minute = int(time_str[3:5])  # 提取分钟
    
    # 扣除15分钟
    total_minutes = hour * 60 + minute - 15
    
    # 处理负值的情况
    if total_minutes < 0:
        total_minutes += 24 * 60  # 加上一天的分钟数

    # 计算新的小时和分钟
    new_hour = total_minutes // 60
    new_minute = total_minutes % 60
    
    # 返回格式化后的字符串
    return f"T{new_hour:02d}{new_minute:02d}"


def run_main() -> None:
    # 1. 读取 Excel 文件
    df = pd.read_excel(file_path)

    # 2. 打印原始数据
    print("原始数据:")
    print(df)

    # 3. 获取数值列（假设从第三列开始）
    numeric_columns = df.columns[2:]  # 获取第 3 列到最后一列的所有列名
    print("数值列名:", numeric_columns)

    # 4. 创建新的列名
    new_column_names = [subtract_15_minutes(col) for col in numeric_columns if col.startswith('T')]
    
    # 5. 修改 DataFrame 的列名
    df.rename(columns=dict(zip(numeric_columns, new_column_names)), inplace=True)

    grouped_df = df.groupby('日期')[new_column_names].sum().reset_index()
    # 6. 打印修改后的数据
    print("修改后的数据:")
    print(grouped_df)
    grouped_df.to_excel(r"C:\Users\juntaox\Desktop\工作\15.山西分时电价成效分析\数据表\数据表-电量总和-20231222.xlsx")

def extract_category_prefix(category: str) -> str:
    # 提取第一个 '-' 之前的字符串
    if '-' in category:
        return category.split('-')[0]
    return category  # 如果没有 '-', 返回原字符串

def run_main1() -> None:
    # 1. 读取 Excel 文件
    df = pd.read_excel(file_path)
    # 3. 获取数值列（假设从第三列开始）
    numeric_columns = df.columns[2:]  # 获取第 3 列到最后一列的所有列名
    new_column_names = [subtract_15_minutes(col) for col in numeric_columns if col.startswith('T')]
    df.rename(columns=dict(zip(numeric_columns, new_column_names)), inplace=True)

    numeric_columns = df.columns[2:]  # 获取第 3 列到最后一列的所有列名
    print("数值列名:", numeric_columns)

    # 4. 创建一个新的类别列，提取第一个 '-' 之前的字符串
    df['提取类别'] = df['分类'].apply(extract_category_prefix)

    # 5. 按 '日期' 列和 '提取类别' 列进行分组，并对数值列求和
    grouped_df = df.groupby(['日期', '提取类别'])[numeric_columns].sum().reset_index()

    grouped_df.to_excel(r"C:\Users\juntaox\Desktop\工作\15.山西分时电价成效分析\原始数据\典型月分时电量_20231222.xlsx")

# 运行主程序
if __name__ == "__main__":
    run_main()
    run_main1()