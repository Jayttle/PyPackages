import pandas as pd 
import re
import json

def get_unique_column(df: pd.DataFrame):
    # 仅筛选产业类别为第二产业或第三产业的行
    df_filtered = df[
        (df['产业类别（第一、二、三产业、居民）'] == '第二产业') |
        (df['产业类别（第一、二、三产业、居民）'] == '第三产业')
    ]
    
    # 获取“行业分类”列中的唯一值
    unique_values = df_filtered['行业分类'].unique()
    
    # 打印唯一值
    print('行业分类中的唯一值:')
    for value in unique_values:
        print(value)
    
    # 统计各个行业的行数
    industry_counts = df_filtered['行业分类'].value_counts()
    
    # 打印每个行业的行数
    print('\n各个行业的行数:')
    for industry, count in industry_counts.items():
        print(f'{industry}: {count} 行')
    
    # 打印总行数
    total_rows = df_filtered.shape[0]
    print(f'\n总行数: {total_rows}')

def read_json_file(json_file: str):
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def split_df_by_category(df: pd.DataFrame, categories: dict) -> dict:
    # 创建一个字典来存储按分类拆分后的 DataFrame
    category_dfs = {}
    
    # 仅筛选产业类别为第二产业或第三产业的行
    df_filtered = df[
        (df['产业类别（第一、二、三产业、居民）'] == '第二产业') |
        (df['产业类别（第一、二、三产业、居民）'] == '第三产业')
    ]
    
    # 用于跟踪哪些行业分类被涵盖了
    covered_categories = set()
    
    # 遍历分类字典
    for key, value in categories.items():
        if callable(value):
            mask = df_filtered['行业分类'].apply(value)
        else:
            mask = df_filtered['行业分类'].isin(value)
        
        # 筛选数据
        filtered_df = df_filtered[mask].copy()
        
        # 存储结果
        category_dfs[key] = filtered_df
        
        # 更新涵盖的行业分类
        covered_categories.update(filtered_df['行业分类'].unique())
    
    # 打印未被涵盖的行业分类
    all_categories = set(df_filtered['行业分类'].unique())
    uncovered_categories = all_categories - covered_categories
    print("未被涵盖的行业分类：", uncovered_categories)
    
    return category_dfs

def analysis_行业分析(df: pd.DataFrame, json_file: str, output_file: str):
    # 读取 JSON 文件
    categories = read_json_file(json_file)
    
    # 将 DataFrame 按分类拆分
    categorized_dfs = split_df_by_category(df, categories)
    
    # 创建一个列表来存储每种类别的分析结果
    results_list = []
    total_electricity_2023 = 0
    total_electricity_2024 = 0
    
    for key, filtered_df in categorized_dfs.items():
        # 找到所有电量列
        electricity_columns = [col for col in filtered_df.columns if col.startswith('电量')]
        
        # 初始化每年电量总和
        yearly_totals = {year: 0 for year in range(2020, 2025)}
        
        # 计算每年电量总和
        for col in electricity_columns:
            match = re.match(r'电量(\d{4})(\d{2})', col)
            if match:
                year = int(match.group(1))
                if year in yearly_totals:
                    filtered_df[col] = pd.to_numeric(filtered_df[col], errors='coerce')
                    yearly_totals[year] += filtered_df[col].sum() / 10000
        
        # 统计2023年和2024年的总电量
        total_electricity_2023 += yearly_totals.get(2023, 0)
        total_electricity_2024 += yearly_totals.get(2024, 0)
        
        # 统计行数
        num_rows = filtered_df.shape[0]
        
        # 存储结果到列表
        results_list.append({
            '类别': key,
            '筛选后的数据行数': num_rows,
            **{f'{year}年电量总和': yearly_totals.get(year, 0) for year in range(2020, 2025)}
        })
    
    # 将所有结果转换为 DataFrame
    df_results = pd.DataFrame(results_list)
    
    # 计算占比
    df_results['2023年用电量占比'] = round(df_results['2023年电量总和'] / total_electricity_2023 * 100, 2) if total_electricity_2023 != 0 else 0
    df_results['2024年用电量占比'] = round(df_results['2024年电量总和'] / total_electricity_2024 * 100, 2) if total_electricity_2024 != 0 else 0
    
    # 写入 Excel 文件
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df_results.to_excel(writer, sheet_name='分析结果', index=False)
    
    print(f"Results have been written to {output_file}")
    print(f"2023年的总用电量: {total_electricity_2023}")
    print(f"2024年的总用电量: {total_electricity_2024}")
    analysis_行业分析_同比(df, json_file)

def analysis_行业分析_同比(df: pd.DataFrame, json_file: str):
    # 读取 JSON 文件
    categories = read_json_file(json_file)
    
    # 将 DataFrame 按分类拆分
    categorized_dfs = split_df_by_category(df, categories)
    
    # 创建一个列表来存储每种类别的分析结果
    results_list = []
    total_electricity_2023 = 0
    total_electricity_2024 = 0
    
    for key, filtered_df in categorized_dfs.items():
        # 找到所有电量列
        electricity_columns = [col for col in filtered_df.columns if col.startswith('电量')]
        
        # 初始化每年每月电量总和
        monthly_totals_2023 = {month: 0 for month in range(1, 9)}
        monthly_totals_2024 = {month: 0 for month in range(1, 9)}
        
        # 计算每月电量总和
        for col in electricity_columns:
            match = re.match(r'电量(\d{4})(\d{2})', col)
            if match:
                year = int(match.group(1))
                month = int(match.group(2))
                if year == 2023 and month in monthly_totals_2023:
                    filtered_df[col] = pd.to_numeric(filtered_df[col], errors='coerce')
                    monthly_totals_2023[month] += filtered_df[col].sum() / 10000
                elif year == 2024 and month in monthly_totals_2024:
                    filtered_df[col] = pd.to_numeric(filtered_df[col], errors='coerce')
                    monthly_totals_2024[month] += filtered_df[col].sum() / 10000
        
        # 计算2023年1月至8月的总电量
        total_electricity_2023 = sum(monthly_totals_2023.values())
        # 计算2024年1月至8月的总电量
        total_electricity_2024 = sum(monthly_totals_2024.values())
        
        # 计算同比
        year_on_year_growth = (
            ((total_electricity_2024 - total_electricity_2023) / total_electricity_2023) * 100
            if total_electricity_2023 != 0 else 0
        )
        
        # 统计行数
        num_rows = filtered_df.shape[0]
        
        # 存储结果到列表
        results_list.append({
            '类别': key,
            '筛选后的数据行数': num_rows,
            '2023年1月至8月电量总和': total_electricity_2023,
            '2024年1月至8月电量总和': total_electricity_2024,
            '同比增长率': round(year_on_year_growth, 2)
        })
    
    # 将所有结果转换为 DataFrame
    df_results = pd.DataFrame(results_list)
    total_2023 = df_results['2023年1月至8月电量总和'].sum()
    total_2024 = df_results['2024年1月至8月电量总和'].sum()
    yay_total = round((total_2024 - total_2023) / total_2023 * 100, 2)

    print(f"total_2023:{total_2023}\ntotal_2024:{total_2024}\nyay_total:{yay_total}\n")
    print(df_results)


def analysis_产业分析(df: pd.DataFrame, output_file: str):
    # 定义类别
    categories = {
        '第二产业': '第二产业',
        '第三产业': '第三产业',
        '其他': lambda x: x not in ['第二产业', '第三产业']
    }
    
    # 创建一个列表来存储每种类别的筛选结果
    results = []
    total_electricity_2023_0108 = 0
    total_electricity_2023 = 0
    total_electricity_2024 = 0
    
    for key, value in categories.items():
        if key == '其他':
            mask = df['产业类别（第一、二、三产业、居民）'].apply(value)
        else:
            mask = df['产业类别（第一、二、三产业、居民）'] == value
        
        # 筛选数据
        filtered_df = df[mask].copy()
        
        # 找到所有电量列
        electricity_columns = [col for col in filtered_df.columns if col.startswith('电量')]
        
        # 初始化每年电量总和
        yearly_totals = {year: {month: 0 for month in range(1, 13)} for year in range(2020, 2025)}
        
        # 计算每月电量总和
        for col in electricity_columns:
            match = re.match(r'电量(\d{4})(\d{2})', col)
            if match:
                year = int(match.group(1))
                month = int(match.group(2))
                if year in yearly_totals and month in yearly_totals[year]:
                    filtered_df[col] = pd.to_numeric(filtered_df[col], errors='coerce')
                    yearly_totals[year][month] += filtered_df[col].sum() / 10000
        
        # 统计2023年和2024年的电量总和
        sum_2020 = sum(yearly_totals[2020][month] for month in range(1, 13))
        sum_2021 = sum(yearly_totals[2021][month] for month in range(1, 13))
        sum_2022 = sum(yearly_totals[2022][month] for month in range(1, 13))
        sum_2023 = sum(yearly_totals[2023][month] for month in range(1, 13))
        sum_2024 = sum(yearly_totals[2024][month] for month in range(1, 13))
        
        sum_2023_0108 = sum(yearly_totals[2023][month] for month in range(1, 9))


        total_electricity_2023_0108 += sum_2023_0108
        total_electricity_2023 += sum_2023
        total_electricity_2024 += sum_2024
        
        yay_sum = round((sum_2024 - sum_2023_0108)/ sum_2023_0108 * 100, 2)

        # 存储结果
        results.append({
            '类别': key,
            '2020年电量总和': sum_2020,
            '2021年电量总和': sum_2021,
            '2022年电量总和': sum_2022,
            '2023年电量总和': sum_2023,
            '2024年电量总和': sum_2024,
            '2024年同比': yay_sum,
        })
    
    # 计算各产业占比
    for result in results:
        result['2023年占比'] = (result['2023年电量总和'] / total_electricity_2023 * 100) if total_electricity_2023 > 0 else 0
        result['2024年占比'] = (result['2024年电量总和'] / total_electricity_2024 * 100) if total_electricity_2024 > 0 else 0
    
    yay = (total_electricity_2024 - total_electricity_2023_0108) / total_electricity_2023_0108 * 100
    print(f'总用电量的同比是：{yay:.2f}')
    # 转换结果为 DataFrame 并写入 Excel
    df_results = pd.DataFrame(results)
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df_results.to_excel(writer, sheet_name='分析结果', index=False)
    
    print(f"Results have been written to {output_file}")


def analysis_第三产业_商业(df: pd.DataFrame):
    category = '第三产业'
    
    mask = (df['产业类别（第一、二、三产业、居民）'] == category)

    # 筛选数据
    filtered_df = df.loc[mask].copy()  # 使用 .copy() 创建副本
    
    # 确保列名以 '电量' 开头
    electricity_columns = [col for col in filtered_df.columns if col.startswith('电量')]
    
    # 只选择需要的列
    selected_columns = [col for col in electricity_columns if '202401' <= col[-6:] <= '202408']
    
    # 将所选列转为数字（处理非数字值或缺失值）
    filtered_df.loc[:, selected_columns] = filtered_df[selected_columns].apply(pd.to_numeric, errors='coerce')
    
    # 计算用电量总额
    total_consumption = filtered_df[selected_columns].sum().sum()
    
    # 统计行数
    num_rows = filtered_df.shape[0]


    
    # 只选择需要的列
    selected_columns_2023 = [col for col in electricity_columns if '202301' <= col[-6:] <= '202308']
    
    # 将所选列转为数字（处理非数字值或缺失值）
    filtered_df.loc[:, selected_columns_2023] = filtered_df[selected_columns_2023].apply(pd.to_numeric, errors='coerce')
    
    # 计算用电量总额
    total_consumption_2023 = filtered_df[selected_columns_2023].sum().sum()
    yay_consumption = (total_consumption - total_consumption_2023) / total_consumption_2023 * 100
    # 输出结果
    print(f'第三产业 2024:\n筛选后的数据行数: {num_rows}')
    print(f'用电量总额: {round(total_consumption / 10000, 2)}万千瓦时')
    print(f'同比：{yay_consumption}%')
    print('---------------------------------------------------------------')


    # 只选择需要的列
    selected_columns = [col for col in electricity_columns if '202301' <= col[-6:] <= '202312']
    
    # 将所选列转为数字（处理非数字值或缺失值）
    filtered_df.loc[:, selected_columns] = filtered_df[selected_columns].apply(pd.to_numeric, errors='coerce')
    
    # 计算用电量总额
    total_consumption = filtered_df[selected_columns].sum().sum()
    
    # 统计行数
    num_rows = filtered_df.shape[0]

    # 输出结果
    print(f'第三产业 2023:\n筛选后的数据行数: {num_rows}')
    print(f'用电量总额: {round(total_consumption / 10000, 2)}万千瓦时')
    print('---------------------------------------------------------------')

def analysis_第二产业(df: pd.DataFrame):
    category = '第二产业'
    
    mask = (df['产业类别（第一、二、三产业、居民）'] == category)

    # 筛选数据
    filtered_df = df.loc[mask].copy()  # 使用 .copy() 创建副本
    
    # 确保列名以 '电量' 开头
    electricity_columns = [col for col in filtered_df.columns if col.startswith('电量')]
    
    # 只选择需要的列
    selected_columns = [col for col in electricity_columns if '202401' <= col[-6:] <= '202408']
    
    # 将所选列转为数字（处理非数字值或缺失值）
    filtered_df.loc[:, selected_columns] = filtered_df[selected_columns].apply(pd.to_numeric, errors='coerce')
    
    # 计算用电量总额
    total_consumption = filtered_df[selected_columns].sum().sum()
    
    # 统计行数
    num_rows = filtered_df.shape[0]


    
    # 只选择需要的列
    selected_columns_2023 = [col for col in electricity_columns if '202301' <= col[-6:] <= '202308']
    
    # 将所选列转为数字（处理非数字值或缺失值）
    filtered_df.loc[:, selected_columns_2023] = filtered_df[selected_columns_2023].apply(pd.to_numeric, errors='coerce')
    
    # 计算用电量总额
    total_consumption_2023 = filtered_df[selected_columns_2023].sum().sum()
    yay_consumption = (total_consumption - total_consumption_2023) / total_consumption_2023 * 100
    # 输出结果
    print(f'第二产业 2024:\n筛选后的数据行数: {num_rows}')
    print(f'用电量总额: {round(total_consumption / 10000, 2)}万千瓦时')
    print(f'同比：{yay_consumption}%')
    print('---------------------------------------------------------------')


    # 只选择需要的列
    selected_columns = [col for col in electricity_columns if '202301' <= col[-6:] <= '202312']
    
    # 将所选列转为数字（处理非数字值或缺失值）
    filtered_df.loc[:, selected_columns] = filtered_df[selected_columns].apply(pd.to_numeric, errors='coerce')
    
    # 计算用电量总额
    total_consumption = filtered_df[selected_columns].sum().sum()
    
    # 统计行数
    num_rows = filtered_df.shape[0]

    # 输出结果
    print(f'第二产业 2023:\n筛选后的数据行数: {num_rows}')
    print(f'用电量总额: {round(total_consumption / 10000, 2)}万千瓦时')
    print('---------------------------------------------------------------')

def analysis_非居民照明(df: pd.DataFrame):
    category = '非居民照明'
    
    mask = (df['用电类别'].str.contains(category)) & ~df['电压等级'].isin([ '交流10kV',  '交流110kV', '交流35kV' ])

    # 筛选数据
    filtered_df = df.loc[mask].copy()  # 使用 .copy() 创建副本
    
    # 确保列名以 '电量' 开头
    electricity_columns = [col for col in filtered_df.columns if col.startswith('电量')]
    
    # 只选择需要的列
    selected_columns = [col for col in electricity_columns if '202401' <= col[-6:] <= '202408']
    
    # 将所选列转为数字（处理非数字值或缺失值）
    filtered_df.loc[:, selected_columns] = filtered_df[selected_columns].apply(pd.to_numeric, errors='coerce')
    
    # 计算用电量总额
    total_consumption = filtered_df[selected_columns].sum().sum()
    
    # 统计行数
    num_rows = filtered_df.shape[0]

    # 输出结果
    print(f'筛选后的数据行数: {num_rows}')
    print(f'用电量总额: {round(total_consumption / 10000, 2)}万千瓦时')
    print('---------------------------------------------------------------')

def analysis_普通工业(df: pd.DataFrame):
    category = '普通工业'
    
    mask = (df['用电类别'].str.contains(category)) & ~df['电压等级'].isin([ '交流10kV',  '交流110kV', '交流35kV' ])

    # 筛选数据
    filtered_df = df.loc[mask].copy()  # 使用 .copy() 创建副本
    
    # 确保列名以 '电量' 开头
    electricity_columns = [col for col in filtered_df.columns if col.startswith('电量')]
    
    # 只选择需要的列
    selected_columns = [col for col in electricity_columns if '202401' <= col[-6:] <= '202408']
    
    # 将所选列转为数字（处理非数字值或缺失值）
    filtered_df.loc[:, selected_columns] = filtered_df[selected_columns].apply(pd.to_numeric, errors='coerce')
    
    # 计算用电量总额
    total_consumption = filtered_df[selected_columns].sum().sum()
    
    # 统计行数
    num_rows = filtered_df.shape[0]

    # 输出结果
    print(f'筛选后的数据行数: {num_rows}')
    print(f'用电量总额: {round(total_consumption / 10000, 2)}万千瓦时')
    print('---------------------------------------------------------------')

def analysis_居民生活用电(df: pd.DataFrame):
    category = '居民生活用电'
    
    mask = (df['用电类别'].str.contains(category)) & ~df['电压等级'].isin([ '交流10kV',  '交流110kV', '交流35kV' ])

    # 筛选数据
    filtered_df = df.loc[mask].copy()  # 使用 .copy() 创建副本
    
    # 确保列名以 '电量' 开头
    electricity_columns = [col for col in filtered_df.columns if col.startswith('电量')]
    
    # 只选择需要的列
    selected_columns = [col for col in electricity_columns if '202401' <= col[-6:] <= '202408']
    
    # 将所选列转为数字（处理非数字值或缺失值）
    filtered_df.loc[:, selected_columns] = filtered_df[selected_columns].apply(pd.to_numeric, errors='coerce')
    
    # 计算用电量总额
    total_consumption = filtered_df[selected_columns].sum().sum()
    
    # 统计行数
    num_rows = filtered_df.shape[0]

    # 输出结果
    print(f'筛选后的数据行数: {num_rows}')
    print(f'用电量总额: {round(total_consumption / 10000, 2)}万千瓦时')
    print('---------------------------------------------------------------')

def analysis_居民生活用电(df: pd.DataFrame):
    category = '居民生活用电'
    
    mask = (df['用电类别'].str.contains(category)) & ~df['电压等级'].isin([ '交流10kV',  '交流110kV', '交流35kV' ])

    # 筛选数据
    filtered_df = df.loc[mask].copy()  # 使用 .copy() 创建副本
    
    # 确保列名以 '电量' 开头
    electricity_columns = [col for col in filtered_df.columns if col.startswith('电量')]
    
    # 只选择需要的列
    selected_columns = [col for col in electricity_columns if '202401' <= col[-6:] <= '202408']
    
    # 将所选列转为数字（处理非数字值或缺失值）
    filtered_df.loc[:, selected_columns] = filtered_df[selected_columns].apply(pd.to_numeric, errors='coerce')
    
    # 计算用电量总额
    total_consumption = filtered_df[selected_columns].sum().sum()
    
    # 统计行数
    num_rows = filtered_df.shape[0]

    # 输出结果
    print(f'筛选后的数据行数: {num_rows}')
    print(f'用电量总额: {round(total_consumption / 10000, 2)}万千瓦时')
    print('---------------------------------------------------------------')

def analysis_非工业(df: pd.DataFrame):
    category = '非工业'
    voltages = '交流10kV'

    # 创建筛选条件
    mask = (df['用电类别'] == category) & (df['电压等级'] == voltages)

    # 筛选数据
    filtered_df = df.loc[mask].copy()  # 使用 .copy() 创建副本
    
    # 确保列名以 '电量' 开头
    electricity_columns = [col for col in filtered_df.columns if col.startswith('电量')]
    
    # 只选择需要的列
    selected_columns = [col for col in electricity_columns if '202401' <= col[-6:] <= '202408']
    
    # 将所选列转为数字（处理非数字值或缺失值）
    filtered_df.loc[:, selected_columns] = filtered_df[selected_columns].apply(pd.to_numeric, errors='coerce')
    
    # 计算用电量总额
    total_consumption = filtered_df[selected_columns].sum().sum()
    
    # 统计行数
    num_rows = filtered_df.shape[0]

    # 输出结果
    print(f'电压等级: {voltages}')
    print(f'筛选后的数据行数: {num_rows}')
    print(f'用电量总额: {round(total_consumption / 10000, 2)}万千瓦时')
    print('---------------------------------------------------------------')
    
    mask = (df['用电类别'] == category) & ~df['电压等级'].isin([ '交流10kV',  '交流110kV', '交流35kV' ])

    # 筛选数据
    filtered_df = df.loc[mask].copy()  # 使用 .copy() 创建副本
    
    # 确保列名以 '电量' 开头
    electricity_columns = [col for col in filtered_df.columns if col.startswith('电量')]
    
    # 只选择需要的列
    selected_columns = [col for col in electricity_columns if '202401' <= col[-6:] <= '202408']
    
    # 将所选列转为数字（处理非数字值或缺失值）
    filtered_df.loc[:, selected_columns] = filtered_df[selected_columns].apply(pd.to_numeric, errors='coerce')
    
    # 计算用电量总额
    total_consumption = filtered_df[selected_columns].sum().sum()
    
    # 统计行数
    num_rows = filtered_df.shape[0]

    # 输出结果
    print(f'筛选后的数据行数: {num_rows}')
    print(f'用电量总额: {round(total_consumption / 10000, 2)}万千瓦时')
    print('---------------------------------------------------------------')

def analysis_大工业用电(df: pd.DataFrame):
    category = '大工业用电'
    voltages = [ '交流10kV']

    # 计算每种电压等级下的用电量总额
    for voltage in voltages:
        # 创建筛选条件
        mask = (df['用电类别'] == category) & (df['电压等级'] == voltage)
        
        # 筛选数据
        filtered_df = df.loc[mask].copy()  # 使用 .copy() 创建副本
        
        # 确保列名以 '电量' 开头
        electricity_columns = [col for col in filtered_df.columns if col.startswith('电量')]
        
        # 只选择需要的列
        selected_columns = [col for col in electricity_columns if '202401' <= col[-6:] <= '202408']
        
        # 将所选列转为数字（处理非数字值或缺失值）
        filtered_df.loc[:, selected_columns] = filtered_df[selected_columns].apply(pd.to_numeric, errors='coerce')
        
        # 计算用电量总额
        total_consumption = filtered_df[selected_columns].sum().sum()
        
        # 统计行数
        num_rows = filtered_df.shape[0]

        # 输出结果
        print(f'电压等级: {voltage}')
        print(f'筛选后的数据行数: {num_rows}')
        print(f'用电量总额: {round(total_consumption / 10000, 2)}万千瓦时')
        print('---------------------------------------------------------------')


def analysis_商业用电(df: pd.DataFrame):
    category = '商业用电'
    voltages = ['交流110kV', '交流35kV', '交流10kV', '交流220V', '交流380V']

    # 计算每种电压等级下的用电量总额
    for voltage in voltages:
        # 创建筛选条件
        mask = (df['用电类别'] == category) & (df['电压等级'] == voltage)
        
        # 筛选数据
        filtered_df = df.loc[mask].copy()  # 使用 .copy() 创建副本
        
        # 确保列名以 '电量' 开头
        electricity_columns = [col for col in filtered_df.columns if col.startswith('电量')]
        
        # 只选择需要的列
        selected_columns = [col for col in electricity_columns if '202401' <= col[-6:] <= '202408']
        
        # 将所选列转为数字（处理非数字值或缺失值）
        filtered_df.loc[:, selected_columns] = filtered_df[selected_columns].apply(pd.to_numeric, errors='coerce')
        
        # 计算用电量总额
        total_consumption = filtered_df[selected_columns].sum().sum()
        
        # 统计行数
        num_rows = filtered_df.shape[0]

        # 输出结果
        print(f'电压等级: {voltage}')
        print(f'筛选后的数据行数: {num_rows}')
        print(f'用电量总额: {round(total_consumption / 10000, 2)}万千瓦时')
        print('---------------------------------------------------------------')

    
if __name__ == '__main__':
    # 读取数据
    print('---------------------------------------------------------------')
    file_path = r"C:\Users\Jayttle\Documents\WeChat Files\wxid_uzs67jx3j0a322\FileStorage\File\2024-09\小陆家嘴.xlsx"
    df = pd.read_excel(file_path)
    # get_unique_column(df)
    analysis_产业分析(df, 'output.xlsx')
    # 假设你的数据表中有以下列：
    # '电压等级', '用电类别', '用电量',

