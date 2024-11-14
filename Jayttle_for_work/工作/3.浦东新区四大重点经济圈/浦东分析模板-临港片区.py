import pandas as pd 
import re
import json
import numpy as np


# 两列数据与列名互换
def swap_df_col(idf: pd.DataFrame, colnum_a: int, colnum_b: int):
    # 修正列索引范围检查
    if 0 <= colnum_a < len(idf.columns) and 0 <= colnum_b < len(idf.columns):
        idf_copy = idf.copy()
        list_cols = idf_copy.columns.tolist()
        # 调试输出，检查列索引和名称
        list_cols[colnum_a], list_cols[colnum_b] = list_cols[colnum_b], list_cols[colnum_a]
        return idf_copy.reindex(columns=list_cols)
    else:
        raise IndexError(f"Invalid column indices: {colnum_a}, {colnum_b}")

def df_swap_col(idf: pd.DataFrame, colname_a: str, colname_b: str):
    if not isinstance(idf, pd.DataFrame):
        raise TypeError("Input must be a DataFrame")
    if colname_a in idf.columns and colname_b in idf.columns:
        return swap_df_col(idf, idf.columns.get_loc(colname_a), idf.columns.get_loc(colname_b))
    else:
        raise ValueError(f"Column names not found: {colname_a} or {colname_b}")

def read_json_file(json_file: str):
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def check_duplicates(category_list):
    seen = set()
    duplicates = set()
    
    for item in category_list:
        if item in seen:
            duplicates.add(item)
        else:
            seen.add(item)
    
    return duplicates
def check_json_duplicates(json_file: str):
    data = read_json_file(json_file)
    
    # 检查每个类别的重复项
    for category, items in data.items():
        duplicates = check_duplicates(items)
        if duplicates:
            print(f"重复项在'{category}'中:")
            for duplicate in duplicates:
                print(f"  {duplicate}")
        else:
            print(f"'{category}'中没有重复项")
    
    # 合并所有类别的项
    all_items = []
    for items in data.values():
        all_items.extend(items)
    
    # 检查合并后的列表中的重复项
    overall_duplicates = check_duplicates(all_items)
    if overall_duplicates:
        print("在所有类别中找到的重复项:")
        for duplicate in overall_duplicates:
            print(f"  {duplicate}")
    else:
        print("在所有类别中没有找到重复项")
    
def split_df_by_category(df: pd.DataFrame, categories: dict) -> dict:
    category_dfs = {}
    df_filtered = df[
        (df['产业类别'] == '第一产业') |
        (df['产业类别'] == '第二产业') |
        (df['产业类别'] == '第三产业')
    ]
    covered_categories = set()
    for key, value in categories.items():
        if callable(value):
            mask = df_filtered['行业分类'].apply(value)
        else:
            mask = df_filtered['行业分类'].isin(value)
        filtered_df = df_filtered[mask].copy()
        category_dfs[key] = filtered_df
        covered_categories.update(filtered_df['行业分类'].unique())
    all_categories = set(df_filtered['行业分类'].unique())
    uncovered_categories = all_categories - covered_categories
    print("未被涵盖的行业分类：", uncovered_categories)
    return category_dfs

def analysis_行业分析(df: pd.DataFrame, json_file: str) -> pd.DataFrame:
    if json_file is None:
        return pd.DataFrame
    categories = read_json_file(json_file)
    categorized_dfs = split_df_by_category(df, categories)
    results_list = []
    total_electricity_2023 = 0
    total_electricity_2024 = 0
    
    for key, filtered_df in categorized_dfs.items():
        electricity_columns = [col for col in filtered_df.columns if col.startswith('电量')]
        yearly_totals = {year: 0 for year in range(2020, 2025)}
        for col in electricity_columns:
            match = re.match(r'电量(\d{4})(\d{2})', col)
            if match:
                year = int(match.group(1))
                if year in yearly_totals:
                    filtered_df[col] = pd.to_numeric(filtered_df[col], errors='coerce')
                    yearly_totals[year] += filtered_df[col].sum() / 10000
        total_electricity_2023 += yearly_totals.get(2023, 0)
        total_electricity_2024 += yearly_totals.get(2024, 0)
        num_rows = filtered_df.shape[0]
        results_list.append({
            '类别': key,
            '筛选后的数据行数': num_rows,
            **{f'{year}年电量总和': yearly_totals.get(year, 0) for year in range(2020, 2025)}
        })
    
    df_results = pd.DataFrame(results_list)
    df_results['2023年用电量占比'] = round(df_results['2023年电量总和'] / total_electricity_2023, 4) if total_electricity_2023 != 0 else 0
    df_results['2024年用电量占比'] = round(df_results['2024年电量总和'] / total_electricity_2024, 4) if total_electricity_2024 != 0 else 0

    return df_results

def analysis_行业分析_同比(df: pd.DataFrame, json_file: str) -> pd.DataFrame:
    if json_file is None:
        return pd.DataFrame
    categories = read_json_file(json_file)
    categorized_dfs = split_df_by_category(df, categories)
    results_list = []
    
    for key, filtered_df in categorized_dfs.items():
        electricity_columns = [col for col in filtered_df.columns if col.startswith('电量')]
        monthly_totals_2023 = {month: 0 for month in range(1, 9)}
        monthly_totals_2024 = {month: 0 for month in range(1, 9)}
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
        total_electricity_2023 = sum(monthly_totals_2023.values())
        total_electricity_2024 = sum(monthly_totals_2024.values())
        year_on_year_growth = (
            ((total_electricity_2024 - total_electricity_2023) / total_electricity_2023) 
            if total_electricity_2023 != 0 else 0
        )
        num_rows = filtered_df.shape[0]
        results_list.append({
            '类别': key,
            '筛选后的数据行数': num_rows,
            '2023年1月至8月电量总和': round(total_electricity_2023, 2),
            '2024年1月至8月电量总和': round(total_electricity_2024, 2),
            '同比增长率': round(year_on_year_growth, 4)
        })
    
    df_results = pd.DataFrame(results_list)
    total_2023 = df_results['2023年1月至8月电量总和'].sum()
    total_2024 = df_results['2024年1月至8月电量总和'].sum()
    yay_total = round((total_2024 - total_2023) / total_2023, 4)
    print(f'analysis_行业分析_同比 总和的同比是:{yay_total}')

    return df_results

def analysis_产业分析(df: pd.DataFrame):
    # 定义类别
    categories = {
        '第一产业': '第一产业',
        '第二产业': '第二产业',
        '第三产业': '第三产业',
        '其他': lambda x: x not in ['第一产业', '第二产业', '第三产业']
    }
    
    # 创建一个列表来存储每种类别的筛选结果
    results = []
    total_electricity_2023_0108 = 0
    total_electricity_2023 = 0
    total_electricity_2024 = 0
    
    for key, value in categories.items():
        if key == '其他':
            mask = df['产业类别'].apply(value)
        else:
            mask = df['产业类别'] == value
        
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
        
        yay_sum = round((sum_2024 - sum_2023_0108) / sum_2023_0108, 4)

        # 存储结果
        results.append({
            '类别': key,
            '2020年电量总和': round(sum_2020, 2),
            '2021年电量总和': round(sum_2021, 2),
            '2022年电量总和': round(sum_2022, 2),
            '2023年电量总和': round(sum_2023, 2),
            '2024年电量总和': round(sum_2024, 2),
            '2024年同比': yay_sum,
        })
    
    # 计算各产业占比
    for result in results:
        result['2023年占比'] = (round(result['2023年电量总和'] / total_electricity_2023 , 4)) if total_electricity_2023 > 0 else 0
        result['2024年占比'] = (round(result['2024年电量总和'] / total_electricity_2024 , 4)) if total_electricity_2024 > 0 else 0
    
    yay = (total_electricity_2024 - total_electricity_2023_0108) / total_electricity_2023_0108 
    results.append({
        '总用电量的同比是': round(yay, 4)
    })
    
    # 转换结果为 DataFrame 并返回
    df_results = pd.DataFrame(results)
    
    return df_results

def analysis_总售电量(df: pd.DataFrame) -> pd.DataFrame:
    years = ['2020', '2021', '2022', '2023', '2024']
    result_data = []

    for year in years:
        electricity_columns = [col for col in df.columns if col.startswith('电量')]
        selected_columns = [col for col in electricity_columns if col[-6:-2] == year]
        df.loc[:, selected_columns] = df[selected_columns].apply(pd.to_numeric, errors='coerce')
        total_consumption = df[selected_columns].sum().sum()
        num_rows = df.shape[0]
        result_data.append({
            '类别': year,
            '筛选后的数据行数': num_rows,
            '年度售电量总额(亿千瓦时)': round(total_consumption / 10000 /10000, 2)
        })

    electricity_columns = [col for col in df.columns if col.startswith('电量')]
    selected_columns = [col for col in electricity_columns if '202301' <= col[-6:] <= '202308']
    df.loc[:, selected_columns] = df[selected_columns].apply(pd.to_numeric, errors='coerce')
    total_consumption = df[selected_columns].sum().sum()
    num_rows = df.shape[0]
    result_data.append({
        '类别': '20230108',
        '筛选后的数据行数': num_rows,
        '年度售电量总额(亿千瓦时)': round(total_consumption / 10000 /10000, 2)
    })

    result_df = pd.DataFrame(result_data)
    # 调整“20230108”行的位置
    result_df = result_df.set_index('类别')
    order = ['2020', '2021', '2022', '2023', '20230108', '2024']
    result_df = result_df.reindex(order).reset_index()
    # 计算同比
    result_df.set_index('类别', inplace=True)
    result_df['同比增长率 (%)'] = round(result_df['年度售电量总额(亿千瓦时)'].pct_change(), 4)
        
    # 设置“20230108”的同比增长率为NaN
    result_df.loc['20230108', '同比 (%)'] = np.nan
    result_df.reset_index(inplace=True)
    return result_df


def analysis(df: pd.DataFrame, category: str) -> pd.DataFrame:
    voltages = ['交流220kV', '交流110kV', '交流35kV', '交流10kV', '低压']
    results = []

    for voltage in voltages:
        if voltage == '低压':
            mask = (df['用电类别'].str.contains(category)) & (df['电压等级'].isin(['交流220V', '交流380V']))
        else:
            mask = (df['用电类别'].str.contains(category)) & (df['电压等级'] == voltage)
        

        filtered_df = df.loc[mask].copy()
        
        electricity_columns = [col for col in filtered_df.columns if col.startswith('电量')]
        selected_columns = [col for col in electricity_columns if '202401' <= col[-6:] <= '202408']
        filtered_df.loc[:, selected_columns] = filtered_df[selected_columns].apply(pd.to_numeric, errors='coerce')
        total_consumption = filtered_df[selected_columns].sum().sum()
        num_rows = filtered_df.shape[0]

        results.append({
            '用电类别': category,
            '电压等级': voltage,
            '筛选后的数据行数': num_rows,
            '用电量总额（万千瓦时）': round(total_consumption / 10000, 2)
        })

    result_df = pd.DataFrame(results)
    return result_df

def save_results_to_excel(df: pd.DataFrame, json_file: str = None):
    categories = ['商业用电', '大工业用电','非工业','居民生活用电' ,'普通工业','非居民照明','农业生产用电', '学校教学用电']
    dfs = [analysis(df, category) for category in categories]
    result_df = pd.concat(dfs, ignore_index=True)
    # 清理筛选后的数据行数为0的数据行
    result_df = result_df[result_df['筛选后的数据行数'] != 0]
    results = {
        '用电类别': result_df,
        '总售电量': analysis_总售电量(df),
        '产业分析': analysis_产业分析(df),
        '行业分析': analysis_行业分析(df, json_file),
        '行业分析同比': analysis_行业分析_同比(df, json_file),
    }
    
    # Merge '行业分析' and '行业分析同比' based on '类别'
    industry_analysis = results['行业分析']
    industry_analysis_yoy = results['行业分析同比']
    
    # 保留行业分析同比中需要的列
    industry_analysis_yoy_filtered = industry_analysis_yoy[['类别', '同比增长率']]
    
    # Merge with keeping '行业分析' order
    merged_analysis = pd.merge(industry_analysis, industry_analysis_yoy_filtered, on='类别', how='outer', suffixes=('_current', '_yoy'))
    
    # Reorder merged_analysis to match the order of industry_analysis
    merged_analysis = merged_analysis.set_index('类别').reindex(industry_analysis['类别']).reset_index()
    df_swap_col(merged_analysis, '2024年电量总和', '2023年用电量占比')
    # Update results with the merged DataFrame
    results['行业分析'] = merged_analysis
    results.pop('行业分析同比')  # Remove the old '行业分析同比' key

    with pd.ExcelWriter('分析结果.xlsx') as writer:
        for sheet_name, result_df in results.items():
            result_df.to_excel(writer, sheet_name=sheet_name, index=False)
            


if __name__ == '__main__':
    # 读取数据
    print('---------------------------------------------------------------')
    file_path = r"C:\Users\juntaox\Documents\WeChat Files\wxid_uzs67jx3j0a322\FileStorage\File\2024-09\临港202001-202408用户电量明细(1).xlsx"
    json_file = r'E:\vscode_proj\临港片区分类.json'
    sheet_name = 'Select code_cls_val'  # 指定工作表名称
    check_json_duplicates(json_file)
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    save_results_to_excel(df, json_file)