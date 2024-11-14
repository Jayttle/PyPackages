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
        (df['产业类别（第一、二、三产业、居民）'] == '第一产业') |
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

def analysis_行业分析_获取行业每月用电量(df: pd.DataFrame, json_file: str, output_file: str):
    # 读取 JSON 文件（假设 read_json_file 已经定义）
    categories = read_json_file(json_file)
    
    # 将 DataFrame 按分类拆分（假设 split_df_by_category 已经定义）
    categorized_dfs = split_df_by_category(df, categories)
    
    # 创建一个列表来存储每种类别的分析结果
    results_list = []
    
    for key, filtered_df in categorized_dfs.items():
        # 找到所有以 '电量' 开头的列，且匹配 '电量(\d{4})(\d{2})' 格式的列
        electricity_columns = [col for col in filtered_df.columns if re.match(r'电量(\d{4})(\d{2})', col)]
        
        # 统计每个分类下符合条件的 '电量' 列的总和
        total_electricity = {}
        
        for col in electricity_columns:
            # 获取每个电量列的总和并转换为原生整数类型
            total_electricity[col] = int(filtered_df[col].sum())
        
        # 将结果存储到 results_list 中
        results_list.append({
            'category': key,
            'total_electricity': total_electricity
        })
    
    # 将结果转换为 DataFrame
    results_df = pd.DataFrame(results_list)
    
    # 展开 'total_electricity' 字典，将每个 '电量' 的列转换为 DataFrame 的独立列
    electricity_df = pd.json_normalize(results_df['total_electricity'])
    # 合并分类信息和展开的电量列
    final_df = pd.concat([results_df['category'], electricity_df], axis=1)
    # 将结果保存到输出文件（例如，Excel 格式）
    final_df.to_excel(output_file, index=False)
    
    return final_df
           
if __name__ == '__main__':
    # 读取数据
    print('---------------------------------------------------------------')
    file_path = r"E:\vscode_proj\小陆家嘴.xlsx"
    df = pd.read_excel(file_path)
    json_file = r"E:\OneDrive\PyPackages\Jayttle_for_work\3.浦东新区四大重点经济圈\陆家嘴行业分类.json"
    out_put_file = r'陆家嘴行业分析结果.xlsx'
    analysis_行业分析_获取行业每月用电量(df, json_file, out_put_file)
