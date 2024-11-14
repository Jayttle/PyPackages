import pandas as pd 
import re
import json

def split_df_by_category(df: pd.DataFrame) -> dict:
    # 创建一个字典来存储按分类拆分后的 DataFrame
    category_dfs = {}
    
    # 仅筛选产业类别为第二产业或第三产业的行
    df_filtered = df[
        (df['产业类别（第一、二、三产业、居民）'] == '第一产业') |
        (df['产业类别（第一、二、三产业、居民）'] == '第二产业') |
        (df['产业类别（第一、二、三产业、居民）'] == '第三产业')
    ]
    
    # 按照产业类别（第二产业和第三产业）拆分数据
    for category in ['第一产业', '第二产业', '第三产业']:
        filtered_df = df_filtered[df_filtered['产业类别（第一、二、三产业、居民）'] == category].copy()
        category_dfs[category] = filtered_df
    
    return category_dfs

def analysis_行业分析_获取行业每月用电量(df: pd.DataFrame, output_file: str):
    
    # 将 DataFrame 按分类拆分（假设 split_df_by_category 已经定义）
    categorized_dfs = split_df_by_category(df)
    
    # 创建一个列表来存储每种类别的分析结果
    results_list = []
    
    # 用于统计所有产业的总用电量
    total_all_industries_electricity = 0
    
    for key, filtered_df in categorized_dfs.items():
        # 找到所有以 '电量' 开头的列，且匹配 '电量(\d{4})(\d{2})' 格式的列
        electricity_columns = [col for col in filtered_df.columns if re.match(r'电量(\d{4})(\d{2})', col)]
        
        # 统计每个分类下符合条件的 '电量' 列的总和
        total_electricity = {}
        
        for col in electricity_columns:
            # 获取每个电量列的总和并转换为原生整数类型
            total_electricity[col] = int(filtered_df[col].sum())
        
        # 计算该产业类别的总电量
        industry_total = sum(total_electricity.values())
        
        # 将结果存储到 results_list 中
        results_list.append({
            'category': key,
            'total_electricity': total_electricity,
            'industry_total': industry_total
        })
        
        # 累加到总电量
        total_all_industries_electricity += industry_total
    
    # 将结果转换为 DataFrame
    results_df = pd.DataFrame(results_list)
    
    # 展开 'total_electricity' 字典，将每个 '电量' 的列转换为 DataFrame 的独立列
    electricity_df = pd.json_normalize(results_df['total_electricity'])
    
    # 合并分类信息和展开的电量列
    final_df = pd.concat([results_df[['category', 'industry_total']], electricity_df], axis=1)
    
    # 计算每个产业的占比并附加到 DataFrame
    final_df['industry_percentage'] = final_df['industry_total'] / total_all_industries_electricity * 100
    
    # 将结果保存到输出文件（例如，Excel 格式）
    final_df.to_excel(output_file, index=False)
    
    return final_df

if __name__ == '__main__':
    # 读取数据
    print('---------------------------------------------------------------')
    region_names = ['川沙', '祝桥', '小陆家嘴']
    for region in region_names:
        file_path = rf"C:\Users\juntaox\Desktop\工作\25.GDP与售电量比值预测\行业预测\原始材料\{region}.xlsx"
        df = pd.read_excel(file_path)
        out_put_file = rf"C:\Users\juntaox\Desktop\工作\25.GDP与售电量比值预测\行业预测\{region}二、三产业.xlsx"
        analysis_行业分析_获取行业每月用电量(df, out_put_file)
