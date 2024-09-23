import pandas as pd 
import re
import json

def analysis_非居民照明(df: pd.DataFrame) -> pd.DataFrame:
    category = '非居民照明'
    mask = (df['用电类别'].str.contains(category)) & ~df['电压等级'].isin(['交流10kV', '交流110kV', '交流35kV'])
    filtered_df = df.loc[mask].copy()
    electricity_columns = [col for col in filtered_df.columns if col.startswith('电量')]
    selected_columns = [col for col in electricity_columns if '202401' <= col[-6:] <= '202408']
    filtered_df.loc[:, selected_columns] = filtered_df[selected_columns].apply(pd.to_numeric, errors='coerce')
    total_consumption = filtered_df[selected_columns].sum().sum()
    num_rows = filtered_df.shape[0]
    result = pd.DataFrame({
        '类别': [category],
        '筛选后的数据行数': [num_rows],
        '用电量总额 (万千瓦时)': [round(total_consumption / 10000, 2)]
    })
    return result

def analysis_普通工业(df: pd.DataFrame) -> pd.DataFrame:
    category = '普通工业'
    mask = (df['用电类别'].str.contains(category)) & ~df['电压等级'].isin(['交流10kV', '交流110kV', '交流35kV'])
    filtered_df = df.loc[mask].copy()
    electricity_columns = [col for col in filtered_df.columns if col.startswith('电量')]
    selected_columns = [col for col in electricity_columns if '202401' <= col[-6:] <= '202408']
    filtered_df.loc[:, selected_columns] = filtered_df[selected_columns].apply(pd.to_numeric, errors='coerce')
    total_consumption = filtered_df[selected_columns].sum().sum()
    num_rows = filtered_df.shape[0]
    result = pd.DataFrame({
        '类别': [category],
        '筛选后的数据行数': [num_rows],
        '用电量总额 (万千瓦时)': [round(total_consumption / 10000, 2)]
    })
    return result

def analysis_居民生活用电(df: pd.DataFrame) -> pd.DataFrame:
    category = '居民生活用电'
    mask = (df['用电类别'].str.contains(category)) & ~df['电压等级'].isin(['交流10kV', '交流110kV', '交流35kV'])
    filtered_df = df.loc[mask].copy()
    electricity_columns = [col for col in filtered_df.columns if col.startswith('电量')]
    selected_columns = [col for col in electricity_columns if '202401' <= col[-6:] <= '202408']
    filtered_df.loc[:, selected_columns] = filtered_df[selected_columns].apply(pd.to_numeric, errors='coerce')
    total_consumption = filtered_df[selected_columns].sum().sum()
    num_rows = filtered_df.shape[0]
    result = pd.DataFrame({
        '类别': [category],
        '筛选后的数据行数': [num_rows],
        '用电量总额 (万千瓦时)': [round(total_consumption / 10000, 2)]
    })
    return result


def analysis_商业用电(df: pd.DataFrame) -> pd.DataFrame:
    category = '商业用电'
    voltages = ['交流110kV', '交流35kV', '交流10kV', '交流220V', '交流380V']

    results = []

    for voltage in voltages:
        mask = (df['用电类别'] == category) & (df['电压等级'] == voltage)
        filtered_df = df.loc[mask].copy()
        
        electricity_columns = [col for col in filtered_df.columns if col.startswith('电量')]
        selected_columns = [col for col in electricity_columns if '202401' <= col[-6:] <= '202408']
        filtered_df.loc[:, selected_columns] = filtered_df[selected_columns].apply(pd.to_numeric, errors='coerce')
        total_consumption = filtered_df[selected_columns].sum().sum()
        num_rows = filtered_df.shape[0]

        results.append({
            '电压等级': voltage,
            '筛选后的数据行数': num_rows,
            '用电量总额（万千瓦时）': round(total_consumption / 10000, 2)
        })

    result_df = pd.DataFrame(results)
    return result_df

def analysis_非工业(df: pd.DataFrame) -> pd.DataFrame:
    category = '非工业'
    voltages = '交流10kV'
    mask = (df['用电类别'] == category) & (df['电压等级'] == voltages)
    filtered_df = df.loc[mask].copy()
    electricity_columns = [col for col in filtered_df.columns if col.startswith('电量')]
    selected_columns = [col for col in electricity_columns if '202401' <= col[-6:] <= '202408']
    filtered_df.loc[:, selected_columns] = filtered_df[selected_columns].apply(pd.to_numeric, errors='coerce')
    total_consumption = filtered_df[selected_columns].sum().sum()
    num_rows = filtered_df.shape[0]
    result_10kV = pd.DataFrame({
        '类别': [category],
        '电压等级': [voltages],
        '筛选后的数据行数': [num_rows],
        '用电量总额 (万千瓦时)': [round(total_consumption / 10000, 2)]
    })
    
    mask = (df['用电类别'] == category) & ~df['电压等级'].isin(['交流10kV', '交流110kV', '交流35kV'])
    filtered_df = df.loc[mask].copy()
    electricity_columns = [col for col in filtered_df.columns if col.startswith('电量')]
    selected_columns = [col for col in electricity_columns if '202401' <= col[-6:] <= '202408']
    filtered_df.loc[:, selected_columns] = filtered_df[selected_columns].apply(pd.to_numeric, errors='coerce')
    total_consumption = filtered_df[selected_columns].sum().sum()
    num_rows = filtered_df.shape[0]
    result_other = pd.DataFrame({
        '类别': [category],
        '电压等级': ['其他'],
        '筛选后的数据行数': [num_rows],
        '用电量总额 (万千瓦时)': [round(total_consumption / 10000, 2)]
    })
    
    return pd.concat([result_10kV, result_other], ignore_index=True)

def save_results_to_excel(df: pd.DataFrame):
    results = {
        '商业用电': analysis_商业用电(df),
        '非居民照明': analysis_非居民照明(df),
        '普通工业': analysis_普通工业(df),
        '居民生活用电': analysis_居民生活用电(df),
        '非工业': analysis_非工业(df)
    }
    
    with pd.ExcelWriter('分析结果.xlsx') as writer:
        for sheet_name, result_df in results.items():
            result_df.to_excel(writer, sheet_name=sheet_name, index=False)

if __name__ == '__main__':
    # 读取数据
    print('---------------------------------------------------------------')
    file_path = r"C:\Users\Jayttle\Documents\WeChat Files\wxid_uzs67jx3j0a322\FileStorage\File\2024-09\小陆家嘴.xlsx"
    df = pd.read_excel(file_path)
    # get_unique_column(df)
    save_results_to_excel(df)
    # 假设你的数据表中有以下列：
    # '电压等级', '用电类别', '用电量',

