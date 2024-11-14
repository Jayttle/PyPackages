import pandas as pd
import json
import os

def read_json_file(json_file: str):
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def process_year_data(year, categories):
    # 读取对应年份的Excel文件
    person_df = pd.read_excel(r"C:\Users\juntaox\Desktop\工作\3.浦东新区四大重点经济圈发展分析\人口统计年鉴.xlsx", sheet_name=str(year))
    
    # 创建一个字典来存储每个区域的统计信息
    region_stats = {region: {'moved_in': 0, 'moved_out': 0, 'net_mobility': 0} for region in categories.keys()}
    
    # 遍历每一行数据，更新各个区域的统计信息
    for index, row in person_df.iterrows():
        # 获取街道/镇名称和对应的迁入、迁出和净迁入数据
        area_name = row['Unnamed: 0'].strip()
        moved_in = row['迁 入\r\nPopulation\r\nMoved-in']
        moved_out = row['迁 出\r\nPopulation\r\nMoved-out']
        net_mobility = row['净迁入\r\nNet Mobility']
        
        # 确定该街道/镇属于哪个区域
        for region, areas in categories.items():
            if area_name in areas:
                region_stats[region]['moved_in'] += moved_in
                region_stats[region]['moved_out'] += moved_out
                region_stats[region]['net_mobility'] += net_mobility
                break  # 找到对应区域后跳出循环

    return region_stats

if __name__ == '__main__':
    # 读取 JSON 文件
    categories = read_json_file(r"E:\vscode_proj\浦东新区区域划分.json")

    # 文件路径
    file_path = r"C:\Users\juntaox\Desktop\工作\3.浦东新区四大重点经济圈发展分析\人口统计年鉴.xlsx"

    # 统计每一年的数据
    years = [2019, 2020, 2021, 2022]
    all_year_stats = {}

    for year in years:
        year_stats = process_year_data(year, categories)
        all_year_stats[year] = year_stats

    # 写入到对应的工作表
    with pd.ExcelWriter(file_path, engine='openpyxl', mode='a' if os.path.exists(file_path) else 'w') as writer:
        for year, stats in all_year_stats.items():
            result_df = pd.DataFrame(stats).T  # 转置以便于写入
            result_df.index.name = '区域'
            result_df.reset_index(inplace=True)

            # 给列添加属性
            result_df.columns = ['区域', '迁入', '迁出', '净人口']

            # 写入对应的年份工作表，命名为 {year}人口迁移情况
            sheet_name = f"{year}人口迁移情况"
            result_df.to_excel(writer, sheet_name=sheet_name, index=False)

            print(f"{year}年的数据已写入到工作表 {sheet_name} 中。")
