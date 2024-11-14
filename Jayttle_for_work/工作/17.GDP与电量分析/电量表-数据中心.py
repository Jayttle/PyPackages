import pandas as pd

# 读取Excel文件
file_path = r"C:\Users\juntaox\Desktop\工作\17.浦东新区售电量与GDP关联分析报告\数据表\电量表-数据中心.xlsx"
df = pd.read_excel(file_path)

# 使用正则表达式筛选列名
filtered_columns = [col for col in df.columns if pd.Series(col).str.match(r'^电量\d{6}$').any()]

# 显示筛选后的列名
print("筛选后的列名:", filtered_columns)

# 创建一个字典来存储每个年份的总和
yearly_sums = {}

# 遍历筛选后的列名并计算每个年份的总和
for col in filtered_columns:
    year = col[2:6]  # 提取年份
    
    # 如果年份在我们关心的范围内，计算总和
    if year in ['2020', '2021', '2022', '2023', '2024']:
        if year not in yearly_sums:
            yearly_sums[year] = 0
        
        # 累加该列的总和
        yearly_sums[year] += df[col].sum() / 10000 / 10000

# 将结果转换为DataFrame
result_df = pd.DataFrame(yearly_sums.items(), columns=['年份', '总和'])

# 显示结果
print(result_df)
