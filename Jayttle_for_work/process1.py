import pandas as pd



def main1():
    # clear_log(log_file_path)
    print("------------------------------run------------------------------")
    
    file_path = r"C:\Users\juntaox\Desktop\用电地址_提取.xlsx"
    
    # 读取数据
    df = pd.read_excel(file_path)
    
    address_to_station_dict = []
    # 确保 '小区站点的地址' 列存在
    if '小区站点的地址' in df.columns:
        # 从 '小区站点的地址' 列中提取 "弄" 字之前的部分
        df['用电地址_提取'] = df['小区站点的地址'].str.split('弄').str[0]
        
        # 删除 '用电地址_提取' 列中前面的“上海市浦东新区”
        df['用电地址_提取'] = df['用电地址_提取'].str.replace('上海市浦东新区', '', regex=False)
        
        # 确保 '小区站点名称' 列存在
        if '小区站点名称' in df.columns:
            # 创建字典，键为 '用电地址_提取' 的值，值为 '小区站点名称' 的值
            address_to_station_dict = df.set_index('用电地址_提取')['小区站点名称'].to_dict()
            
            # 打印或处理字典
            print(address_to_station_dict)
        else:
            print("列 '小区站点名称' 不存在")
    else:
        print("列 '小区站点的地址' 不存在")
    return address_to_station_dict


def main() -> None:
    print("------------------------------run------------------------------")
    
    file_path = r"C:\Users\juntaox\Desktop\用电地址_提取_top2.xlsx"
    
    # 读取数据
    df = pd.read_excel(file_path)
    
    # 统计 '用电地址_提取' 列的值频率
    unique_values = df['用电地址_提取'].value_counts()
    
    # 转换为 DataFrame
    frequency_df = unique_values.reset_index()
    frequency_df.columns = ['用电地址_提取', '频率']
    
    # 创建一个字典，用于存储每个关键字的 DataFrame
    keywords = ['惠南镇', '康桥镇', '周浦镇', '三林镇', '川沙新镇', '航头镇']
    filtered_dfs = {}
    
    for keyword in keywords:
        # 过滤出包含关键字的行
        filtered_df = frequency_df[frequency_df['用电地址_提取'].str.contains(keyword, na=False)]
        filtered_dfs[keyword] = filtered_df
    
    # 计算未匹配到关键字的频次数据
    matched_df = pd.concat(filtered_dfs.values())
    other_df = frequency_df[~frequency_df['用电地址_提取'].isin(matched_df['用电地址_提取'])]
    
    # 保存到 Excel 文件的不同工作表中
    output_file_path = r"C:\Users\juntaox\Desktop\用电地址_提取_频率_by_keyword.xlsx"
    with pd.ExcelWriter(output_file_path) as writer:
        for keyword, filtered_df in filtered_dfs.items():
            filtered_df.to_excel(writer, sheet_name=keyword, index=False)
        
        # 保存未匹配到的频次数据到 "其他" 工作表
        other_df.to_excel(writer, sheet_name='其他', index=False)
    
    print(f"频率统计已保存到 {output_file_path}")

if __name__ == '__main__':
    main()