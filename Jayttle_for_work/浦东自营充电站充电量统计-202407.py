import pandas as pd

def main() -> None:
    # 打印运行信息
    print("------------------------------run------------------------------")
    
    # 文件路径
    file_path = r"C:\Users\juntaox\Desktop\工作待办\浦东月报\浦东自营充电站充电量统计-202407.xlsx"
    
    # 读取 Excel 文件
    df = pd.read_excel(file_path)
    
    # 为第一列添加表头
    df.columns = ['月份'] + df.columns[1:].tolist()
    
    # 确保 '月份' 列为字符串类型
    df['月份'] = df['月份'].astype(str).str.strip()
    
    # 打印数据以确认修改
    print(df.head())
    
    # 过滤月份范围 202401 到 202407
    df_filtered_2024 = df[(df['月份'] >= '202401') & (df['月份'] <= '202407')]
    
    # 过滤月份范围 202301 到 202307
    df_filtered_2023 = df[(df['月份'] >= '202301') & (df['月份'] <= '202307')]
    

    # 计算所需列的总和（2024年数据）
    total_dc_public_2024 = round(df_filtered_2024['直流城市公共'].sum() / 10000, 3)
    total_ac_public_2024 = round(df_filtered_2024['交流城市公共'].sum() / 10000, 3)
    total_dc_bus_2024 = round(df_filtered_2024['直流公交'].sum() / 10000, 3)
    total_dc_fast_2024 = round(df_filtered_2024['直流高速'].sum() / 10000, 3)
    total_ac_residential_2024 = round(df_filtered_2024['交流小区'].sum() / 10000, 3)
    total_dc_internal_2024 = round(df_filtered_2024['直流单位内部'].sum() / 10000, 3)
    total_ac_internal_2024 = round(df_filtered_2024['交流单位内部'].sum() / 10000, 3)

    # 计算所需列的总和（2023年数据）
    total_dc_public_2023 = round(df_filtered_2023['直流城市公共'].sum() / 10000, 3)
    total_ac_public_2023 = round(df_filtered_2023['交流城市公共'].sum() / 10000, 3)
    total_dc_bus_2023 = round(df_filtered_2023['直流公交'].sum() / 10000, 3)
    total_dc_fast_2023 = round(df_filtered_2023['直流高速'].sum() / 10000, 3)
    total_ac_residential_2023 = round(df_filtered_2023['交流小区'].sum() / 10000, 3)
    total_dc_internal_2023 = round(df_filtered_2023['直流单位内部'].sum() / 10000, 3)
    total_ac_internal_2023 = round(df_filtered_2023['交流单位内部'].sum() / 10000, 3)
    # 计算城市公共总和
    total_city_public_2024 = round(total_dc_public_2024 + total_ac_public_2024, 3)
    total_city_public_2023 = round(total_dc_public_2023 + total_ac_public_2023, 3)
    
    
    total_internal_2024 = round(total_dc_internal_2024 + total_ac_internal_2024, 3)
    total_internal_2023 = round(total_dc_internal_2023 + total_ac_internal_2023, 3)
    
    sum_total_2024 = round(total_internal_2024 + total_city_public_2024 + total_dc_bus_2024 + total_dc_fast_2024 + total_ac_residential_2024, 3)
    sum_total_2023 = round(total_internal_2023 + total_city_public_2023 + total_dc_bus_2023 + total_dc_fast_2023 + total_ac_residential_2023, 3)
    # 计算同比
    yoy_city_public = round((total_city_public_2024 - total_city_public_2023) / total_city_public_2023 * 100, 2) if total_city_public_2023 != 0 else float('inf')
    yoy_interal = round((total_internal_2024 - total_internal_2023) / total_internal_2023 * 100, 2) if total_internal_2023 != 0 else float('inf')
    yoy_dc_bus = round((total_dc_bus_2024 - total_dc_bus_2023) / total_dc_bus_2023 * 100, 2) if total_dc_bus_2023 != 0 else float('inf')
    yoy_dc_fast = round((total_dc_fast_2024 - total_dc_fast_2023) / total_dc_fast_2023 * 100, 2) if total_dc_fast_2023 != 0 else float('inf')
    yoy_ac_residential = round((total_ac_residential_2024 - total_ac_residential_2023) / total_ac_residential_2023 * 100, 2) if total_ac_residential_2023 != 0 else float('inf')
    yoy_sum_total = round((sum_total_2024 - sum_total_2023) / sum_total_2023 * 100, 2) if sum_total_2023 != 0 else float('inf')



    total_sum_2024 = total_city_public_2024 + total_dc_bus_2024 + total_dc_fast_2024 + total_ac_residential_2024 + total_dc_internal_2024 + total_ac_internal_2024

    pct_city_public = (total_city_public_2024 / total_sum_2024 * 100) if total_sum_2024 != 0 else 0
    pct_dc_bus = (total_dc_bus_2024 / total_sum_2024 * 100) if total_sum_2024 != 0 else 0
    pct_dc_fast = (total_dc_fast_2024 / total_sum_2024 * 100) if total_sum_2024 != 0 else 0
    pct_ac_residential = (total_ac_residential_2024 / total_sum_2024 * 100) if total_sum_2024 != 0 else 0
    pct_dc_internal = (total_dc_internal_2024 / total_sum_2024 * 100) if total_sum_2024 != 0 else 0
    pct_ac_internal = (total_ac_internal_2024 / total_sum_2024 * 100) if total_sum_2024 != 0 else 0
    pct_internal = pct_dc_internal + pct_ac_internal
    pct_total = pct_internal + pct_ac_residential + pct_dc_fast + pct_dc_bus + pct_city_public


    # 打印结果
    print(f"2024年 公共: {total_dc_public_2024}\t公共: {total_ac_public_2024}\t同比 城市公共: {yoy_city_public:.2f}%\t占比: {pct_city_public:.2f}%")
    print(f"2024年 公交: {total_dc_bus_2024}\t同比 公交: {yoy_dc_bus:.2f}%\t 占比: {pct_dc_bus:.2f}%")
    print(f"2024年 高速: {total_dc_fast_2024}\t同比 高速: {yoy_dc_fast:.2f}%\t占比: {pct_dc_fast:.2f}%")
    print(f"2024年 单位内部: {total_dc_internal_2024}\t单位内部: {total_ac_internal_2024}\t同比:{yoy_interal:.2f}%\t占比:{pct_internal:.2f}%")
    print(f"2024年 小区共享: {total_ac_residential_2024}\t同比 小区: {yoy_ac_residential:.2f}%\t占比: {pct_ac_residential:.2f}%")
    print(f"合计:   {sum_total_2024}\t同比: {yoy_sum_total:.2f}%\t占比: {pct_total:.2f}%")

def main2() -> None:
    # 打印运行信息
    print("------------------------------run------------------------------")
    
    # 文件路径
    file_path = r"C:\Users\juntaox\Desktop\工作待办\浦东月报\浦东自营充电站充电量统计-202407.xlsx"
    
    # 读取 Excel 文件
    df = pd.read_excel(file_path)
    
    # 为第一列添加表头
    df.columns = ['月份'] + df.columns[1:].tolist()
    
    # 确保 '月份' 列为字符串类型
    df['月份'] = df['月份'].astype(str).str.strip()
    
    # 打印数据以确认修改
    print(df.head())
    
    # 过滤月份范围 202401 到 202407
    df_filtered_2024 = df[(df['月份'] == '202407')]
    
    # 过滤月份范围 202301 到 202307
    df_filtered_2023 = df[(df['月份'] == '202307')]
    

    # 计算所需列的总和（2024年数据）
    total_dc_public_2024 = round(df_filtered_2024['直流城市公共'].sum() / 10000, 3)
    total_ac_public_2024 = round(df_filtered_2024['交流城市公共'].sum() / 10000, 3)
    total_dc_bus_2024 = round(df_filtered_2024['直流公交'].sum() / 10000, 3)
    total_dc_fast_2024 = round(df_filtered_2024['直流高速'].sum() / 10000, 3)
    total_ac_residential_2024 = round(df_filtered_2024['交流小区'].sum() / 10000, 3)
    total_dc_internal_2024 = round(df_filtered_2024['直流单位内部'].sum() / 10000, 3)
    total_ac_internal_2024 = round(df_filtered_2024['交流单位内部'].sum() / 10000, 3)

    # 计算所需列的总和（2023年数据）
    total_dc_public_2023 = round(df_filtered_2023['直流城市公共'].sum() / 10000, 3)
    total_ac_public_2023 = round(df_filtered_2023['交流城市公共'].sum() / 10000, 3)
    total_dc_bus_2023 = round(df_filtered_2023['直流公交'].sum() / 10000, 3)
    total_dc_fast_2023 = round(df_filtered_2023['直流高速'].sum() / 10000, 3)
    total_ac_residential_2023 = round(df_filtered_2023['交流小区'].sum() / 10000, 3)
    total_dc_internal_2023 = round(df_filtered_2023['直流单位内部'].sum() / 10000, 3)
    total_ac_internal_2023 = round(df_filtered_2023['交流单位内部'].sum() / 10000, 3)
    # 计算城市公共总和
    total_city_public_2024 = round(total_dc_public_2024 + total_ac_public_2024, 3)
    total_city_public_2023 = round(total_dc_public_2023 + total_ac_public_2023, 3)
    
    
    total_internal_2024 = round(total_dc_internal_2024 + total_ac_internal_2024, 3)
    total_internal_2023 = round(total_dc_internal_2023 + total_ac_internal_2023, 3)
    
    sum_total_2024 = round(total_internal_2024 + total_city_public_2024 + total_dc_bus_2024 + total_dc_fast_2024 + total_ac_residential_2024, 3)
    sum_total_2023 = round(total_internal_2023 + total_city_public_2023 + total_dc_bus_2023 + total_dc_fast_2023 + total_ac_residential_2023, 3)
    # 计算同比
    yoy_city_public = round((total_city_public_2024 - total_city_public_2023) / total_city_public_2023 * 100, 2) if total_city_public_2023 != 0 else float('inf')
    yoy_interal = round((total_internal_2024 - total_internal_2023) / total_internal_2023 * 100, 2) if total_internal_2023 != 0 else float('inf')
    yoy_dc_bus = round((total_dc_bus_2024 - total_dc_bus_2023) / total_dc_bus_2023 * 100, 2) if total_dc_bus_2023 != 0 else float('inf')
    yoy_dc_fast = round((total_dc_fast_2024 - total_dc_fast_2023) / total_dc_fast_2023 * 100, 2) if total_dc_fast_2023 != 0 else float('inf')
    yoy_ac_residential = round((total_ac_residential_2024 - total_ac_residential_2023) / total_ac_residential_2023 * 100, 2) if total_ac_residential_2023 != 0 else float('inf')
    yoy_sum_total = round((sum_total_2024 - sum_total_2023) / sum_total_2023 * 100, 2) if sum_total_2023 != 0 else float('inf')



    total_sum_2024 = total_city_public_2024 + total_dc_bus_2024 + total_dc_fast_2024 + total_ac_residential_2024 + total_dc_internal_2024 + total_ac_internal_2024

    pct_city_public = (total_city_public_2024 / total_sum_2024 * 100) if total_sum_2024 != 0 else 0
    pct_dc_bus = (total_dc_bus_2024 / total_sum_2024 * 100) if total_sum_2024 != 0 else 0
    pct_dc_fast = (total_dc_fast_2024 / total_sum_2024 * 100) if total_sum_2024 != 0 else 0
    pct_ac_residential = (total_ac_residential_2024 / total_sum_2024 * 100) if total_sum_2024 != 0 else 0
    pct_dc_internal = (total_dc_internal_2024 / total_sum_2024 * 100) if total_sum_2024 != 0 else 0
    pct_ac_internal = (total_ac_internal_2024 / total_sum_2024 * 100) if total_sum_2024 != 0 else 0
    pct_internal = pct_dc_internal + pct_ac_internal
    pct_total = pct_internal + pct_ac_residential + pct_dc_fast + pct_dc_bus + pct_city_public


    # 打印结果
    print(f"公共: {total_dc_public_2024}\t公共: {total_ac_public_2024}\t同比 城市公共: {yoy_city_public:.2f}%\t占比: {pct_city_public:.2f}%")
    print(f"公交: {total_dc_bus_2024}\t同比 公交: {yoy_dc_bus:.2f}%\t 占比: {pct_dc_bus:.2f}%")
    print(f"高速: {total_dc_fast_2024}\t同比 高速: {yoy_dc_fast:.2f}%\t占比: {pct_dc_fast:.2f}%")
    print(f"单位内部: {total_dc_internal_2024}\t单位内部: {total_ac_internal_2024}\t同比:{yoy_interal:.2f}%\t占比:{pct_internal:.2f}%")
    print(f"小区: {total_ac_residential_2024}\t同比 小区: {yoy_ac_residential:.2f}%\t占比: {pct_ac_residential:.2f}%")
    print(f"合计:   {sum_total_2024}\t同比: {yoy_sum_total:.2f}%\t占比: {pct_total:.2f}%")
    

if __name__ == "__main__":
    # main()
    main2()
