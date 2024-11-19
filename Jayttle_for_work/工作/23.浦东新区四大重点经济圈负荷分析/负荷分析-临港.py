import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import re

def run_main_00(read_file_path, save_file_path):
    """
    读取庞大的原始文件并生成只有符合特定条件的列（日期和符合条件的PO.P列）的表格。
    """
    df = pd.read_excel(read_file_path)
    
    # 转换‘日期’列
    if '日期' in df.columns:
        df['日期'] = df['日期'].str.split(' ').str[0].str.strip()  # 清理日期列
        df['日期'] = pd.to_datetime(df['日期'].astype(str), format='%Y/%m/%d', errors='coerce')  # 转换日期
    # 找到以'PO.P'开头，且数值部分是4的倍数加1的列名
    time_columns = []
    pattern = r'PO\.P(\d+)\*PO\.CT\*PO\.PT'  # 正则表达式，匹配PO.P后面的数字部分

    for col in df.columns:
        match = re.match(pattern, col)  # 使用正则表达式匹配列名
        if match:
            number_part = int(match.group(1))  # 提取数字部分并转换为整数
            if number_part % 4 == 1:
                time_columns.append(col)  # 如果满足条件，则加入time_columns列表

    # 提取所需的列，包括日期和符合条件的PO.P开头的列
    selected_columns = ['户号'] if '户号'  in df.columns else []  # 确保日期列存在
    # 如果存在日期列，添加到所选列中
    if '日期' in df.columns:
        selected_columns.append('日期')

    selected_columns.extend(time_columns)

    # 创建新的DataFrame，只包含所需的列
    new_df = df[selected_columns]

    # 保存到新的Excel文件
    new_df.to_excel(save_file_path, index=False)

def run_main_0(read_file_path, save_file_path):
    """
    用于读取庞大的原始文件，生成只有日负荷数据总和的表格
    """
    df = pd.read_excel(read_file_path)
    # 转换‘日期’列
    df['日期'] = df['日期'].str.split(' ').str[0].str.strip()  # 清理日期列
    df['日期'] = pd.to_datetime(df['日期'].astype(str), format='%Y/%m/%d', errors='coerce')  # 转换日期
    # 初始化结果数据框

    # 找到以'PO.P'开头，且在'PO.P'后面的字符串以'*'为分隔符的列名，并且提取数字部分 %4 == 0
    time_columns = []
    pattern = r'PO\.P(\d+)\*PO\.CT\*PO\.PT'  # 正则表达式，匹配PO.P后面的数字部分

    for col in df.columns:
        match = re.match(pattern, col)  # 使用正则表达式匹配列名
        if match:
            number_part = int(match.group(1))  # 提取数字部分并转换为整数
            if number_part % 4 == 1:
                time_columns.append(col)  # 如果是4的倍数，则加入time_columns列表

    # 将time_columns的列名替换为0, 1, 2...
    renamed_columns = {col: str(index) for index, col in enumerate(time_columns)}
    df.rename(columns=renamed_columns, inplace=True)

    # 将负荷数据列转换为数值格式，以处理类似.0039的情况
    for col in renamed_columns.values():
        df[col] = pd.to_numeric(df[col], errors='coerce')  # 转换为数值，非数值将被设置为NaN

    # 去除负值，将负值设为NaN
    for col in renamed_columns.values():
        df[col] = df[col].where(df[col] >= 0)  # 负值设为NaN

    # 按日期进行分组并对时刻的负荷求和
    summed_load = df.groupby('日期')[list(renamed_columns.values())].sum().reset_index()
    summed_load.to_excel(save_file_path, index=False, sheet_name='整体')

# 定义一个函数来统计高峰和低谷时间的出现次数
def get_peak_valley_stats(df, peak_col, valley_col, key):
    # 获取高峰和低谷的统计信息
    peak_counts = df[peak_col].value_counts()
    valley_counts = df[valley_col].value_counts()
    
    # 创建一个DataFrame来合并这些统计结果
    peak_valley_stats = pd.DataFrame({
        '时间': list(peak_counts.index) + list(valley_counts.index),
        '类型': ['高峰'] * len(peak_counts) + ['低谷'] * len(valley_counts),
        '次数': list(peak_counts.values) + list(valley_counts.values),
        '分类': [key] * (len(peak_counts) + len(valley_counts))
    })
    
    return peak_valley_stats

def run_main_1(file_path):
    """
    基于只有日负荷数据总和的表格，生成多个sheet，分别是整体、工作日、非工作日的负荷分析
    """
    # 读取 Excel 文件
    df = pd.read_excel(file_path)

    print(df['日期'])
    # 将 '日期' 列设置为索引
    df.set_index('日期', inplace=True)

    # 初始化结果数据框
    result_df = pd.DataFrame(index=df.index)

    # 计算白天（6点到18点）和晚上的日均值
    daytime_avg = df.iloc[:, 6:18].mean(axis=1)  # 白天平均值
    nighttime_avg = df.iloc[:, [0, 1, 2, 3, 4, 5, 18, 19, 20, 21, 22, 23]].mean(axis=1)  # 晚上平均值

    # 将计算结果添加到数据框中
    result_df['白天日均值（6~18点）'] = daytime_avg
    result_df['晚上日均值（18点~次日6点）'] = nighttime_avg

    # 计算每天的高峰和低谷
    result_df['日高峰'] = df.max(axis=1)  # 计算每天的最大值
    result_df['日低谷'] = df.min(axis=1)  # 计算每天的最小值

    # 获取高峰和低谷对应的列名
    result_df['高峰时间'] = df.idxmax(axis=1)  # 对应最大值的列名
    result_df['低谷时间'] = df.idxmin(axis=1)  # 对应最小值的列名

    # 计算峰谷差
    result_df['峰谷差'] = (result_df['日高峰'] - result_df['日低谷'])
    # 计算峰谷差率：峰谷差与最高负荷的比率
    result_df['峰谷差/最高负荷'] = (result_df['日高峰'] - result_df['日低谷']) / result_df['日高峰']

    # 读取非工作日 JSON 文件
    with open(r'E:\OneDrive\PyPackages\Jayttle_for_work\23.浦东新区四大重点经济圈负荷分析\非工作日-临港.json', 'r', encoding='utf-8') as json_file:
        non_working_days_data = json.load(json_file)

    non_working_days = set(non_working_days_data['non_working_days'])
    holiday_days = set(non_working_days_data['holiday_days'])
    adjust_days = set(non_working_days_data['adjust_days'])
    all_days = set(df.index.strftime('%Y-%m-%d'))
    # 计算工作日
    working_days = all_days - non_working_days - holiday_days - adjust_days

    # 从 df 中筛选出非工作日、工作日、节假日、调休日数据
    non_working_days_indices = df.index[df.index.strftime('%Y-%m-%d').isin(non_working_days)]
    non_working_days_df = df[df.index.isin(non_working_days_indices)]

    working_days_indices = df.index[df.index.strftime('%Y-%m-%d').isin(working_days)]
    working_days_df = df[df.index.isin(working_days_indices)]

    holiday_days_indices = df.index[df.index.strftime('%Y-%m-%d').isin(holiday_days)]
    holiday_days_df = df[df.index.isin(holiday_days_indices)]


    adjust_days_indices = df.index[df.index.strftime('%Y-%m-%d').isin(adjust_days)]
    adjust_days_df = df[df.index.isin(adjust_days_indices)]

    # 保存数据到新的工作表
    with pd.ExcelWriter(file_path, engine='openpyxl', mode='a') as writer:
        # 保存非工作日和工作日的原始数据
        if '非工作日' in writer.book.sheetnames:
            del writer.book['非工作日']
        non_working_days_df.to_excel(writer, sheet_name='非工作日', index=True)

        if '工作日' in writer.book.sheetnames:
            del writer.book['工作日']
        working_days_df.to_excel(writer, sheet_name='工作日', index=True)

        if '节假日' in writer.book.sheetnames:
            del writer.book['节假日']
        holiday_days_df.to_excel(writer, sheet_name='节假日', index=True)

        if '调休日' in writer.book.sheetnames:
            del writer.book['调休日']
        adjust_days_df.to_excel(writer, sheet_name='调休日', index=True)

        # 删除已有的相关工作表（如果存在）
        if '负荷分析' in writer.book.sheetnames:
            del writer.book['负荷分析']
        result_df.to_excel(writer, sheet_name='负荷分析', index=True)  # 保留索引

        # 保存非工作日负荷分析
        non_working_days_analysis = result_df[result_df.index.isin(non_working_days_indices)]
        if '非工作日负荷分析' in writer.book.sheetnames:
            del writer.book['非工作日负荷分析']
        non_working_days_analysis.to_excel(writer, sheet_name='非工作日负荷分析', index=True)

        # 保存工作日负荷分析
        working_days_analysis = result_df[result_df.index.isin(working_days_indices)]
        if '工作日负荷分析' in writer.book.sheetnames:
            del writer.book['工作日负荷分析']
        working_days_analysis.to_excel(writer, sheet_name='工作日负荷分析', index=True)

        # 保存节假日负荷分析
        holiday_days_analysis = result_df[result_df.index.isin(holiday_days_indices)]
        if '节假日负荷分析' in writer.book.sheetnames:
            del writer.book['节假日负荷分析']
        holiday_days_analysis.to_excel(writer, sheet_name='节假日负荷分析', index=True)

        # 保存调休日负荷分析
        adjust_days_analysis = result_df[result_df.index.isin(adjust_days_indices)]
        if '调休日负荷分析' in writer.book.sheetnames:
            del writer.book['调休日负荷分析']
        adjust_days_analysis.to_excel(writer, sheet_name='调休日负荷分析', index=True)

        # 对非工作日数据进行统计
        non_working_peak_valley_stats = get_peak_valley_stats(non_working_days_analysis, '高峰时间', '低谷时间', '非工作日')

        # 对工作日数据进行统计
        working_peak_valley_stats = get_peak_valley_stats(working_days_analysis, '高峰时间', '低谷时间', '工作日')

        # 对节假日数据进行统计
        holiday_peak_valley_stats = get_peak_valley_stats(holiday_days_analysis, '高峰时间', '低谷时间', '节假日')

        # 合并所有数据
        combined_peak_valley_stats = pd.concat([non_working_peak_valley_stats, working_peak_valley_stats, holiday_peak_valley_stats])

        # 将统计结果写入Excel文件的'峰谷情况'工作表
        if '峰谷情况' in writer.book.sheetnames:
            del writer.book['峰谷情况']
        combined_peak_valley_stats.to_excel(writer, sheet_name='峰谷情况', index=False)


    print("数据已保存到 Excel 文件的...数据")

    # 输出工作日和非工作日天数
    working_days_count = len(working_days)
    non_working_days_count = len(non_working_days)
    print(f"工作日天数: {working_days_count}")
    print(f"非工作日天数: {non_working_days_count}")

    return working_days_count, non_working_days_count

def run_main_2(file_path, sheet_name):
    """
    读取 Excel 文件的指定工作表，处理负荷数据并统计分析。
    :param sheet_name: 要读取的工作表名称
    """
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # 将 '日期' 列转换为 datetime 类型并只保留年月日
    df['日期'] = pd.to_datetime(df['日期'])

    # 将 '日期' 列设置为索引，确保索引为 DatetimeIndex
    df.set_index('日期', inplace=True)

    hourly_stats = df.groupby(df.index.to_period('M')).mean()  # 计算每月每小时的平均值

    # 将结果添加到数据框，统计结果是每月每小时的平均值
    hourly_stats.reset_index(inplace=True)

    # 保存数据到新的工作表
    with pd.ExcelWriter(file_path, engine='openpyxl', mode='a') as writer:
        # 删除已有的 "负荷月分析" 工作表，如果存在
        if f'{sheet_name}负荷月分析' in writer.book.sheetnames:
            del writer.book[f'{sheet_name}负荷月分析']
        hourly_stats.to_excel(writer, sheet_name=f'{sheet_name}负荷月分析', index=False)  # 保存不带索引

    print("按年月统计的小时平均值已保存至 Excel 文件。")


def run_main_3(fuhe_file_path, yongdian_file_path, save_file_path = False, working_days_count=None,  non_working_days_count=None):
    """
    读取负荷的清单和用电的清单，用于匹配电压等级；
    """
    # 读取两个 Excel 文件
    fh_df = pd.read_excel(fuhe_file_path)
    yd_df = pd.read_excel(yongdian_file_path)

    # 通过分割字符串获取 {region}
    region = file_path.split("\\")[-1].split("负荷")[0]
    if region != "1.陆家嘴":
        # 将 '户号' 列按照需要的规则进行处理
        fh_df['户号'] = fh_df['户号'].apply(lambda x: '31' + '0' * (11 - len(str(x))) + str(x) if str(x).isdigit() else str(x))

    # 确保 '户号' 列的类型一致（转换为字符串）
    fh_df['户号'] = fh_df['户号'].astype(str)
    yd_df['户号'] = yd_df['户号'].astype(str)

    # 使用 merge 将 '户号' 相同的行匹配，并将 '电压等级' 列加到 fh_df 中
    merged_df = pd.merge(fh_df, yd_df[['户号', '电压等级']], on='户号', how='left')

    # 检查 '电压等级' 列是否存在空值，并打印出对应的户号
    missing_voltage = merged_df[merged_df['电压等级'].isnull()]
    
    if not missing_voltage.empty:
        print("以下户号没有对应的电压等级:")
        print(missing_voltage['户号'].tolist())
    else:
        print("所有户号都成功匹配到电压等级。")
        print(merged_df.head())


    # 找到以'P'开头的列名
    time_columns = [col for col in merged_df.columns if col.startswith('P')]

    # 将time_columns的列名替换为0, 1, 2...
    renamed_columns = {col: str(index) for index, col in enumerate(time_columns)}
    merged_df.rename(columns=renamed_columns, inplace=True)

    # 将负荷数据列转换为数值格式，以处理类似.0039的情况
    for col in renamed_columns.values():
        merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')  # 转换为数值，非数值将被设置为NaN

    # 去除负值，将负值设为NaN
    for col in renamed_columns.values():
        merged_df[col] = merged_df[col].where(merged_df[col] >= 0)  # 负值设为NaN

    # 读取非工作日 JSON 文件
    with open(r'E:\OneDrive\PyPackages\Jayttle_for_work\工作\23.浦东新区四大重点经济圈负荷分析\非工作日-临港.json', 'r', encoding='utf-8') as json_file:
        non_working_days_data = json.load(json_file)

    non_working_days = set(non_working_days_data['non_working_days'])
    holiday_days = set(non_working_days_data['holiday_days'])
    adjust_days = set(non_working_days_data['adjust_days'])

    # 将 '日期' 列转换为 datetime 类型
    merged_df['日期'] = pd.to_datetime(merged_df['日期'], format='%Y_%m_%d %H:%M:%S', errors='coerce')

    # 将 datetime 转换为 'YYYY-MM-DD' 格式
    merged_df['日期'] = merged_df['日期'].dt.strftime('%Y-%m-%d')

    non_working_day_data = merged_df[merged_df['日期'].isin(non_working_days)]
    # 过滤出那些在 non_working_days, holiday_days 和 adjust_days 中的日期
    working_day_data = merged_df[~merged_df['日期'].isin(non_working_days | holiday_days | adjust_days)]

    # 按电压等级对工作日数据进行分组并求和
    working_day_avg = working_day_data.groupby('电压等级')[list(renamed_columns.values())].sum().reset_index()

    # 按电压等级对非工作日数据进行分组并求和
    non_working_day_avg = non_working_day_data.groupby('电压等级')[list(renamed_columns.values())].sum().reset_index()

    # 计算平均负荷：除以工作日和非工作日的数量
    if working_days_count is not None:
        working_day_avg[list(renamed_columns.values())] = working_day_avg[list(renamed_columns.values())].div(working_days_count)

    if non_working_days_count is not None:
        non_working_day_avg[list(renamed_columns.values())] = non_working_day_avg[list(renamed_columns.values())].div(non_working_days_count)

    # 保存数据到新的工作表
    with pd.ExcelWriter(save_file_path, engine='openpyxl', mode='a') as writer:
        if '电压等级工作日负荷' in writer.book.sheetnames:
            del writer.book['电压等级工作日负荷']
        working_day_avg.to_excel(writer, sheet_name='电压等级工作日负荷', index=True)

        if '电压等级非工作日负荷' in writer.book.sheetnames:
            del writer.book['电压等级非工作日负荷']
        non_working_day_avg.to_excel(writer, sheet_name='电压等级非工作日负荷', index=True)

if __name__ == '__main__':
    print('------------------------run-----------------------')
    region_names = ["2.临港"]
    region = region_names[0]

    read_file_path = rf"C:\Users\juntaox\Desktop\工作\23.浦东四大重点区域负荷分析\原始材料\临港负荷数据.xlsx"
    save_file_path = rf"C:\Users\juntaox\Desktop\工作\23.浦东四大重点区域负荷分析\原始材料\浦东四大重点区域负荷数据-{region}.xlsx"
    # run_main_00(read_file_path, save_file_path)
    file_path = rf"C:\Users\juntaox\Desktop\工作\23.浦东四大重点区域负荷分析\{region}负荷.xlsx"
    # run_main_0(read_file_path, file_path)
    working_days_count, non_working_days_count = run_main_1(file_path)
    # sheet_names = ['整体', '非工作日', "工作日", '节假日', '调休日']
    # for sheet in sheet_names:
    #     run_main_2(file_path, sheet)
    fuhe_file_path = save_file_path
    yongdian_file_path = rf"C:\Users\juntaox\Desktop\工作\23.浦东四大重点区域负荷分析\原始材料\用电-{region}.xlsx"
    run_main_3(fuhe_file_path, yongdian_file_path, file_path, working_days_count, non_working_days_count)