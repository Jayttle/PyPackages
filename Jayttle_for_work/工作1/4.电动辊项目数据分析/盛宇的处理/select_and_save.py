import pandas as pd
import numpy as np
def save_to_excel():

    # 读取CSV文件
    input_file = r"D:\Code\data-analysis\一车间能源消耗数据分析\故障分析\B18_DGTPM_20241114_20241115.csv"  # 替换为你的CSV文件路径
    output_file = r"D:\Code\data-analysis\一车间能源消耗数据分析\4752-1-1114.xlsx"  # 输出的Excel文件名

    # 读取CSV数据
    data = pd.read_csv(input_file)

    # 筛选所需列
    columns_to_keep = ["DateTime", "Motor_ID", "Sensor_State", "Motor_Run", "Motor_Speed", "Motor_Temperature", "Motor_Torque"]
    filtered_data = data[columns_to_keep]

    # 筛选 Motor_ID 为 '4752-1' 的行
    filtered_data = filtered_data[filtered_data["Motor_ID"] == "'4752-1'"]
    #print(filtered_data)
    filtered_data = filtered_data[filtered_data["Sensor_State"] == 1]
    #print(filtered_data)
    filtered_data = filtered_data[filtered_data["Motor_Run"] == 1]
    # 保存到新的Excel文件
    filtered_data.to_excel(output_file, index=False)

    print(f"筛选后的数据已保存到 {output_file}")

def save_to_excel_min():
    import pandas as pd

    # 文件路径
    input_file = r"D:\Code\data-analysis\一车间能源消耗数据分析\故障分析\4752-1.xlsx"  # 输入文件路径
    output_file = "4752-1-min.xlsx"  # 输出文件路径

    # 读取Excel数据
    data = pd.read_excel(input_file)

    # 去除 DateTime 列中的引号并转换为时间格式
    data["DateTime"] = data["DateTime"].str.strip("'")  # 去除首尾单引号
    data["DateTime"] = pd.to_datetime(data["DateTime"])  # 转换为 datetime 格式

    # 按分钟取平均值
    data["Minute"] = data["DateTime"].dt.floor("T")  # 按分钟截取时间
    averaged_data = data.groupby("Minute")[["Motor_Speed", "Motor_Temperature", "Motor_Torque"]].mean().reset_index()

    # 保存到新的Excel文件
    averaged_data.to_excel(output_file, index=False)

    print(f"Averaged data has been saved to {output_file}")

def save_to_excel_hour():
    import pandas as pd

    # 文件路径
    input_file = r"D:\Code\data-analysis\一车间能源消耗数据分析\4752-1-1114.xlsx"  # 输入文件路径
    output_file = r"D:\Code\data-analysis\一车间能源消耗数据分析\4752-1-1114-hour.xlsx"  # 输出文件路径

    # 读取Excel数据
    data = pd.read_excel(input_file)

    # 去除 DateTime 列中的引号并转换为时间格式
    data["DateTime"] = data["DateTime"].str.strip("'")  # 去除首尾单引号
    data["DateTime"] = pd.to_datetime(data["DateTime"])  # 转换为 datetime 格式

    # 按小时取平均值
    data["Hour"] = data["DateTime"].dt.floor("H")  # 按小时截取时间
    hourly_averaged_data = data.groupby("Hour")[
        ["Motor_Speed", "Motor_Temperature", "Motor_Torque"]].mean().reset_index()

    # 保存到新的Excel文件
    hourly_averaged_data.to_excel(output_file, index=False)

    print(f"Hourly averaged data has been saved to {output_file}")

def save_to_excel_history_hour():
    import pandas as pd

    # 输入和输出文件路径
    input_file = r"D:\Code\data-analysis\一车间能源消耗数据分析\故障分析\4717-1-11-electric_roller_analysis_202411211441.csv"  # 替换为你的CSV文件路径
    output_file = "data1121/4717-1-11-hour.xlsx"  # 输出的Excel文件名

    # 读取CSV数据
    data = pd.read_csv(input_file)

    # 筛选需要的列
    columns_to_keep = ["datetime", "motor_speed", "motor_temperature", "motor_torque","sensor_state",'motor_run']
    filtered_data = data[columns_to_keep]
    filtered_data = filtered_data[filtered_data["sensor_state"] == 1]
    #print(filtered_data)
    filtered_data = filtered_data[filtered_data["motor_run"] == 1]
    # 确保 DateTime 是时间格式并提取小时
    filtered_data["datetime"] = pd.to_datetime(filtered_data["datetime"], errors="coerce")  # 转换为 datetime 格式
    filtered_data = filtered_data.dropna(subset=["datetime"])  # 删除无效时间行

    # 筛选时间
    #time_range_1 = ("2024-09-10 00:00:00", "2024-09-14 00:00:00")
    #time_range_1 = ("2024-10-15 00:00:00", "2024-10-17 00:00:00")
    time_range_1 = ("2024-11-19 00:00:00", "2024-11-21 00:00:00")


    filtered_data = filtered_data[(filtered_data["datetime"] >= time_range_1[0]) & (filtered_data["datetime"] <= time_range_1[1])]

    filtered_data["Hour"] = filtered_data["datetime"].dt.floor("H")  # 按小时截取时间

    # 按小时计算平均值
    hourly_data = filtered_data.groupby("Hour")[
        ["motor_speed", "motor_temperature", "motor_torque"]].mean().reset_index()

    # 保存到新的Excel文件
    #hourly_data.to_excel(output_file, index=False)

    print(f"按小时计算的平均值已保存到 {output_file}")

def save_to_excel_10_hour():
    import pandas as pd

    # 输入和输出文件路径
    #input_file = r"D:\work\上烟\电动轨数据分析\建模PCA\20241001-1016electric_roller_analysis_202411211628.csv"
    input_files = [
        r"D:\work\上烟\电动轨数据分析\建模PCA\20241001-1016electric_roller_analysis_202411211628.csv",
        r"D:\work\上烟\电动轨数据分析\建模PCA\20241017-1031electric_roller_analysis_202411211630.csv"
    ]
    # 替换为你的CSV文件路径
    output_file = r"D:\Code\data-analysis\一车间能源消耗数据分析\4752-1-10-hour.xlsx"  # 输出的Excel文件名

    # 筛选需要的列
    columns_to_keep = ["datetime", "motor_speed", "motor_temperature", "motor_torque","sensor_state",'motor_run']

    # 存储所有文件的处理结果
    all_data = []

    for file in input_files:
        # 读取CSV数据
        data = pd.read_csv(file)

        # 筛选需要的列
        filtered_data = data[columns_to_keep]
        filtered_data = filtered_data[filtered_data["sensor_state"] == 1]  # 过滤条件
        filtered_data = filtered_data[filtered_data["motor_run"] == 1]  # 过滤条件

        # 确保 datetime 是时间格式
        filtered_data["datetime"] = pd.to_datetime(filtered_data["datetime"], errors="coerce")
        filtered_data = filtered_data.dropna(subset=["datetime"])  # 删除无效时间行

        all_data.append(filtered_data)

    # 合并所有数据
    merged_data = pd.concat(all_data, ignore_index=True)
    print(len(merged_data))
    # 筛选时间
    #time_range_1 = ("2024-09-10 00:00:00", "2024-09-14 00:00:00")
    #time_range_1 = ("2024-10-15 00:00:00", "2024-10-17 00:00:00")
    #time_range_1 = ("2024-11-19 00:00:00", "2024-11-21 00:00:00")


    #filtered_data = filtered_data[(filtered_data["datetime"] >= time_range_1[0]) & (filtered_data["datetime"] <= time_range_1[1])]

    merged_data["Hour"] = merged_data["datetime"].dt.floor("H")  # 按小时截取时间
    #
    # 按小时计算平均值
    hourly_data = merged_data.groupby("Hour")[
        ["motor_speed", "motor_temperature", "motor_torque"]].mean().reset_index()

    # 保存到新的Excel文件
    #hourly_data.to_excel(output_file, index=False)

    print(f"按小时计算的平均值已保存到 {output_file}")


def plot():
    import pandas as pd
    import matplotlib.pyplot as plt

    # 读取保存的Excel文件
    file_path = r"D:\Code\data-analysis\一车间能源消耗数据分析\故障分析\4752-1-hour.xlsx"  # 替换为你的文件路径

    # 读取Excel数据
    data = pd.read_excel(file_path)
    print("加载完成")
    # 取绝对值并获取所需列
    data["Motor_Speed"] = data["Motor_Speed"].abs()
    data["Motor_Temperature"] = data["Motor_Temperature"].abs()
    data["Motor_Torque"] = data["Motor_Torque"].abs()
    print("取绝对值完成")
    # 设置 x 轴为 DateTime
    x = data["DateTime"]
    # 绘制 Motor_Speed 曲线
    plt.figure(figsize=(10, 6))
    plt.plot(x, data["Motor_Speed"], label="Motor Speed", color="blue")
    plt.title("4752-1 Motor Speed")
    plt.xlabel("DateTime")
    plt.ylabel("Motor Speed")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 绘制 Motor_Temperature 曲线
    plt.figure(figsize=(10, 6))
    plt.plot(x, data["Motor_Temperature"], label="Motor Temperature", color="red")
    plt.title("4752-1 Motor Temperature")
    plt.xlabel("DateTime")
    plt.ylabel("Motor Temperature")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 绘制 Motor_Torque 曲线
    plt.figure(figsize=(10, 6))
    plt.plot(x, data["Motor_Torque"], label="Motor Torque", color="green")
    plt.title("4752-1 Motor Torque")
    plt.xlabel("DateTime")
    plt.ylabel("Motor Torque")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()
def diff_plot_2():
    import pandas as pd
    import matplotlib.pyplot as plt

    # 文件路径
    hourly_file_1 = r"D:\Code\data-analysis\一车间能源消耗数据分析\故障分析\data1121\4753-1-10-hour.xlsx"  # 第一个Excel文件路径
    hourly_file_2 = r"D:\Code\data-analysis\一车间能源消耗数据分析\故障分析\data1121\4753-1-11-hour.xlsx"  # 第二个Excel文件路径

    # 时间范围
    time_range_1 = ("2024-11-14 01:00:00", "2024-11-15 11:00:00")
    time_range_2 = ("2024-10-15 01:00:00", "2024-10-16 11:00:00")

    # 读取第一个Excel文件
    data_1 = pd.read_excel(hourly_file_1)
    data_1["Hour"] = pd.to_datetime(data_1["Hour"])  # 确保时间为datetime格式
    #filtered_data_1 = data_1[(data_1["Hour"] >= time_range_1[0]) & (data_1["Hour"] <= time_range_1[1])]

    # 读取第二个Excel文件
    data_2 = pd.read_excel(hourly_file_2)
    data_2["Hour"] = pd.to_datetime(data_2["Hour"])  # 确保时间为datetime格式
    #filtered_data_2 = data_2[(data_2["Hour"] >= time_range_2[0]) & (data_2["Hour"] <= time_range_2[1])]
    # 找出较长和较短的数据
    max_length = max(len(data_1), len(data_2))
    print(len(data_1))
    filtered_data_1 = data_1.reindex(range(max_length)).reset_index(drop=True)
    print(len(filtered_data_1))
    #filtered_data_2 = data_2.reindex(range(max_length)).reset_index(drop=True)
    filtered_data_2 = data_2
    # 确保两个数据的行数一致
    assert len(filtered_data_1) == len(filtered_data_2), "两个数据的行数不一致，请检查时间范围。"

    # 提取需要的列
    x = filtered_data_2["Hour"]
    # y1_speed = filtered_data_1["Motor_Speed"].abs()
    # y1_temperature = filtered_data_1["Motor_Temperature"]
    # y1_torque = filtered_data_1["Motor_Torque"]
    y1_speed = filtered_data_1["motor_speed"].abs()
    y1_temperature = filtered_data_1["motor_temperature"]
    y1_torque = filtered_data_1["motor_torque"]

    y2_speed = filtered_data_2["motor_speed"].abs()
    y2_temperature = filtered_data_2["motor_temperature"]
    y2_torque = filtered_data_2["motor_torque"]

    # 绘制对比图
    plt.figure(figsize=(10, 6))
    plt.plot(x, y1_speed, label="4753-1-10", color="blue")
    plt.plot(x, y2_speed, label="4753-1-11", color="cyan", linestyle="--")
    plt.title("Speed Comparison")
    plt.xlabel("Hour")
    plt.ylabel("Speed")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(x, y1_temperature, label="4753-1-10", color="red")
    plt.plot(x, y2_temperature, label="4753-1-11", color="orange", linestyle="--")
    plt.title("Temperature Comparison")
    plt.xlabel("Hour")
    plt.ylabel("Temperature")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(x, y1_torque, label="4753-1-10", color="green")
    plt.plot(x, y2_torque, label="4753-1-11", color="lime", linestyle="--")
    plt.title("Torque Comparison")
    plt.xlabel("Hour")
    plt.ylabel("Torque")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def diff_plot_3():
    import pandas as pd
    import matplotlib.pyplot as plt

    # 文件路径
    hourly_file_1 = r"D:\Code\data-analysis\一车间能源消耗数据分析\故障分析\data1121\4717-1-9-hour.xlsx"  # 第一个Excel文件路径
    hourly_file_2 = r"D:\Code\data-analysis\一车间能源消耗数据分析\故障分析\data1121\4717-1-10-hour.xlsx"  # 第二个Excel文件路径
    hourly_file_3 = r"D:\Code\data-analysis\一车间能源消耗数据分析\故障分析\data1121\4717-1-11-hour.xlsx"  # 第三个Excel文件路径

    # 读取文件
    data_1 = pd.read_excel(hourly_file_1)
    data_2 = pd.read_excel(hourly_file_2)
    data_3 = pd.read_excel(hourly_file_3)
    x = data_1["Hour"]
    # 转换时间格式（只保留时分秒），统一为 HH:MM:SS 格式
    data_1["Hour"] = pd.to_datetime(data_1["Hour"]).dt.strftime('%H:%M:%S')
    data_2["Hour"] = pd.to_datetime(data_2["Hour"]).dt.strftime('%H:%M:%S')
    data_3["Hour"] = pd.to_datetime(data_3["Hour"]).dt.strftime('%H:%M:%S')

    max_length = max(len(data_1), len(data_2),len(data_3))
    #print(len(data_3))
    data_3 = data_3.reindex(range(max_length)).reset_index(drop=True)
    # 提取绘图数据


    y1_speed = data_1["motor_speed"].abs()
    y2_speed = data_2["motor_speed"].abs()
    # y3_speed = data_3["Motor_Speed"].abs()
    y3_speed = data_3["motor_speed"].abs()


    y1_temperature = data_1["motor_temperature"]
    y2_temperature = data_2["motor_temperature"]
    # y3_temperature = data_3["Motor_Temperature"]
    y3_temperature = data_3["motor_temperature"]

    y1_torque = data_1["motor_torque"]
    y2_torque = data_2["motor_torque"]
    # y3_torque = data_3["Motor_Torque"]
    y3_torque = data_3["motor_torque"]


    # 绘制对比图 - Speed
    plt.figure(figsize=(10, 6))
    plt.plot(x, y1_speed, label="4717-1-9", color="red",linestyle="solid")
    plt.plot(x, y2_speed, label="4717-1-10", color="green", linestyle="--")
    plt.plot(x, y3_speed, label="4717-1-11", color="blue", linestyle="dotted")
    plt.title("Speed Comparison")
    plt.xlabel("Time (HH:MM:SS)")
    plt.ylabel("Speed")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 绘制对比图 - Temperature
    plt.figure(figsize=(10, 6))
    plt.plot(x, y1_temperature, label="4717-1-9", color="red",linestyle="solid")
    plt.plot(x, y2_temperature, label="4717-1-10", color="green", linestyle="--")
    plt.plot(x, y3_temperature, label="4717-1-11", color="blue", linestyle="dotted")
    plt.title("Temperature Comparison")
    plt.xlabel("Time (HH:MM:SS)")
    plt.ylabel("Temperature")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 绘制对比图 - Torque
    plt.figure(figsize=(10, 6))
    plt.plot(x, y1_torque, label="4717-1-9", color="red",linestyle="solid")
    plt.plot(x, y2_torque, label="4717-1-10", color="green", linestyle="--")
    plt.plot(x, y3_torque, label="4717-1-11", color="blue", linestyle="dotted")
    plt.title("Torque Comparison")
    plt.xlabel("Time (HH:MM:SS)")
    plt.ylabel("Torque")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    save_to_excel_10_hour()