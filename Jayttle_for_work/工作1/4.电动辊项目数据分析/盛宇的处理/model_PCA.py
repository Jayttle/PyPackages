import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
import matplotlib.dates as mdates
def smooth_data(data, method="moving_average", window_size=5, mode="reflect"):
    """
    对健康度数据进行平滑处理。

    参数：
    - data: np.array，健康度数据。
    - method: str，平滑方法 ('moving_average' 或 'savitzky_golay')。
    - window_size: int，平滑窗口大小。
    - data: np.array，输入数据。
    - window_size: int，滑动窗口大小。
    - mode: str，边缘填充方式，默认 "reflect"（反射填充）。
    返回：
    - smoothed_data: np.array，平滑后的数据。
    """

    # 使用 np.pad 进行边缘填充
    pad_width = window_size // 2
    data = np.pad(data, pad_width, mode=mode)

    if method == "moving_average":
        # 移动平均平滑
        smoothed_data = np.convolve(data, np.ones(window_size)/window_size, mode='same')# valid
    elif method == "savitzky_golay":
        # Savitzky-Golay 滤波器平滑
        smoothed_data = savgol_filter(data, window_length=window_size, polyorder=2)
    else:
        raise ValueError("不支持的平滑方法！")

        # 裁剪填充的部分，返回与原始数据长度一致的部分
    smoothed_data = smoothed_data[pad_width: -pad_width] if pad_width > 0 else smoothed_data
    return smoothed_data

    # if method == "moving_average":
    #     # 保证平滑后数据长度与原始一致，mode='same'
    #     return np.convolve(data, np.ones(window_size) / window_size, mode="same")
    # elif method == "savitzky_golay":
    #     from scipy.signal import savgol_filter
    #     # Savgol Filter 需要奇数窗口
    #     if window_size % 2 == 0:
    #         window_size += 1
    #     return savgol_filter(data, window_length=window_size, polyorder=2)
    # else:
    #     return data  # 如果未指定方法，返回原始数据

def process_folder_with_filewise_standardization(folder_path,
                                                 relevant_columns,
                                                 timestamp_column='Hour',
                                                 timestamp_filter='2024-10-11 16:00:00',
                                                 smoothing_method="moving_average",
                                                 smoothing_window=5):
    """
    处理文件夹中的多个辊数据，单独标准化每个文件，计算全局权重，并按时间段独立计算健康度。

    参数：
    - folder_path: str，文件夹路径。
    - relevant_columns: list[str]，需要提取的列名（如 ['speed', 'torque', 'temperature']）。
    - timestamp_column: str，时间戳列名（默认 'hour'）。
    - timestamp_filter: str，时间过滤条件（默认 '2024-10-11 17:00:00'）。
    - smoothing_method: str，平滑方法（'moving_average' 或 'savitzky_golay'）。
    - smoothing_window: int，平滑窗口大小。

    返回：
    - file_health_scores: dict，每个文件的健康度。
    - weights: list，全局 PCA 提取的权重。
    """

    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.xlsx')]
    if not files:
        print("指定文件夹中没有找到任何 Excel 文件。")
        return None, None
    name_list = ['4707_1','4712_1','4714_1','4716_1','4721_1',
                 '4724_1','4728_1','4731_1','4733_1','4735_1','4737_1','4740_1','4742_1','4744_1']
    all_standardized_data = []  # 用于存储所有文件的标准化数据以计算全局权重
    file_data_dict = {}  # 保存每个文件的原始和标准化数据
    file_colors = {}  # 保存每个文件的颜色

    # 遍历文件夹中的文件，筛选数据并标准化
    for idx, file in enumerate(files):
        # name = os.path.basename(file)
        # name = name[7:-5]
        # if name not in name_list:
        #     continue
        #print(f"正在处理文件 {file}...")
        try:
            # 读取 Excel 文件
            data = pd.read_excel(file)

            # 检查是否包含必要的列
            if timestamp_column not in data.columns or not all(col in data.columns for col in relevant_columns):
                print(f"文件 {file} 缺少必要的列，已跳过。")
                break

            # 转换时间列为 datetime 类型
            data[timestamp_column] = pd.to_datetime(data[timestamp_column], errors='coerce')
            data['motor_speed'] = data['motor_speed'].abs()

            # 筛选符合条件的时间段
            #filtered_data = data[data[timestamp_column] > pd.Timestamp(timestamp_filter)]
            filtered_data = data
            # 去除空值或无效值
            filtered_data = filtered_data[relevant_columns + [timestamp_column]].replace(0, np.nan).dropna()

            if not filtered_data.empty:
                # 对单个文件的特定列进行标准化
                scaler = StandardScaler()
                # standardized_data = scaler.fit_transform(filtered_data[relevant_columns])
                # standardized_df = pd.DataFrame(standardized_data, columns=relevant_columns)
                #scaler = MinMaxScaler(feature_range=(0, 1))  # 也可以设置为 (0, 100) 进行归一化到 [0, 100]
                standardized_data = scaler.fit_transform(filtered_data[relevant_columns])
                standardized_df = pd.DataFrame(standardized_data, columns=relevant_columns)

                # 输出计算得到的均值和标准差
                with open(r'D:\Code\data-analysis\一车间能源消耗数据分析\mean_scale_10.txt', 'a+') as f:
                    #f.write(f"{os.path.basename(file)[7:-5]}:" + str(np.abs(scaler.data_min_))+str(scaler.data_max_)+'\n')
                    f.write(f"{os.path.basename(file)[7:-5]}:" + str(scaler.mean_)+str(scaler.scale_)+'\n')

                    print("write mean and scale")
                # 保存单个文件的原始数据及其标准化数据
                filtered_data.reset_index(drop=True, inplace=True)
                standardized_df[timestamp_column] = filtered_data[timestamp_column]
                file_data_dict[file] = standardized_df  # 标准化后的数据
                all_standardized_data.append(standardized_df[relevant_columns])  # 用于全局权重计算

                # 为每个文件分配颜色
                file_colors[file] = plt.cm.get_cmap('tab10', len(files))(idx)

        except Exception as e:
            print(f"文件 {file} 处理时出错：{e}")

    if not all_standardized_data:
        print("没有符合条件的数据。")
        return None, None

    # 合并所有标准化数据并进行 PCA
    combined_data = pd.concat(all_standardized_data, ignore_index=True)
    print(combined_data.columns)
    pca = PCA()
    pca.fit(combined_data)
    weights = pca.explained_variance_ratio_

    # 计算健康度
    file_health_scores = {}
    for file, standardized_df in file_data_dict.items():
        # 计算健康度
        standardized_values = standardized_df[relevant_columns].values
        health_scores = np.dot(standardized_values, weights)

        # 归一化健康度为百分比形式
        health_scores_normalized = (health_scores - health_scores.min()) / (health_scores.max() - health_scores.min()) * 100
        # for idx in range(len(health_scores_normalized)):
        #     item = health_scores_normalized[idx]
            # if item < 40:
            #     health_scores_normalized[idx] = health_scores_normalized[idx]+40
            # elif 90>item>40:
            #     health_scores_normalized[idx] = health_scores_normalized[idx]*1.2
            #     item = health_scores_normalized[idx]
            #     if item>=90:
            #         health_scores_normalized[idx] = health_scores_normalized[idx]*0.8
            #         if health_scores_normalized[idx]>=100:
            #             health_scores_normalized[idx] = health_scores_normalized[idx]*0.2
            #         print(health_scores_normalized[idx])
        standardized_df['health_score'] = health_scores_normalized

        with open(r'D:\Code\data-analysis\一车间能源消耗数据分析\health_score_10.txt', 'a+') as f:
            f.write(f"{os.path.basename(file)[0:-5]}:" + str(health_scores.min()) +" "+ str(health_scores.max()) + '\n')
            print(f"{os.path.basename(file)[0:-5]}:" + str(health_scores.min()) + str(health_scores.max()))

        # 对健康度数据进行平滑
        smoothed_health_scores = smooth_data(health_scores_normalized, method=smoothing_method, window_size=7)

        # 保存健康度及时间列
        standardized_df['smoothed_health_score'] = smoothed_health_scores
        file_health_scores[file] = standardized_df[[timestamp_column, 'smoothed_health_score']]
        #file_health_scores[file] = standardized_df[[timestamp_column, 'health_score']]
    # 绘制每个辊的健康度曲线
    #plt.figure(figsize=(12, 6))
    plt.figure(figsize=(15, 10))
    for file, health_data in file_health_scores.items():
        timestamps = health_data[timestamp_column].values
        smoothed_health_scores = health_data['smoothed_health_score'].values

       # health_scores = health_data['health_score'].values
        #smoothed_health_scores = health_scores
        color = file_colors[file]
        #label = os.path.basename(file)  # 每个文件使用相同的标签
        name = os.path.basename(file)
        name = name[0:-5]
        label =name
        # 找到连续的时间段
        gaps = np.diff(timestamps) > pd.Timedelta("2 hour")  # 假设间隔大于1小时为中断

        segments = np.split(np.arange(len(timestamps)), np.where(gaps)[0] + 1)

        # 分段绘制每个文件的健康度曲线，确保颜色一致
        #for segment in segments:
        for idx, segment in enumerate(segments):

            if idx ==0:
                plt.plot(
                    timestamps[segment],
                    smoothed_health_scores[segment],
                    label=label,
                    color=color  # 使用固定的颜色
                )
            else:
                plt.plot(
                    timestamps[segment],
                    smoothed_health_scores[segment],
                    #label=label,
                    color=color  # 使用固定的颜色
                )
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H"))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())  # 自动调整时间间隔

    plt.title('Smoothed Equipment Health Over Time', fontsize=16)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Smoothed Health Score (%)', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return file_health_scores, weights

def main():
    # ===== 示例调用 =====
    folder_path = r"D:\Code\data-analysis\一车间能源消耗数据分析"  # 文件夹路径
    speed = 'motor_speed'
    torque = 'motor_torque'
    temperature = 'motor_temperature'
    relevant_columns = [speed, temperature, torque]  # 需要的列

    file_health_scores, weights = process_folder_with_filewise_standardization(folder_path, relevant_columns, smoothing_method="savitzky_golay", smoothing_window=7)

    # 输出结果
    if file_health_scores:
        print("\n各文件健康度结果：")
        for file, health_data in file_health_scores.items():
            print(f"\n文件: {file}")
            print(health_data.head())
        print("\n全局权重 (PCA 权重):", weights)

def compute_health_score():
    file = r'D:\Code\data-analysis\一车间能源消耗数据分析\4752-1-1119-1120-hour.xlsx'
    # 读取 Excel 文件
    data = pd.read_excel(file)
    timestamp_column = "Hour"
    # speed = 'Motor_Speed'
    # torque = 'Motor_Torque'
    # temperature = 'Motor_Temperature'
    speed = 'motor_speed'
    torque = 'motor_torque'
    temperature = 'motor_temperature'
    relevant_columns = [speed, temperature, torque]  # 需要的列

    # 检查是否包含必要的列
    if  not all(col in data.columns for col in relevant_columns):
        print(f"文件 {file} 缺少必要的列，已跳过。")
        return

    # 转换时间列为 datetime 类型
    data['Hour'] = pd.to_datetime(data["Hour"], errors='coerce')
    # data['Motor_Speed'] = data['Motor_Speed'].abs()
    data['motor_speed'] = data['motor_speed'].abs()


    filtered_data = data
    # 去除空值或无效值
    #filtered_data = filtered_data[relevant_columns + ["Hour"]].replace(0, np.nan).dropna()
    if filtered_data.empty:
        print("新数据无有效数据，无法计算健康度。")
        return
    print(filtered_data.columns)
    # 加载均值和标准差
    mean = [789.97364499, 59.04133107, 66.64320535]
    scale = [1.67499015, 2.32244835, 6.84288333]
    max_health_score = 1.0846991513155575
    min_health_score = -4.6680481418296855
    # 使用保存的均值和标准差进行标准化
    standardized_data = (filtered_data[relevant_columns] - mean) / scale

    # 假设权重均等（可以根据具体务调整）
    weights = [0.5587,0.3384,0.1029]

    # 计算健康度
    health_scores = np.dot(standardized_data, weights)

    # 利用历史数据 归一化健康度为百分比
    health_scores_normalized = (health_scores - min_health_score) / (max_health_score - min_health_score) * 100
    # 为了防止极大值或极小值出现 再根据现有值进行归一化
    if health_scores_normalized.min() < 0:
        # 数据异常则再次归一化显示
        health_scores_normalized[health_scores_normalized < 0] = 0
        max = np.max(health_scores_normalized)
        min = np.min(health_scores_normalized)
        health_scores_normalized = (health_scores_normalized - min) / (max - min) * 100
    #print(health_scores_normalized)
    # 平滑健康度数据
    smoothed_health_scores = smooth_data(health_scores_normalized, method="moving_average",
                                         window_size=3)

    # 绘制健康度曲线
    plt.figure(figsize=(12, 6))
    plt.plot(filtered_data[timestamp_column], smoothed_health_scores, label="Smoothed Health Score", color="blue")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H"))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())  # 自动调整时间间隔
    plt.title(f"Health Score for 4752-1-11", fontsize=16)
    plt.xlabel("Time (Month-Day Hour)", fontsize=12)
    plt.ylabel("Smoothed Health Score (%)", fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()