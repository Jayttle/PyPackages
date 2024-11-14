import pandas as pd
import matplotlib.pyplot as plt
import pandas.plotting as pd_plotting


def save_dataframe_as_image(df: pd.DataFrame, file_name: str):
    fig, ax = plt.subplots(figsize=(12, len(df) * 0.3 + 2))  # Adjust size based on the number of rows
    ax.axis('off')
    
    # Create table and add it to the figure
    table = pd_plotting.table(ax, df, loc='center', cellLoc='center', colWidths=[0.1] * len(df.columns))
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)  # Adjust scale if needed
    
    # Save the figure
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0.1)
    plt.close()

def avr_pdProcess(file_path: str, frequency: int = None):
    avr_column = ['time', 'stationID', 'fixmode', 'yaw', 'tilt', 'range', 'pdop', 'sat_num']
    df = pd.read_csv(file_path, names=avr_column, header=None)
    
    # 统计缺失值
    missing_values = df.isnull().sum()
    print("缺失值统计：")
    print(missing_values)
    
    # 统计重复值
    duplicate_rows = df.duplicated().sum()
    print(f"\n重复行的数量: {duplicate_rows}")
    
    # 统计不同 stationID 的数据量
    stationID_counts = df['stationID'].value_counts()
    print("\n不同 stationID 的数据量：")
    print(stationID_counts)
    
    if frequency is not None:
        # 计算一天的秒数和应有数据量
        total_seconds = 86400
        expected_data_count = total_seconds // frequency
        print(f"\n一天的应有数据量（每{frequency}秒一个数据）: {expected_data_count}")
        
        # 计算不同 stationID 的数据占有率百分比
        print("\n不同 stationID 的数据占有率：")
        for stationID, count in stationID_counts.items():
            percentage = (count / expected_data_count) * 100
            print(f"stationID {stationID}: {percentage:.2f}%")


    # 创建一个 DataFrame 来存储统计分析数据
    stats_data = []
    stats_columns = ['stationID', 'column', 'mean', 'std_err', 'median', 'max', 'min']

    for stationID in stationID_counts.index:
        station_df: pd.DataFrame = df[df['stationID'] == stationID]
        for column in ['yaw', 'tilt', 'range']:
            if column in station_df.columns:
                # 计算统计数据
                mean = round(station_df[column].mean(), 4)
                std_err = round(station_df[column].sem(), 4)  # 标准误差
                median = round(station_df[column].median(), 4)
                max_value = round(station_df[column].max(), 4)
                min_value = round(station_df[column].min(), 4)
                
                # 将结果添加到列表中
                stats_data.append({
                    'stationID': stationID,
                    'column': column,
                    'mean': mean,
                    'std_err': std_err,
                    'median': median,
                    'max': max_value,
                    'min': min_value
                })
    
    # 将列表转换为 DataFrame
    stats_df = pd.DataFrame(stats_data, columns=stats_columns)
    
    print("\n统计分析结果：")
    print(stats_df)

    # 比较不同 stationID 的数据
    comparison_df = compare_duplicate_columns(stats_df)
    
    # 打印比较结果
    print("\n不同 stationID 之间的比较结果：")
    print(comparison_df)

    save_dataframe_as_image(comparison_df, 'comparison_df_image.png')

def compare_duplicate_columns(stats_df: pd.DataFrame) -> pd.DataFrame:
    comparison_data = []
    
    for column_name, group in stats_df.groupby('column'):
        stations = group['stationID'].unique()
        for i in range(len(stations)):
            for j in range(i + 1, len(stations)):
                station1 = stations[i]
                station2 = stations[j]
                
                # Extract data for station1 and station2
                data1 = group[group['stationID'] == station1]
                data2 = group[group['stationID'] == station2]
                
                # Extract statistics for station1
                mean1 = data1['mean'].values[0]
                std_err1 = data1['std_err'].values[0]
                median1 = data1['median'].values[0]
                max1 = data1['max'].values[0]
                min1 = data1['min'].values[0]
                
                # Extract statistics for station2
                mean2 = data2['mean'].values[0]
                std_err2 = data2['std_err'].values[0]
                median2 = data2['median'].values[0]
                max2 = data2['max'].values[0]
                min2 = data2['min'].values[0]
                
                # Calculate differences
                mean_diff = round(mean2 - mean1, 4)
                std_err_diff = round(std_err2 - std_err1, 4)
                median_diff = round(median2 - median1, 4)
                max_diff = round(max2 - max1, 4)
                min_diff = round(min2 - min1, 4)
                
                # Calculate percentage changes
                mean_percent_change = round((mean_diff / mean1) * 100, 2) if mean1 != 0 else float('inf')
                std_err_percent_change = round((std_err_diff / std_err1) * 100, 2) if std_err1 != 0 else float('inf')
                median_percent_change = round((median_diff / median1) * 100, 2) if median1 != 0 else float('inf')
                max_percent_change = round((max_diff / max1) * 100, 2) if max1 != 0 else float('inf')
                min_percent_change = round((min_diff / min1) * 100, 2) if min1 != 0 else float('inf')
                
                # Add comparison results to list
                comparison_data.append({
                    'column': column_name,
                    'station1': station1,
                    'station1_mean': mean1,
                    'station1_std_err': std_err1,
                    'station1_median': median1,
                    'station1_max': max1,
                    'station1_min': min1,
                    'station2': station2,
                    'station2_mean': mean2,
                    'station2_std_err': std_err2,
                    'station2_median': median2,
                    'station2_max': max2,
                    'station2_min': min2,
                    'mean_diff': mean_diff,
                    'mean_percent': mean_percent_change,
                    'std_err_diff': std_err_diff,
                    'std_err_percent': std_err_percent_change,
                    'median_diff': median_diff,
                    'median_percent': median_percent_change,
                    'max_diff': max_diff,
                    'max_percent': max_percent_change,
                    'min_diff': min_diff,
                    'min_percent': min_percent_change
                })
    
    # Convert list to DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    return comparison_df

if __name__ == '__main__':
    file_path = r"D:\Program Files (x86)\Software\OneDrive\PyPackages\avr_yesterday_data.txt"
    frequency = 1  # 频率参数，单位为秒
    avr_pdProcess(file_path, frequency)
