import pandas as pd
from sklearn.ensemble import IsolationForest
import numpy as np

def DIF(df: pd.DataFrame):
    # 过滤出motor_run为True的数据
    df_motor_running = df[df['motor_run'] == True]
    
    # 提取需要分析的特征
    features = ['motor_speed', 'motor_temperature', 'motor_torque']
    
    # 处理潜在的缺失值
    df_filtered = df_motor_running[features].dropna()
    
    # 使用Isolation Forest进行异常检测
    iso_forest = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
    df_filtered['anomaly'] = iso_forest.fit_predict(df_filtered)

    # 输出筛选后的结果，-1为异常，1为正常
    anomalies = df_filtered[df_filtered['anomaly'] == -1]
    print("Anomaly data points:")
    print(anomalies)
    anomalies.to_excel('temp.xlsx')

if __name__ == "__main__":
    file_path = r"C:\Users\juntaox\Desktop\4702-1\4702-1_0829_0917.csv"
    df = pd.read_csv(file_path)

    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values(by='datetime')
    # 执行DIF函数来查找异常
    DIF(df)
