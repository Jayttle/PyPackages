import pandas as pd
from sklearn.model_selection import train_test_split
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import numpy as np

# 读取 Excel 文件
file_path = r"C:\Users\juntaox\Desktop\最新计算数据（含19-24年四个季度.xlsx"  # 替换为你的 Excel 文件路径
df = pd.read_excel(file_path)

# 查看数据的前几行，确保正确读取
print(df.head())

# 创建一个空的 DataFrame 来保存结果
result_df = pd.DataFrame(columns=["名称", "AIC", "BIC", "2024年Q4预测值"])

for index, row in df.iterrows():
    # 打印每行数据
    print(f"Data: {row.iloc[0]}")
    if row.iloc[0] == '信息传输、软件和信息技术服务业' or row.iloc[0] == '租赁和商务服务业':
        data = {
            'Q1': [row.iloc[5], row.iloc[9], row.iloc[13], row.iloc[17]],
            'Q2': [row.iloc[6], row.iloc[10], row.iloc[14], row.iloc[18]],
            'Q3': [row.iloc[7], row.iloc[11], row.iloc[15], row.iloc[19]],
            'Q4': [row.iloc[8], row.iloc[12], row.iloc[16], row.iloc[20]],
        }
    else:
        data = {
            'Q1': [row.iloc[1], row.iloc[5], row.iloc[9], row.iloc[13], row.iloc[17]],
            'Q2': [row.iloc[2], row.iloc[6], row.iloc[10], row.iloc[14], row.iloc[18]],
            'Q3': [row.iloc[3], row.iloc[7], row.iloc[11], row.iloc[15], row.iloc[19]],
            'Q4': [row.iloc[4], row.iloc[8], row.iloc[12], row.iloc[16], row.iloc[20]],
        }

    data_df = pd.DataFrame(data)
    
    # 以 Q1, Q2, Q3 为时间序列的输入数据，Q4 为目标
    y = data_df['Q4']
    X = data_df[['Q1', 'Q2', 'Q3']]  # 使用 Q1, Q2, Q3 作为输入特征

    # 假设2024年Q1, Q2, Q3的值
    Q1_2024 = row.iloc[21]
    Q2_2024 = row.iloc[22]
    Q3_2024 = row.iloc[23]

    # 使用SARIMA模型来预测Q4
    # 这里选择一个简单的SARIMA(1, 1, 1)x(1, 1, 1, 4)，你可以根据实际情况调整
    exog_data = data_df[['Q1', 'Q2', 'Q3']].values

    # 使用外生变量
    model = SARIMAX(y, order=(1, 1, 1), seasonal_order=(1, 1, 1, 4), exog=exog_data)
    model_fit = model.fit(disp=False)
    model_fit = model.fit(disp=False)

    # 获取 AIC 和 BIC
    aic = model_fit.aic
    bic = model_fit.bic

    forecast = model_fit.forecast(steps=1)
    if isinstance(forecast, pd.Series):
        forecast_value = forecast.iloc[0]  # 使用iloc访问第一个元素
    else:
        forecast_value = forecast[0]  # 如果返回是numpy数组，直接使用[0]
    print(forecast_value)

    # # 提供外生变量预测值
    # future_exog = np.array([[Q1_2024, Q2_2024, Q3_2024]])  # 预测年份的特征

    # # 进行预测
    # forecast = model_fit.forecast(steps=1, exog=future_exog)
    # forecast_value = forecast.iloc[0]  # 包装成预测值的单一值
    # 将结果添加到 result_df
    new_row = pd.DataFrame([{
        "名称": row.iloc[0],  # 使用 iloc 来获取名称列
        "AIC": aic,
        "BIC": bic,
        "2024年Q4预测值": forecast
    }])

    # 使用 pd.concat() 合并原 DataFrame 和新行
    result_df = pd.concat([result_df, new_row], ignore_index=True)

# 将结果保存到新的 Excel 文件
result_df.to_excel(r"C:\Users\juntaox\Desktop\数据结果-SARIMA.xlsx", index=False)
