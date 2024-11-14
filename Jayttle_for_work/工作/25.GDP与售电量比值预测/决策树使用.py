import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 读取 Excel 文件
file_path = r"C:\Users\juntaox\Desktop\最新计算数据（含19-24年四个季度.xlsx"  # 替换为你的 Excel 文件路径
df = pd.read_excel(file_path)

# 查看数据的前几行，确保正确读取
print(df.head())

# 创建一个空的 DataFrame 来保存结果
result_df = pd.DataFrame(columns=["名称", "Q1、Q2、Q3特征重要性", "MSE", "2024年Q4预测值"])

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
    
    # 特征和目标变量
    X = data_df[['Q1', 'Q2', 'Q3']]  # 前三个季度
    y = data_df['Q4']  # 第四季度

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 创建决策树回归模型
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)

    # 对测试集进行预测
    y_pred = model.predict(X_test)

    # 计算MSE
    mse = mean_squared_error(y_test, y_pred)

    # 获取特征重要性
    feature_importances = model.feature_importances_

    # 假设2024年Q1, Q2, Q3的值
    Q1_2024 = row.iloc[21]
    Q2_2024 = row.iloc[22]
    Q3_2024 = row.iloc[23]

    # 使用 DataFrame 传递数据，保持列名一致
    X_2024 = pd.DataFrame([[Q1_2024, Q2_2024, Q3_2024]], columns=['Q1', 'Q2', 'Q3'])

    # 根据回归模型预测2024年Q4的值
    Q4_2024 = model.predict(X_2024)[0]

    # 将结果添加到result_df
    new_row = pd.DataFrame([{
        "名称": row.iloc[0],  # 使用 iloc 来获取名称列
        "Q1、Q2、Q3特征重要性": feature_importances.tolist(),
        "MSE": mse,
        "2024年Q4预测值": Q4_2024
    }])

    # 使用 pd.concat() 合并原 DataFrame 和新行
    result_df = pd.concat([result_df, new_row], ignore_index=True)

# 将结果保存到新的 Excel 文件
result_df.to_excel(r"C:\Users\juntaox\Desktop\数据结果-决策树.xlsx", index=False)

