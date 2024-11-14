import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 读取 Excel 文件
file_path = r"C:\Users\juntaox\Desktop\最新计算数据（含19-24年四个季度.xlsx" # 替换为你的 Excel 文件路径
df = pd.read_excel(file_path)

# 假设 Excel 中的数据列名称是：'名称', '2019年1季度', '2019年2季度', ..., '2024年3季度'
# 查看数据的前几行，确保正确读取
print(df.head())
# 创建一个空的 DataFrame 来保存结果
result_df = pd.DataFrame(columns=["名称", "回归系数", "截距", "MSE", "2024年Q4预测值"])

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

    # 创建线性回归模型
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 对测试集进行预测
    y_pred = model.predict(X_test)

    # 计算MSE和R²
    mse = mean_squared_error(y_test, y_pred)

    # 假设2024年Q1, Q2, Q3的值
    Q1_2024 = row.iloc[21]
    Q2_2024 = row.iloc[22]
    Q3_2024 = row.iloc[23]

    # 根据回归系数和截距计算2024年Q4的预测值
    Q4_2024 = model.intercept_ + model.coef_[0] * Q1_2024 + model.coef_[1] * Q2_2024 + model.coef_[2] * Q3_2024

    # 将结果添加到result_df
    # 创建一个新的 DataFrame 来表示要添加的数据
    new_row = pd.DataFrame([{
        "名称": row.iloc[0],  # 注意这里使用了 iloc
        "回归系数": model.coef_.tolist(),
        "截距": model.intercept_,
        "MSE": mse,
        "2024年Q4预测值": Q4_2024
    }])

    # 使用 pd.concat() 合并原 DataFrame 和新行
    result_df = pd.concat([result_df, new_row], ignore_index=True)
result_df.to_excel(r"C:\Users\juntaox\Desktop\数据结果-线性回归.xlsx")

# # 示例数据框
# data = {
#     'Q1': [1.2, 2.1, 3.4, 2.3, 1.8],
#     'Q2': [1.5, 2.5, 3.0, 3.3, 1.7],
#     'Q3': [1.4, 2.3, 2.8, 3.1, 2.0],
#     'Q4': [1.6, 2.6, 3.5, 3.4, 2.1]
# }

# df = pd.DataFrame(data)

# # 特征和目标变量
# X = df[['Q1', 'Q2', 'Q3']]  # 前三个季度
# y = df['Q4']  # 第四季度

# # 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 创建线性回归模型
# model = LinearRegression()
# model.fit(X_train, y_train)

# # 对测试集进行预测
# y_pred = model.predict(X_test)

# # 输出结果
# print("Coefficients:", model.coef_)  # 回归系数
# print("Intercept:", model.intercept_)  # 截距
# print("Mean squared error (MSE): %.2f" % mean_squared_error(y_test, y_pred))
# print("Coefficient of determination (R^2): %.2f" % r2_score(y_test, y_pred))

# # 解释结果
# print("\n影响程度（回归系数）:")
# for i, col in enumerate(X.columns):
#     print(f"{col}: {model.coef_[i]:.4f}")

# # 假设2024年Q1, Q2, Q3的值
# Q1_2024 = 2.2
# Q2_2024 = 2.5
# Q3_2024 = 2.9

# # 根据回归系数和截距计算2024年Q4的预测值
# Q4_2024 = model.intercept_ + model.coef_[0] * Q1_2024 + model.coef_[1] * Q2_2024 + model.coef_[2] * Q3_2024

# print(f"2024年Q4的预测值: {Q4_2024:.4f}")
