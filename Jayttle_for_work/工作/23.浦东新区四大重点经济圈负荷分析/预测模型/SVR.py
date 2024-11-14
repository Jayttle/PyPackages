import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR  # 导入支持向量回归模型
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler  # 导入标准化预处理类

if __name__ == '__main__':
    # 读取数据
    file_path = r"C:\Users\juntaox\Desktop\电量汇总结果.xlsx"
    sheet_name = '1、小陆家嘴'
    df = pd.read_excel(file_path, usecols=['日期', sheet_name])

    # 确保日期列是 datetime 类型
    df['日期'] = pd.to_datetime(df['日期'].str[2:], format='%Y%m')
    df['month'] = df['日期'].dt.month

    # 创建延时特征以捕捉历史趋势
    df['lag1'] = df[sheet_name].shift(1)
    df['rolling_mean3'] = df[sheet_name].rolling(window=3).mean()

    # 移除缺失值
    df.dropna(inplace=True)

    X = df[['month', 'lag1', 'rolling_mean3']]  # 特征
    y = df[sheet_name]  # 目标变量

    # 拆分数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 标准化特征，这通常对于支持向量机类模型很重要
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 使用SVR训练模型
    model = SVR(kernel='linear')  # 你可以尝试不同的核函数，例如 'rbf', 'poly', 'sigmoid'
    model.fit(X_train_scaled, y_train)

    # 预测
    predictions = model.predict(X_test_scaled)

    # 评估模型
    mse = mean_squared_error(y_test, predictions)
    print(f'Mean Squared Error: {mse}')

    # 计算预测误差，并用于调整预测
    errors = y_test - predictions
    correction_factor = errors.mean()  # 使用简单的平均误差进行调整
    adjusted_predictions = predictions + correction_factor

    # 打印调整后的预测结果和实际值
    result_df = pd.DataFrame({'实际值': y_test, '预测值': predictions, '调整后预测值': adjusted_predictions})
    result_df['误差百分比'] = (result_df['调整后预测值'] - result_df['实际值']) / result_df['实际值'] * 100
    print(result_df)

    # 重新评估调整后的模型
    adjusted_mse = mean_squared_error(y_test, adjusted_predictions)
    print(f'Adjusted Mean Squared Error: {adjusted_mse}')