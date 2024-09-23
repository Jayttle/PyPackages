import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error


if __name__ == '__main__':
    # 读取数据
    print('---------------------------------------------------------------')
    file_path = r"C:\Users\juntaox\Desktop\电量汇总结果.xlsx"  # 替换为实际的文件路径
    sheet_names = '1、小陆家嘴'
    columns_to_read = ['日期', sheet_names]  # 替换为你需要的列名
    df = pd.read_excel(file_path, usecols=columns_to_read)

    # 确保日期列是 datetime 类型
    if '日期' in df.columns:
        df['日期'] = pd.to_datetime(df['日期'].str[2:], format='%Y%m')
    else:
        raise KeyError("DataFrame 中找不到 '日期' 列")
    
    # 处理数据
    df['month'] = df['日期'].dt.month  # 提取月份作为特征
    X = df[['month']]  # 特征
    y = df[sheet_names]  # 目标变量

    # 拆分数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 训练决策树回归模型
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)

    # 预测
    predictions = model.predict(X_test)

    # 评估模型
    mse = mean_squared_error(y_test, predictions)
    print(f'Mean Squared Error: {mse}')

    # 打印预测值和实际值
    result_df = pd.DataFrame({'实际值': y_test, '预测值': predictions})

    # 计算误差值百分比
    result_df['误差百分比'] = (result_df['预测值'] - result_df['实际值']) / result_df['实际值'] * 100

    # 计算所有误差值的绝对值的平均值
    result_df['绝对误差'] = (result_df['误差百分比']).abs().mean()

    # 打印结果
    print(result_df[['实际值', '预测值', '误差百分比', '绝对误差']])
