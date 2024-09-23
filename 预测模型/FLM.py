import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl

if __name__ == '__main__':
    # 读取数据
    file_path = r"C:\Users\juntaox\Desktop\电量汇总结果.xlsx"
    sheet_name = '1、小陆家嘴'
    df = pd.read_excel(file_path, usecols=['日期', sheet_name])

    # 确保日期列是 datetime 类型
    df['日期'] = pd.to_datetime(df['日期'].str[2:], format='%Y%m')
    df['month'] = df['日期'].dt.month
    df['lag1'] = df[sheet_name].shift(1)
    df['rolling_mean3'] = df[sheet_name].rolling(window=3).mean()
    df.dropna(inplace=True)

    # 定义模糊变量
    month = ctrl.Antecedent(np.arange(1, 13, 1), 'month')
    lag1 = ctrl.Antecedent(np.arange(df[sheet_name].min(), df[sheet_name].max(), 1), 'lag1')
    rolling_mean3 = ctrl.Antecedent(np.arange(df[sheet_name].min(), df[sheet_name].max(), 1), 'rolling_mean3')
    electric = ctrl.Consequent(np.arange(df[sheet_name].min(), df[sheet_name].max(), 1), 'electric')

    # 为每个模糊变量定义隶属度函数
    month.automf(3)
    lag1.automf(3)
    rolling_mean3.automf(3)
    electric.automf(3)

    # 定义模糊规则
    rule1 = ctrl.Rule(month['poor'] & lag1['poor'] & rolling_mean3['poor'], electric['poor'])
    rule2 = ctrl.Rule(month['average'] | lag1['average'], electric['average'])
    rule3 = ctrl.Rule(month['good'] & lag1['good'] & rolling_mean3['good'], electric['good'])

    # 创建和运行模糊逻辑系统
    electric_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
    electric_sim = ctrl.ControlSystemSimulation(electric_ctrl)

    # 使用测试数据进行预测
    predictions = []
    for _, row in df.iterrows():
        electric_sim.input['month'] = row['month']
        electric_sim.input['lag1'] = row['lag1']
        electric_sim.input['rolling_mean3'] = row['rolling_mean3']
        electric_sim.compute()
        predictions.append(electric_sim.output['electric'])

    df['predicted'] = predictions
    print(df[['日期', sheet_name, 'predicted']])

    # 评估模型（根据实际应用情况选择合适的评估方法）
    # 例如使用均方误差评估
    mse = mean_squared_error(df[sheet_name], df['predicted'])
    print(f'Mean Squared Error: {mse}')
