import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 示例时间序列数据（示意性）
data = {
    'Year': [2020, 2020, 2020, 2020, 2021, 2021, 2021, 2021, 2022, 2022, 2022, 2022],
    'Quarter': ['Q1', 'Q2', 'Q3', 'Q4',
                'Q1', 'Q2', 'Q3', 'Q4',
                'Q1', 'Q2', 'Q3', 'Q4'],
    'Value': [1.5, 1.6, 1.7, 1.8, 
              2.0, 2.1, 2.2, 2.3, 
              2.4, 2.5, 2.6, 2.7]
}

df = pd.DataFrame(data)
print(df)
# 创建特征和目标
# 因为我们要用前三个季度来预测第四季度，所以每四行是一组数据
features = []
targets = []

for year in df['Year'].unique():
    yearly_data = df[df['Year'] == year]['Value'].values
    features.append(yearly_data[:3])  # 前三个季度数据
    targets.append(yearly_data[3])    # 第四季度数据

X = np.array(features)
y = np.array(targets)
print(X)
print(y)
# 数据拆分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立随机森林模型
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 模型预测
y_pred = model.predict(X_test)

# 评估模型
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
# 获得特征重要性
importances = model.feature_importances_

# 可视化特征重要性
quarters = ['Q1', 'Q2', 'Q3']

print(importances)
plt.figure(figsize=(10, 6))
plt.barh(quarters, importances, color='skyblue')
plt.xlabel('Feature Importance')
plt.title('Feature Importance by Random Forest')
plt.show()


# 假设2024年Q1, Q2, Q3的值
Q1_2024 = 2.2
Q2_2024 = 2.5
Q3_2024 = 2.9
# 使用模型来预测2024年Q4的值
Q4_2024_pred = model.predict([[Q1_2024, Q2_2024, Q3_2024]])

print(f"2024年Q4的预测值: {Q4_2024_pred[0]:.4f}")
