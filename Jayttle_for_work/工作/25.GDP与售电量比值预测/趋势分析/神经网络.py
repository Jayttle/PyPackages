import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 示例时间序列数据（示意性）
data = {
    'Year': [2020, 2020, 2020, 2020, 
             2021, 2021, 2021, 2021, 
             2022, 2022, 2022, 2022],
    'Quarter': ['Q1', 'Q2', 'Q3', 'Q4',
                'Q1', 'Q2', 'Q3', 'Q4',
                'Q1', 'Q2', 'Q3', 'Q4'],
    'Value': [1.5, 1.6, 1.7, 1.8, 
              2.0, 2.1, 2.2, 2.3, 
              2.4, 2.5, 2.6, 2.7]
}

df = pd.DataFrame(data)

# 创建特征和目标
features = []
targets = []

for year in df['Year'].unique():
    yearly_data = df[df['Year'] == year]['Value'].values
    features.append(yearly_data[:3])  # 前三个季度数据
    targets.append(yearly_data[3])    # 第四季度数据

X = np.array(features)
y = np.array(targets)

# 数据拆分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征缩放
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 构建优化后的模型
model = Sequential([
    Dense(128, activation='relu', input_shape=(3,)),  # 增加神经元数量和层数
    Dropout(0.2),  # Dropout层用于正则化
    Dense(64, activation='relu'),
    Dropout(0.2),  # Dropout层，用于减少过拟合
    Dense(32, activation='relu'),
    Dense(1)  # 输出层
])

# 编译模型
model.compile(optimizer='adam', loss='mse')

# 使用早停（EarlyStopping）防止过拟合
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 训练模型
history = model.fit(X_train_scaled, y_train, epochs=200, batch_size=1, validation_split=0.1, verbose=1, callbacks=[early_stopping])

# 预测
y_pred = model.predict(X_test_scaled)

# 计算MSE、MAE和R²
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')
print(f'R² Score: {r2}')
