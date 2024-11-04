import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
def _df_read_excel() -> pd.DataFrame: 
    df = pd.read_excel(r"C:\Users\Jayttle\Desktop\2024917_desktop\小陆家嘴.xlsx")
    columns = [col for col in df.columns if col.startswith('电量')]

    df_result = {
        'Date': [],
        'values': []
    }
    for col in columns:
        data_str = col[2:]
        dt = pd.to_datetime(data_str, format='%Y%m')
        df_result['Date'].append(dt)
        df_result['values'].append(df[col].sum() / 10000 / 10000)

    df_result = pd.DataFrame(df_result)

    return df_result

def calculate_mape(actual, predicted):
    # Avoid division by zero
    return (abs(actual - predicted) / actual).mean() * 100

def train_svr_on_residuals(X_train, y_train):
    # 标准化
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))

    svr_model = SVR(kernel='rbf')
    svr_model.fit(X_train_scaled, y_train_scaled.ravel())

    return svr_model, scaler_X, scaler_y

def predict_with_svr(svr_model, scaler_X, scaler_y, X_test):
    X_test_scaled = scaler_X.transform(X_test)
    residuals_pred_scaled = svr_model.predict(X_test_scaled)
    return scaler_y.inverse_transform(residuals_pred_scaled.reshape(-1, 1)).flatten()

if __name__ == '__main__':
    df = _df_read_excel()
    
    df.set_index('Date', inplace=True)

    df_train = df.iloc[:-6].copy()
    df_test = df.iloc[-6:]


    # Fit the SARIMA model
    model_sarima = SARIMAX(df_train['values'], order=(1, 1, 2), seasonal_order=(1, 1, 0, 12))
    results_sarima = model_sarima.fit(disp=False)

    # Make predictions
    df_train['L_hat'] = results_sarima.predict(start=df_train.index[0], end=df_train.index[-1])

    # 残差
    df_train['E_t'] = df_train['values'] - df_train['L_hat']

    # Prepare data for SVR
    # 这里我们使用时间序列的索引或其它特征来构建X
    df_train['t'] = range(len(df_train))  # 添加时间特征
    X_train = df_train[['t']]  # 特征（时间）
    y_train = df_train['E_t']  # 残差

    # Train SVR on residuals
    svr_model, scaler_X, scaler_y = train_svr_on_residuals(X_train, y_train)

    # Predict residuals for the test set
    df_test['t'] = range(len(df_train), len(df_train) + len(df_test))  # 添加时间特征
    X_test = df_test[['t']]
    residuals_pred = predict_with_svr(svr_model, scaler_X, scaler_y, X_test)

    # Combine predictions
    df_test['L_hat_optimized'] = results_sarima.forecast(steps=6) + residuals_pred
    # Calculate MAPE for optimized predictions
    mape_value_optimized = calculate_mape(df_test['values'], df_test['L_hat_optimized'])
    print(df_test)
    print(f'Optimized MAPE: {mape_value_optimized:.2f}%')
