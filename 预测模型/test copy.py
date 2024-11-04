import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam

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

def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step)])
        y.append(data[i + time_step])
    return np.array(X), np.array(y)

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

    # Calculate residuals
    df_train['E_t'] = df_train['values'] - df_train['L_hat']

    # Prepare data for LSTM
    residuals = df_train['E_t'].values
    residuals = residuals.reshape(-1, 1)  # Reshape for LSTM

    # Create dataset
    time_step = 1  # You can adjust this value
    X, y = create_dataset(residuals, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)  # Reshape for LSTM

    # Build LSTM model
    model_lstm = Sequential()
    model_lstm.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model_lstm.add(LSTM(50))
    model_lstm.add(Dense(1))
    model_lstm.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    # Train the model
    model_lstm.fit(X, y, epochs=100, batch_size=16)

    # Prepare test data for prediction
    last_residuals = residuals[-time_step:].reshape(1, time_step, 1)
    lstm_predictions = []

    for _ in range(6):  # Predict 6 steps ahead
        pred = model_lstm.predict(last_residuals)
        lstm_predictions.append(pred[0][0])
        last_residuals = np.append(last_residuals[:, 1:, :], pred.reshape(1, 1, 1), axis=1)



    # Update the predictions
    df_test['L_hat_optimized'] = results_sarima.forecast(steps=6) + np.array(lstm_predictions)
    actual_values = df.iloc[-6:].values.ravel()  # Converts to 1D
    predicted_values = df_test['L_hat_optimized'].values.ravel()  # Converts to 1D
    df_test['MAPE'] = np.abs((actual_values - predicted_values) / actual_values) * 100

    print(df_test)
