from tensorflow.keras.layers import SimpleRNN

def train_RNN_model(df: pd.DataFrame):
    # Extract and normalize sales data
    sales = df['总和'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    sales_scaled = scaler.fit_transform(sales)

    # Prepare the input sequences and target values
    X, y = [], []
    for i in range(len(sales_scaled) - 10):
        X.append(sales_scaled[i:i + 10, 0])
        y.append(sales_scaled[i + 10, 0])

    X, y = np.array(X), np.array(y)

    # Reshape the data for RNN model
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Build the RNN model
    model = Sequential()
    model.add(SimpleRNN(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(SimpleRNN(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=50, batch_size=32)

    # Save the model
    model.save(f'model_RNN.h5')

    # Evaluate the model on the training data
    train_predictions = model.predict(X)
    train_predictions = scaler.inverse_transform(train_predictions)
    true_sales = df['总和'].values[10:]
    mse = mean_squared_error(true_sales, train_predictions)
    mae = mean_absolute_error(true_sales, train_predictions)

    print(f"Mean Squared Error on Training Data: {mse}")
    print(f"Mean Absolute Error on Training Data: {mae}")

def load_model_predict_RNN(file_path, sheet_names, date_str):
    # Load the saved model
    model = load_model(f'model_RNN.h5')

    # Read the CSV file
    df = pd.read_excel(file_path)
    # 确保日期列是 datetime 类型
    if '日期' in df.columns:
        df['日期'] = pd.to_datetime(df['日期'].str[2:], format='%Y%m')
    else:
        raise KeyError("DataFrame 中找不到 '日期' 列")
    
    # Convert the input date string to datetime
    input_date = datetime.datetime.strptime(date_str, '%Y/%m')

    # Filter data for the given product and date
    product_df = df[(df['日期'] <= input_date)].copy()

    # Extract and normalize sales data
    sales = product_df[sheet_names].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    sales_scaled = scaler.fit_transform(sales)

    # Prepare the input sequence for prediction
    input_sequence = sales_scaled[-10:].reshape(1, 10, 1)

    # Make predictions
    predicted_sales_scaled = model.predict(input_sequence)

    # Inverse transform to get the actual sales values
    predicted_sales = scaler.inverse_transform(predicted_sales_scaled)

    return predicted_sales[0, 0]

if __name__ == '__main__':
    print('---------------------------------------------------------------')
    # 示例调用
    file_path = r"C:\Users\juntaox\Desktop\电量汇总结果.xlsx"  # 替换为实际的文件路径
    sheet_names = '1、小陆家嘴'
    results = {}
    columns_to_read = ['日期', sheet_names]  # 替换为你需要的列名
    df_train, df_test = read_and_prepare_data(file_path, columns_to_read)
    # Uncomment to train RNN model
    # train_RNN_model(df_train)
    prediction_date = '2024/9'
    predicted_sales = load_model_predict_RNN(file_path, sheet_names, prediction_date)
    print(predicted_sales)
