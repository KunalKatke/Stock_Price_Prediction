import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Function to get historical data from Yahoo Finance
def get_historical_data(ticker, start_date, end_date):
    data = yf.download(tickers=ticker, start=start_date, end=end_date)
    return data

# Function to preprocess the data
def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    return scaled_data, scaler

# Function to create the dataset for LSTM
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

# Function to create the LSTM model
def create_model(look_back):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to evaluate the model
def evaluate_model(model, X_test, Y_test, scaler):
    Y_pred = model.predict(X_test)
    Y_pred = scaler.inverse_transform(Y_pred)
    Y_test = scaler.inverse_transform(Y_test.reshape(-1, 1))
    mse = mean_squared_error(Y_test, Y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(Y_test, Y_pred)
    return mse, rmse, r2

# Main function
def main():
    # Set the start and end dates for historical data
    start_date = '2010-01-01'
    end_date = '2022-04-11'

    # Get historical data for a stock
    ticker = 'BTC-USD'
    data = get_historical_data(ticker, start_date, end_date)

    # Preprocess the data
    scaled_data, scaler = preprocess_data(data)

    # Create the dataset for LSTM
    look_back = 60
    X, Y = create_dataset(scaled_data, look_back)

    # Split the dataset into training and testing sets
    train_size = int(len(X) * 0.8)
    X_train, Y_train = X[:train_size], Y[:train_size]
    X_test, Y_test = X[train_size:], Y[train_size:]

    # Reshape the input data for LSTM
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Create the LSTM model
    model = create_model(look_back)

    # Train the LSTM model
    model.fit(X_train, Y_train, epochs=50, batch_size=64, verbose=1)

    # Evaluate the model
    mse, rmse, r2 = evaluate_model(model, X_test, Y_test, scaler)

    # Print the results
    print('Mean squared error:', mse)
    print('Root mean squared error:', rmse)
    print('R-squared score:', r2)

if __name__ == '__main__':
    main()
