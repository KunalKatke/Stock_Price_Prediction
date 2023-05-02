import streamlit as st

def predict_price(ticker, days):
    # Importing required libraries
    import numpy as np
    import pandas as pd
    import yfinance as yf
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM

    # Defining the function to get real-time data
    def get_realtime_data(ticker, interval='1m', range='1d'):
        data = yf.download(tickers=ticker, interval=interval, period=range)
        return data

    # Preprocessing the data
    def preprocess_data(data):
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
        return scaled_data, scaler

    # Creating the dataset for LSTM
    def create_dataset(dataset, look_back=1, days=1):
        X, Y = [], []
        for i in range(len(dataset)-look_back-days):
            a = dataset[i:(i+look_back), 0]
            X.append(a)
            Y.append(dataset[i + look_back:i + look_back + days, 0])
        return np.array(X), np.array(Y)

    # Creating the LSTM model
    def create_model():
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
        model.add(LSTM(50, return_sequences=True))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    # Defining the function to predict stock prices
    def predict_prices(model, data, scaler):
        last_data = data.tail(look_back)
        last_data_scaled = scaler.transform(last_data['Close'].values.reshape(-1, 1))
        X_test = np.array([last_data_scaled])
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        prediction = model.predict(X_test)
        prediction = scaler.inverse_transform(prediction)
        return prediction[0][0] # return the actual predicted value

    # Defining the main function
    def main(ticker):
        data = get_realtime_data(ticker)
        scaled_data, scaler = preprocess_data(data)
        X, Y = create_dataset(scaled_data, look_back)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        model = create_model()
        model.fit(X, Y, batch_size=64, epochs=100,verbose=1)
        prediction = predict_prices(model, data, scaler)
        return prediction # return the actual predicted value

    # Call the main function and return the predicted price
    look_back = 60 
    predicted_price = main(ticker)
    return predicted_price

def main():
    st.title("Real-Time Stock Price Prediction")

    # Create input fields for entering the stock symbol and the number of days to predict
    symbol = st.text_input("Enter stock symbol")
    days = st.slider("Enter number of days to predict", min_value=1, max_value=30, value=1)

    # Create a button to trigger the prediction
    if st.button("Predict"):
        predicted_price = predict_price(symbol, days)
        st.success(f"Predicted price for {symbol} in {days} days: {predicted_price:.2f}")


if __name__ == "__main__":
    
    st.set_page_config(
        page_title="STOCK PRICE PREDICTION",
        page_icon=":chart_with_upwards_trend:",
        layout="wide",
        initial_sidebar_state="collapsed",
        )
    
    main()
    
    
