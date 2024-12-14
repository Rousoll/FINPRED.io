from flask import Flask, render_template, request, jsonify
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# Initialize Flask app
app = Flask(__name__)

# Function to fetch and predict stock price
def get_stock_data(ticker):
    data = yf.download(ticker, start="2020-01-01", end="2024-12-11")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    def create_dataset(data, time_step=60):
        X, y = [], []
        for i in range(time_step, len(data)):
            X.append(data[i-time_step:i, 0])  # Features
            y.append(data[i, 0])  # Target
        return np.array(X), np.array(y)

    X, y = create_dataset(scaled_data, time_step=60)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)

    return predictions[-1][0]

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# API route to handle stock prediction
@app.route('/predict', methods=['POST'])
def predict():
    ticker = request.form['ticker']
    predicted_price = get_stock_data(ticker)
    return jsonify({'predicted_price': predicted_price})

if __name__ == '__main__':
    app.run(debug=True)
