from flask import Flask, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import load_model  # nopep8
# Get data for multiple stocks


def get_data_for_stocks(symbols, start_date, end_date):
    stock_data = {}
    for symbol in symbols:
        df = yf.download(symbol, start_date, end_date)
        df = df.dropna()
        df = df.drop_duplicates()
        df = df[['Close']]
        stock_data[symbol] = df
    return stock_data

# Forecasting function


def forecasting(df, days_to_forecast, model, seq_length=100):
    data = df[['Close']]

    # Normalize the data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data.values.reshape(-1, 1))

    X, Y = [], []
    for i in range(len(data) - seq_length):
        seq = data_scaled[i: i + seq_length]
        label = data_scaled[i + seq_length]
        X.append(seq)
        Y.append(label)

    X = np.array(X)
    Y = np.array(Y)

    # Forecasting the next period
    def forecast_next_period(model, data, seq_length, n_forecasts, scaler):
        forecasts = []
        last_sequence = data[-seq_length:]

        for _ in range(n_forecasts):
            last_sequence_scaled = scaler.transform(
                last_sequence.reshape(-1, 1)).reshape(1, seq_length, 1)
            forecast_scaled = model.predict(last_sequence_scaled)
            forecast = scaler.inverse_transform(forecast_scaled)[0, 0]
            forecasts.append(forecast)
            last_sequence = np.append(last_sequence[1:], forecast)

        return forecasts

    # Forecast the next period
    forecasts = forecast_next_period(
        model, data.values, seq_length, days_to_forecast, scaler)
    return forecasts[-1]  # Return the last forecasted value

# Recommendation function


def get_recommendation(stock_data, days_to_forecast, ROI_target, amount, model_path='CNN_LSTM_120_days.h5'):
    model = load_model(model_path)
    recommendations = []

    for symbol, df in stock_data.items():
        # Perform forecasting
        last_forecast = forecasting(df, days_to_forecast, model)
        last_observation = df['Close'].values[-1]

        # Calculate ROI
        ROI = (last_forecast - last_observation) / last_observation * 100

        # Check if stock meets criteria
        if ROI >= ROI_target and last_forecast <= amount:
            recommendations.append(
                {'symbol': symbol, 'forecast': last_forecast, 'ROI': ROI})

    return recommendations


# Example usage
symbols = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'
    # 'V', 'MA', 'PYPL', 'DIS', 'NKE',
    # 'HD', 'KO', 'PEP', 'MRNA', 'PFE',
    # 'CRM', 'ADBE', 'INTC', 'AMD', 'CSCO', 'QCOM', 'ORCL', 'IBM', 'MU', 'TXN',
    # 'WMT', 'COST', 'TGT', 'MCD', 'SBUX', 'XOM', 'CVX', 'BP', 'WFC', 'JPM',
    # 'GS', 'MS', 'BAC', 'SPGI', 'BLK', 'CAT', 'GE', 'MMM', 'DE', 'UPS'
]

start_date = '2024-01-01'
end_date = '2024-08-21'
# Input for the number of days to forecast
days_to_forecast_options = [30, 60, 90, 120]
# while True:
#     try:
#         days_to_forecast = int(input(
#             f"Enter the number of days to forecast (choose from {days_to_forecast_options}): "))
#         if days_to_forecast in days_to_forecast_options:
#             break
#         else:
#             print(f"Please choose from {days_to_forecast_options}.")
#     except ValueError:
#         print("Invalid input. Please enter an integer.")

# # Input for ROI target
# while True:
#     try:
#         ROI_target = int(
#             input("Enter the ROI target percentage (e.g., 1 for 1%): "))
#         break
#     except ValueError:
#         print("Invalid input. Please enter an integer.")

# # Input for amount
# while True:
#     try:
#         amount = int(
#             input("Enter the investment amount per stock (e.g., 1000 for $1000): "))
#         break
#     except ValueError:
#         print("Invalid input. Please enter an integer.")
# stock_data = get_data_for_stocks(symbols, start_date, end_date)
# recommendations = get_recommendation(
#     stock_data, days_to_forecast, ROI_target, amount)

# # Print recommendations
# for rec in recommendations:
#     # print("hi")
#     print(f"Stock: {rec['symbol']}, Forecasted Price: {
#           rec['forecast']:.2f}, ROI: {rec['ROI']:.2f}%")


app = Flask(__name__)


@app.route("/getRecommendation", methods=['POST', 'GET'])
def getUserInput():
    ROI_target = request.json["roi"]
    amount = request.json["maxInvestment"]
    days_to_forecast = request.json["period"]
    time.sleep(1)
    stock_data = get_data_for_stocks(symbols, start_date, end_date)
    recommendations = get_recommendation(
        stock_data, days_to_forecast, ROI_target, amount)
    print(recommendations)
    print(ROI_target)
    print(amount)
    print(days_to_forecast)
    output = []
    for rec in recommendations:
        output.append(f"Stock: {rec['symbol']}\nForecasted Price: {rec['forecast']:.2f}\nROI: {rec['ROI']:.2f}%\n")
    if len(output) == 0:
        output.append("No recommendations available")
        # output.append("Test")
    return output


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
