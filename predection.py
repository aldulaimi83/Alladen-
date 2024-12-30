from flask import Flask, jsonify, request
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error

app = Flask(__name__)

# Prediction function
def predict_prices(ticker, days=5):
    try:
        # Fetch stock data
        stock = yf.Ticker(ticker)
        data = stock.history(period="1y")
        if data.empty:
            return {"error": f"No data found for ticker: {ticker}"}
        
        data['Date'] = pd.to_datetime(data.index)

        # Add technical indicators
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        data['BB_middle'] = data['Close'].rolling(window=20).mean()
        data['BB_std'] = data['Close'].rolling(window=20).std()
        data['BB_upper'] = data['BB_middle'] + (2 * data['BB_std'])
        data['BB_lower'] = data['BB_middle'] - (2 * data['BB_std'])
        data.dropna(inplace=True)

        # Prepare data
        data['Date_ordinal'] = data['Date'].map(pd.Timestamp.toordinal)
        features = data[['Date_ordinal', 'Volume', 'SMA_20', 'SMA_50', 'RSI', 'BB_upper', 'BB_middle', 'BB_lower']]
        target = data['Close']

        # Train model
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Predict future prices
        last_row = features.iloc[-1].values
        future_predictions = []
        for _ in range(days):
            prediction = model.predict(np.array([last_row]).reshape(1, -1))[0]
            future_predictions.append(prediction)
            last_row[0] += 1  # Increment date ordinal
            last_row[2:4] = (last_row[2:4] + prediction) / 2  # Adjust SMA_20 and SMA_50
            last_row[4] = min(max(0, last_row[4] + np.random.uniform(-5, 5)), 100)  # Adjust RSI
            last_row[5:] = prediction  # Adjust Bollinger bands

        return {"predictions": future_predictions}

    except Exception as e:
        return {"error": str(e)}

@app.route("/")
def home():
    return jsonify({"message": "Welcome to the Stock Prediction API! Use /predict?ticker=<TICKER> to get predictions."})

@app.route("/predict", methods=['GET'])
def predict():
    ticker = request.args.get("ticker")
    days = request.args.get("days", default=5, type=int)
    if not ticker:
        return jsonify({"error": "Please provide a stock ticker symbol using the 'ticker' query parameter."})

    result = predict_prices(ticker, days)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
