import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_percentage_error
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Email Configuration
EMAIL_ADDRESS = "aldulaimi_ahmed83@yahoo.com"  # Replace with your email
EMAIL_PASSWORD = "SSDD12345ssdd"          # Replace with your email password
TO_EMAIL = "aldulaimi_ahmed@yahoo.com"  # Replace with recipient's email

prediction_days = 5 #Globalizing for the chart
# Send Email Notification
def send_email(subject, body):
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = TO_EMAIL
        msg['Subject'] = subject
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Connect to the server and send the email
        with smtplib.SMTP('smtp.gmail.com', 587) as server:  # Adjust SMTP settings for your email provider
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
        
        print(f"Email sent to {TO_EMAIL}.")
    except Exception as e:
        print(f"Failed to send email: {e}")

# Fetch SOFI stock data
def get_stock_data(ticker, period="1y"):
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)
    data['Date'] = data.index
    data['Date'] = pd.to_datetime(data['Date'])
    return data

# Add technical indicators manually
def add_technical_indicators(data):
    # Simple Moving Averages (SMA)
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    
    # Relative Strength Index (RSI)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    data['BB_middle'] = data['Close'].rolling(window=20).mean()
    data['BB_std'] = data['Close'].rolling(window=20).std()
    data['BB_upper'] = data['BB_middle'] + (2 * data['BB_std'])
    data['BB_lower'] = data['BB_middle'] - (2 * data['BB_std'])
    
    # Drop rows with NaN values created during rolling calculations
    return data.dropna()

# Prepare data for training
def prepare_data(data):
    data['Date_ordinal'] = data['Date'].map(pd.Timestamp.toordinal)
    features = data[['Date_ordinal', 'Volume', 'SMA_20', 'SMA_50', 'RSI', 'BB_upper', 'BB_middle', 'BB_lower']]
    target = data['Close']
    return features, target

# Train the model
def train_model(features, target):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    mape = mean_absolute_percentage_error(y_test, predictions)
    print(f"Model trained with R²: {r2:.4f}, MAPE: {mape:.4f}")
    return model

# Predict future prices
def predict_future(model, last_row, days=prediction_days):
    future_predictions = []
    for _ in range(days):
        features = np.array([last_row]).reshape(1, -1)
        prediction = model.predict(features)[0]
        future_predictions.append(prediction)
        # Simulate the next day with new values (adjust simulation as needed)
        last_row[0] += 1  # Increment date ordinal
        last_row[2] = (last_row[2] + prediction) / 2  # Adjust SMA_20 simplistically
        last_row[3] = (last_row[3] + prediction) / 2  # Adjust SMA_50 simplistically
        last_row[4] = min(max(0, last_row[4] + np.random.uniform(-5, 5)), 100)  # Adjust RSI within bounds
        last_row[5:] = prediction  # Adjust Bollinger bands to follow the predicted price
    return future_predictions

#Chart plotting for prediction
def chart(future_predictions):
    t_days = []
    for i in range(prediction_days):
        t_days.append(i+1)
    plt.figure(figsize=(10, 6))
    plt.plot(t_days, future_predictions, marker='o')

    plt.xlabel("Days")
    plt.ylabel("Share Prediciton Value")
    plt.title(f"Prediction of {ticker} over {prediction_days} days")
    plt.tight_layout()
    plt.show()
# Main function
def main():
    global ticker
    ticker = "SOFI"
    print(f"Fetching data for {ticker}...")

    while True:
        # Fetch and process data
        data = get_stock_data(ticker)
        print("Adding technical indicators...")
        data = add_technical_indicators(data)

        print("Preparing data...")
        features, target = prepare_data(data)

        print("Training model...")
        model = train_model(features, target)

        print("Predicting future prices...")
        last_row = features.iloc[-1].values  # Use the last row of features as a starting point
        future_predictions = predict_future(model, last_row, days=prediction_days)

        #Adding chart visibility with matplotlib
        chart(future_predictions)
        print(f"Predicted prices for the next 5 days: {future_predictions}")

        # Send email with predictions
        subject = "SOFI Stock Prediction Update"
        body = f"Predicted prices for the next 5 days: {future_predictions}"
        send_email(subject, body)

        print("Waiting 5 minutes before fetching new data...\n")
        time.sleep(300)  # Wait for 5 minutes (300 seconds)

if __name__ == "__main__":
    main()
