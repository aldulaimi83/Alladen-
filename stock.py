import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import datetime
import talib as ta
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Download data from yfinance
def fetch_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        print(f"No data found for ticker {ticker}")
        return None

    if len(data) < 10:
        print(f"Insufficient data for ticker {ticker}. Minimum 10 data points required\n")
        return None
    return data
# Step 2: Feature Engineering (using previous day's data)


# Enhanced feature creation with more technical indicators
def create_features(df):
    if len(df) < 5:
        print("Not enough data for rolling mean or standard deviation.")
        return df

    # Previous day prices
    df['Prev Close'] = df['Close'].shift(1)
    df['Prev High'] = df['High'].shift(1)
    df['Prev Low'] = df['Low'].shift(1)
    df['Prev Open'] = df['Open'].shift(1)
    
    # Moving averages
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # Exponential moving average
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    
    # Ensure 'Close' column is a 1-dimensional array
    close_prices = df['Close'].values.astype(float).reshape(-1)

    
    # RSI (Relative Strength Index)
    df['RSI'] = ta.RSI(close_prices, timeperiod=14)

    
    # MACD (Moving Average Convergence Divergence)
    df['MACD'], df['MACD_signal'], _ = ta.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
    
    # Bollinger Bands
    df['BB_upper'], df['BB_middle'], df['BB_lower'] = ta.BBANDS(close_prices, timeperiod=20)

    # Rolling statistics
    df['Rolling Mean'] = df['Close'].rolling(window=5).mean()
    df['Rolling Std'] = df['Close'].rolling(window=5).std()

    # Drop rows with missing values after features creation
    df = df.dropna()

    return df


# Step 3: Prepare Data (Features and Labels)
def prepare_data(df):
    # Features: Using various lagged features
    features = df[['Prev Close', 'Prev High', 'Prev Low', 'Prev Open', 'Rolling Mean', 'Rolling Std']]
    
    # Labels: The next day's closing price
    labels = df['Close']
    
    # Split the data into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, shuffle=False)
    
    # Check if data is not empty
    if X_train.empty or X_test.empty:
        raise ValueError("Training or testing data is empty!")
    
    # Feature Scaling (Standardization)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler



def train_xgboost_model(X_train, y_train):
    model = xgb.XGBRegressor(objective="reg:squarederror", colsample_bytree=0.3, learning_rate=0.1,
                             max_depth=5, alpha=10, n_estimators=100)
    model.fit(X_train, y_train)
    return model

def tune_random_forest(X_train, y_train):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    print("Best parameters found: ", grid_search.best_params_)
    return grid_search.best_estimator_

# Step 4: Train the Model
def train_model(X_train, y_train):
    model = tune_random_forest(X_train, y_train)
    return model

# Step 5: Evaluate the Model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    y_test_reshaped = y_test.values.astype(float).reshape(-1) #Need to become 1D array for evaluation
    mse = mean_squared_error(y_test_reshaped, y_pred)
    mae = mean_absolute_error(y_test_reshaped, y_pred)
    r2 = r2_score(y_test_reshaped, y_pred)
    
    print(f'MSE: {mse}')
    print(f'MAE: {mae}')
    print(f'R² Score: {r2}')
    
    # Plotting the actual vs predicted values
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test, label='Actual', color='blue')
    plt.plot(y_test.index, y_pred, label='Predicted', color='red')
    plt.title(f"{ticker} Price Prediction: Actual vs Predicted")
    plt.xlabel('Date')
    plt.ylabel('Price')
    ax = plt.gca()
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=10))
    plt.legend()
    plt.show()




def time_series_cross_val(X, y):
    tscv = TimeSeriesSplit(n_splits=5)
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        model = train_model(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        print(f'MSE for fold: {mse}')

# Step 6: Predict Future Price (if user inputs a ticker)
def predict_future(model, scaler, ticker):
    # Fetch the most recent data
    latest_data = yf.download(ticker, period="5d")
    
    if latest_data.empty:
        print(f"No data found for ticker {ticker}")
        return
    
    # Prepare features for prediction
    latest_data['Prev Close'] = latest_data['Close'].shift(1)
    latest_data['Prev High'] = latest_data['High'].shift(1)
    latest_data['Prev Low'] = latest_data['Low'].shift(1)
    latest_data['Prev Open'] = latest_data['Open'].shift(1)
    latest_data['Rolling Mean'] = latest_data['Close'].rolling(window=5).mean()
    latest_data['Rolling Std'] = latest_data['Close'].rolling(window=5).std()
    latest_data = latest_data.dropna()

    # If there is no data after cleaning, return early
    if latest_data.empty:
        print("No valid data available for prediction.")
        return
    
    # Extract features
    features = latest_data[['Prev Close', 'Prev High', 'Prev Low', 'Prev Open', 'Rolling Mean', 'Rolling Std']].tail(1)
    
    # Scale the features
    features_scaled = scaler.transform(features)
    
    # Predict the next day's closing price
    predicted_price = model.predict(features_scaled)
    print(f"Predicted next day closing price for {ticker}: {predicted_price[0]:.2f}")

# Main Function to Run the Model
def main():
    # Input ticker from user
    #ticker = input("Enter the ticker symbol: ").upper()
    global ticker
    ticker = 'RGTI'
    # Define the period for the data (e.g., last 1 year)
    end_date = datetime.datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.datetime.today() - datetime.timedelta(days=1095)).strftime('%Y-%m-%d')
    
    # Step 1: Fetch historical data
    df = fetch_data(ticker, start_date, end_date)
    
    # Step 2: Create features
    if df.empty:
        print(f"No data available for ticker {ticker}.")
        return

    df = create_features(df)
    
    # Step 3: Prepare Data (Train-Test Split & Scaling)
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)
    y_train_reshaped = y_train.values.astype(float).reshape(-1)
    # Step 4: Train the Random Forest model
    model = train_model(X_train, y_train_reshaped)
    
    # Step 5: Evaluate the model
    evaluate_model(model, X_test, y_test)

    X = df.drop(columns=['Close'])
    y = df['Close']
    time_series_cross_val(X.values, y.values.astype(float).reshape(-1))
    # Step 6: Predict the future price for the input ticker
    predict_future(model, scaler, ticker)

if __name__ == "__main__":
    main()
