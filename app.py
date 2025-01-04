from flask import Flask, request, render_template
import datetime
import os
from stock import fetch_data, create_features, prepare_data, train_model, evaluate_model, predict_future

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    ticker = request.form['ticker']
    # Define the period for the data (e.g., last 1 year)
    end_date = datetime.datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.datetime.today() - datetime.timedelta(days=1095)).strftime('%Y-%m-%d')
    
    # Step 1: Fetch historical data
    df = fetch_data(ticker, start_date, end_date)
    
    # Step 2: Create features
    if df is None or df.empty:
        return f"No data available for ticker {ticker}."
    
    df = create_features(df)
    
    # Step 3: Prepare Data (Train-Test Split & Scaling)
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)
    
    # Step 4: Train the Random Forest model
    model = train_model(X_train, y_train)
    
    # Step 5: Evaluate the model and generate the plot
    plot_path, mse, mae, r2 = evaluate_model(model, X_test, y_test, ticker)
    
    # Step 6: Predict the future price for the input ticker
    predict_future(model, scaler, ticker)
    
    return render_template('result.html', ticker=ticker, plot_url=plot_path, mse=mse, mae=mae, r2=r2)

if __name__ == '__main__':
    app.run(debug=True)