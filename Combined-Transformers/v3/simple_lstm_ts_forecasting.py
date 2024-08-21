"""
simple_lstm_ts_forecasting.py

This script implements a simple LSTM model for stock price forecasting.
It includes data preparation, model definition, training, evaluation, and utility functions.

Usage:
Can be run directly or imported and called via main():
python simple_lstm_ts_forecasting.py [--batch_size BATCH_SIZE] [--version_num VERSION_NUM] [--ticker TICKER] [--start_date START_DATE] [--end_date END_DATE]

Requirements:
- Python 3.7+
- Libraries: yfinance, numpy, pandas, matplotlib, scikit-learn, torch
"""

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging
import os
import argparse
import csv
from datetime import datetime

logging.basicConfig(level=logging.INFO)

def get_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            raise ValueError(f"No data retrieved for {ticker}")
        return data
    except Exception as e:
        logging.error(f"Error retrieving data for {ticker}: {str(e)}")
        return None

def prepare_data(data: pd.DataFrame, sequence_length: int = 20):
    scaler = RobustScaler()
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    scaled_data = scaler.fit_transform(data[features])
    
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:i+sequence_length])
        y.append(scaled_data[i+sequence_length, features.index('Close')])
    
    X, y = np.array(X), np.array(y)
    return X, y, scaler

class SimpleLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(SimpleLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def train_and_evaluate(X, y, scaler, batch_size=32, epochs=100, test_size=0.2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.FloatTensor(y_train).to(device)
    X_test = torch.FloatTensor(X_test).to(device)
    y_test = torch.FloatTensor(y_test).to(device)
    
    train_data = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    model = SimpleLSTM(input_dim=X.shape[2], hidden_dim=64, num_layers=2, output_dim=1).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_X).squeeze()
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}')
    
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test).cpu().numpy().flatten()
        y_test = y_test.cpu().numpy()
    
    test_predictions = scaler.inverse_transform(np.column_stack([test_predictions, np.zeros((len(test_predictions), 4))]))[: , 0]
    y_test = scaler.inverse_transform(np.column_stack([y_test, np.zeros((len(y_test), 4))]))[: , 0]
    
    return y_test, test_predictions

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape
    }

def plot_results(y_true, y_pred, title, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Actual', linewidth=2)
    plt.plot(y_pred, label='Predicted', linewidth=2, alpha=0.7)
    plt.title(title, fontsize=16)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Stock Price', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    plt.savefig(os.path.join('plots', filename))
    print(f"Plot saved as 'plots/{filename}'")
    plt.close()

def save_metrics(metrics, filename):
    if not os.path.exists('metrics'):
        os.makedirs('metrics')
    
    file_path = os.path.join('metrics', filename)
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Metric', 'Value'])
        for key, value in metrics.items():
            writer.writerow([key, value])
    
    print(f"Metrics saved to '{file_path}'")

def print_metrics(metrics):
    print("Evaluation Metrics:")
    print("-" * 30)
    print(f"{'Metric':<10} {'Value':<10}")
    print("-" * 30)
    for metric, value in metrics.items():
        print(f"{metric:<10} {value:<10.4f}")
    print("-" * 30)

def generate_summary(metrics, ticker, start_date, end_date, version_num):
    summary = f"""
Stock Price Forecast Summary for {ticker} (Simple LSTM)
Version: {version_num}
Period: {start_date} to {end_date}

Evaluation Metrics:
MSE:  {metrics['MSE']:.4f}
RMSE: {metrics['RMSE']:.4f}
MAE:  {metrics['MAE']:.4f}
R2:   {metrics['R2']:.4f}
MAPE: {metrics['MAPE']:.4f}%

Key Insights:
- Model explains {metrics['R2']*100:.2f}% of the variance in stock price.
- Average prediction error: ${metrics['RMSE']:.2f} (RMSE)
- Predictions are off by an average of {metrics['MAPE']:.2f}% (MAPE)

Note: Please refer to the generated plot and metrics file for more details.
"""
    print(summary)
    
    if not os.path.exists('summaries'):
        os.makedirs('summaries')
    summary_filename = f"{version_num}_{ticker}_{start_date}_{end_date}_lstm_summary.txt"
    with open(os.path.join('summaries', summary_filename), 'w') as f:
        f.write(summary)
    print(f"Summary saved as 'summaries/{summary_filename}'")

def run_forecast(ticker="AAPL", start_date='2022-01-01', end_date='2024-06-10', batch_size=32, version_num='v1'):
    print(f"Fetching stock data for {ticker}...")
    stock_data = get_stock_data(ticker, start_date, end_date)
    
    if stock_data is not None:
        print("Preparing data...")
        X, y, scaler = prepare_data(stock_data, sequence_length=20)
        
        print("Training model and making predictions...")
        y_true, y_pred = train_and_evaluate(X, y, scaler, batch_size=batch_size)
        
        metrics = evaluate_model(y_true, y_pred)
        print_metrics(metrics)
        
        print("Plotting results...")
        plot_filename = f"{version_num}_{ticker}_{start_date}_{end_date}_lstm.png"
        plot_results(y_true, y_pred, f"{ticker} Stock Price Forecast (Simple LSTM)", plot_filename)
        
        metrics_filename = f"{version_num}_{ticker}_{start_date}_{end_date}_lstm_metrics.csv"
        save_metrics(metrics, metrics_filename)
        
        print("Generating summary...")
        generate_summary(metrics, ticker, start_date, end_date, version_num)
        
        print(f"LSTM forecast complete. Results, metrics, and summary saved.")
        
        return metrics, y_true, y_pred
    else:
        print("Failed to retrieve stock data. Please check your inputs and try again.")
        return None, None, None

def main(batch_size=32, version_num='v1', args=None):
    if args is None:
        ticker = "AAPL"
        start_date = "2022-01-01"
        end_date = datetime.now().strftime('%Y-%m-%d')
    else:
        ticker = args.ticker
        start_date = args.start_date
        end_date = args.end_date
        batch_size = args.batch_size

    return run_forecast(ticker=ticker, start_date=start_date, end_date=end_date, 
                        batch_size=batch_size, version_num=version_num)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run stock price forecasting with Simple LSTM")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--version_num", type=str, default='v1', help="Version number for the output file")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker symbol")
    parser.add_argument("--start_date", type=str, default="2022-01-01", help="Start date for historical data")
    parser.add_argument("--end_date", type=str, default=datetime.now().strftime('%Y-%m-%d'), help="End date for historical data")
    args = parser.parse_args()
    
    main(batch_size=args.batch_size, version_num=args.version_num, args=args)
