"""
transformer_ts_forecasting_core.py

This file contains the core functionality for the Transformer-based time series forecasting model.
It includes data preparation, model definition, training, evaluation, and utility functions.

This file should be imported by the main script that handles command-line arguments and executes the forecast.
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

class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, nhead, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.transformer = nn.Transformer(d_model=hidden_dim, nhead=nhead, num_encoder_layers=num_layers, dropout=dropout)
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, output_dim)

    def forward(self, src):
        src = self.encoder(src)
        src = src.permute(1, 0, 2)  # Change shape to (seq_len, batch, features)
        output = self.transformer(src, src)
        output = output.permute(1, 0, 2)  # Change shape back to (batch, seq_len, features)
        output = self.decoder(output[:, -1, :])  # Only use the last output for prediction
        return output

def train_and_evaluate(X, y, scaler, batch_size=32, epochs=100, test_size=0.2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.FloatTensor(y_train).to(device)
    X_test = torch.FloatTensor(X_test).to(device)
    y_test = torch.FloatTensor(y_test).to(device)
    
    # Create data loaders
    train_data = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    # Initialize the model
    model = TransformerModel(input_dim=X.shape[2], hidden_dim=64, output_dim=1, num_layers=3, nhead=4).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
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
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test).cpu().numpy().flatten()
        y_test = y_test.cpu().numpy()
    
    # Inverse transform the predictions and true values
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
    
    # Close the plot to free up memory
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
Stock Price Forecast Summary for {ticker}
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
    
    # Save summary to file
    if not os.path.exists('summaries'):
        os.makedirs('summaries')
    summary_filename = f"{version_num}_{ticker}_{start_date}_{end_date}_summary.txt"
    with open(os.path.join('summaries', summary_filename), 'w') as f:
        f.write(summary)
    print(f"Summary saved as 'summaries/{summary_filename}'")
