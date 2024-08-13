"""
Improved Vision Transformer for Time Series Forecasting

This script implements an enhanced Vision Transformer model for stock price forecasting
with strategies to prevent overfitting and improve generalization.

Features:
- Fetches stock data from Yahoo Finance
- Implements k-fold cross-validation
- Uses data augmentation techniques
- Adds L2 regularization
- Implements early stopping based on validation performance
- Evaluates the model performance using multiple metrics
- Plots and saves the results
- Generates and saves summary of results
- Allows flexible parameter passing from external calls

Usage:
Can be run directly or imported and called via main():
python vision_transformer_ts_forecasting.py [--batch_size BATCH_SIZE] [--version_num VERSION_NUM] [--ticker TICKER] [--start_date START_DATE] [--end_date END_DATE]

Requirements:
- Python 3.7+
- Libraries: yfinance, numpy, pandas, matplotlib, scikit-learn, torch, vit-pytorch

Caveats:
- The script is designed for educational purposes and may not be suitable for production use.
 - This script has been tested only from a Googl colab scaffolding and may require modifications to run in other environments.

"""

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from vit_pytorch import ViT
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
    
    sqrt_seq_len = int(np.sqrt(sequence_length))
    X = X.transpose(0, 2, 1)
    X = X.reshape(X.shape[0], len(features), sqrt_seq_len, sqrt_seq_len)
    
    return X, y, scaler

def augment_data(X, y, num_augmentations=1):
    augmented_X, augmented_y = [], []
    for _ in range(num_augmentations):
        noise = np.random.normal(0, 0.01, X.shape)
        augmented_X.append(X + noise)
        augmented_y.append(y)
    return np.concatenate([X] + augmented_X), np.concatenate([y] + augmented_y)

class VisionTransformerRegressor(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=5, dropout=0.1):
        super().__init__()
        self.vit = ViT(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            channels=channels,
            dropout=dropout
        )
        self.regressor = nn.Sequential(
            nn.Linear(num_classes, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.vit(x)
        return self.regressor(x)

def train_and_evaluate(X, y, scaler, batch_size=32, epochs=300, n_splits=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    all_predictions = []
    all_true_values = []
    
    for fold, (train_index, val_index) in enumerate(kf.split(X)):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        X_train, y_train = augment_data(X_train, y_train)
        
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        model = VisionTransformerRegressor(
            image_size=X.shape[2],
            patch_size=1,
            num_classes=128,
            dim=256,
            depth=6,
            heads=8,
            mlp_dim=512,
            channels=X.shape[1],
            dropout=0.1
        ).to(device)
        
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, verbose=True)
        
        best_val_loss = float('inf')
        patience = 20
        counter = 0
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            
            model.eval()
            with torch.no_grad():
                val_predictions = model(torch.FloatTensor(X_val).to(device)).cpu().numpy().flatten()
                val_loss = criterion(torch.FloatTensor(val_predictions), torch.FloatTensor(y_val)).item()
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                torch.save(model.state_dict(), f'best_model_fold_{fold}.pth')
            else:
                counter += 1
            
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
            
            if (epoch + 1) % 10 == 0:
                print(f"Fold {fold+1}, Epoch [{epoch+1}/{epochs}], Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        model.load_state_dict(torch.load(f'best_model_fold_{fold}.pth'))
        model.eval()
        with torch.no_grad():
            fold_predictions = model(torch.FloatTensor(X_val).to(device)).cpu().numpy().flatten()
        
        all_predictions.extend(fold_predictions)
        all_true_values.extend(y_val)
    
    all_predictions = np.array(all_predictions)
    all_true_values = np.array(all_true_values)
    
    all_predictions_inv = scaler.inverse_transform(np.column_stack([all_predictions, np.zeros((len(all_predictions), 4))]))[: , 0]
    all_true_values_inv = scaler.inverse_transform(np.column_stack([all_true_values, np.zeros((len(all_true_values), 4))]))[: , 0]
    
    return all_true_values_inv, all_predictions_inv

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
    plt.show()

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

def run_forecast(ticker="AAPL", start_date='2022-01-01', end_date='2024-06-10', batch_size=32, version_num='v1'):
    print(f"Fetching stock data for {ticker}...")
    stock_data = get_stock_data(ticker, start_date, end_date)
    
    if stock_data is not None:
        print("Preparing data...")
        X, y, scaler = prepare_data(stock_data, sequence_length=16)
        
        print("Training model and making predictions...")
        y_true, y_pred = train_and_evaluate(X, y, scaler, batch_size=batch_size)
        
        metrics = evaluate_model(y_true, y_pred)
        print_metrics(metrics)
        
        print("Plotting results...")
        plot_filename = f"{version_num}_{ticker}_{start_date}_{end_date}.png"
        plot_results(y_true, y_pred, f"{ticker} Stock Price Forecast", plot_filename)
        
        metrics_filename = f"{version_num}_{ticker}_{start_date}_{end_date}_metrics.csv"
        save_metrics(metrics, metrics_filename)
        
        print("Generating summary...")
        generate_summary(metrics, ticker, start_date, end_date, version_num)
        
        print(f"Forecast complete. Results, metrics, and summary saved.")
        
        return metrics, y_true, y_pred
    else:
        print("Failed to retrieve stock data. Please check your inputs and try again.")
        return None, None, None

def main(batch_size=32, version_num='v1', args=None):
    if args is None:
        # Use default values if args is not provided
        ticker = "AAPL"
        start_date = "2022-01-01"
        end_date = datetime.now().strftime('%Y-%m-%d')
    else:
        # Use values from args if provided
        ticker = args.ticker
        start_date = args.start_date
        end_date = args.end_date

    return run_forecast(ticker=ticker, start_date=start_date, end_date=end_date, 
                        batch_size=batch_size, version_num=version_num)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run improved stock price forecasting with Vision Transformer")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--version_num", type=str, default='v1', help="Version number for the output file")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker symbol")
    parser.add_argument("--start_date", type=str, default="2022-01-01", help="Start date for historical data")
    parser.add_argument("--end_date", type=str, default=datetime.now().strftime('%Y-%m-%d'), help="End date for historical data")
    args = parser.parse_args()
    
    main(batch_size=args.batch_size, version_num=args.version_num, args=args)