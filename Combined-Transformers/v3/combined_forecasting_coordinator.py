"""
combined_forecasting_coordinator.py

This script coordinates the execution of Transformer, Vision Transformer, and Simple LSTM
time series forecasting models. It imports the necessary functions from the
existing implementation files and runs all models for comparison.

Usage:
python combined_forecasting_coordinator.py [--batch_size BATCH_SIZE] [--version_num VERSION_NUM] [--ticker TICKER] [--start_date START_DATE] [--end_date END_DATE]

Requirements:
- Python 3.7+
- Existing implementation files: 
  - transformer_ts_forecasting_core.py
  - transformer_ts_forecasting_main.py
  - vision_transformer_ts_forecasting_v32.py
  - simple_lstm_ts_forecasting.py
"""

import argparse
from datetime import datetime
import os
import csv
import matplotlib.pyplot as plt
import numpy as np

# Import functions from existing files
from transformer_ts_forecasting_core import get_stock_data, prepare_data as prepare_data_transformer, train_and_evaluate as train_and_evaluate_transformer, evaluate_model
from vision_transformer_ts_forecasting_v32 import prepare_data as prepare_data_vit, train_and_evaluate as train_and_evaluate_vit
from simple_lstm_ts_forecasting import prepare_data as prepare_data_lstm, train_and_evaluate as train_and_evaluate_lstm

def plot_results(y_true, y_pred_dict, title, filename, ticker):
    # Determine overall min and max for x and y axes
    min_x = 0
    max_x = len(y_true)
    min_y = min(np.min(y_true), min(np.min(y_pred) for y_pred in y_pred_dict.values()))
    max_y = max(np.max(y_true), max(np.max(y_pred) for y_pred in y_pred_dict.values()))

    for model_name, y_pred in y_pred_dict.items():
        plt.figure(figsize=(12, 6))
        plt.plot(y_true, label='Actual', linewidth=2)
        plt.plot(y_pred, label=f'{model_name} Predicted', linewidth=2, alpha=0.7)
        plt.title(f"{ticker} Stock Price Forecast ({model_name})", fontsize=16)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Stock Price', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xlim(min_x, max_x)
        plt.ylim(min_y, max_y)
        plt.tight_layout()
        
        if not os.path.exists('plots'):
            os.makedirs('plots')
        
        model_filename = f"{filename}_{model_name.lower().replace(' ', '_')}.png"
        plt.savefig(os.path.join('plots', model_filename))
        print(f"Plot saved as 'plots/{model_filename}'")
        
        # Display the plot on screen
        plt.show()
        
        plt.close()

def create_and_save_plots(y_true, y_pred_dict, ticker, version_num):
    filename = f"{version_num}_{ticker}_forecast"
    plot_results(y_true, y_pred_dict, f"{ticker} Stock Price Forecast", filename, ticker)

def save_combined_metrics(metrics_dict, filename):
    if not os.path.exists('metrics'):
        os.makedirs('metrics')
    
    file_path = os.path.join('metrics', filename)
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Metric'] + list(metrics_dict.keys()))
        for metric in metrics_dict[list(metrics_dict.keys())[0]].keys():
            writer.writerow([metric] + [metrics_dict[model][metric] for model in metrics_dict.keys()])
    
    print(f"Combined metrics saved to '{file_path}'")

def generate_combined_summary(metrics_dict, ticker, start_date, end_date, version_num):
    summary = f"""
Stock Price Forecast Summary for {ticker}
Version: {version_num}
Period: {start_date} to {end_date}

"""
    for model_name, metrics in metrics_dict.items():
        summary += f"{model_name} Model Metrics:\n"
        summary += f"MSE:  {metrics['MSE']:.4f}\n"
        summary += f"RMSE: {metrics['RMSE']:.4f}\n"
        summary += f"MAE:  {metrics['MAE']:.4f}\n"
        summary += f"R2:   {metrics['R2']:.4f}\n"
        summary += f"MAPE: {metrics['MAPE']:.4f}%\n\n"

    summary += "Comparative Summary:\n"
    best_r2_model = max(metrics_dict, key=lambda x: metrics_dict[x]['R2'])
    best_rmse_model = min(metrics_dict, key=lambda x: metrics_dict[x]['RMSE'])
    best_mape_model = min(metrics_dict, key=lambda x: metrics_dict[x]['MAPE'])

    summary += f"- The {best_r2_model} model explains the highest variance at {metrics_dict[best_r2_model]['R2']*100:.2f}%.\n"
    summary += f"- The {best_rmse_model} model has the lowest average prediction error at ${metrics_dict[best_rmse_model]['RMSE']:.2f} (RMSE).\n"
    summary += f"- The {best_mape_model} model's predictions are the most accurate with an average error of {metrics_dict[best_mape_model]['MAPE']:.2f}%.\n"
    summary += f"- Overall, the {best_r2_model} model performs best in terms of R2 score and prediction accuracy.\n\n"

    # Comparison between Transformer and Vision Transformer
    summary += "Transformer vs Vision Transformer Comparison:\n"
    t_metrics = metrics_dict['Transformer']
    vt_metrics = metrics_dict['Vision Transformer']

    r2_diff = t_metrics['R2'] - vt_metrics['R2']
    rmse_diff = t_metrics['RMSE'] - vt_metrics['RMSE']
    mape_diff = t_metrics['MAPE'] - vt_metrics['MAPE']

    summary += f"- R2 Score: The {'Transformer' if r2_diff > 0 else 'Vision Transformer'} model performs better by {abs(r2_diff)*100:.2f}%.\n"
    summary += f"- RMSE: The {'Vision Transformer' if rmse_diff > 0 else 'Transformer'} model has lower prediction error by ${abs(rmse_diff):.2f}.\n"
    summary += f"- MAPE: The {'Vision Transformer' if mape_diff > 0 else 'Transformer'} model is more accurate by {abs(mape_diff):.2f}%.\n"

    if r2_diff > 0 and rmse_diff < 0 and mape_diff < 0:
        summary += "The Transformer model shows better overall performance across all metrics.\n"
    elif r2_diff < 0 and rmse_diff > 0 and mape_diff > 0:
        summary += "The Vision Transformer model shows better overall performance across all metrics.\n"
    else:
        summary += "The performance comparison between Transformer and Vision Transformer is mixed, with each model showing strengths in different areas.\n"

    if abs(r2_diff) < 0.05 and abs(rmse_diff) < 1 and abs(mape_diff) < 1:
        summary += "The performance difference between the two models is relatively small, suggesting that both approaches are viable for this forecasting task.\n"
    
    summary += "\nNote: Please refer to the generated plots and metrics file for more details.\n"

    print(summary)
    
    if not os.path.exists('summaries'):
        os.makedirs('summaries')
    summary_filename = f"{version_num}_{ticker}_{start_date}_{end_date}_combined_summary.txt"
    with open(os.path.join('summaries', summary_filename), 'w') as f:
        f.write(summary)
    print(f"Combined summary saved as 'summaries/{summary_filename}'")

def run_combined_forecast(ticker, start_date, end_date, batch_size, version_num):
    print(f"Fetching stock data for {ticker}...")
    stock_data = get_stock_data(ticker, start_date, end_date)
    
    if stock_data is not None:
        print("Preparing data for Transformer model...")
        X_transformer, y_transformer, scaler_transformer = prepare_data_transformer(stock_data)
        
        print("Preparing data for Vision Transformer model...")
        X_vit, y_vit, scaler_vit = prepare_data_vit(stock_data, sequence_length=20)
        
        print("Preparing data for Simple LSTM model...")
        X_lstm, y_lstm, scaler_lstm = prepare_data_lstm(stock_data, sequence_length=20)
        
        print("Training and evaluating Transformer model...")
        y_true_transformer, y_pred_transformer = train_and_evaluate_transformer(X_transformer, y_transformer, scaler_transformer, batch_size=batch_size)
        
        print("Training and evaluating Vision Transformer model...")
        y_true_vit, y_pred_vit = train_and_evaluate_vit(X_vit, y_vit, scaler_vit, batch_size=batch_size)
        
        print("Training and evaluating Simple LSTM model...")
        y_true_lstm, y_pred_lstm = train_and_evaluate_lstm(X_lstm, y_lstm, scaler_lstm, batch_size=batch_size)
        
        metrics_transformer = evaluate_model(y_true_transformer, y_pred_transformer)
        metrics_vit = evaluate_model(y_true_vit, y_pred_vit)
        metrics_lstm = evaluate_model(y_true_lstm, y_pred_lstm)
        
        metrics_dict = {
            'Transformer': metrics_transformer,
            'Vision Transformer': metrics_vit,
            'Simple LSTM': metrics_lstm
        }
        
        y_pred_dict = {
            'Transformer': y_pred_transformer,
            'Vision Transformer': y_pred_vit,
            'Simple LSTM': y_pred_lstm
        }
        
        print("Creating and saving plots...")
        create_and_save_plots(y_true_transformer, y_pred_dict, ticker, version_num)
        
        print("Saving combined metrics...")
        metrics_filename = f"{version_num}_{ticker}_{start_date}_{end_date}_combined_metrics.csv"
        save_combined_metrics(metrics_dict, metrics_filename)
        
        print("Generating combined summary...")
        generate_combined_summary(metrics_dict, ticker, start_date, end_date, version_num)
        
        print("Combined forecast complete. Results, metrics, and summary saved.")
        
        return metrics_dict, y_true_transformer, y_pred_dict
    else:
        print("Failed to retrieve stock data. Please check your inputs and try again.")
        return None, None, None

def main(batch_size=32, version_num='v1', args=None):
    if args is None:
        # This branch won't be used when called from the scaffolding,
        # but it's good for backwards compatibility and direct script execution
        ticker = "AAPL"
        start_date = "2022-01-01"
        end_date = datetime.now().strftime('%Y-%m-%d')
    else:
        ticker = args.ticker
        start_date = args.start_date
        end_date = args.end_date
        # Use the batch_size from args, as it's explicitly passed in the scaffolding
        batch_size = args.batch_size

    return run_combined_forecast(ticker, start_date, end_date, batch_size, version_num)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run combined stock price forecasting with Transformer, Vision Transformer, and Simple LSTM")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--version_num", type=str, default='v1', help="Version number for the output file")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker symbol")
    parser.add_argument("--start_date", type=str, default="2022-01-01", help="Start date for historical data")
    parser.add_argument("--end_date", type=str, default=datetime.now().strftime('%Y-%m-%d'), help="End date for historical data")
    args = parser.parse_args()
    
    main(batch_size=args.batch_size, version_num=args.version_num, args=args)
