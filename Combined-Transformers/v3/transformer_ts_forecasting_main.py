"""
transformer_ts_forecasting_main.py

This file contains the main execution logic for the Transformer-based time series forecasting model.
It imports functions from transformer_ts_forecasting_core.py and handles command-line arguments.

Usage:
python transformer_ts_forecasting_main.py [--batch_size BATCH_SIZE] [--version_num VERSION_NUM] [--ticker TICKER] [--start_date START_DATE] [--end_date END_DATE]

Requirements:
- Python 3.7+
- transformer_ts_forecasting_core.py file in the same directory
"""

import argparse
from datetime import datetime
from transformer_ts_forecasting_core import (
    get_stock_data,
    prepare_data,
    train_and_evaluate,
    evaluate_model,
    plot_results,
    save_metrics,
    print_metrics,
    generate_summary
)

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
    parser = argparse.ArgumentParser(description="Run improved stock price forecasting with Transformer")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--version_num", type=str, default='v1', help="Version number for the output file")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker symbol")
    parser.add_argument("--start_date", type=str, default="2022-01-01", help="Start date for historical data")
    parser.add_argument("--end_date", type=str, default=datetime.now().strftime('%Y-%m-%d'), help="End date for historical data")
    args = parser.parse_args()
    
    main(batch_size=args.batch_size, version_num=args.version_num, args=args)
