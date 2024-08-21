# Time Series Forecasting Project Documentation

## Table of Contents
1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Models](#models)
   - [Transformer](#transformer)
   - [Vision Transformer](#vision-transformer)
   - [Simple LSTM](#simple-lstm)
4. [Key Functions](#key-functions)
5. [Control Flow](#control-flow)
6. [Possible Roadmap](#possible-roadmap)

## Introduction

This project implements and compares three different models for time series forecasting of stock prices:
1. Transformer
2. Vision Transformer (ViT)
3. Simple LSTM

The project uses historical stock data from Yahoo Finance and evaluates the models using various metrics such as MSE, RMSE, MAE, R2, and MAPE.

## Project Structure

The project consists of five main Python files:

1. `combined_forecasting_coordinator.py`: Coordinates the execution of all models.
2. `transformer_ts_forecasting_core.py`: Core functionality for the Transformer model.
3. `transformer_ts_forecasting_main.py`: Main execution logic for the Transformer model.
4. `vision_transformer_ts_forecasting_v32.py`: Implementation of the Vision Transformer model.
5. `simple_lstm_ts_forecasting.py`: Implementation of the Simple LSTM model.

## Models

### Transformer

The Transformer model is implemented in `transformer_ts_forecasting_core.py`. It uses a custom `TransformerModel` class that combines a transformer encoder with linear layers for regression.

Key components:
- `TransformerModel`: Custom PyTorch module for time series forecasting.
- Uses self-attention mechanism to capture temporal dependencies.
- Configurable number of layers, heads, and hidden dimensions.

### Vision Transformer

The Vision Transformer model is implemented in `vision_transformer_ts_forecasting_v32.py`. It uses the `ViT` class from the `vit-pytorch` library.

Key components:
- `VisionTransformerRegressor`: Custom PyTorch module that combines ViT with a regression head.
- Treats time series data as 2D images.
- Uses patch embedding and positional encoding.
- Implements k-fold cross-validation for robust evaluation.

### Simple LSTM

The Simple LSTM model is implemented in `simple_lstm_ts_forecasting.py`. It uses a standard LSTM architecture.

Key components:
- `SimpleLSTM`: Custom PyTorch module with LSTM layers and a linear output layer.
- Configurable number of layers and hidden dimensions.

## Key Functions

1. `get_stock_data(ticker, start_date, end_date)`: Fetches stock data from Yahoo Finance.
2. `prepare_data(data, sequence_length)`: Prepares time series data for model input.
3. `train_and_evaluate(X, y, scaler, batch_size, epochs)`: Trains and evaluates the model.
4. `evaluate_model(y_true, y_pred)`: Calculates performance metrics.
5. `plot_results(y_true, y_pred, title, filename)`: Plots actual vs predicted values.
6. `save_metrics(metrics, filename)`: Saves evaluation metrics to a CSV file.
7. `generate_summary(metrics, ticker, start_date, end_date, version_num)`: Generates a summary of the forecast results.
8. `run_forecast(ticker, start_date, end_date, batch_size, version_num)`: Orchestrates the entire forecasting process.

## Control Flow

1. The main entry point is `combined_forecasting_coordinator.py`.
2. It imports necessary functions from other files.
3. The `run_combined_forecast` function:
   a. Fetches stock data
   b. Prepares data for each model
   c. Trains and evaluates each model
   d. Computes metrics and generates plots
   e. Saves results and summaries
4. The `main` function handles command-line arguments and calls `run_combined_forecast`.
5. Each model-specific file (`transformer_ts_forecasting_main.py`, `vision_transformer_ts_forecasting_v32.py`, `simple_lstm_ts_forecasting.py`) can also be run independently.

## Possible Roadmap

1. **Model Enhancements**:
   - Implement more advanced Transformer architectures (e.g., Informer, Autoformer).
   - Experiment with different ViT configurations and patch sizes.
   - Try hybrid models combining LSTM with attention mechanisms.

2. **Data Processing**:
   - Implement more sophisticated data augmentation techniques.
   - Explore feature engineering specific to financial time series.

3. **Hyperparameter Optimization**:
   - Implement automated hyperparameter tuning (e.g., Bayesian optimization, genetic algorithms).
   - Create a configuration file for easy parameter adjustments.

4. **Evaluation and Visualization**:
   - Add more advanced evaluation metrics (e.g., Directional Accuracy, Sharpe Ratio).
   - Implement interactive visualizations (e.g., using Plotly or Bokeh).

5. **Scalability and Performance**:
   - Optimize data loading and preprocessing for larger datasets.
   - Implement distributed training for faster experimentation.

6. **Deployment and Production**:
   - Create a web API for real-time predictions.
   - Implement model versioning and A/B testing infrastructure.

7. **Additional Features**:
   - Add support for multi-variate time series forecasting.
   - Implement ensemble methods combining predictions from multiple models.
   - Add explainability features to interpret model predictions.

8. **Documentation and Testing**:
   - Expand inline code documentation and add type hints.
   - Implement comprehensive unit and integration tests.
   - Create user guides and API documentation.

By following this roadmap, the project can evolve into a more robust, scalable, and feature-rich time series forecasting framework.
