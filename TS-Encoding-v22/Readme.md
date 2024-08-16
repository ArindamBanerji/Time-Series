# Stock Price Prediction App Documentation

## Table of Contents
1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Module Functionality](#module-functionality)
   - [main_app_complete_final.py](#main_app_complete_finalpy)
   - [data_preparation_updated_v3.py](#data_preparation_updated_v3py)
   - [models_updated_v3.py](#models_updated_v3py)
   - [train_predict_updated_v3.py](#train_predict_updated_v3py)
   - [evaluation_complete_v3.py](#evaluation_complete_v3py)
4. [Architecture](#architecture)
5. [Data Flow](#data-flow)
6. [Next Steps](#next-steps)
7. [Developer Guidelines](#developer-guidelines)

## Overview

This project implements a comparative performance experiment on different approaches for using image data in time series forecasting, specifically for stock price prediction. The application is built using Python and Streamlit, with PyTorch as the deep learning framework. It compares four main approaches:

1. Simple LSTM (base estimator)
2. CNN-GADF (Convolutional Neural Network with Gramian Angular Difference Field)
3. LSTM-Image
4. ResNet-LSTM

## Project Structure

The project consists of five main Python files:

1. `main_app_complete_final.py`: The main Streamlit application
2. `data_preparation_updated_v3.py`: Data retrieval and preparation functions
3. `models_updated_v3.py`: Model architectures
4. `train_predict_updated_v3.py`: Training and prediction functions
5. `evaluation_complete_v3.py`: Evaluation metrics and visualization

## Module Functionality

### main_app_complete_final.py

This is the main Streamlit application file. It handles the user interface, orchestrates the prediction process, and visualizes results.

#### Functions:

- `plot_predictions(y_true: np.ndarray, predictions: dict, title: str) -> plt.Figure`:
  - Plots actual vs predicted stock prices for all models
  - Parameters: true values, predictions dictionary, plot title
  - Returns: matplotlib Figure object

- `train_model_with_progress(train_func, x_train, y_train, model_name, epochs=100, batch_size=32) -> nn.Module`:
  - Trains a model with a progress bar
  - Parameters: training function, training data, model name, epochs, batch size
  - Returns: trained PyTorch model

- `load_data(ticker: str, period: str) -> pd.DataFrame`:
  - Caches and loads stock data
  - Parameters: stock ticker, time period
  - Returns: DataFrame with stock data

- `create_metrics_table(metrics: Dict[str, Dict[str, float]]) -> pd.DataFrame`:
  - Creates a comparison table of model metrics
  - Parameters: dictionary of metrics for each model
  - Returns: DataFrame with formatted metrics

- `main()`:
  - Main application logic
  - Handles user input, data loading, model training, prediction, and result visualization

### data_preparation_updated_v3.py

This module is responsible for data retrieval and preparation.

#### Functions:

- `get_stock_data(ticker: str, period: str = '5y') -> pd.DataFrame`:
  - Retrieves stock data using yfinance
  - Parameters: stock ticker, time period
  - Returns: DataFrame with stock data

- `prepare_time_series_data(data: pd.DataFrame, sequence_length: int = 60) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], MinMaxScaler]`:
  - Prepares data for Simple LSTM
  - Parameters: stock data, sequence length
  - Returns: train data, test data, scaler

- `prepare_image_data(data: pd.DataFrame, sequence_length: int = 60) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], MinMaxScaler]`:
  - Prepares data for CNN-GADF and LSTM-Image models
  - Parameters: stock data, sequence length
  - Returns: train data, test data, scaler

- `prepare_resnet_data(data: pd.DataFrame, sequence_length: int = 60) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], MinMaxScaler]`:
  - Prepares data for ResNet-LSTM model
  - Parameters: stock data, sequence length
  - Returns: train data, test data, scaler

- `inverse_transform(scaler: MinMaxScaler, data: np.ndarray) -> np.ndarray`:
  - Inverse transforms scaled data
  - Parameters: scaler, scaled data
  - Returns: original scale data

### models_updated_v3.py

This file contains the PyTorch model architectures.

#### Classes:

- `SimpleLSTM(nn.Module)`:
  - Basic LSTM model
  - Methods:
    - `__init__(self, input_size=1, hidden_size=50, num_layers=1)`
    - `forward(self, x)`

- `CNNGADF(nn.Module)`:
  - CNN model for Gramian Angular Difference Field images
  - Methods:
    - `__init__(self)`
    - `forward(self, x)`

- `LSTMImage(nn.Module)`:
  - LSTM model for flattened images
  - Methods:
    - `__init__(self, input_size, hidden_size=50, num_layers=1)`
    - `forward(self, x)`

- `ResNetLSTM(nn.Module)`:
  - Combined ResNet and LSTM model
  - Methods:
    - `__init__(self, sequence_length, hidden_size=50, num_layers=1)`
    - `forward(self, x)`

### train_predict_updated_v3.py

This module handles the training and prediction processes.

#### Functions:

- `train_model(model: nn.Module, x_train: np.ndarray, y_train: np.ndarray, epochs: int = 100, batch_size: int = 32, callback: Callable = None) -> nn.Module`:
  - Generic training function for all models
  - Parameters: model, training data, epochs, batch size, callback function
  - Returns: trained model

- `predict_model(model: nn.Module, x_test: np.ndarray) -> np.ndarray`:
  - Generic prediction function for all models
  - Parameters: model, test data
  - Returns: predictions

- `train_simple_lstm(x_train, y_train, callback=None) -> nn.Module`:
  - Trains Simple LSTM model
  - Parameters: training data, callback function
  - Returns: trained model

- `train_cnn_gadf(x_train, y_train, callback=None) -> nn.Module`:
  - Trains CNN-GADF model
  - Parameters: training data, callback function
  - Returns: trained model

- `train_lstm_image(x_train, y_train, callback=None) -> nn.Module`:
  - Trains LSTM-Image model
  - Parameters: training data, callback function
  - Returns: trained model

- `train_resnet_lstm(x_train, y_train, callback=None) -> nn.Module`:
  - Trains ResNet-LSTM model
  - Parameters: training data, callback function
  - Returns: trained model

- `predict_simple_lstm(model: nn.Module, x_test: np.ndarray) -> np.ndarray`:
  - Predicts using Simple LSTM model
  - Parameters: model, test data
  - Returns: predictions

- `predict_cnn_gadf(model: nn.Module, x_test: np.ndarray) -> np.ndarray`:
  - Predicts using CNN-GADF model
  - Parameters: model, test data
  - Returns: predictions

- `predict_lstm_image(model: nn.Module, x_test: np.ndarray) -> np.ndarray`:
  - Predicts using LSTM-Image model
  - Parameters: model, test data
  - Returns: predictions

- `predict_resnet_lstm(model: nn.Module, x_test: np.ndarray) -> np.ndarray`:
  - Predicts using ResNet-LSTM model
  - Parameters: model, test data
  - Returns: predictions

### evaluation_complete_v3.py

This file contains functions for evaluating and comparing model performance.

#### Functions:

- `evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]`:
  - Calculates various evaluation metrics
  - Parameters: true values, predicted values
  - Returns: dictionary of evaluation metrics

- `print_evaluation_results(metrics: Dict[str, float]) -> None`:
  - Prints formatted evaluation results
  - Parameters: dictionary of evaluation metrics

- `compare_models(model_metrics: Dict[str, Dict[str, float]]) -> None`:
  - Compares performance across different models
  - Parameters: dictionary of metrics for each model

## Architecture

The application follows a modular architecture:

1. **Data Layer**: Handled by `data_preparation_updated_v3.py`
2. **Model Layer**: Defined in `models_updated_v3.py`
3. **Training and Prediction Layer**: Managed by `train_predict_updated_v3.py`
4. **Evaluation Layer**: Implemented in `evaluation_complete_v3.py`
5. **Presentation Layer**: The Streamlit app in `main_app_complete_final.py`

This separation of concerns allows for easy maintenance and extension of the codebase.

## Data Flow

1. User inputs parameters in the Streamlit app
2. Stock data is retrieved and prepared based on the selected models
3. Models are trained on the prepared data
4. Predictions are made using the trained models
5. Results are evaluated and compared
6. Visualizations and metrics are presented to the user

## Next Steps

To extend and improve the project, consider the following steps:

1. **Hyperparameter Tuning**: Implement automated hyperparameter tuning for each model.
2. **Additional Models**: Integrate more advanced models or ensemble methods.
3. **Feature Engineering**: Add more features beyond just the closing price.
4. **Real-time Updates**: Implement real-time data fetching and model updating.
5. **Performance Optimization**: Optimize the code for faster training and prediction, possibly using GPU acceleration.
6. **Error Handling**: Enhance error handling and user feedback for edge cases.
7. **Testing**: Add unit tests and integration tests for improved reliability.
8. **Documentation**: Add docstrings and improve inline comments for better code understanding.

## Developer Guidelines

When modifying or extending the code:

1. **Modular Design**: Maintain the modular structure. Add new functionalities in appropriate modules.
2. **Consistent Naming**: Follow the existing naming conventions for functions and variables.
3. **Type Hinting**: Use type hints for function arguments and return values.
4. **Error Handling**: Implement proper error handling and logging.
5. **Performance**: Consider the performance implications of your changes, especially for large datasets.
6. **Testing**: Add tests for any new functionality you implement.
7. **Documentation**: Update this documentation and add inline comments for significant changes.
8. **Dependencies**: If adding new dependencies, update the project's requirements file.

Remember to thoroughly test any changes before merging them into the main branch.
