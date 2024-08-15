# Vision Transformer for Time Series Forecasting

This document provides a detailed description of the Python implementation for stock price forecasting using a Vision Transformer model. The code is designed for developers and researchers working on time series forecasting tasks.

Note: Call the vision_transformer_ts_forecasting.py file from the colab scaffolding provided

## Table of Contents

1. [Overview](#overview)
2. [Dependencies](#dependencies)
3. [Code Structure](#code-structure)
4. [Key Components](#key-components)
5. [Usage](#usage)
6. [Performance Metrics](#performance-metrics)
7. [Planned Enhancements](#planned-enhancements)

## Overview

This implementation adapts a Vision Transformer (ViT) architecture for time series forecasting, specifically for predicting stock prices. It includes data preparation, model training with cross-validation, and evaluation components.

## Dependencies

- Python 3.7+
- yfinance
- numpy
- pandas
- matplotlib
- scikit-learn
- torch
- vit-pytorch

## Code Structure

The code is organized into several main functions:

1. `get_stock_data`: Fetches stock data from Yahoo Finance
2. `prepare_data`: Prepares and scales the data for the model
3. `augment_data`: Performs data augmentation
4. `VisionTransformerRegressor`: Defines the ViT model architecture
5. `train_and_evaluate`: Implements the training and evaluation loop
6. `evaluate_model`: Calculates performance metrics
7. `plot_results`: Visualizes the predictions
8. `save_metrics`: Saves evaluation metrics to a CSV file
9. `generate_summary`: Creates a summary of the forecast results
10. `run_forecast`: Orchestrates the entire forecasting process
11. `main`: Entry point for script execution

## Key Components

### Data Preparation

- Uses `yfinance` to download stock data
- Applies `RobustScaler` for feature scaling
- Converts time series to image-like format (4x4 grid for each feature)

```python
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
```

### Model Architecture

The `VisionTransformerRegressor` class defines the model:

- Uses `ViT` from `vit-pytorch` as the base
- Adds a regression head for price prediction

```python
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
```

### Training and Evaluation

The `train_and_evaluate` function implements:

- K-fold cross-validation
- Data augmentation
- Early stopping
- Learning rate scheduling

```python
def train_and_evaluate(X, y, scaler, batch_size=32, epochs=300, n_splits=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # ... (training loop implementation)
    
    return all_true_values_inv, all_predictions_inv
```

## Usage

The script can be run from the command line with various arguments:

```bash
python vision_transformer_ts_forecasting.py --ticker AAPL --start_date 2022-01-01 --end_date 2023-12-31 --batch_size 32 --version_num v1
```

Alternatively, you can import and use the `main` function in your Python code:

```python
from vision_transformer_ts_forecasting import main

metrics, y_true, y_pred = main(batch_size=32, version_num='v1')
```

## Performance Metrics

The code calculates and reports the following metrics:

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R-squared (R2)
- Mean Absolute Percentage Error (MAPE)

## Planned Enhancements

1. Multi-step forecasting: Extend the model to predict multiple time steps into the future.
2. Hyperparameter tuning: Implement automated hyperparameter optimization.
3. Ensemble methods: Combine predictions from multiple models for improved accuracy.
4. Attention visualization: Add functionality to visualize the attention weights of the ViT.
5. Pre-training: Implement pre-training on a large dataset of stock prices before fine-tuning on specific stocks.
6. Feature importance: Add methods to interpret the model's decisions and identify important features.
7. Real-time predictions: Modify the code to allow for real-time stock price predictions using streaming data.
8. Multi-stock modeling: Extend the model to simultaneously predict multiple stock prices, capturing inter-stock relationships.

To contribute or suggest enhancements, please open an issue or submit a pull request on the project's repository.
