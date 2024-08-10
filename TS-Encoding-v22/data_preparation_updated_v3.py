import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pyts.image import GramianAngularField
from typing import Tuple

def get_stock_data(ticker: str, period: str = '5y') -> pd.DataFrame:
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        if data.empty:
            raise ValueError(f"No data retrieved for {ticker}")
        return data
    except Exception as e:
        print(f"Error retrieving data for {ticker}: {str(e)}")
        return None

def prepare_time_series_data(data: pd.DataFrame, sequence_length: int = 60) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], MinMaxScaler]:
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    
    x, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        x.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])
    
    x, y = np.array(x), np.array(y)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))
    
    train_size = int(len(x) * 0.8)
    x_train, x_test = x[:train_size], x[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return (x_train, y_train), (x_test, y_test), scaler

def prepare_image_data(data: pd.DataFrame, sequence_length: int = 60) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], MinMaxScaler]:
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    
    gadf = GramianAngularField(image_size=sequence_length)
    x, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        img = gadf.fit_transform(scaled_data[i-sequence_length:i].T)
        x.append(img[0])
        y.append(scaled_data[i, 0])
    
    x, y = np.array(x), np.array(y)
    
    # Reshape x to (samples, height, width)
    x = x.reshape(x.shape[0], sequence_length, sequence_length)
    
    train_size = int(len(x) * 0.8)
    x_train, x_test = x[:train_size], x[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return (x_train, y_train), (x_test, y_test), scaler

def prepare_resnet_data(data: pd.DataFrame, sequence_length: int = 60) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], MinMaxScaler]:
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    
    x, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        x.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])
    
    x, y = np.array(x), np.array(y)
    
    # Reshape x to (samples, sequence_length, 1)
    x = x.reshape(x.shape[0], sequence_length, 1)
    
    train_size = int(len(x) * 0.8)
    x_train, x_test = x[:train_size], x[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return (x_train, y_train), (x_test, y_test), scaler

def inverse_transform(scaler: MinMaxScaler, data: np.ndarray) -> np.ndarray:
    return scaler.inverse_transform(data.reshape(-1, 1)).flatten()
