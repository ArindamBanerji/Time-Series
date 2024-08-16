import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Callable
from models_updated_v3 import SimpleLSTM, CNNGADF, LSTMImage, ResNetLSTM

def train_model(model, x_train: np.ndarray, y_train: np.ndarray, epochs: int = 100, batch_size: int = 32, callback: Callable = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    x_train_tensor = torch.FloatTensor(x_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i in range(0, len(x_train), batch_size):
            batch_x = x_train_tensor[i:i+batch_size]
            batch_y = y_train_tensor[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            # Ensure outputs and batch_y have the same shape
            outputs = outputs.view(-1)
            batch_y = batch_y.view(-1)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        avg_loss = total_loss / (len(x_train) / batch_size)
        if callback:
            callback.update(epoch, avg_loss)

    return model

def predict_model(model, x_test: np.ndarray) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        x_test_tensor = torch.FloatTensor(x_test).to(device)
        predictions = model(x_test_tensor).cpu().numpy()
    return predictions.flatten()

def train_simple_lstm(x_train, y_train, callback=None):
    model = SimpleLSTM(input_size=1)
    return train_model(model, x_train, y_train, callback=callback)

def train_cnn_gadf(x_train, y_train, callback=None):
    model = CNNGADF()
    x_train = np.expand_dims(x_train, axis=1)  # Add channel dimension
    return train_model(model, x_train, y_train, callback=callback)

def train_lstm_image(x_train, y_train, callback=None):
    input_size = x_train.shape[1] * x_train.shape[2]  # Flatten the image
    model = LSTMImage(input_size=input_size)
    x_train = x_train.reshape(x_train.shape[0], -1)  # Flatten images
    return train_model(model, x_train, y_train, callback=callback)

def train_resnet_lstm(x_train, y_train, callback=None):
    sequence_length = x_train.shape[1]
    model = ResNetLSTM(sequence_length)
    return train_model(model, x_train, y_train, callback=callback)

def predict_simple_lstm(model, x_test):
    return predict_model(model, x_test)

def predict_cnn_gadf(model, x_test):
    x_test = np.expand_dims(x_test, axis=1)  # Add channel dimension
    return predict_model(model, x_test)

def predict_lstm_image(model, x_test):
    x_test = x_test.reshape(x_test.shape[0], -1)  # Flatten images
    return predict_model(model, x_test)

def predict_resnet_lstm(model, x_test):
    return predict_model(model, x_test)
