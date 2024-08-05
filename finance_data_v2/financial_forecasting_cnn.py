# Set environment variables to reduce TensorFlow warnings
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyts.image import GramianAngularField
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def generate_sample_data(start_date, end_date, seed=42):
    """
    Generate more realistic sample financial data with trends and seasonality.
    
    :param start_date: Start date for the time series
    :param end_date: End date for the time series
    :param seed: Random seed for reproducibility
    :return: DataFrame with date and price columns
    """
    np.random.seed(seed)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n = len(dates)
    
    # Generate trend
    trend = np.linspace(0, 100, n)
    
    # Generate seasonality
    seasonality = 10 * np.sin(np.arange(n) * (2 * np.pi / 365.25))
    
    # Generate noise
    noise = np.random.normal(0, 5, n)
    
    # Combine components
    price = 100 + trend + seasonality + noise
    
    return pd.DataFrame({'date': dates, 'price': price})

def prepare_data(data, sequence_length=60):
    """
    Prepare the data for model training with increased sequence length.
    
    :param data: DataFrame with price data
    :param sequence_length: Length of input sequences
    :return: Scaled input sequences and target values
    """
    time_series = data['price'].values
    scaler = MinMaxScaler(feature_range=(-1, 1))
    time_series_scaled = scaler.fit_transform(time_series.reshape(-1, 1)).flatten()
    
    X = np.array([time_series_scaled[i:i+sequence_length] for i in range(len(time_series_scaled)-sequence_length)])
    y = time_series[sequence_length:]
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

def create_gaf_images(X_train, X_test):
    """
    Convert time series data to Gramian Angular Field images.
    
    :param X_train: Training input sequences
    :param X_test: Testing input sequences
    :return: GAF images for training and testing data
    """
    gaf = GramianAngularField()
    X_train_gaf = gaf.fit_transform(X_train)
    X_test_gaf = gaf.transform(X_test)
    return X_train_gaf, X_test_gaf

def build_model(input_shape):
    """
    Build and compile an improved CNN model.
    
    :param input_shape: Shape of the input data
    :return: Compiled Keras model
    """
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def train_model(model, X_train, y_train, epochs=100, batch_size=32):
    """
    Train the model on the provided data with early stopping.
    
    :param model: Keras model to train
    :param X_train: Training input data
    :param y_train: Training target data
    :param epochs: Maximum number of training epochs
    :param batch_size: Batch size for training
    :return: Training history
    """
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    return model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=0
    )

def evaluate_model(model, X_test, y_test, batch_size=32):
    """
    Evaluate the model and make predictions.
    
    :param model: Trained Keras model
    :param X_test: Testing input data
    :param y_test: Testing target data
    :param batch_size: Batch size for prediction (default: 32)
    :return: Predictions and Mean Squared Error
    """
    y_pred = model.predict(X_test, batch_size=batch_size, verbose=0).flatten()
    mse = np.mean((y_test - y_pred)**2)
    return y_pred, mse

def plot_original_time_series(data):
    """
    Plot the original time series data.
    
    :param data: DataFrame with date and price columns
    :return: Figure and axes objects
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data['date'], data['price'])
    ax.set_title('Original Time Series')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    return fig, ax

def plot_encoded_images(X_train_gaf, num_images=5):
    """
    Plot a set of encoded GAF images.
    
    :param X_train_gaf: GAF images of training data
    :param num_images: Number of images to plot
    :return: Figure and axes objects
    """
    fig, axes = plt.subplots(1, num_images, figsize=(4*num_images, 4))
    for i in range(num_images):
        axes[i].imshow(X_train_gaf[i], cmap='viridis')
        axes[i].set_title(f'GAF Image {i+1}')
        axes[i].axis('off')
    plt.tight_layout()
    return fig, axes  # Fixed: return fig, axes instead of fig, ax

def plot_training_history(history):
    """
    Plot the training history of the model.
    
    :param history: Training history object
    :return: Figure and axes objects
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history.history['loss'], label='Training Loss')
    ax.plot(history.history['val_loss'], label='Validation Loss')
    ax.set_title('Model Training History')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    return fig, ax

def plot_predictions(y_test, y_pred):
    """
    Plot predictions vs actual values.
    
    :param y_test: Actual test values
    :param y_pred: Predicted test values
    :return: Figure and axes objects
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, y_pred, alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_title('Predictions vs Actual')
    ax.set_xlabel('Actual Price')
    ax.set_ylabel('Predicted Price')
    return fig, ax

def main(start_date, end_date, sequence_length=60, epochs=100, batch_size=32):
    """
    Main function to run the entire process.
    
    :param start_date: Start date for the time series
    :param end_date: End date for the time series
    :param sequence_length: Length of input sequences
    :param epochs: Maximum number of training epochs
    :param batch_size: Batch size for training
    :return: Dictionary containing all results and figures
    """
    # Generate and prepare data
    data = generate_sample_data(start_date, end_date)
    X_train, X_test, y_train, y_test = prepare_data(data, sequence_length)
    
    # Create GAF images
    X_train_gaf, X_test_gaf = create_gaf_images(X_train, X_test)
    
    # Build and train model
    model = build_model(input_shape=(X_train_gaf.shape[1], X_train_gaf.shape[2], 1))
    history = train_model(model, X_train_gaf, y_train, epochs, batch_size)
    
    # Evaluate model
    y_pred, mse = evaluate_model(model, X_test_gaf, y_test, batch_size)
    
    # Generate plots
    original_ts_fig, _ = plot_original_time_series(data)
    encoded_images_fig, _ = plot_encoded_images(X_train_gaf)
    training_history_fig, _ = plot_training_history(history)
    predictions_fig, _ = plot_predictions(y_test, y_pred)
    
    return {
        'data': data,
        'model': model,
        'X_test_gaf': X_test_gaf,
        'y_test': y_test,
        'y_pred': y_pred,
        'mse': mse,
        'original_ts_fig': original_ts_fig,
        'encoded_images_fig': encoded_images_fig,
        'training_history_fig': training_history_fig,
        'predictions_fig': predictions_fig
    }

if __name__ == "__main__":
    results = main('2020-01-01', '2023-12-31')
    print(f"Mean Squared Error: {results['mse']:.2f}")
    plt.show()
