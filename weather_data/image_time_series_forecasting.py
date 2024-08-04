import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D, Flatten, Dense, Reshape, Input
from scipy.ndimage import gaussian_filter


import tensorflow as tf

# Suppress TensorFlow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def generate_weather_sequence(num_samples, image_size, num_channels):
    """
    Generate a sequence of weather-like images showing explicit evolution over time.

    Args:
    num_samples (int): Number of images in the time series.
    image_size (tuple): Size of each image (height, width).
    num_channels (int): Number of color channels (1 for grayscale, 3 for RGB).

    Returns:
    numpy.ndarray: Array of sample weather-like images.
    """
    images = []
    
    # Parameters for weather evolution
    cloud_center = np.array([0, 0], dtype=float)
    cloud_speed = np.array([image_size[0] / num_samples, image_size[1] / num_samples])
    cloud_size = 5
    cloud_growth = 0.2  # growth rate per frame
    
    for i in range(num_samples):
        image = np.zeros((image_size[0], image_size[1]))
        
        # Move and grow the cloud
        cloud_center += cloud_speed
        cloud_size += cloud_growth
        
        # Create cloud pattern
        y, x = np.ogrid[:image_size[0], :image_size[1]]
        cloud = np.exp(-((x - cloud_center[1])**2 + (y - cloud_center[0])**2) / (2 * cloud_size**2))
        
        # Add cloud to image
        image += cloud
        
        # Add some random noise for texture
        image += np.random.randn(image_size[0], image_size[1]) * 0.05
        
        # Normalize to [0, 1] range
        image = (image - image.min()) / (image.max() - image.min())
        
        # Apply Gaussian filter for smooth, cloud-like shapes
        image = gaussian_filter(image, sigma=1)
        
        # Create image with specified number of channels
        image = np.repeat(image[:, :, np.newaxis], num_channels, axis=2)
        
        images.append(image)
    
    return np.array(images)

def create_model(input_shape):
    """
    Create a ConvLSTM model for time series forecasting.

    Args:
    input_shape (tuple): Shape of input image sequences (time_steps, height, width, channels).

    Returns:
    tensorflow.keras.models.Sequential: The created model.
    """
    model = Sequential([
        Input(shape=input_shape),
        ConvLSTM2D(64, (3, 3), activation='relu', return_sequences=False),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(np.prod(input_shape[1:])),
        Reshape(input_shape[1:])
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def prepare_data(images, sequence_length):
    """
    Prepare data for training by creating input sequences and target outputs.

    Args:
    images (numpy.ndarray): Array of images in the time series.
    sequence_length (int): Number of images to use as input for each prediction.

    Returns:
    tuple: (X, y) where X is the input sequences and y is the target outputs.
    """
    X, y = [], []
    for i in range(len(images) - sequence_length):
        X.append(images[i:i+sequence_length])
        y.append(images[i+sequence_length])
    return np.array(X), np.array(y)

def forecast_next_images(model, input_sequence, num_forecasts):
    """
    Forecast the next images in the time series.

    Args:
    model (tensorflow.keras.models.Sequential): Trained forecasting model.
    input_sequence (numpy.ndarray): Input sequence of images.
    num_forecasts (int): Number of images to forecast.

    Returns:
    list: List of forecasted images.
    """
    forecasts = []
    current_sequence = input_sequence.copy()
    
    for _ in range(num_forecasts):
        next_image = model.predict(current_sequence[np.newaxis])[0]
        forecasts.append(next_image)
        current_sequence = np.roll(current_sequence, -1, axis=0)
        current_sequence[-1] = next_image
    
    return forecasts

def train_model(num_samples, image_size, num_channels, sequence_length):
    """
    Train the model with the given parameters.

    Args:
    num_samples (int): Number of images in the time series.
    image_size (tuple): Size of each image (height, width).
    num_channels (int): Number of color channels (1 for grayscale, 3 for RGB).
    sequence_length (int): Number of images to use as input for each prediction.

    Returns:
    tuple: (model, X_test) The trained model and a sample input sequence for testing.
    """
    images = generate_weather_sequence(num_samples, image_size, num_channels)
    X, y = prepare_data(images, sequence_length)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = create_model(X.shape[1:])
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

    return model, X_test[0]

if __name__ == "__main__":
    # This block is for testing purposes only
    num_samples = 100
    image_size = (64, 64)
    num_channels = 1
    sequence_length = 5
    num_forecasts = 3

    model, input_sequence = train_model(num_samples, image_size, num_channels, sequence_length)
    forecasts = forecast_next_images(model, input_sequence, num_forecasts)

    # Visualize results
    fig, axes = plt.subplots(2, num_forecasts + 1, figsize=(15, 6))
    for i in range(num_forecasts + 1):
        if i < sequence_length:
            axes[0, i].imshow(input_sequence[i, :, :, 0], cmap='viridis')
            axes[0, i].set_title(f'Input {i+1}')
        if i > 0:
            axes[1, i].imshow(forecasts[i-1][:, :, 0], cmap='viridis')
            axes[1, i].set_title(f'Forecast {i}')
    plt.tight_layout()
    plt.show()
