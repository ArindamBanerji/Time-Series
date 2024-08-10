import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from data_preparation_updated_v3 import get_stock_data, prepare_time_series_data, prepare_image_data, prepare_resnet_data, inverse_transform
from train_predict_updated_v3 import (
    train_simple_lstm, predict_simple_lstm,
    train_cnn_gadf, predict_cnn_gadf,
    train_lstm_image, predict_lstm_image,
    train_resnet_lstm, predict_resnet_lstm
)
from evaluation_complete_v3 import evaluate_predictions

logging.basicConfig(level=logging.INFO)

def plot_predictions(y_true: np.ndarray, predictions: dict, title: str):
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Actual Prices', color='black')
    colors = ['red', 'blue', 'green', 'purple']
    for (model_name, y_pred), color in zip(predictions.items(), colors):
        plt.plot(y_pred, label=f'{model_name} Predictions', color=color, alpha=0.7)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    return plt

def train_model_with_progress(train_func, x_train, y_train, model_name, epochs=100, batch_size=32):
    progress_bar = st.progress(0)
    epoch_progress = st.empty()
    
    class ProgressCallback:
        def __init__(self, total_epochs):
            self.total_epochs = total_epochs
        
        def update(self, epoch, loss):
            progress = (epoch + 1) / self.total_epochs
            progress_bar.progress(progress)
            epoch_progress.text(f"{model_name} - Epoch [{epoch+1}/{self.total_epochs}], Loss: {loss:.4f}")

    callback = ProgressCallback(epochs)
    model = train_func(x_train, y_train, callback=callback)
    progress_bar.empty()
    epoch_progress.empty()
    return model

@st.cache_data
def load_data(ticker, period):
    return get_stock_data(ticker, period)

def create_metrics_table(metrics):
    # Create a DataFrame from the metrics dictionary
    df = pd.DataFrame(metrics).T
    
    # Round all numeric values to 4 decimal places
    df = df.round(4)
    
    # Move the index (model names) to a column
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Model'}, inplace=True)
    
    return df

def main():
    st.title("Stock Price Prediction App")
    
    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL):", "AAPL")
    period = st.selectbox("Select Time Period:", ["1y", "2y", "5y", "10y"], index=1)
    models_to_run = st.multiselect("Select Models to Run:", 
                                   ["Simple LSTM", "CNN-GADF", "LSTM Image", "ResNet-LSTM"],
                                   default=["Simple LSTM"])
    
    epochs = st.slider("Number of training epochs:", min_value=10, max_value=200, value=100, step=10)
    batch_size = st.select_slider("Batch size:", options=[16, 32, 64, 128], value=32)
    sequence_length = st.selectbox("Select Sequence Length:", [60, 224], index=0)
    
    if st.button("Predict Stock Prices"):
        stock_data = load_data(ticker, period)
        
        if stock_data is not None and len(stock_data) > 0:
            try:
                st.write("Data retrieved successfully. Processing...")
                
                predictions = {}
                metrics = {}
                
                for model_name in models_to_run:
                    st.write(f"Training {model_name} model...")
                    
                    if model_name == "Simple LSTM":
                        (x_train, y_train), (x_test, y_test), scaler = prepare_time_series_data(stock_data, sequence_length=sequence_length)
                        model = train_model_with_progress(train_simple_lstm, x_train, y_train, model_name, epochs, batch_size)
                        predictions[model_name] = predict_simple_lstm(model, x_test)
                    elif model_name == "CNN-GADF":
                        (x_train, y_train), (x_test, y_test), scaler = prepare_image_data(stock_data, sequence_length=sequence_length)
                        model = train_model_with_progress(train_cnn_gadf, x_train, y_train, model_name, epochs, batch_size)
                        predictions[model_name] = predict_cnn_gadf(model, x_test)
                    elif model_name == "LSTM Image":
                        (x_train, y_train), (x_test, y_test), scaler = prepare_image_data(stock_data, sequence_length=sequence_length)
                        model = train_model_with_progress(train_lstm_image, x_train, y_train, model_name, epochs, batch_size)
                        predictions[model_name] = predict_lstm_image(model, x_test)
                    elif model_name == "ResNet-LSTM":
                        (x_train, y_train), (x_test, y_test), scaler = prepare_resnet_data(stock_data, sequence_length=sequence_length)
                        model = train_model_with_progress(train_resnet_lstm, x_train, y_train, model_name, epochs, batch_size)
                        predictions[model_name] = predict_resnet_lstm(model, x_test)
                
                y_test_inv = inverse_transform(scaler, y_test)
                
                st.write("Evaluating predictions...")
                for model_name, pred in predictions.items():
                    pred_inv = inverse_transform(scaler, pred)
                    metrics[model_name] = evaluate_predictions(y_test_inv, pred_inv)
                    predictions[model_name] = pred_inv  # Store inverse-transformed predictions
                
                st.subheader("Model Performance Metrics:")
                if len(metrics) > 1:
                    metrics_df = create_metrics_table(metrics)
                    st.table(metrics_df)
                else:
                    for model_name, model_metrics in metrics.items():
                        st.write(f"\n{model_name}:")
                        for metric, value in model_metrics.items():
                            st.write(f"  {metric}: {value:.4f}")
                
                st.subheader("Stock Price Predictions")
                fig = plot_predictions(y_test_inv, predictions, f"{ticker} Stock Price Predictions")
                st.pyplot(fig)
                
            except Exception as e:
                logging.error(f"An error occurred during processing: {str(e)}", exc_info=True)
                st.error(f"An error occurred during processing: {str(e)}")
                st.write("Please try again with different parameters or contact support if the issue persists.")
        else:
            st.error("Unable to retrieve stock data or insufficient data. Please check the ticker symbol and try again.")

if __name__ == "__main__":
    main()
