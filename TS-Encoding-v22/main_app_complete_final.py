import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
from datetime import datetime
from data_preparation_updated_v3 import get_stock_data, prepare_time_series_data, prepare_image_data, prepare_resnet_data, inverse_transform
from train_predict_updated_v3 import (
    train_simple_lstm, predict_simple_lstm,
    train_cnn_gadf, predict_cnn_gadf,
    train_lstm_image, predict_lstm_image,
    train_resnet_lstm, predict_resnet_lstm
)
from evaluation_complete_v3 import evaluate_predictions

logging.basicConfig(level=logging.INFO)

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def generate_unique_filename(base_name, approach_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{approach_name}_{timestamp}"

def plot_predictions(y_true: np.ndarray, predictions: dict, title: str, save_path: str):
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Actual Prices', color='black')
    colors = ['red', 'blue', 'green', 'purple']
    for (model_name, y_pred), color in zip(predictions.items(), colors):
        plt.plot(y_pred, label=f'{model_name} Predictions', color=color, alpha=0.7)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.savefig(save_path)
    st.pyplot(plt)
    plt.close()

def plot_individual_prediction(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, title: str, save_path: str):
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Actual Prices', color='black')
    plt.plot(y_pred, label=f'{model_name} Predictions', color='red', alpha=0.7)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.savefig(save_path)
    st.pyplot(plt)
    plt.close()

def save_metrics(metrics: dict, save_path: str):
    df = pd.DataFrame(metrics).T
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Model'}, inplace=True)
    df.to_csv(save_path, index=False)

def create_summary_analysis(metrics: dict, save_path: str):
    with open(save_path, 'w') as f:
        f.write("Summary Analysis:\n\n")
        best_model = min(metrics, key=lambda x: metrics[x]['MSE'])
        worst_model = max(metrics, key=lambda x: metrics[x]['MSE'])
        f.write(f"• Best performing model: {best_model}\n")
        f.write(f"• Worst performing model: {worst_model}\n")
        f.write(f"• Performance difference (MSE): {metrics[worst_model]['MSE'] - metrics[best_model]['MSE']:.4f}\n")
        f.write("\n• Model rankings based on MSE:\n")
        for i, (model, model_metrics) in enumerate(sorted(metrics.items(), key=lambda x: x[1]['MSE']), 1):
            f.write(f"  {i}. {model}: {model_metrics['MSE']:.4f}\n")
        f.write("\n• Key observations:\n")
        f.write(f"  - The {best_model} model outperforms other models, suggesting it may be the most suitable for this particular stock and time period.\n")
        f.write(f"  - There's a significant performance gap between the best and worst models ({metrics[worst_model]['MSE'] - metrics[best_model]['MSE']:.4f} MSE difference).\n")
        
        # Check for any unexpected patterns
        if worst_model in ['Simple LSTM', 'ResNet-LSTM'] or best_model in ['CNN-GADF', 'LSTM Image']:
            f.write("  - Unexpectedly, a simpler model outperformed more complex ones. This might indicate overfitting in the complex models or that the stock's behavior is relatively simple to predict.\n")
        
        # Suggest improvements
        f.write("\n• Potential improvements and next steps:\n")
        f.write("  1. Fine-tune hyperparameters, especially for the underperforming models.\n")
        f.write("  2. Experiment with different sequence lengths to capture optimal time dependencies.\n")
        f.write("  3. Incorporate additional features such as trading volume or technical indicators.\n")
        f.write("  4. Test the models on different stocks and time periods to assess their generalization capabilities.\n")
        f.write("  5. Consider ensemble methods to combine predictions from multiple models.\n")

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
    df = pd.DataFrame(metrics).T
    df = df.round(4)
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
                
                # Create directories
                create_directory("plots")
                create_directory("metrics")
                create_directory("summary_analysis")
                
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
                
                # Save metrics
                metrics_filename = generate_unique_filename("metrics", "_".join(models_to_run))
                save_metrics(metrics, f"metrics/{metrics_filename}.csv")
                
                # Create and save summary analysis
                summary_filename = generate_unique_filename("summary", "_".join(models_to_run))
                create_summary_analysis(metrics, f"summary_analysis/{summary_filename}.txt")
                
                st.subheader("Stock Price Predictions")
                
                # Composite plot
                composite_filename = generate_unique_filename("composite_plot", "_".join(models_to_run))
                plot_predictions(y_test_inv, predictions, f"{ticker} Stock Price Predictions", f"plots/{composite_filename}.png")
                
                # Individual plots
                for model_name, pred in predictions.items():
                    individual_filename = generate_unique_filename("individual_plot", model_name)
                    plot_individual_prediction(y_test_inv, pred, model_name, f"{ticker} Stock Price Prediction - {model_name}", f"plots/{individual_filename}.png")
                
            except Exception as e:
                logging.error(f"An error occurred during processing: {str(e)}", exc_info=True)
                st.error(f"An error occurred during processing: {str(e)}")
                st.write("Please try again with different parameters or contact support if the issue persists.")
        else:
            st.error("Unable to retrieve stock data or insufficient data. Please check the ticker symbol and try again.")

if __name__ == "__main__":
    main()
