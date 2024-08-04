import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from image_time_series_forecasting import train_model, forecast_next_images, generate_weather_sequence

import tensorflow as tf

# Suppress TensorFlow warnings - needs to change
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def main():
    st.title("Explicit Weather-like Image Time Series Forecasting")

    # Sidebar for input parameters
    st.sidebar.header("Input Parameters")
    num_samples = st.sidebar.slider("Number of Samples", 50, 500, 100)
    image_size = st.sidebar.slider("Image Size", 32, 128, 64)
    num_channels = st.sidebar.selectbox("Number of Channels", [1, 3], index=0)
    sequence_length = st.sidebar.slider("Sequence Length", 3, 10, 5)
    num_forecasts = st.sidebar.slider("Number of Forecasts", 1, 5, 3)

    # Generate sample sequence
    if st.sidebar.button("Generate Sample Sequence"):
        images = generate_weather_sequence(num_samples, (image_size, image_size), num_channels)
        st.session_state.sample_sequence = images
        st.success("Sample sequence generated!")

    # Display sample sequence
    if 'sample_sequence' in st.session_state:
        st.subheader("Sample Sequence")
        fig, axes = plt.subplots(1, 5, figsize=(15, 3))
        for i in range(5):
            axes[i].imshow(st.session_state.sample_sequence[i*num_samples//5, :, :, 0], cmap='viridis')
            axes[i].axis('off')
            axes[i].set_title(f'Frame {i*num_samples//5}')
        st.pyplot(fig)

    # Train model button
    if st.sidebar.button("Train Model"):
        with st.spinner("Training model..."):
            model, input_sequence = train_model(num_samples, (image_size, image_size), num_channels, sequence_length)
            st.session_state.model = model
            st.session_state.input_sequence = input_sequence
        st.success("Model trained successfully!")

    # Forecast button
    if st.button("Generate Forecast"):
        if 'model' not in st.session_state:
            st.error("Please train the model first.")
        else:
            forecasts = forecast_next_images(st.session_state.model, st.session_state.input_sequence, num_forecasts)
            
            # Visualize results
            fig, axes = plt.subplots(2, num_forecasts + 1, figsize=(15, 6))
            for i in range(num_forecasts + 1):
                if i < sequence_length:
                    axes[0, i].imshow(st.session_state.input_sequence[i, :, :, 0], cmap='viridis')
                    axes[0, i].set_title(f'Input {i+1}')
                    axes[0, i].axis('off')
                if i > 0:
                    axes[1, i].imshow(forecasts[i-1][:, :, 0], cmap='viridis')
                    axes[1, i].set_title(f'Forecast {i}')
                    axes[1, i].axis('off')
            plt.tight_layout()
            st.pyplot(fig)

if __name__ == "__main__":
    main()
