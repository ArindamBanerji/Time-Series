import streamlit as st
import pandas as pd
from financial_forecasting_cnn import main as run_forecast

def run_streamlit_app():
    st.title("Improved Financial Time Series Forecasting")
    
    # User input for date range
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
    with col2:
        end_date = st.date_input("End Date", value=pd.to_datetime("2023-12-31"))
    
    # User input for model parameters
    sequence_length = st.slider("Sequence Length", min_value=30, max_value=120, value=60)
    epochs = st.slider("Maximum Number of Epochs", min_value=50, max_value=300, value=100)
    batch_size = st.slider("Batch Size", min_value=16, max_value=128, value=32)
    
    if st.button("Run Forecast"):
        with st.spinner("Running forecast... This may take a few minutes."):
            # Run the forecast
            results = run_forecast(
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d'),
                sequence_length,
                epochs,
                batch_size
            )
            
            # Display results
            st.success(f"Forecast completed! Mean Squared Error: {results['mse']:.2f}")
            
            # Display original time series
            st.subheader("Original Time Series")
            st.pyplot(results['original_ts_fig'])
            
            # Display encoded images
            st.subheader("Sample Encoded GAF Images")
            st.pyplot(results['encoded_images_fig'])
            
            # Display training history
            st.subheader("Model Training History")
            st.pyplot(results['training_history_fig'])
            
            # Display predictions vs actual
            st.subheader("Predictions vs Actual")
            st.pyplot(results['predictions_fig'])
            
            # Display sample of predictions
            st.subheader("Sample Predictions")
            results_df = pd.DataFrame({'Actual': results['y_test'], 'Predicted': results['y_pred']})
            st.write(results_df.head(10))

if __name__ == "__main__":
    run_streamlit_app()
