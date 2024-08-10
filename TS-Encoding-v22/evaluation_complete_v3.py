import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Evaluate the predictions using various metrics.
    
    Args:
    y_true (np.ndarray): True values
    y_pred (np.ndarray): Predicted values
    
    Returns:
    Dict[str, float]: A dictionary containing the evaluation metrics
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "MAPE": mape
    }

def print_evaluation_results(metrics: Dict[str, float]) -> None:
    """
    Print the evaluation results in a formatted manner.
    
    Args:
    metrics (Dict[str, float]): Dictionary containing the evaluation metrics
    """
    print("Evaluation Results:")
    print(f"Mean Squared Error (MSE): {metrics['MSE']:.4f}")
    print(f"Root Mean Squared Error (RMSE): {metrics['RMSE']:.4f}")
    print(f"Mean Absolute Error (MAE): {metrics['MAE']:.4f}")
    print(f"R-squared (R2) Score: {metrics['R2']:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {metrics['MAPE']:.2f}%")

def compare_models(model_metrics: Dict[str, Dict[str, float]]) -> None:
    """
    Compare the performance of different models.
    
    Args:
    model_metrics (Dict[str, Dict[str, float]]): Dictionary containing metrics for each model
    """
    print("Model Comparison:")
    models = list(model_metrics.keys())
    metrics = list(next(iter(model_metrics.values())).keys())
    
    for metric in metrics:
        print(f"\n{metric}:")
        for model in models:
            print(f"  {model}: {model_metrics[model][metric]:.4f}")

if __name__ == "__main__":
    # Example usage
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 2.2, 2.9, 4.2, 5.1])
    
    metrics = evaluate_predictions(y_true, y_pred)
    print_evaluation_results(metrics)
    
    # Example of comparing multiple models
    model_metrics = {
        "Model A": evaluate_predictions(y_true, y_pred),
        "Model B": evaluate_predictions(y_true, y_pred * 1.1)  # Simulating a different model
    }
    compare_models(model_metrics)
