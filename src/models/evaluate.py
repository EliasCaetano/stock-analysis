import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_model(y_true, y_pred):
    """
    Calculates performance metrics
    """

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print("\nModel Evaluation:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }

def plot_predictions(y_true, y_pred):
    """
    Real X Predicted Plot
    """

    plt.figure(figsize=(12,6))

    plt.plot(y_true.values, label='Real')
    plt.plot(y_pred, label='Predict')

    plt.title('Real vs Predicted')
    plt.xlabel('Observations')
    plt.ylabel('Price')

    plt.legend()
    plt.grid(True)

    plt.show()

def plot_feature_importance(feature_importance):
    """
    Plots feature importance
    """

    plt.figure(figsize=(10,6))

    plt.barh(
        feature_importance['Feature'],
        feature_importance['Importance']
    )

    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Features')

    plt.gca().invert_yaxis()

    plt.grid(True)

    plt.show()

def create_results_df(y_true, y_pred):
    """
    DataFrame comparing values
    """

    results = pd.DataFrame({
        'Actual': y_true.values,
        'Predicted': y_pred
    })

    return results

def compare_with_baseline(y_true, y_pred, y_baseline):

    """
    Compares the model with the baseline
    """

    def metrics(y_true, y_hat):
        rmse = np.sqrt(mean_squared_error(y_true, y_hat))
        mae = mean_absolute_error(y_true, y_hat)

        return rmse, mae
    
    model_rmse, model_mae = metrics(y_true, y_pred)
    base_rmse, base_mae = metrics(y_true, y_baseline)

    print("\nComparison with Baseline:")
    print(f"Model  -> RMSE: {model_rmse:.4f} | MAE: {model_mae:.4f}")
    print(f"Baseline-> RMSE: {base_rmse:.4f} | MAE: {base_mae:.4f}")

def plot_model_vs_baseline(y_true, y_pred, y_baseline):
    """
    Creates a bar plot to compare model vs baseline
    """
    plt.figure(figsize=(12,6))

    plt.plot(y_true.values, label='Real')
    plt.plot(y_pred, label='Model')
    plt.plot(y_baseline.values, label='Baseline', linestyle='--')

    plt.title('Modelo vs Baseline')
    plt.xlabel('Observations')
    plt.ylabel('Price')

    plt.legend()
    plt.grid(True)

    plt.show()