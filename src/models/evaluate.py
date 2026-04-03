import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

PLOT_DIR = Path(__file__).resolve().parents[2] / "model_plots"


def _save_plot(filename=None):
    """
    Saves the current plot in the model_plots folder when a filename is provided.
    """

    if not filename:
        return

    PLOT_DIR.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=300, bbox_inches="tight")


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


def plot_predictions(y_true, y_pred, filename=None):
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

    _save_plot(filename)
    plt.show()


def plot_feature_importance(feature_importance, filename=None):
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

    _save_plot(filename)
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


def plot_model_vs_baseline(y_true, y_pred, y_baseline, filename=None):
    """
    Creates a bar plot to compare model vs baseline
    """
    plt.figure(figsize=(12,6))

    plt.plot(y_true.values, label='Real')
    plt.plot(y_pred, label='Model')
    plt.plot(y_baseline.values, label='Baseline', linestyle='--')

    plt.title('Model vs Baseline')
    plt.xlabel('Observations')
    plt.ylabel('Price')

    plt.legend()
    plt.grid(True)

    _save_plot(filename)
    plt.show()


def plot_model_performance_comparison(y_true, rf_pred, gb_pred, y_baseline, filename=None):
    """
    Plots a comparison of performance metrics for Random Forest,
    Gradient Boosting and Baseline.
    """

    model_names = ['Random Forest', 'Gradient Boosting', 'Baseline']
    predictions = [rf_pred, gb_pred, y_baseline]

    metrics_df = pd.DataFrame([
        {
            'Model': model_name,
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred)
        }
        for model_name, y_pred in zip(model_names, predictions)
    ])

    x = np.arange(len(model_names))
    width = 0.25

    plt.figure(figsize=(12, 6))
    plt.bar(x - width, metrics_df['RMSE'], width=width, label='RMSE')
    plt.bar(x, metrics_df['MAE'], width=width, label='MAE')
    plt.bar(x + width, metrics_df['R2'], width=width, label='R2')

    plt.title('Model Performance Comparison')
    plt.xlabel('Model')
    plt.ylabel('Metric Value')
    plt.xticks(x, model_names, rotation=10)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.4)

    _save_plot(filename)
    plt.show()


def compare_models(y_test, rf_pred, gb_pred, y_baseline):
    print("\nModel Comparison:")

    print("\nRandom Forest:")
    evaluate_model(y_test, rf_pred)

    print("\nGradient Boosting:")
    evaluate_model(y_test, gb_pred)

    print("\nBaseline:")
    evaluate_model(y_test, y_baseline)
