from src.preprocessing import preprocess_raw_data
from src.models.train import (
    train_model_random_forest,
    train_model_gradient_boosting,
    get_feature_importance
)
from src.models.evaluate import (
    plot_predictions,
    evaluate_model,
    plot_feature_importance,
    compare_with_baseline,
    plot_model_vs_baseline,
    compare_models,
    plot_model_performance_comparison
)


def main():
    '''
    Runs the model training and comparison
    '''

    # Load and preprocess data
    df = preprocess_raw_data()

    # Random Forest
    print("\nTraining Random Forest...")
    rf_model, X_test, X_train, y_test, rf_pred, y_baseline = train_model_random_forest(df)

    print("\nRandom Forest Evaluation:")
    evaluate_model(y_test, rf_pred)

    plot_predictions(y_test, rf_pred, filename="random_forest_predictions.png")

    rf_feature_importance = get_feature_importance(rf_model, X_train.columns)

    print("\nFeature Importance (Random Forest):")
    print(rf_feature_importance)

    plot_feature_importance(
        rf_feature_importance,
        filename="random_forest_feature_importance.png"
    )

    compare_with_baseline(y_test, rf_pred, y_baseline)

    plot_model_vs_baseline(
        y_test,
        rf_pred,
        y_baseline,
        filename="random_forest_vs_baseline.png"
    )

    # Gradient Boosting
    print("\nTraining Gradient Boosting...")
    gb_model, _, _, _, gb_pred, _ = train_model_gradient_boosting(df)

    print("\nGradient Boosting Evaluation:")
    evaluate_model(y_test, gb_pred)

    plot_predictions(y_test, gb_pred, filename="gradient_boosting_predictions.png")

    gb_feature_importance = get_feature_importance(gb_model, X_train.columns)

    print("\nFeature Importance (Gradient Boosting):")
    print(gb_feature_importance)

    plot_feature_importance(
        gb_feature_importance,
        filename="gradient_boosting_feature_importance.png"
    )

    compare_with_baseline(y_test, gb_pred, y_baseline)

    plot_model_vs_baseline(
        y_test,
        gb_pred,
        y_baseline,
        filename="gradient_boosting_vs_baseline.png"
    )

    #Final Comparison
    print("\nFinal Comparison Summary:")
    print("Random Forest vs Gradient Boosting vs Baseline")

    print("\nRandom Forest:")
    evaluate_model(y_test, rf_pred)

    print("\nGradient Boosting:")
    evaluate_model(y_test, gb_pred)

    print("\nBaseline:")
    evaluate_model(y_test, y_baseline)

    plot_model_performance_comparison(
        y_test,
        rf_pred,
        gb_pred,
        y_baseline,
        filename="model_performance_comparison.png"
    )


if __name__ == "__main__":
    main()
