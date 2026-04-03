from src.preprocessing import preprocess_raw_data
from src.models.train import train_model, get_feature_importance
from src.models.evaluate import plot_predictions, evaluate_model, plot_feature_importance, compare_with_baseline, plot_model_vs_baseline

def main():
    '''
    Runs the model's training
    '''

    df = preprocess_raw_data()

    model, X_test, X_train, y_test, y_pred, y_baseline = train_model(df)

    evaluate_model(y_test, y_pred)

    plot_predictions(y_test, y_pred)

    feature_names = X_train.columns

    feature_importance = get_feature_importance(model, feature_names)

    print(feature_importance)

    plot_feature_importance(feature_importance)

    compare_with_baseline(y_test, y_pred, y_baseline)

    plot_model_vs_baseline(y_test, y_pred, y_baseline)


if __name__ == "__main__":
    main()