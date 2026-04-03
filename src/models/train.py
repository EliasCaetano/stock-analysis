import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from src.models.utils import create_target, time_series_split, select_features, create_baseline


def train_model_random_forest(df):
    """
    Training Pipeline
    """

    # Creating Target
    df = create_target(df)

    # Removing NaNs
    df = df.dropna()

    # Time Split
    train_df, test_df = time_series_split(df)

    # Spliting Features
    X_train, y_train = select_features(train_df)
    X_test, y_test = select_features(test_df)

    # Adding simple baseline
    y_baseline = create_baseline(test_df)

    # Model
    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    # Train
    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)

    # Save model
    joblib.dump(model, 'src/models/random_forest.pkl')
    print("\nModel saved in src/models/random_forest.pkl")

    return model, X_test, X_train, y_test, y_pred, y_baseline


def train_model_gradient_boosting(df):
    """
    Training Pipeline - Gradient Boosting
    """

    # Creating Target
    df = create_target(df)

    # Removing NaNs
    df = df.dropna()

    # Time Split
    train_df, test_df = time_series_split(df)

    # Splitting Features
    X_train, y_train = select_features(train_df)
    X_test, y_test = select_features(test_df)

    # Adding simple baseline
    y_baseline = create_baseline(test_df)

    # Model
    model = GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )

    # Train
    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)

    # Save model
    joblib.dump(model, 'src/models/gradient_boosting.pkl')
    print("\nModel saved in src/models/gradient_boosting.pkl")

    return model, X_test, X_train, y_test, y_pred, y_baseline


def get_feature_importance(model, feature_names):
    """
    Features Importance
    """
    import pandas as pd

    importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    return importance