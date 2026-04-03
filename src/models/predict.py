import joblib


def load_model(path='data/models/random_forest.pkl'):
    """
    Loads saved model
    """
    model = joblib.load(path)
    return model


def predict(model, X):
    """
    Predicts using the model
    """
    return model.predict(X)