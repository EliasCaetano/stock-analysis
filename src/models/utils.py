import pandas as pd


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates the target (next day's return)
    """
    df = df.sort_values(['Ticker', 'Date'])
    df['Target'] = df.groupby('Ticker')['Return'].shift(-1)
    return df


def time_series_split(df: pd.DataFrame, split_ratio: float = 0.8):
    """
    Split the data into train and test sets based on time
    """
    df = df.sort_values('Date')

    split_index = int(len(df) * split_ratio)

    train = df.iloc[:split_index]
    test = df.iloc[split_index:]

    return train, test


def select_features(df: pd.DataFrame):
    """
    Selects features and target
    """
    features = [
        'Return',
        'MA_10',
        'MA_50',
        'Vol_10',
        'Return_Lag_1',
        'Return_Lag_2',
        'Return_Lag_3'
    ]

    X = df[features]
    y = df['Target']

    return X, y


def create_baseline(df, column='Return'):
    """
    Creates a baseline prediction (next day's price = today's price)
    """
    return df[column]