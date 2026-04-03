import pandas as pd
import numpy as np


def add_daily_return(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds the colum Return with the daily return by asset
    """
    df['Return'] = df.groupby('Ticker')['Close'].pct_change()

    return df

def add_moving_averages_volatility(df: pd.DataFrame, short_window: int = 10, long_window: int = 50) -> pd.DataFrame:
    """
    Adds two moving averages columns and a volatility column
    """
    for ticker, group in df.groupby('Ticker'):
        idx = group.index
        df.loc[idx, f'MA_{short_window}'] = group['Close'].rolling(short_window).mean()
        df.loc[idx, f'MA_{long_window}'] = group['Close'].rolling(long_window).mean()
        df.loc[idx, f'Vol_{short_window}'] = group['Return'].rolling(short_window).std()
    
    return df

def add_accumulated_return(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds the column Accum_Return with the accumulated return
    """
    df = df.sort_values(['Ticker', 'Date'])
    df['Accum_Return'] = (
        df
        .groupby('Ticker')['Return']
        .transform(lambda x: (1 + x).cumprod() -1)
    )

    return df

def add_drawdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds the column Drawdown
    """
    df = df.sort_values(['Ticker', 'Date'])
    df['Date'] = pd.to_datetime(df['Date'])

    df['Peak'] = (df.groupby('Ticker')['Accum_Return'].transform(lambda x: x.cummax()))

    df['Drawdown'] = (
        (df['Accum_Return'] - df['Peak']) /
        (1 + df['Peak'])
    )

    return df

def add_lag_features(df: pd.DataFrame, lags: list = [1, 2, 3]) -> pd.DataFrame:
    """
    Adds 3 Lag columns
    """
    df = df.sort_values(['Ticker', 'Date'])

    for lag in lags:
        df[f'Return_Lag_{lag}'] = df.groupby('Ticker')['Return'].shift(lag)

    return df