import yfinance as yf
import pandas as pd
import numpy as np
import os

def download_raw_data(tickers: list[str], time_period: str, folder='data/raw', filename='raw_data.csv'):

    data = yf.download(tickers, period=time_period, group_by='ticker', threads=True)

    # Builiding df long

    ticker_data = []

    for t in tickers:
        df_t = data[t].copy()

        # Expected columns
        expected = ['Open', 'High', 'Low', 'Close', 'Volume']

        # Add NaN if missing
        for c in expected:
            if c not in df_t.columns:
                df_t[c] = np.nan
        
        df_t = df_t[expected]
        df_t = df_t.reset_index().rename(columns={'index': 'Date'})
        df_t['Ticker'] = t
        ticker_data.append(df_t)

    if not ticker_data:
        raise RuntimeError('No valid data was downloaded for the provided tickers')

    df_long = pd.concat(ticker_data, ignore_index=True)

    # Order and adjust types

    df_long['Date'] = pd.to_datetime(df_long['Date'])
    df_long = df_long.sort_values(['Ticker','Date']).reset_index(drop=True)

    # Setting output

    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, filename)
    df_long.to_csv(file_path, index=False)

    # Final Check

    print(df_long.head())
    print(df_long.dtypes)
    print('Raw dataset successfully created!')
