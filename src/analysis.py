import pandas as pd
import numpy as np
from src.feature_engineering import add_drawdown

def get_total_return(df: pd.DataFrame) -> pd.DataFrame:

    df_sorted = df.sort_values(['Ticker', 'Date'])
    
    df_total_ret = (
        df_sorted
        .groupby('Ticker')['Accum_Return']
        .last()
        .reset_index(name='Total Accumulated Return')
    )

    return df_total_ret

def get_accumulated_return_yearly(df: pd.DataFrame) -> pd.DataFrame:

    df_accum_ret_year = df.copy()

    df_accum_ret_yearly = (
    df_accum_ret_year.groupby(['Ticker', 'Year'])['Return']
    .apply(lambda x: (1 + x).prod() -1)
    .reset_index(name='Accum_Return')
    )

    return df_accum_ret_yearly

def get_volatility(df: pd.DataFrame) -> pd.DataFrame:

    volatility = (
        df.groupby('Ticker')['Return'].std().reset_index(name='Volatility')
    )

    return volatility

def get_annualized_volatility(df:pd.DataFrame) -> pd.DataFrame:

    df_annual_vol = get_volatility(df)
    df_annual_vol['Annualized_Volatility'] = df_annual_vol['Volatility'] * np.sqrt(252)

    return df_annual_vol

def get_volatility_summary(df: pd.DataFrame) -> pd.DataFrame:

    summary = (
    df.groupby('Ticker')['Return']
    .agg(['mean','std'])
    .reset_index()
    )

    summary['Vol_Annual'] = summary['std'] * np.sqrt(252)
    summary['Return_Annual'] = summary['mean'] * 252

    return summary

def get_max_drawdown(df: pd.DataFrame) -> pd.DataFrame:

    df_max_drawdown = add_drawdown(df)

    max_dd = (df_max_drawdown.groupby('Ticker')['Drawdown']
    .min()
    .reset_index(name='Max_Drawdown')
    )

    max_dd['Max_Drawdown'] = max_dd['Max_Drawdown'].abs() * 100

    return max_dd

def get_monthly_close_price(df: pd.DataFrame) -> pd.DataFrame:

    df_sorted = df.sort_values(['Ticker','Date'])
    df_sorted['Date'] = pd.to_datetime(df_sorted['Date'])

    df_monthly_close_price = (
        df_sorted.set_index('Date').groupby('Ticker')['Close'].resample('ME').last().reset_index()
    )

    return df_monthly_close_price