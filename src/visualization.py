import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_accumulated_return(df):

    plt.figure(figsize=(10,6))
    vis = sns.barplot(
    data=df,
    x='Ticker',
    y='Total Accumulated Return',
    hue='Ticker',
    palette='tab10'
    )

    for p in vis.patches:
        vis.annotate(
            format(p.get_height(), '.2f'),
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='bottom',
            fontsize=10
        )

    plt.title('Total Accumulated Return by Asset')
    plt.xlabel('Ticker')
    plt.ylabel('Accumulated Return')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

def plot_accumulated_return_yearly(df):

    plt.figure(figsize=(10,6))
    sns.lineplot(
        data=df,
        x='Year',
        y='Accum_Return',
        hue='Ticker'
    )
    plt.title('Accumulated Return by Year')
    plt.xlabel('Year')
    plt.ylabel('Accumulated Return')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_price_evolution(df):

    plt.figure(figsize=(10,6))
    sns.lineplot(
        data=df,
        x='Date',
        y='Close',
        hue='Ticker'
    )
    plt.title('Monthly Close Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

def plot_return_distribution(df):

    plt.figure(figsize=(10,6))
    sns.histplot(
        data=df['Return'],
        bins=100,
        kde=True
    )
    plt.title('Return Distribution')
    plt.show()

def plot_volatility(df):

    plt.figure(figsize=(10,6))
    vis = sns.barplot(
        data=df,
        x='Ticker',
        y='Volatility',
        hue='Ticker',
        palette='tab10'
    )

    for p in vis.patches:
        vis.annotate(
            format(p.get_height(), '.4f'),
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='bottom',
            fontsize=10
        )

    plt.title('Volatility by Asset')
    plt.ylabel('Volatility (Std of Returns)')
    plt.xlabel('Ticker')
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

def plot_annualized_volatility(df):

    plt.figure(figsize=(10,6))
    vis = sns.barplot(
    data=df,
    x='Ticker',
    y='Annualized_Volatility',
    hue='Ticker',
    palette='tab10'
    )

    for p in vis.patches:
        vis.annotate(
            format(p.get_height(), '.4f'),
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='bottom',
            fontsize=10
        )

    plt.title('Annualized Volatility by Asset')
    plt.ylabel('Volatility (Annualized)')
    plt.xlabel('Ticker')
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

def plot_rolling_volatility(df):

    df['Date'] = pd.to_datetime(df['Date'])
    plt.figure(figsize=(14,6))
    sns.lineplot(
        data=df,
        x='Date',
        y='Vol_10',
        hue='Ticker'
    )
    plt.ylabel('Volatility')
    plt.title('Rolling Volatility (10-day)')
    plt.show()

def plot_volatility_summary(df):

    plt.figure(figsize=(10,6))
    sns.scatterplot(
        data=df,
        x='Vol_Annual',
        y='Return_Annual',
        hue='Ticker',
        s=100
    )

    plt.title('Risk vs Return')
    plt.xlabel('Volatility (Risk)')
    plt.ylabel('Expected Return')
    plt.grid(True)
    plt.show()

def plot_asset_correlation(df):

    df_pivot = df.pivot(index='Date', columns='Ticker', values='Return')
    corr_matrix = df_pivot.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    plt.figure(figsize=(10,6))
    sns.heatmap(
    corr_matrix,
    annot=True,
    cmap='coolwarm',
    center=0,
    fmt='.2f',
    mask=mask
    )
    plt.title('Correlation Matrix of Asset Returns')
    plt.tight_layout()
    plt.show()

def plot_drawdown(df):

    plt.figure(figsize=(15,6))
    sns.lineplot(
        data=df,
        x='Date',
        y='Drawdown',
        hue='Ticker'
    )
    plt.show()

def plot_max_drawdown(df):

    plt.figure(figsize=(10,6))
    vis = sns.barplot(
    data=df,
    x='Ticker',
    y='Max_Drawdown',
    hue='Ticker',
    palette='tab10'
    )

    for p in vis.patches:
        vis.annotate(
        format(p.get_height(), '.4f'),
        (p.get_x() + p.get_width() / 2., p.get_height()),
        ha='center', va='bottom',
        fontsize=10,
    )

    plt.title('Max Drawdown by Asset')
    vis.set_ylabel('Max Drawdown (%)')
    vis.set_xlabel('Asset')
    plt.show()