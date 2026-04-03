from src import data_collection, preprocessing, feature_engineering, analysis, visualization

def main():
    # Stage 1: data collection
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    time_period = '5y'  # ou o período que preferir

    # Raw data download
    data_collection.download_raw_data(tickers, time_period)

    # Stage 2: Preprocessing + Feature Engineering
    df = preprocessing.preprocess_raw_data()

    # Stage 3: Analyses (aggregated calculations)
    df_accum_return = analysis.get_total_return(df)
    df_accum_yearly = analysis.get_accumulated_return_yearly(df)
    df_volatility = analysis.get_volatility(df)
    df_annual_volatility = analysis.get_annualized_volatility(df)
    df_vol_summary = analysis.get_volatility_summary(df)
    df_drawdown = feature_engineering.add_drawdown(df)
    df_max_drawdown = analysis.get_max_drawdown(df)

    print(df_accum_yearly)

    # Stage 4: Visualization
    visualization.plot_accumulated_return(df_accum_return)
    visualization.plot_accumulated_return_yearly(df_accum_yearly)
    visualization.plot_price_evolution(df)
    visualization.plot_return_distribution(df)
    visualization.plot_volatility(df_volatility)
    visualization.plot_annualized_volatility(df_annual_volatility)
    visualization.plot_rolling_volatility(df)
    visualization.plot_volatility_summary(df_vol_summary)
    visualization.plot_asset_correlation(df)
    visualization.plot_drawdown(df_drawdown)
    visualization.plot_max_drawdown(df_max_drawdown)

if __name__ == "__main__":
    main()