import pandas as pd
import os
from src.feature_engineering import add_daily_return, add_moving_averages_volatility, add_accumulated_return, add_lag_features

def load_data(df_path: str):
   """
   Loads the dataframe
   """
   data = pd.read_csv(df_path)
   df = pd.DataFrame(data)
   return df

def load_raw_data(path: str = 'data/raw/raw_data.csv') -> pd.DataFrame:
    """
    Loads the raw data from path
    """
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"The file {path} doesn't exist.")
    
    df = pd.read_csv(path)
    
    # Adjusting types
    df['Date'] = pd.to_datetime(df['Date'])
    df['Ticker'] = df['Ticker'].astype(str)
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def handle_missing_data(df: pd.DataFrame, method: str = 'drop') -> pd.DataFrame:
    """
    Handles missing data by dropping them as default
    """
    df_processed = df.copy()
    price_cols = ['Open', 'High', 'Low', 'Close', 'Return', 'MA_10', 'MA_50', 'Vol_10']

    if method == 'drop':
        df_processed = df_processed.dropna(subset=price_cols)
    elif method == "ffill":
        df_processed[price_cols + ['Volume']] = df_processed[price_cols + ['Volume']].ffill()
    
    return df_processed

def preprocess_raw_data(raw_path = 'data/raw/raw_data.csv', processed_folder = 'data/processed', missing_method = 'drop') -> pd.DataFrame:
        
    # Loading dataset
    df = load_raw_data()

    # Order by ticker and date
    df = df.sort_values(['Ticker','Date']).reset_index(drop=True)

    # Add daily return
    df = add_daily_return(df)

    # Add MA and volatility
    df = add_moving_averages_volatility(df)

    # Add lag features
    df = add_lag_features(df)

    # Handling missing data
    df = handle_missing_data(df, method='drop')

    # Updating Date column dtype
    df['Date'] = pd.to_datetime(df['Date'])

    # Adding 'Year' column
    df['Year'] = df['Date'].dt.year

    # Add Accumulated return
    df = add_accumulated_return(df)

    # Creating outfile folder
    os.makedirs(processed_folder, exist_ok=True)
    processed_path = os.path.join(processed_folder, 'processed_data.csv')

    # Saving processed dataset
    df.to_csv(processed_path, index=False)

    # Quick validation
    print(df.head())
    print(df.info())
    print(f'Processed dataset saved at {processed_path}')
    
    return df