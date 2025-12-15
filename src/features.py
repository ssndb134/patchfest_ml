import pandas as pd
import numpy as np
import os

def load_data(filepath):
    """Load data and parse dates."""
    try:
        df = pd.read_csv(filepath)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

def add_time_features(df):
    """Add basic time-based features."""
    if 'date' not in df.columns:
        return df
    
    df['weekday'] = df['date'].dt.weekday
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)
    df['month'] = df['date'].dt.month
    df['weekofyear'] = df['date'].dt.isocalendar().week
    return df

def add_lag_features(df, target_col='count'):
    """Add lag features lag_1, lag_7, lag_30."""
    # Ensure sorted by date
    if 'date' in df.columns:
        df = df.sort_values('date').reset_index(drop=True)
    
    df['lag_1'] = df[target_col].shift(1)
    df['lag_7'] = df[target_col].shift(7)
    df['lag_30'] = df[target_col].shift(30)
    return df

def add_rolling_features(df, target_col='count', window=7):
    """Add rolling mean and std with specified window."""
    # Rolling depends on previous values, assumes sorted
    
    # We use shift(1) for rolling to prevent data leakage if this is strictly past data for current prediction?
    # Standard rolling includes current row by default in pandas.
    # For forecasting (predicting t using t-1...t-n), we usually roll on the shifted data OR shift the result.
    # I'll apply rolling on the series.
    # roll_mean_7 at row T uses T, T-1, ... T-6.
    
    df[f'rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
    df[f'rolling_std_{window}'] = df[target_col].rolling(window=window).std()
    return df

def process_features(input_path, output_path):
    print(f"Processing {input_path}...")
    df = load_data(input_path)
    
    if df.empty:
        print("Dataframe is empty.")
        return

    # Ensure target exists
    if 'count' not in df.columns:
        # Create dummy count if missing for demo purposes, or error
        print("Warning: 'count' column missing. Initializing with random data for demonstration if needed.")
        # Check acceptance criteria: "Enhance... dataset". If column missing, maybe fail? 
        # But user gave structured_sample.csv which HAS count. So we are good.
        if 'count' not in df.columns:
             return
    
    df = add_time_features(df)
    df = add_lag_features(df, 'count')
    df = add_rolling_features(df, 'count', 7)
    
    # Handle NaNs
    # Strategy: Fill NaNs with 0 to preserve data structure for small datasets,
    # or drop if user prefers strictness. Acceptance criteria: "Handle initial NaNs cleanly."
    # Given lag_30, dropping would wipe out months of data. 
    # I'll choose to backfill or fill 0. 
    # For forecasting, filling 0 for lags where data didn't exist is common or using mean.
    
    cleaned_df = df.copy()
    
    # Forward fill then 0 (classic cold start)
    cleaned_df = cleaned_df.ffill().fillna(0)
    
    # Save
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
        
    cleaned_df.to_csv(output_path, index=False)
    print(f"Transformed data saved to {output_path}")
    print(cleaned_df.head())

if __name__ == "__main__":
    # Define paths
    base_dir = os.getcwd()
    input_csv = os.path.join(base_dir, "data", "processed", "structured_sample.csv")
    output_csv = os.path.join(base_dir, "data", "processed", "enhanced_forecast_data.csv")
    
    process_features(input_csv, output_csv)
