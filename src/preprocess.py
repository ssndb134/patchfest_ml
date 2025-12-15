import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import MinMaxScaler
import os

# Setup module-level logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def correct_outliers(df, target_col='count', window=7, z_thresh=3.0):
    """
    Corrects outliers by replacing extreme values with the local median (±3 days).
    
    Parameters:
    - df: DataFrame containing the time series.
    - target_col: The column to check for outliers.
    - window: Size of the rolling window (default 7 for ±3 days).
    - z_thresh: Threshold for Z-score to consider a point an outlier.
    
    Returns:
    - df_corrected: DataFrame with outliers replaced.
    """
    if target_col not in df.columns:
        logger.warning(f"Column {target_col} not found. Skipping outlier correction.")
        return df

    # Work on a copy
    df_corrected = df.copy()
    
    # Ensure sorted by date if possible
    if 'date' in df_corrected.columns:
        df_corrected['date'] = pd.to_datetime(df_corrected['date'])
        df_corrected = df_corrected.sort_values('date').reset_index(drop=True)
    
    series = df_corrected[target_col]
    
    # Calculate local median (robust trend)
    # min_periods=1 ensures we get values at edges
    rolling_median = series.rolling(window=window, center=True, min_periods=1).median()
    
    # Calculate local standard deviation (resid variance)
    # Estimate sigma usually via MAD to be robust, or just rolling std
    rolling_std = series.rolling(window=window, center=True, min_periods=1).std()
    
    # Handle zero std (flat lines)
    rolling_std = rolling_std.replace(0, 1e-6) # Avoid division by zero
    
    # Z-score based on local neighborhood
    z_scores = (series - rolling_median).abs() / rolling_std
    
    # Identify outliers
    outlier_mask = z_scores > z_thresh
    
    # Log and Replace
    outliers_idx = df_corrected.index[outlier_mask]
    
    if len(outliers_idx) > 0:
        logger.info(f"detected {len(outliers_idx)} outliers in '{target_col}'. Correcting...")
        
        # Log details
        log_file = "logs/outlier_correction.log"
        os.makedirs("logs", exist_ok=True)
        with open(log_file, 'a') as f:
            f.write(f"--- Outlier Correction Run: {pd.Timestamp.now()} ---\n")
            f.write(f"Index, Date, Original, corrected(Median)\n")
            
            for idx in outliers_idx:
                date_val = df_corrected.loc[idx, 'date'] if 'date' in df_corrected.columns else idx
                orig_val = df_corrected.loc[idx, target_col]
                new_val = rolling_median.loc[idx]
                
                # Replace
                df_corrected.loc[idx, target_col] = new_val
                
                # Log
                log_msg = f"{idx}, {date_val}, {orig_val:.2f}, {new_val:.2f}"
                f.write(log_msg + "\n")
                logger.info(f"Corrected: {log_msg}")
    else:
        logger.info(f"No outliers detected in '{target_col}' given threshold {z_thresh}.")

    return df_corrected

def fill_missing(df):
    """Fills missing values with median for numeric and mode for categorical."""
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        else:
            if df[col].isnull().any():
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df[col] = df[col].fillna(mode_val[0])
    return df

def normalize(df, cols):
    """MinMax normalization."""
    scaler = MinMaxScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df, scaler

if __name__ == "__main__":
    # Test/Demo
    data = {
        'date': pd.date_range(start='2023-01-01', periods=20),
        'count': [100, 102, 98, 105, 500, 101, 99, 103, 100, 97, 1000, 102, 101, 99, 98, 20, 100, 101, 99, 102]
        # 500 is outlier, 1000 is outlier, 20 maybe outlier
    }
    df_test = pd.DataFrame(data)
    print("Original Data (Subset):")
    print(df_test['count'].values)
    
    df_clean = correct_outliers(df_test, target_col='count', window=7, z_thresh=2.0)
    
    print("\nCorrected Data (Subset):")
    print(df_clean['count'].values)
