import pandas as pd
import numpy as np
from typing import List

def add_lag_features(df: pd.DataFrame, column: str, lags: List[int]) -> pd.DataFrame:
    df_copy = df.copy()
    for lag in lags:
        df_copy[f"{column}_lag_{lag}"] = df_copy[column].shift(lag)
    return df_copy

def add_rolling_window_features(
    df: pd.DataFrame,
    column: str,
    window: int,
    aggregations: List[str] = ['mean', 'std']
) -> pd.DataFrame:
    df_copy = df.copy()
    rolling_series = df_copy[column].rolling(window=window, min_periods=1)

    for agg in aggregations:
        if agg == 'mean':
            df_copy[f"{column}_roll_mean_{window}"] = rolling_series.mean()
        elif agg == 'std':
            df_copy[f"{column}_roll_std_{window}"] = rolling_series.std()

    return df_copy

def run_feature_engineering(df: pd.DataFrame, target_column: str = 'target') -> pd.DataFrame:
    df_transformed = add_lag_features(df, target_column, [1, 7, 30])
    df_transformed = add_rolling_window_features(
        df_transformed,
        target_column,
        window=7,
        aggregations=['mean', 'std']
    )
    return df_transformed

if __name__ == '__main__':
    print("Starting feature engineering...")

    date_range = pd.date_range(start='2025-01-01', periods=40, freq='D')
    df = pd.DataFrame({
        'timestamp': date_range,
        'target': np.random.randint(100, 500, size=40)
    }).set_index('timestamp')

    df_final = run_feature_engineering(df, 'target')
    df_final.fillna(0, inplace=True)

    output_path = 'data/processed/features_df.csv'
    df_final.to_csv(output_path)

    print(f"Successfully saved transformed dataset to {output_path}")
    print(df_final.head(10).to_markdown())
