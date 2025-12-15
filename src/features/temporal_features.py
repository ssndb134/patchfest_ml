import pandas as pd
import numpy as np
import json
import os

HOLIDAYS_CONFIG_PATH = 'config/holidays.json'

def load_holidays(config_path: str) -> set:
    try:
        with open(config_path, 'r') as f:
            holiday_data = json.load(f)
            return set(holiday_data.keys())
    except FileNotFoundError:
        print(f"WARNING: Holiday config not found at {config_path}. No holidays will be flagged.")
        return set()

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df_features = df.copy()
    if not isinstance(df_features.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a Pandas DatetimeIndex.")
    temporal_cols = ['weekday', 'is_weekend', 'month', 'week_number', 'is_holiday']
    df_features = df_features.drop(columns=temporal_cols, errors='ignore')
    index = df_features.index
    df_features['weekday'] = index.dayofweek
    df_features['is_weekend'] = (index.dayofweek >= 5).astype(int)
    df_features['month'] = index.month
    df_features['week_number'] = index.isocalendar().week.astype(int)
    holiday_dates = load_holidays(HOLIDAYS_CONFIG_PATH)
    df_features['is_holiday'] = df_features.index.strftime('%Y-%m-%d').isin(holiday_dates).astype(int)
    return df_features

def run_temporal_feature_test():
    dates = pd.to_datetime([
        '2025-01-01',
        '2025-01-04',
        '2025-01-06',
        '2025-12-25',
        '2025-12-31',
        '2026-01-01'
    ])
    df = pd.DataFrame({'target': np.random.randint(100, 500, len(dates))}, index=dates)
    df_final = add_temporal_features(df)
    print("--- Temporal Feature Engineering Results ---")
    print(df_final[['target', 'weekday', 'is_weekend', 'month', 'week_number', 'is_holiday']].to_markdown())

if __name__ == '__main__':
    run_temporal_feature_test()
