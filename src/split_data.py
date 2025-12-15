import pandas as pd
import numpy as np
import os
from typing import Dict

def create_time_aware_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> Dict[str, pd.DataFrame]:
    total_len = len(df)

    train_end = int(total_len * train_ratio)
    val_end = train_end + int(total_len * val_ratio)

    return {
        "train": df.iloc[:train_end],
        "val": df.iloc[train_end:val_end],
        "test": df.iloc[val_end:]
    }

def print_split_summary(splits: Dict[str, pd.DataFrame]) -> None:
    print("\n--- Split Summary (Date Ranges) ---")

    for name, split_df in splits.items():
        if split_df.empty:
            print(f"{name.capitalize()} Set: Empty")
            continue

        start = split_df.index.min().strftime('%Y-%m-%d')
        end = split_df.index.max().strftime('%Y-%m-%d')

        print(f"{name.capitalize()} Set:")
        print(f"  Rows: {len(split_df)}")
        print(f"  Range: {start} to {end}")

if __name__ == '__main__':
    print("Starting time-aware data splitting...")

    days = 365
    dates = pd.date_range(start='2025-01-01', periods=days, freq='D')

    df_raw = pd.DataFrame(
        {
            'target': np.random.randint(100, 500, size=days),
            'feature_lag_1': np.random.rand(days)
        },
        index=dates
    )

    splits = create_time_aware_split(df_raw, train_ratio=0.7, val_ratio=0.15)
    print_split_summary(splits)

    output_dir = 'data/processed'
    os.makedirs(output_dir, exist_ok=True)

    for name, split_df in splits.items():
        path = os.path.join(output_dir, f"{name}_set.csv")
        split_df.to_csv(path)
        print(f"Saved {name} set to {path}")
