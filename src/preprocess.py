import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List

def identify_outliers_iqr(series: pd.Series) -> pd.Series:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 3.0 * iqr
    upper = q3 + 3.0 * iqr
    return (series < lower) | (series > upper)

def correct_outliers_local_median(
    df: pd.DataFrame,
    column: str,
    window_size: int = 7
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    df_corrected = df.copy()
    rolling_median = (
        df_corrected[column]
        .rolling(window=window_size, center=True, min_periods=1)
        .median()
    )

    outliers = identify_outliers_iqr(df_corrected[column])
    log = []

    for idx, flag in outliers.items():
        if flag:
            original = df_corrected.loc[idx, column]
            corrected = rolling_median.loc[idx]
            df_corrected.loc[idx, column] = corrected

            log.append({
                "timestamp": str(idx.date()),
                "original_value": round(original, 2),
                "corrected_value": round(corrected, 2),
                "change": round(corrected - original, 2)
            })

    return df_corrected, log

if __name__ == '__main__':
    print("Starting outlier correction simulation...")

    days = 50
    dates = pd.date_range(start='2025-01-01', periods=days, freq='D')

    baseline = 100 + 5 * np.sin(np.linspace(0, 10 * np.pi, days))
    noise = np.random.normal(0, 2, days)
    values = baseline + noise

    values[15] = 250
    values[35] = 10

    df = pd.DataFrame({'target': values}, index=dates)

    df_clean, changes = correct_outliers_local_median(df, 'target')

    print("\n--- Corrected Outlier Log ---")
    if changes:
        for c in changes:
            print(
                f"Corrected {c['timestamp']}: "
                f"{c['original_value']} -> {c['corrected_value']} "
                f"(Change: {c['change']})"
            )
    else:
        print("No outliers detected.")

    print("\nOriginal Data Summary:")
    print(df['target'].describe())
    print("\nCleaned Data Summary:")
    print(df_clean['target'].describe())
