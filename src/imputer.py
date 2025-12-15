import pandas as pd
import numpy as np

def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df_imputed = df.copy()
    imputation_summary = {}

    for col in df_imputed.columns:
        if df_imputed[col].isnull().sum() > 0:
            dtype = str(df_imputed[col].dtype)
            if 'float' in dtype or 'int' in dtype:
                fill_value = df_imputed[col].median()
                fill_method = 'median'
            elif 'object' in dtype or 'category' in dtype:
                fill_value = df_imputed[col].mode()[0]
                fill_method = 'mode'
            else:
                continue
            count_imputed = df_imputed[col].isnull().sum()
            df_imputed[col].fillna(fill_value, inplace=True)
            imputation_summary[col] = {
                'count': count_imputed,
                'method': fill_method,
                'value': fill_value
            }

    print("\n--- Imputation Summary ---")
    if imputation_summary:
        for col, summary in imputation_summary.items():
            print(f"Column '{col}': Imputed {summary['count']} values with {summary['method']} ({summary['value']}).")
    else:
        print("No missing values found for imputation.")

    return df_imputed

def run_imputer_test():
    data = {
        'target_count': [100, 120, np.nan, 150, 110, 140, np.nan],
        'price': [12.5, 10.0, 15.0, np.nan, 10.0, 12.5, 15.0],
        'movie_genre': ['Sci-Fi', 'Action', 'Action', 'Drama', 'Action', np.nan, 'Sci-Fi'],
        'screen_type': ['IMAX', 'Standard', np.nan, 'IMAX', 'Standard', 'Standard', 'Standard'],
        'date': pd.to_datetime(['2025-01-01', '2025-01-02', '2025-01-03', '2025-01-04', '2025-01-05', '2025-01-06', '2025-01-07'])
    }
    df = pd.DataFrame(data).set_index('date')
    print("--- Original DataFrame Info ---")
    print(df.isnull().sum())
    df_clean = impute_missing_values(df)
    print("\n--- Cleaned DataFrame Info ---")
    print(df_clean.isnull().sum())
    print("\nCleaned DataFrame Sample:")
    print(df_clean.to_markdown())

if __name__ == '__main__':
    run_imputer_test()
