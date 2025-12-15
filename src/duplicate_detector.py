import pandas as pd
import numpy as np
import os
from typing import List

DUPLICATE_FIELDS = [
    'movie_name', 
    'date', 
    'time', 
    'seat_number', 
    'screen_number'
]
REPORT_PATH = 'reports/duplicates.csv'

def detect_duplicates(df: pd.DataFrame, key_fields: List[str]) -> pd.DataFrame:
    duplicate_mask = df.duplicated(subset=key_fields, keep=False)
    duplicates_df = df[duplicate_mask].sort_values(by=key_fields)
    return duplicates_df

def run_duplicate_detection():
    print("Starting duplicate detection simulation...")
    data = [
        {'movie_name': 'Interstellar', 'date': '2025-12-20', 'time': '20:00', 'seat_number': 'A-01', 'screen_number': 5, 'id': 1},
        {'movie_name': 'Inception', 'date': '2025-12-20', 'time': '20:00', 'seat_number': 'B-02', 'screen_number': 5, 'id': 2},
        {'movie_name': 'Interstellar', 'date': '2025-12-20', 'time': '20:00', 'seat_number': 'A-01', 'screen_number': 5, 'id': 3},
        {'movie_name': 'Interstellar', 'date': '2025-12-20', 'time': '20:00', 'seat_number': 'C-03', 'screen_number': 5, 'id': 4},
        {'movie_name': 'Interstellar', 'date': '2025-12-20', 'time': '20:00', 'seat_number': 'C-03', 'screen_number': 5, 'id': 5},
        {'movie_name': 'Matrix', 'date': '2025-12-21', 'time': '10:00', 'seat_number': 'D-04', 'screen_number': 1, 'id': 6},
        {'movie_name': 'Interstellar', 'date': '2025-12-20', 'time': '22:00', 'seat_number': 'E-05', 'screen_number': 5, 'id': 7},
        {'movie_name': 'Avatar', 'date': '2025-12-22', 'time': '15:00', 'seat_number': 'F-06', 'screen_number': 3, 'id': 8},
    ]
    df = pd.DataFrame(data)
    duplicates_df = detect_duplicates(df, DUPLICATE_FIELDS)
    num_duplicates = len(duplicates_df)
    print(f"\n✅ Detection Complete: {num_duplicates} rows identified as duplicates.")
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    duplicates_df.to_csv(REPORT_PATH, index=False)
    print(f"Successfully saved duplicate rows to {REPORT_PATH}")
    print("\nSample Duplicates Detected:")
    print(duplicates_df[['movie_name', 'seat_number', 'id']].to_markdown(index=False))

if __name__ == '__main__':
    run_duplicate_detection()
