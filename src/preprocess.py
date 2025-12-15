import pandas as pd
import os

from sklearn.preprocessing import MinMaxScaler

def fill_missing(df):
    for col in df.columns:
        if df[col].dtype!='object':
            df[col]=df[col].fillna(df[col].median())
        else:
            df[col]=df[col].fillna(df[col].mode()[0])
    return df

def normalize(df, cols):
    scaler=MinMaxScaler()
    df[cols]=scaler.fit_transform(df[cols])
    return df, scaler

def detect_duplicates(df):
    key_cols = ["movie_name", "date", "time", "seat", "screen"]
    duplicates = df[df.duplicated(subset=key_cols, keep=False)]
    os.makedirs("reports", exist_ok=True)
    duplicates.to_csv("reports/duplicates.csv", index=False)
    print(f"Duplicate tickets: {len(duplicates)}")
    return duplicates
