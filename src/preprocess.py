import pandas as pd
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
    dup=df.duplicated()
    return df[dup]

import pickle
from pathlib import Path

ARTIFACTS_DIR = Path("artifacts")
SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"

def normalize_features(df, feature_cols, target_col):

    #ensuring that target is not normalised
    feature_cols = [col for col in feature_cols if col != target_col]

    #applying normalisation
    df, scaler = normalize(df, feature_cols)

    #creating artifacts directory if not exists
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    #printing transformed values
    print("✅ Feature range after MinMax scaling:")
    print(f"Min: {df[feature_cols].min().min():.4f}")
    print(f"Max: {df[feature_cols].max().max():.4f}")

    return df
