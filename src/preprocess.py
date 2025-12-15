import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path //importing path library

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

OUTLIER_PATH = Path("reports/outliers.csv")


def detect_zscore_outliers(df, threshold=3.0):
    #detects ticket count
    ticket_col = None
    for col in df.columns:
        if "ticket" in col.lower() or "count" in col.lower() or "quantity" in col.lower():
            ticket_col = col
            break

    if ticket_col is None:
        raise KeyError("No ticket count column found for Z-score detection")

    #converts to numerics
    df[ticket_col] = pd.to_numeric(df[ticket_col], errors="coerce")
    valid_df = df.dropna(subset=[ticket_col]).copy()

    mean = valid_df[ticket_col].mean()
    std = valid_df[ticket_col].std()

    if std == 0:
        raise ValueError("Standard deviation is zero; cannot compute Z-score")

    #calculating z-score
    valid_df["z_score"] = (valid_df[ticket_col] - mean) / std

    valid_df["is_outlier"] = valid_df["z_score"].abs() > threshold

    #defining outliers
    outliers = valid_df[valid_df["is_outlier"]].copy()

    #ensuring that directory exists
    OUTLIER_PATH.parent.mkdir(parents=True, exist_ok=True)

    outliers.to_csv(OUTLIER_PATH, index=False)

    return outliers
