import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def fill_missing(df):
    summary = {}
    for col in df.columns:
        mb = df[col].isna().sum()
        if df[col].dtype!='object':
            df[col]=df[col].fillna(df[col].median())
        else:
            df[col]=df[col].fillna(df[col].mode()[0])
        if mb>0:
            summary[col] = mb

    if summary: 
        print ("Summary:")
        for col, count in summary.items():
            print(f"{col}: {count} values filled")
            
    return df

def normalize(df, cols):
    scaler=MinMaxScaler()
    df[cols]=scaler.fit_transform(df[cols])
    return df, scaler

def detect_duplicates(df):
    dup=df.duplicated()
    return df[dup]
