import pandas as pd
from preprocess import detect_zscore_outliers

def main():
    df = pd.read_csv("data/processed/structured_sample.csv")
    detect_zscore_outliers(df)

if __name__ == "__main__":
    main()
