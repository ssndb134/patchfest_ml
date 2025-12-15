import pandas as pd
from preprocess import normalize_features

def main():
    df = pd.read_csv("data/processed/structured_sample.csv")

    feature_cols = ["price", "count"]   # example numeric features
    target_col = "count"                 # target should NOT be normalized

    normalize_features(df, feature_cols, target_col)

if __name__ == "__main__":
    main()
