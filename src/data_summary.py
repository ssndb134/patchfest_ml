import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/processed/structured_sample.csv")
REPORT_PATH = Path("reports/data_summary.txt")

def generate_data_summary():
    # checking if dataset exists
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    # loading dataset
    df = pd.read_csv(DATA_PATH)

    row_count = df.shape[0] #setting row count and column count
    column_count = df.shape[1]

    missing_values = df.isnull().sum()

    #for counting number of unique movie names
    if "movie_name" not in df.columns:
        raise KeyError("Column 'movie_name' not found in dataset")

    unique_movies = df["movie_name"].nunique()

    #generating report
    report = []
    report.append("DATASET SUMMARY REPORT\n")
    report.append(f"Row count: {row_count}")
    report.append(f"Column count: {column_count}\n")

    report.append("Missing values per column:")
    for col, count in missing_values.items():
        report.append(f"  {col}: {count}")

    report.append(f"\nUnique movie names count: {unique_movies}")

    #making sure report directory exists
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    #writing report
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print("✅ Data summary report generated at reports/data_summary.txt")

if __name__ == "__main__":
    generate_data_summary()
