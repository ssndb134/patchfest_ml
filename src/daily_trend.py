import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_PATH = Path("data/processed/structured_sample.csv")
PLOT_PATH = Path("plots/daily_trend.png")

def plot_daily_trend():
    #loading data
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    #detecting date column
    date_col = None
    for col in df.columns:
        if "date" in col.lower():
            date_col = col
            break

    if date_col is None:
        raise KeyError("No date column found in dataset")

    #parsing dates
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    #detecting ticket column
    ticket_col = None
    for col in df.columns:
        if "ticket" in col.lower() or "count" in col.lower() or "quantity" in col.lower():
            ticket_col = col
            break

    if ticket_col is None:
        raise KeyError("No ticket count column found in dataset")

    #sorting
    daily_counts = (
        df.groupby(date_col)[ticket_col]
        .sum()
        .reset_index()
        .sort_values(by=date_col)
    )

    #plotting the graph
    plt.figure(figsize=(10, 5))
    plt.plot(daily_counts[date_col], daily_counts[ticket_col])
    plt.xlabel("Date")
    plt.ylabel("Ticket Count")
    plt.title("Daily Ticket Booking Trend")
    plt.tight_layout()

    #ensuring existence
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)

    #saving the plot
    plt.savefig(PLOT_PATH)
    plt.close()

    print("✅ Daily trend plot saved to plots/daily_trend.png")

if __name__ == "__main__":
    plot_daily_trend()
