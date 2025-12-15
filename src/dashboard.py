import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

OUTPUT_DIR = 'plots/dashboard'

def create_hourly_heatmap(df: pd.DataFrame, output_path: str) -> None:
    df_plot = df.copy()
    df_plot['hour'] = df_plot.index.hour
    df_plot['day_of_week'] = df_plot.index.dayofweek

    heatmap_data = (
        df_plot
        .groupby(['day_of_week', 'hour'])['target']
        .mean()
        .unstack(level=1)
    )

    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    heatmap_data.index = day_names[:len(heatmap_data.index)]

    plt.figure(figsize=(12, 6))
    sns.heatmap(
        heatmap_data,
        cmap="YlGnBu",
        annot=True,
        fmt=".0f",
        linewidths=.5,
        cbar_kws={'label': 'Average Ticket Count'}
    )
    plt.title('Day × Hour Heatmap (Average Ticket Count)')
    plt.xlabel('Hour of Day')
    plt.ylabel('Day of Week')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def create_monthly_trend(df: pd.DataFrame, output_path: str) -> None:
    monthly_data = df['target'].resample('M').sum()

    plt.figure(figsize=(12, 6))
    monthly_data.plot(marker='o')
    plt.title('Monthly Ticket Count Trend')
    plt.xlabel('Date')
    plt.ylabel('Total Ticket Count')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def create_seasonal_curve(df: pd.DataFrame, output_path: str) -> None:
    df_plot = df.copy()
    df_plot['day_of_week'] = df_plot.index.dayofweek

    seasonal_data = df_plot.groupby('day_of_week')['target'].mean()
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    seasonal_data.index = day_names[:len(seasonal_data.index)]

    plt.figure(figsize=(8, 5))
    seasonal_data.plot(kind='bar', color='skyblue')
    plt.title('Weekly Seasonal Curve (Average Ticket Count)')
    plt.xlabel('Day of Week')
    plt.ylabel('Average Ticket Count')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def run_dashboard_generation():
    np.random.seed(42)
    hours = 365 * 24
    dates = pd.date_range(start='2024-01-01', periods=hours, freq='H')

    trend = np.linspace(10, 20, hours)
    weekly = np.sin(2 * np.pi * dates.dayofweek / 7) * 5
    monthly = np.sin(2 * np.pi * dates.day / 30) * 3
    noise = np.random.normal(0, 2, hours)

    values = trend + weekly + monthly + noise
    target = np.round(np.maximum(0, values)).astype(int)

    df = pd.DataFrame({'target': target}, index=dates)
    df.fillna(0, inplace=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Generating Day x Hour Heatmap...")
    create_hourly_heatmap(df, os.path.join(OUTPUT_DIR, 'day_hour_heatmap.png'))

    print("Generating Monthly Trend Chart...")
    create_monthly_trend(df, os.path.join(OUTPUT_DIR, 'monthly_trend.png'))

    print("Generating Seasonal Curve (Weekly) Chart...")
    create_seasonal_curve(df, os.path.join(OUTPUT_DIR, 'weekly_seasonal_curve.png'))

    print(f"Dashboard generation complete. All plots saved to {OUTPUT_DIR}/")

if __name__ == '__main__':
    run_dashboard_generation()
