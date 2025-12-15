import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

def load_data(filepath):
    """Load data and ensure datetime parsing."""
    if not os.path.exists(filepath):
        print(f"File {filepath} not found.")
        return pd.DataFrame()
    
    df = pd.read_csv(filepath)
    
    # Parse Date
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Parse Time if exists to get Hour
    if 'time' in df.columns:
        # standardizing time format often tricky, infer assuming standard HH:MM
        # Some are "18:30", some might be "07:00 PM" based on parser.
        # Let's try flexible parsing or just taking first 2 chars if consistent?
        # Safer: pd.to_datetime combined
        pass
    
    return df

def create_heatmap(df, output_dir):
    """Day of Week x Hour Heatmap."""
    if df.empty: return

    # Ensure Day of Week
    if 'weekday' not in df.columns and 'date' in df.columns:
        df['weekday'] = df['date'].dt.day_name()
    elif 'weekday' in df.columns:
        # map int to name if needed, or keep int
        pass

    # Extract Hour
    if 'hour' not in df.columns:
        if 'time' in df.columns:
            # Clean time string, remove PM/AM for parsing or use flexible
            # regex clean?
            try:
                # Approach: Convert to datetime, extract hour
                # Coerce errors to NaT
                df['temp_time'] = pd.to_datetime(df['time'], format='mixed', errors='coerce')
                df['hour'] = df['temp_time'].dt.hour
            except:
                # Fallback: simple string slice for 24h or numeric regex
                # If "18:30", slice :2
                df['hour'] = df['time'].astype(str).str.extract(r'(\d{1,2})').astype(float)
        else:
            print("No time info for heatmap.")
            return

    # Pivot: Index=Hour, Cols=Day
    # Aggfunc = mean count (or sum)
    if 'hour' not in df.columns or df['hour'].isnull().all():
        print("Could not extract hours for heatmap.")
        return
        
    pivot = df.pivot_table(index='hour', columns='weekday', values='count', aggfunc='mean')
    
    # Reorder days if names
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    # Filter only days present
    existing_days = [d for d in days_order if d in pivot.columns]
    if existing_days:
        pivot = pivot[existing_days]

    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot, cmap="YlGnBu", annot=True, fmt=".1f", linewidths=.5)
    plt.title("Average Ticket Demand: Day vs Hour")
    plt.ylabel("Hour of Day")
    plt.xlabel("Day of Week")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "heatmap_day_hour.png"))
    plt.close()

def create_monthly_trend(df, output_dir):
    """Monthly Trend Line Plot."""
    if 'date' not in df.columns: return

    # Group by Month
    # df['month_year'] = df['date'].dt.to_period('M') # period returns obj
    # easier to resample if index is date
    
    temp = df.set_index('date')
    monthly = temp.resample('ME')['count'].mean() # ME is month end alias in newer pandas, M deprecated
    
    plt.figure(figsize=(12, 5))
    monthly.plot(marker='o', linestyle='-', color='teal', linewidth=2)
    plt.title("Monthly Booking Trend (Average)")
    plt.ylabel("Avg Tickets")
    plt.xlabel("Month")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "monthly_trend.png"))
    plt.close()

def create_seasonal_curve(df, output_dir):
    """Seasonal Curve (e.g. Day of Year or Month-based seasonality)."""
    if 'month' not in df.columns and 'date' in df.columns:
        df['month'] = df['date'].dt.month
        
    if 'month' not in df.columns: return

    # Agg by month (1-12)
    seasonal = df.groupby('month')['count'].mean()
    std = df.groupby('month')['count'].std().fillna(0)

    plt.figure(figsize=(10, 5))
    x = seasonal.index
    y = seasonal.values
    
    plt.plot(x, y, marker='s', color='#d62728', label='Mean')
    plt.fill_between(x, y - std, y + std, color='#d62728', alpha=0.2, label='Std Dev')
    
    plt.title("Seasonality Curve (Yearly)")
    plt.xlabel("Month")
    plt.ylabel("Ticket Demand")
    plt.xticks(range(1, 13))
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "seasonal_curve.png"))
    plt.close()

def run_dashboard():
    base_dir = os.getcwd()
    processed_data = os.path.join(base_dir, "data", "processed", "enhanced_forecast_data.csv")
    output_dir = os.path.join(base_dir, "plots", "dashboard")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Loading data...")
    df = load_data(processed_data)
    
    if df.empty:
        print("Dataframe empty, generating dummy data for dashboard demo.")
        # Gen dummy
        dates = pd.date_range('2023-01-01', periods=365, freq='D')
        df = pd.DataFrame({'date': dates})
        df['count'] = np.random.poisson(50, 365) + 10 * np.sin(np.linspace(0, 3.14*2, 365))
        df['weekday'] = df['date'].dt.day_name()
        df['time'] = np.random.choice(["10:00", "14:00", "18:30", "21:00"], 365)
        df['month'] = df['date'].dt.month

    # 1. Heatmap
    print("Generating Heatmap...")
    create_heatmap(df, output_dir)
    
    # 2. Monthly
    print("Generating Monthly Trend...")
    create_monthly_trend(df, output_dir)
    
    # 3. Seasonal
    print("Generating Seasonal Curve...")
    create_seasonal_curve(df, output_dir)
    
    print(f"Dashboard generated in {output_dir}")

if __name__ == "__main__":
    run_dashboard()
