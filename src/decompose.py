import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL

def run_stl_decomposition(data: pd.Series, period: int, output_path: str) -> None:
    stl = STL(data, period=period, robust=True)
    res = stl.fit()

    fig = res.plot()
    fig.suptitle('STL Decomposition (Trend, Seasonal, Residual)', fontsize=16, y=1.02)

    try:
        fig.savefig(output_path)
        plt.close(fig)
        print(f"Successfully saved STL decomposition plot to {output_path}")
    except Exception as e:
        print(f"Error saving figure to {output_path}: {e}")

if __name__ == '__main__':
    print("Starting STL Decomposition...")

    np.random.seed(42)
    days = 365
    date_range = pd.date_range(start='2025-01-01', periods=days, freq='D')

    trend = np.linspace(100, 200, days)
    seasonal = np.sin(2 * np.pi * date_range.dayofweek / 7) * 20
    residual = np.random.normal(0, 5, days)

    ts = pd.Series(trend + seasonal + residual, index=date_range)

    output_file = 'plots/stl_decomposition.png'

    run_stl_decomposition(
        data=ts,
        period=7,
        output_path=output_file
    )
