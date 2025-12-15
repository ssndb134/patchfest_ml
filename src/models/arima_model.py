import pandas as pd
import numpy as np
import pmdarima as pm
import pickle
import matplotlib.pyplot as plt
import os
from typing import Dict

def create_arima_baseline(
    y_train: pd.Series,
    model_output_path: str,
    plot_output_path: str
) -> pm.ARIMA:
    print("Fitting Auto-ARIMA model...")

    model = pm.auto_arima(
        y_train,
        seasonal=False,
        stepwise=True,
        suppress_warnings=True,
        error_action='ignore'
    )

    print(model.summary())

    try:
        with open(model_output_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Successfully saved model to {model_output_path}")
    except Exception as e:
        print(f"Error saving model: {e}")

    try:
        fig = model.plot_diagnostics(figsize=(12, 8))
        fig.savefig(plot_output_path)
        plt.close(fig)
        print(f"Successfully saved residual plot to {plot_output_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")

    return model

if __name__ == '__main__':
    days = 200
    dates = pd.date_range(start='2025-01-01', periods=days, freq='D')

    trend = np.linspace(50, 200, days)
    noise = np.random.normal(0, 10, days)

    y_train = pd.Series(trend + noise, index=dates)

    model_path = 'models/arima.pkl'
    plot_path = 'plots/arima_residuals.png'

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)

    fitted_model = create_arima_baseline(y_train, model_path, plot_path)
