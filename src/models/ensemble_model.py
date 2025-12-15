import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.metrics import mean_squared_error
from typing import Dict, Tuple

MODEL_PATH = 'models/ensemble.pkl'
PLOT_PATH = 'plots/ensemble_compare.png'

def simulate_predictions(data_size: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    np.random.seed(23)

    y_true = np.random.randint(200, 800, size=data_size) + np.arange(data_size) * 0.8

    arima_pred = y_true * 0.9 + np.random.normal(0, 40, data_size)
    xgb_pred = y_true * 1.05 + np.random.normal(0, 20, data_size)
    lstm_pred = y_true * 1.0 + np.random.normal(0, 10, data_size)

    return y_true, {
        'ARIMA': arima_pred,
        'XGBoost': xgb_pred,
        'LSTM': lstm_pred
    }

def build_ensemble(predictions: Dict[str, np.ndarray]) -> np.ndarray:
    stacked = np.stack(list(predictions.values()), axis=1)
    return np.mean(stacked, axis=1)

def save_ensemble(ensemble_pred: np.ndarray, model_path: str) -> None:
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(ensemble_pred, f)

def plot_comparison(y_true: np.ndarray, predictions: Dict[str, np.ndarray], plot_path: str) -> None:
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)

    df = pd.DataFrame(predictions)
    df['True'] = y_true
    df['Ensemble'] = df.mean(axis=1)

    plt.figure(figsize=(14, 7))
    plt.plot(df['True'], linewidth=3)
    plt.plot(df['ARIMA'], linestyle='--', alpha=0.5)
    plt.plot(df['XGBoost'], linestyle=':', alpha=0.5)
    plt.plot(df['LSTM'], linestyle='-.', alpha=0.5)
    plt.plot(df['Ensemble'], linewidth=2.5)

    plt.xlabel('Time Step')
    plt.ylabel('Ticket Count')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

if __name__ == '__main__':
    data_size = 100
    y_true, base_predictions = simulate_predictions(data_size)

    ensemble_pred = build_ensemble(base_predictions)

    base_scores = {k: calculate_rmse(y_true, v) for k, v in base_predictions.items()}
    ensemble_score = calculate_rmse(y_true, ensemble_pred)

    save_ensemble(ensemble_pred, MODEL_PATH)
    plot_comparison(y_true, base_predictions, PLOT_PATH)
