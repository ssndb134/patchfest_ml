import pandas as pd
import numpy as np
import json
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Tuple

def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true_safe = np.where(y_true == 0, np.finfo(float).eps, y_true)
    return np.mean(np.abs((y_true_safe - y_pred) / y_true_safe)) * 100

def get_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    }

def evaluate_models(
    model_predictions: Dict[str, Tuple[np.ndarray, np.ndarray]],
    report_path: str
) -> Dict[str, Dict[str, float]]:
    full_report = {}

    for model_name, (y_true, y_pred) in model_predictions.items():
        metrics = get_metrics(y_true, y_pred)
        full_report[model_name] = metrics
        print(f"Metrics for {model_name}: {json.dumps(metrics, indent=4)}")

    try:
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(full_report, f, indent=4)
        print(f"\nSuccessfully saved metrics report to {report_path}")
    except Exception as e:
        print(f"Error saving report: {e}")

    return full_report

if __name__ == '__main__':
    np.random.seed(42)
    days = 100

    y_true = np.random.randint(100, 500, size=days) + np.arange(days) * 0.5
    arima_pred = y_true * 0.95 + np.random.normal(0, 15, days)
    xgb_pred = y_true + np.random.normal(0, 5, days)

    predictions = {
        "ARIMA_Baseline": (y_true, arima_pred),
        "XGBoost_Regressor": (y_true, xgb_pred)
    }

    report_path = 'reports/metrics.json'
    evaluate_models(predictions, report_path)
