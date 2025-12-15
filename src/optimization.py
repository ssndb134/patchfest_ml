import pandas as pd
import numpy as np
import optuna
import xgboost as xgb
import json
import os
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

N_TRIALS = 25
TARGET_COLUMN = 'target'
ARTIFACTS_DIR = 'artifacts'
PLOTS_DIR = 'plots'
BEST_PARAMS_PATH = os.path.join(ARTIFACTS_DIR, 'best_params.json')
OPTUNA_HISTORY_PATH = os.path.join(PLOTS_DIR, 'optuna_history.png')

def load_dummy_data() -> pd.DataFrame:
    np.random.seed(42)
    days = 300
    dates = pd.date_range(start='2025-01-01', periods=days, freq='D')

    df = pd.DataFrame(
        {
            TARGET_COLUMN: np.random.randint(100, 500, size=days) + np.arange(days) * 0.5,
            'target_lag_1': np.random.rand(days),
            'target_roll_mean_7': np.random.rand(days),
            'day_of_week': dates.dayofweek
        },
        index=dates
    )

    df.fillna(df.mean(numeric_only=True), inplace=True)
    return df

def objective_xgboost(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
    params = {
        'objective': 'reg:squarederror',
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
        'seed': 42
    }

    kf = KFold(n_splits=5, shuffle=False)
    scores = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        scores.append(np.sqrt(mean_squared_error(y_val, preds)))

    return float(np.mean(scores))

def save_artifacts(study: optuna.Study, params_path: str, plot_path: str) -> None:
    os.makedirs(os.path.dirname(params_path), exist_ok=True)
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)

    with open(params_path, 'w') as f:
        json.dump(
            {
                "best_score_rmse": study.best_value,
                "best_params": study.best_params
            },
            f,
            indent=4
        )

    try:
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_image(plot_path)
    except Exception:
        pass

if __name__ == '__main__':
    print("Starting Hyperparameter Optimization...")

    df = load_dummy_data()
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    study = optuna.create_study(direction='minimize', study_name='XGBoost_HPO')
    study.optimize(
        lambda trial: objective_xgboost(trial, X, y),
        n_trials=N_TRIALS,
        show_progress_bar=True
    )

    save_artifacts(study, BEST_PARAMS_PATH, OPTUNA_HISTORY_PATH)

    print("Optimization complete.")
