import optuna
import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor
from warnings import filterwarnings

# Suppress warnings
filterwarnings('ignore')

# --- Data Loading ---
def load_data(filepath):
    if not os.path.exists(filepath):
        # Generate dummy if missing (as per previous step fallback)
        dates = pd.date_range(start='2020-01-01', periods=200)
        df = pd.DataFrame({
            'date': dates,
            'count': np.sin(np.linspace(0, 20, 200)) * 100 + 150 + np.random.normal(0, 10, 200)
        })
        # Add lags
        df['lag_1'] = df['count'].shift(1).fillna(0)
        df['lag_7'] = df['count'].shift(7).fillna(0)
        df['lag_30'] = df['count'].shift(30).fillna(0)
        df['rolling_mean_7'] = df['count'].rolling(7).mean().fillna(0)
        df['rolling_std_7'] = df['count'].rolling(7).std().fillna(0)
        return df
    return pd.read_csv(filepath)

# --- Objective Functions ---

def objective_arima(trial, train_series, val_series):
    # ARIMA tuning (p, d, q)
    p = trial.suggest_int('p', 0, 5)
    d = trial.suggest_int('d', 0, 2)
    q = trial.suggest_int('q', 0, 5)
    
    try:
        model = ARIMA(train_series, order=(p, d, q))
        model_fit = model.fit()
        preds = model_fit.forecast(steps=len(val_series))
        
        # Check for NaN/Inf in preds
        if np.isnan(preds).any():
             return float('inf')
        
        error = mean_squared_error(val_series, preds)
        return np.sqrt(error) # RMSE
    except Exception as e:
        return float('inf')

def objective_xgboost(trial, X_train, y_train, X_val, y_val):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'n_jobs': 1
    }
    
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    error = mean_squared_error(y_val, preds)
    return np.sqrt(error)

def objective_lstm(trial, train_dataset, val_dataset):
    # Tuning: hidden_size, num_layers, lr, dropout
    
    # Imports inside to avoid early failures if torch issues (though handled)
    from src.models.lstm_model import LSTMModel
    from torch.utils.data import DataLoader
    
    hidden_size = trial.suggest_categorical('hidden_size', [32, 64, 128])
    num_layers = trial.suggest_int('num_layers', 1, 3)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    
    # Setup
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # Val shuffle false
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Get input size from dataset
    x_samp, _ = train_dataset[0]
    input_size = x_samp.shape[1]
    
    model = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=30, dropout=dropout)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Short training for tuning (e.g. 5 epochs)
    epochs = 5 
    
    model.train()
    for epoch in range(epochs):
        for X_b, y_b in train_loader:
             optimizer.zero_grad()
             out = model(X_b)
             loss = criterion(out, y_b)
             loss.backward()
             optimizer.step()
    
    # Validation
    model.eval()
    total_loss = 0
    count = 0
    with torch.no_grad():
        for X_b, y_b in val_loader:
            out = model(X_b)
            loss = criterion(out, y_b)
            total_loss += loss.item() * X_b.size(0)
            count += X_b.size(0)
            
    return np.sqrt(total_loss / count) if count > 0 else float('inf')


# --- Main Driver ---

if __name__ == "__main__":
    # Setup paths
    base_dir = os.getcwd()
    data_path = os.path.join(base_dir, "data", "processed", "enhanced_forecast_data.csv")
    artifacts_dir = os.path.join(base_dir, "artifacts")
    plots_dir = os.path.join(base_dir, "plots")
    
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load Data
    df = load_data(data_path)
    
    # Prepare Data for different models
    # ARIMA: Univariate 'count'
    target_col = 'count'
    series = df[target_col].values
    split = int(len(series) * 0.8)
    train_series = series[:split]
    val_series = series[split:]
    
    # XGBoost: Tabular
    # Drop date, ensure numeric
    feature_cols = [c for c in df.columns if c not in ['date', target_col]]
    if not feature_cols: 
         # Fallback if no features
         df['lag_1'] = df[target_col].shift(1).fillna(0)
         feature_cols = ['lag_1']

    # Must align X and y. 
    # Current 'features.csv' has aligned rows? 
    # Actually, for features like lag, the first few rows are 0 or NaN.
    # XGBoost can handle it.
    
    X = df[feature_cols].select_dtypes(include=np.number).values
    y = df[target_col].values
    
    X_train_xgb = X[:split]
    y_train_xgb = y[:split]
    X_val_xgb = X[split:]
    y_val_xgb = y[split:]
    
    # LSTM: Sliding Window
    from src.models.lstm_model import SlidingWindowDataset
    # Note: SlidingWindowDataset takes df and extracts features internally.
    # We should pass the same DF?
    # Yes. But we need to split DF first to avoid leakage in window creation if we want strict split.
    train_df = df.iloc[:split].reset_index(drop=True)
    val_df = df.iloc[split:].reset_index(drop=True)
    
    try:
        train_dataset = SlidingWindowDataset(train_df, seq_length=30, horizon=30)
        val_dataset = SlidingWindowDataset(val_df, seq_length=30, horizon=30)
        
        # Check if datasets valid
        if len(train_dataset) == 0 or len(val_dataset) == 0:
            print("Warning: Dataset too small for LSTM windowing. Skipping LSTM tuning.")
            do_lstm = False
        else:
            do_lstm = True
    except Exception as e:
        print(f"LSTM Dataset Error: {e}")
        do_lstm = False

    best_params = {}
    
    # --- Tune ARIMA ---
    print("Tuning ARIMA...")
    study_arima = optuna.create_study(direction='minimize')
    study_arima.optimize(lambda t: objective_arima(t, train_series, val_series), n_trials=10)
    best_params['arima'] = study_arima.best_params
    print(f"Best ARIMA: {study_arima.best_params}")
    
    # --- Tune XGBoost ---
    print("Tuning XGBoost...")
    study_xgb = optuna.create_study(direction='minimize')
    study_xgb.optimize(lambda t: objective_xgboost(t, X_train_xgb, y_train_xgb, X_val_xgb, y_val_xgb), n_trials=10)
    best_params['xgboost'] = study_xgb.best_params
    print(f"Best XGBoost: {study_xgb.best_params}")
    
    # --- Tune LSTM ---
    if do_lstm:
        print("Tuning LSTM...")
        study_lstm = optuna.create_study(direction='minimize')
        study_lstm.optimize(lambda t: objective_lstm(t, train_dataset, val_dataset), n_trials=5) # fewer trials for DL
        best_params['lstm'] = study_lstm.best_params
        print(f"Best LSTM: {study_lstm.best_params}")
    else:
        best_params['lstm'] = "Skipped"

    # --- Save Results ---
    with open(os.path.join(artifacts_dir, 'best_params.json'), 'w') as f:
        json.dump(best_params, f, indent=4)
        
    print("Saved best params.")
    
    # --- Plotting ---
    # Plot optimization history for XGBoost (usually most interesting) or all 3
    # We'll stick to XGBoost history sample prompt
    try:
        fig = optuna.visualization.matplotlib.plot_optimization_history(study_xgb)
        fig.figure.savefig(os.path.join(plots_dir, 'optuna_history.png'))
        print("Saved optimization plot.")
    except Exception as e:
        print(f"Plotting failed: {e}")
        # Manually plot if optuna viz fails
        plt.figure()
        plt.plot([t.value for t in study_xgb.trials if t.value is not None])
        plt.title('XGBoost Optimization History')
        plt.xlabel('Trial')
        plt.ylabel('RMSE')
        plt.savefig(os.path.join(plots_dir, 'optuna_history.png'))
