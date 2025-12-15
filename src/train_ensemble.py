import numpy as np
import pandas as pd
import json
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from src.models.ensemble_model import EnsembleForecaster

def run_ensemble_pipeline():
    base_dir = os.getcwd()
    data_path = os.path.join(base_dir, "data", "processed", "enhanced_forecast_data.csv")
    params_path = os.path.join(base_dir, "artifacts", "best_params.json")
    
    # Load Data
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
    else:
        df = pd.DataFrame() 
    
    if len(df) < 100:
        print("Data too small, generating synthetic data for ensemble demonstration.")
        dates = pd.date_range(start='2020-01-01', periods=300)
        t = np.linspace(0, 50, 300)
        count = 100 + 10 * np.sin(t) + np.random.normal(0, 5, 300) + 0.5 * t
        df = pd.DataFrame({'date': dates, 'count': count})
        df['lag_1'] = df['count'].shift(1).fillna(0)
        df['lag_7'] = df['count'].shift(7).fillna(0)
        df['rolling_mean_7'] = df['count'].rolling(7).mean().fillna(0)

    # Load Params
    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            best_params = json.load(f)
    else:
        best_params = {'arima': {}, 'xgb': {}, 'lstm': {}}

    # Split
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    val_df = df.iloc[split_idx:].reset_index(drop=True)
    
    # Init Ensemble
    ensemble = EnsembleForecaster(
        arima_params=best_params.get('arima', {}),
        xgb_params=best_params.get('xgb', {}),
        lstm_params=best_params.get('lstm', {}) if best_params.get('lstm') != "Skipped" else {},
        weights={'arima': 0.2, 'xgb': 0.6, 'lstm': 0.2} 
    )
    
    # Fit
    ensemble.fit(train_df, target_col='count')
    
    # Predict
    final_pred, sub_preds = ensemble.predict(val_df, history_df=train_df)
    
    # Evaluate
    y_true = val_df['count'].values
    min_len = min(len(y_true), len(final_pred))
    y_true = y_true[:min_len]
    final_pred = final_pred[:min_len]
    
    rmse = np.sqrt(mean_squared_error(y_true, final_pred))
    print(f"Ensemble RMSE: {rmse:.4f}")
            
    # Save Model
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    with open(os.path.join(models_dir, "ensemble.pkl"), "wb") as f:
        pickle.dump(ensemble, f)
    print("Ensemble model saved to models/ensemble.pkl")
    
    # Plot
    plots_dir = os.path.join(base_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Actual', color='black', linewidth=2)
    plt.plot(final_pred, label='Ensemble', color='purple', linestyle='--', linewidth=2)
    if np.any(sub_preds['xgb']): plt.plot(sub_preds['xgb'][:min_len], label='XGBoost', alpha=0.5)
    if np.any(sub_preds['arima']): plt.plot(sub_preds['arima'][:min_len], label='ARIMA', alpha=0.5)
    plt.title('Ensemble Forecast vs Actual')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, "ensemble_compare.png"))
    print("Comparison chart saved.")

if __name__ == "__main__":
    run_ensemble_pipeline()
