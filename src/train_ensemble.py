import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import pickle
import os
import json
from sklearn.metrics import mean_squared_error
from src.models.lstm_model import LSTMModel, SlidingWindowDataset

# --- Ensemble Class ---
class EnsembleForecaster:
    def __init__(self, arima_params, xgb_params, lstm_params, weights=None):
        self.arima_params = arima_params
        self.xgb_params = xgb_params
        self.lstm_params = lstm_params
        self.weights = weights if weights else {'arima': 0.33, 'xgb': 0.33, 'lstm': 0.34}
        
        self.arima_model = None
        self.xgb_model = None
        self.lstm_model = None
        self.feature_cols = None
        self.seq_length = 30
        
    def fit(self, df, target_col='count'):
        # Prepare Data
        self.target_col = target_col
        y = df[target_col].values
        
        # 1. Fit ARIMA
        print("Fitting ARIMA...")
        try:
            order = (self.arima_params.get('p', 1), self.arima_params.get('d', 1), self.arima_params.get('q', 1))
            self.arima_model = ARIMA(y, order=order).fit()
        except Exception as e:
            print(f"ARIMA Fit Failed: {e}")
            self.arima_model = None
            
        # 2. Fit XGBoost
        print("Fitting XGBoost...")
        self.feature_cols = [c for c in df.columns if c not in ['date', target_col]]
        # Handle if no features
        if not self.feature_cols:
             df_temp = df.copy()
             df_temp['lag_1'] = df_temp[target_col].shift(1).fillna(0)
             self.feature_cols = ['lag_1']
             X = df_temp[self.feature_cols].values
        else:
             X = df[self.feature_cols].select_dtypes(include=np.number).values
        
        self.xgb_model = XGBRegressor(**self.xgb_params)
        self.xgb_model.fit(X, y)
        
        # 3. Fit LSTM
        print("Fitting LSTM...")
        # Prepare Dataset
        dataset = SlidingWindowDataset(df, seq_length=self.seq_length, horizon=30)
        # Check size
        if len(dataset) > 0:
            loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
            
            # Init Model
            input_size = dataset[0][0].shape[1]
            hidden_size = self.lstm_params.get('hidden_size', 64)
            num_layers = self.lstm_params.get('num_layers', 2)
            
            self.lstm_model = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=30)
            
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(self.lstm_model.parameters(), lr=self.lstm_params.get('lr', 0.001))
            
            epochs = 10 # Short training for ensemble demo
            self.lstm_model.train()
            for ep in range(epochs):
                for X_b, y_b in loader:
                    optimizer.zero_grad()
                    out = self.lstm_model(X_b)
                    loss = criterion(out, y_b)
                    loss.backward()
                    optimizer.step()
        else:
            print("Skipping LSTM (Data too small)")
            self.lstm_model = None

    def predict(self, df_val, history_df=None):
        """
        Make predictions on the validation set.
        df_val: DataFrame containing the validation data (features + target for metric calc, or just features)
        history_df: DataFrame containing training data (needed for LSTM context/ARIMA extension)
        """
        preds = {'arima': [], 'xgb': [], 'lstm': []}
        
        # We assume df_val immediately follows history_df
        n_forecast = len(df_val)
        
        # 1. ARIMA Prediction
        if self.arima_model:
            # For ARIMA, valid prediction is usually 'forecast' from end of training
            # or 'predict' with dynamic adjustment?
            # We'll use forecast (out of sample)
            try:
                arima_pred = self.arima_model.forecast(steps=n_forecast)
                preds['arima'] = arima_pred
            except:
                preds['arima'] = np.zeros(n_forecast)
        else:
            preds['arima'] = np.zeros(n_forecast)

        # 2. XGBoost Prediction
        # Needs features from df_val
        X_val = df_val[self.feature_cols].select_dtypes(include=np.number).values
        if self.xgb_model:
            preds['xgb'] = self.xgb_model.predict(X_val)
        else:
            preds['xgb'] = np.zeros(n_forecast)
            
        # 3. LSTM Prediction
        # LSTM predicts 30 day horizon.
        # Ideally we use the LAST window of history_df to predict the NEXT 30 days (which is df_val).
        # We only produce ONE forecast sequence of length 30 (or min(30, n_forecast)).
        if self.lstm_model and history_df is not None:
             self.lstm_model.eval()
             # Get last window
             full_data = pd.concat([history_df, df_val]).reset_index(drop=True)
             # Actually, we just need the last `seq_length` rows from history to predict first step of val?
             # But the model predicts a generic 30-step horizon.
             # So we take the very last window of TRAINING data.
             
             last_window_data = history_df.tail(self.seq_length)
             if len(last_window_data) == self.seq_length:
                 # Extract features using same logic as Dataset
                 # Re-instantiate dataset class to use its feature extraction logic safely?
                 # Or manual:
                 ds = SlidingWindowDataset(history_df, seq_length=self.seq_length, horizon=30)
                 # Get last item?
                 # SlidingWindowDataset logic: idx is start of window.
                 # We want the window ending at the end of history_df.
                 # i.e. start index = len(history) - seq_length.
                 
                 feat_cols = ds.features
                 x_window = history_df[feat_cols].tail(self.seq_length).values.astype(np.float32)
                 x_tensor = torch.tensor(x_window).unsqueeze(0) # Batch dim
                 
                 with torch.no_grad():
                     lstm_out = self.lstm_model(x_tensor)
                     lstm_pred = lstm_out.numpy().flatten()
                     
                 # Trim to match n_forecast
                 if len(lstm_pred) > n_forecast:
                     lstm_pred = lstm_pred[:n_forecast]
                 elif len(lstm_pred) < n_forecast:
                     # Pad?
                     lstm_pred = np.pad(lstm_pred, (0, n_forecast - len(lstm_pred)))
                 
                 preds['lstm'] = lstm_pred
             else:
                 preds['lstm'] = np.zeros(n_forecast)
        else:
            preds['lstm'] = np.zeros(n_forecast)

        # Blending
        final_pred = (
            preds['arima'] * self.weights['arima'] +
            preds['xgb'] * self.weights['xgb'] +
            preds['lstm'] * self.weights['lstm']
        )
        
        return final_pred, preds

# --- Main Script ---
def run_ensemble_pipeline():
    base_dir = os.getcwd()
    data_path = os.path.join(base_dir, "data", "processed", "enhanced_forecast_data.csv")
    params_path = os.path.join(base_dir, "artifacts", "best_params.json")
    
    # Load Data
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
    else:
        df = pd.DataFrame() # fail gracefully or gen dummy
    
    # Generate robust dummy data if needed for demonstration
    if len(df) < 100:
        print("Data too small, generating synthetic data for ensemble demonstration.")
        dates = pd.date_range(start='2020-01-01', periods=300)
        t = np.linspace(0, 50, 300)
        count = 100 + 10 * np.sin(t) + np.random.normal(0, 5, 300) + 0.5 * t
        df = pd.DataFrame({'date': dates, 'count': count})
        # Add basic features
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
        weights={'arima': 0.2, 'xgb': 0.6, 'lstm': 0.2} # Prioritize XGB usually
    )
    
    # Fit
    ensemble.fit(train_df, target_col='count')
    
    # Predict
    final_pred, sub_preds = ensemble.predict(val_df, history_df=train_df)
    
    # Evaluate
    y_true = val_df['count'].values
    
    # Handle mismatches in length if any (e.g. LSTM horizon constraint)
    min_len = min(len(y_true), len(final_pred))
    y_true = y_true[:min_len]
    final_pred = final_pred[:min_len]
    
    rmse = np.sqrt(mean_squared_error(y_true, final_pred))
    print(f"Ensemble RMSE: {rmse:.4f}")
    
    # Metrics for sub-models
    for name, pred in sub_preds.items():
        if len(pred) >= min_len:
            p = pred[:min_len]
            err = np.sqrt(mean_squared_error(y_true, p))
            print(f"{name.upper()} RMSE: {err:.4f}")
            
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
    # Plot subs
    # Limit noise: plot only if valid
    if np.any(sub_preds['xgb']): plt.plot(sub_preds['xgb'][:min_len], label='XGBoost', alpha=0.5)
    if np.any(sub_preds['arima']): plt.plot(sub_preds['arima'][:min_len], label='ARIMA', alpha=0.5)
    
    plt.title('Ensemble Forecast vs Actual')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, "ensemble_compare.png"))
    print("Comparison chart saved to plots/ensemble_compare.png")

if __name__ == "__main__":
    run_ensemble_pipeline()
