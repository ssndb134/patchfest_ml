import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from src.models.lstm_model import LSTMModel, SlidingWindowDataset

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
            
            epochs = 10 
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
        preds = {'arima': [], 'xgb': [], 'lstm': []}
        n_forecast = len(df_val)
        
        # 1. ARIMA
        if self.arima_model:
            try:
                preds['arima'] = self.arima_model.forecast(steps=n_forecast)
            except:
                preds['arima'] = np.zeros(n_forecast)
        else:
            preds['arima'] = np.zeros(n_forecast)

        # 2. XGBoost
        X_val = df_val[self.feature_cols].select_dtypes(include=np.number).values
        if self.xgb_model:
            preds['xgb'] = self.xgb_model.predict(X_val)
        else:
            preds['xgb'] = np.zeros(n_forecast)
            
        # 3. LSTM
        if self.lstm_model and history_df is not None:
             self.lstm_model.eval()
             last_window_data = history_df.tail(self.seq_length)
             if len(last_window_data) == self.seq_length:
                 ds = SlidingWindowDataset(history_df, seq_length=self.seq_length, horizon=30)
                 feat_cols = ds.features
                 x_window = history_df[feat_cols].tail(self.seq_length).values.astype(np.float32)
                 x_tensor = torch.tensor(x_window).unsqueeze(0)
                 
                 with torch.no_grad():
                     lstm_out = self.lstm_model(x_tensor)
                     lstm_pred = lstm_out.numpy().flatten()
                     
                 if len(lstm_pred) > n_forecast:
                     lstm_pred = lstm_pred[:n_forecast]
                 elif len(lstm_pred) < n_forecast:
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
