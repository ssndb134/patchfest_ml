import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple

LOOKBACK_WINDOW = 30
FORECAST_HORIZON = 30
MODEL_PATH = 'models/lstm.keras'

def create_sliding_window_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    lookback: int,
    horizon: int
) -> Tuple[np.ndarray, np.ndarray]:
    X_out, y_out = [], []

    for i in range(lookback, len(X) - horizon + 1):
        X_out.append(X.iloc[i - lookback:i].values)
        y_out.append(y.iloc[i])

    return np.array(X_out), np.array(y_out)

def build_lstm_model(input_shape: Tuple[int, int]) -> tf.keras.Model:
    model = Sequential([
        LSTM(128, activation='relu', input_shape=input_shape),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def run_lstm_forecasting():
    print("Preparing data...")

    np.random.seed(42)
    features = 5

    days_train = 250
    X_train_raw = np.random.rand(days_train, features)
    y_train_raw = np.random.rand(days_train) * 100 + np.sin(np.arange(days_train) / 30 * 2) * 50

    days_val = 50
    X_val_raw = np.random.rand(days_val, features)
    y_val_raw = np.random.rand(days_val) * 100 + np.sin(np.arange(days_val) / 30 * 2) * 50

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_scaled = scaler_X.fit_transform(X_train_raw)
    y_train_scaled = scaler_y.fit_transform(y_train_raw.reshape(-1, 1))

    X_val_scaled = scaler_X.transform(X_val_raw)
    y_val_scaled = scaler_y.transform(y_val_raw.reshape(-1, 1))

    X_train_df = pd.DataFrame(X_train_scaled)
    y_train_series = pd.Series(y_train_scaled.flatten())
    X_val_df = pd.DataFrame(X_val_scaled)
    y_val_series = pd.Series(y_val_scaled.flatten())

    X_train_win, y_train_win = create_sliding_window_dataset(
        X_train_df, y_train_series, LOOKBACK_WINDOW, FORECAST_HORIZON
    )
    X_val_win, y_val_win = create_sliding_window_dataset(
        X_val_df, y_val_series, LOOKBACK_WINDOW, FORECAST_HORIZON
    )

    print("Building model...")
    model = build_lstm_model((LOOKBACK_WINDOW, features))

    stopper = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    print("Training model...")
    model.fit(
        X_train_win,
        y_train_win,
        epochs=100,
        batch_size=32,
        validation_data=(X_val_win, y_val_win),
        callbacks=[stopper],
        verbose=0
    )

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    try:
        model.save(MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")
    except Exception as e:
        print(f"Error saving model: {e}")

if __name__ == '__main__':
    try:
        run_lstm_forecasting()
    except ModuleNotFoundError as e:
        print(f"Missing dependency: {e}")
