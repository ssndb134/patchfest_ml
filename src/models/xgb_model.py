import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder

def load_data():
    np.random.seed(42)
    days = 300
    dates = pd.date_range(start='2025-01-01', periods=days, freq='D')

    df = pd.DataFrame(
        {
            'target': np.random.randint(100, 500, size=days) + np.arange(days) * 0.5,
            'target_lag_1': np.random.rand(days),
            'target_lag_7': np.random.rand(days),
            'target_lag_30': np.random.rand(days),
            'target_roll_mean_7': np.random.rand(days),
            'target_roll_std_7': np.random.rand(days),
            'day_of_week': dates.dayofweek,
            'is_weekend': (dates.dayofweek >= 5).astype(int),
            'category_feature': np.random.choice(['A', 'B', 'C'], size=days)
        },
        index=dates
    )

    df.fillna(df.mean(numeric_only=True), inplace=True)
    return df

def preprocess_data(df: pd.DataFrame, target_column: str):
    df_processed = df.copy()

    if 'category_feature' in df_processed.columns:
        encoder = LabelEncoder()
        df_processed['category_feature'] = encoder.fit_transform(
            df_processed['category_feature']
        )

    X = df_processed.drop(columns=[target_column])
    y = df_processed[target_column]

    return X, y

def train_and_save_model(X: pd.DataFrame, y: pd.Series, model_path: str):
    print("Training XGBoost Regressor...")

    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )

    model.fit(X, y)

    try:
        model.save_model(model_path)
        print(f"Successfully saved XGBoost model to {model_path}")
    except Exception as e:
        print(f"Error saving model: {e}")

    return model

def plot_feature_importance(model: xgb.XGBRegressor, X: pd.DataFrame, plot_path: str):
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        xgb.plot_importance(
            model,
            importance_type='weight',
            max_num_features=10,
            ax=ax,
            title='XGBoost Feature Importance'
        )
        fig.tight_layout()
        fig.savefig(plot_path)
        plt.close(fig)
        print(f"Successfully saved feature importance plot to {plot_path}")
    except Exception as e:
        print(f"Error saving feature importance plot: {e}")

if __name__ == '__main__':
    print("Starting XGBoost Model Training...")

    target_col = 'target'
    model_path = 'models/xgb.json'
    plot_path = 'plots/feature_importance.png'

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)

    df_train = load_data()
    X_train, y_train = preprocess_data(df_train, target_col)

    model = train_and_save_model(X_train, y_train, model_path)
    plot_feature_importance(model, X_train, plot_path)

    print("XGBoost training complete.")
