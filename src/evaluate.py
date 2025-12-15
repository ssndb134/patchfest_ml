import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import os
import pickle
import sys

# Ensure src imports work
sys.path.append(os.getcwd())

class ModelEvaluator:
    def __init__(self, output_dir="reports"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
    def compute_metrics(self, y_true, y_pred):
        """
        Computes RMSE, MAE, MAPE, R2.
        Handles zeros in y_true for MAPE by replacing with small epsilon or excluding.
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Filter NaNs if any
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        if len(y_true) == 0:
            return {"rmse": None, "mae": None, "mape": None, "r2": None}

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # MAPE: Mean Absolute Percentage Error
        # Avoid division by zero
        non_zero = y_true != 0
        if np.any(non_zero):
            mape = np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100
        else:
            mape = None
            
        return {
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "mape": round(mape, 4) if mape is not None else None,
            "r2": round(r2, 4)
        }

    def evaluate_models(self, models_dict, y_true):
        """
        models_dict: {'model_name': y_pred_array}
        """
        results = {}
        for name, preds in models_dict.items():
            print(f"Evaluating {name}...")
            metrics = self.compute_metrics(y_true, preds)
            results[name] = metrics
            
        return results

    def save_metrics(self, results, filename="metrics.json"):
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Metrics saved to {filepath}")


if __name__ == "__main__":
    # Integration with existing pipeline data
    base_dir = os.getcwd()
    data_path = os.path.join(base_dir, "data", "processed", "enhanced_forecast_data.csv")
    models_path = os.path.join(base_dir, "models", "ensemble.pkl")
    
    if os.path.exists(data_path) and os.path.exists(models_path):
        # Load Data
        df = pd.read_csv(data_path)
        
        # Load Ensemble
        with open(models_path, 'rb') as f:
            ensemble = pickle.load(f)
            
        # Split (Same logic as training to get validation set)
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx].reset_index(drop=True)
        val_df = df.iloc[split_idx:].reset_index(drop=True)
        
        target_col = 'count'
        y_true = val_df[target_col].values
        
        # Get Predictions
        # Note: 'predict' in our ensemble class returns blended and subs
        final_pred, sub_preds = ensemble.predict(val_df, history_df=train_df)
        
        # Prepare dictionary
        predictions = {
            "Ensemble": final_pred,
            "ARIMA": sub_preds.get('arima', []),
            "XGBoost": sub_preds.get('xgb', []),
            "LSTM": sub_preds.get('lstm', [])
        }
        
        # Trim to min length
        min_len = len(y_true)
        for k in predictions:
            if len(predictions[k]) > min_len:
                predictions[k] = predictions[k][:min_len]
            elif len(predictions[k]) < min_len:
                # Pad for shape consistency or trim truth?
                # Trim truth for evaluation per model logic is complex
                # We'll just assume alignment or trim y_true specifically for that model?
                # Simpler: Trim y_true to prediction length if pred is shorter
                pass
                
        evaluator = ModelEvaluator()
        
        # Compute for each
        all_metrics = {}
        for k, v in predictions.items():
            # Align
            curr_len = min(len(v), len(y_true))
            if curr_len == 0: continue
            
            m = evaluator.compute_metrics(y_true[:curr_len], v[:curr_len])
            all_metrics[k] = m
            
        evaluator.save_metrics(all_metrics)
        print(json.dumps(all_metrics, indent=2))
        
    else:
        print("Data or Model not found. Cannot run standalone evaluation.")
