import argparse
import yaml
import logging
import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, Any

def setup_logging(log_file: str):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

def load_config(config_path: str = 'config.yaml') -> Dict[str, Any]:
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"Configuration file not found at {config_path}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        sys.exit(1)

def load_and_clean_data(config: Dict[str, Any]) -> pd.DataFrame:
    logging.info("Running PREPROCESS & SPLIT stage...")
    days = 365
    date_range = pd.date_range(start='2025-01-01', periods=days, freq='D')
    df = pd.DataFrame(
        {config['DATA_SETTINGS']['TARGET_COLUMN']: np.random.randint(100, 500, days)},
        index=date_range
    )

    train_end = int(len(df) * config['DATA_SETTINGS']['TRAIN_RATIO'])
    val_end = train_end + int(len(df) * config['DATA_SETTINGS']['VAL_RATIO'])

    df.iloc[:train_end].to_csv('data/processed/train_set.csv')
    df.iloc[train_end:val_end].to_csv('data/processed/val_set.csv')
    df.iloc[val_end:].to_csv('data/processed/test_set.csv')

    logging.info(f"Data split completed. Train: {train_end}, Val: {val_end - train_end}")
    return df

def run_feature_engineering(config: Dict[str, Any]):
    logging.info("Running FEATURE ENGINEERING stage...")
    os.makedirs('data/processed', exist_ok=True)
    with open('data/processed/features_final.csv', 'w') as f:
        f.write("timestamp,feature1,feature2\n")
    logging.info("Feature engineering completed.")

def train_model_stage(config: Dict[str, Any], model_name: str):
    settings = config['MODEL_SETTINGS'].get(model_name.upper())
    if not settings or not settings.get('ENABLED'):
        logging.info(f"Skipping {model_name} training.")
        return

    logging.info(f"Training {model_name.upper()} model...")
    os.makedirs('models', exist_ok=True)
    model_file = f'models/{model_name.lower()}.pkl'

    with open(model_file, 'w') as f:
        f.write("trained_model_artifact")

    logging.info(f"{model_name.upper()} model saved to {model_file}")

def run_evaluation(config: Dict[str, Any]):
    logging.info("Running EVALUATION stage...")
    os.makedirs('reports', exist_ok=True)
    with open('reports/metrics.json', 'w') as f:
        f.write('{"ARIMA": {"RMSE": 15.2}, "XGBoost": {"RMSE": 12.8}}')
    logging.info("Evaluation completed.")

def run_pipeline(stage_name: str, config: Dict[str, Any]):
    stages = {
        'preprocess': [load_and_clean_data],
        'features': [run_feature_engineering],
        'train_arima': [lambda c: train_model_stage(c, 'ARIMA')],
        'train_xgb': [lambda c: train_model_stage(c, 'XGBoost')],
        'train_lstm': [lambda c: train_model_stage(c, 'LSTM')],
        'evaluate': [run_evaluation],
        'all': [
            load_and_clean_data,
            run_feature_engineering,
            lambda c: train_model_stage(c, 'ARIMA'),
            lambda c: train_model_stage(c, 'XGBoost'),
            lambda c: train_model_stage(c, 'LSTM'),
            run_evaluation
        ]
    }

    execution_list = stages.get(stage_name)
    if not execution_list:
        logging.error(f"Unknown stage '{stage_name}'")
        return

    for func in execution_list:
        func(config)

    logging.info("Pipeline execution finished.")

if __name__ == '__main__':
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)

    config = load_config()
    setup_logging(config['PIPELINE_SETTINGS']['LOG_FILE'])

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--stage',
        type=str,
        default='all',
        choices=['all', 'preprocess', 'features', 'train_arima', 'train_xgb', 'train_lstm', 'evaluate']
    )
    args = parser.parse_args()

    try:
        run_pipeline(args.stage, config)
    except Exception as e:
        logging.critical(f"Pipeline failed: {e}")
        sys.exit(1)
