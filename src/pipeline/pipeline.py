import logging
import os
import sys
import time

LOG_FILE = 'logs/pipeline.log'

def setup_logger(log_file: str):
    os.makedirs(os.path.dirname(log_file) or '.', exist_ok=True)
    logger = logging.getLogger('MLOpsPipeline')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger

def run_pipeline_stage(logger: logging.Logger, stage_name: str, success_rate: float):
    logger.info(f"Starting stage: {stage_name.upper()}...")
    time.sleep(0.5)
    if success_rate > 0.8:
        logger.info(f"Stage {stage_name.upper()} completed successfully.")
    elif success_rate > 0.5:
        logger.warning(f"Stage {stage_name.upper()} completed with warnings (e.g., partial feature extraction).")
    else:
        logger.error(f"Stage {stage_name.upper()} FAILED due to critical error (Simulated).")
    time.sleep(0.1)

if __name__ == '__main__':
    logger = setup_logger(LOG_FILE)
    logger.info("--- Pipeline Runner Initialized ---")
    run_pipeline_stage(logger, "preprocess_data", 0.95)
    run_pipeline_stage(logger, "feature_engineering", 0.60)
    run_pipeline_stage(logger, "model_training", 0.20)
    logger.info("--- Pipeline execution finished. Check logs/pipeline.log for full details. ---")
