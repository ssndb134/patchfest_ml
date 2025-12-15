import argparse
import sys
import subprocess
import logging
import yaml
import os
import datetime

# --- Setup Logging ---
if not os.path.exists("logs"):
    os.makedirs("logs")

log_filename = f"logs/pipeline_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/pipeline.log"), # Fixed name as requested
        logging.FileHandler(log_filename),        # Timestamped for history
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_config(config_path="pipeline_config.yaml"):
    if not os.path.exists(config_path):
        logger.error(f"Config file {config_path} not found.")
        sys.exit(1)
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_stage(stage_name, script_path):
    logger.info(f"=== Starting Stage: {stage_name} ===")
    logger.info(f"Executing script: {script_path}")
    
    # Use current python executable
    cmd = [sys.executable, script_path]
    
    # Ensure PYTHONPATH includes current directory
    env = os.environ.copy()
    if 'PYTHONPATH' not in env:
        env['PYTHONPATH'] = os.getcwd()
    else:
        env['PYTHONPATH'] = os.getcwd() + os.pathsep + env['PYTHONPATH']

    try:
        # Check if file exists
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"Script {script_path} not found.")

        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            env=env,
            check=True
        )
        
        # Log Output
        if result.stdout:
            logger.info(f"[{stage_name}] Output:\n{result.stdout.strip()}")
        
        logger.info(f"=== Stage {stage_name} COMPLETED Successfully ===")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"=== Stage {stage_name} FAILED ===")
        logger.error(f"Exit Code: {e.returncode}")
        logger.error(f"Error Output:\n{e.stderr.strip()}")
        if e.stdout:
            logger.info(f"Standard Output before failure:\n{e.stdout.strip()}")
        return False
        
    except Exception as e:
        logger.error(f"=== Stage {stage_name} FAILED with Exception: {e} ===")
        return False

def main():
    parser = argparse.ArgumentParser(description="ML Pipeline Orchestrator")
    parser.add_argument("--stage", type=str, required=True, help="Stage to run (e.g., preprocess, train_lstm, all)")
    args = parser.parse_args()

    config = load_config()
    stages_config = config.get('stages', {})
    
    # Define execution order for 'all'
    # Order: preprocess -> tune -> train_lstm -> train_ensemble
    execution_order = ['preprocess', 'train_lstm', 'train_ensemble'] 
    # omitted 'tune' from default 'all' as it takes long, usually optional? 
    # Or include it. Prompt asks for specific stages. I'll stick to a logical flow.
    
    stages_to_run = []
    
    if args.stage.lower() == 'all':
        stages_to_run = execution_order
    else:
        if args.stage not in stages_config:
            logger.error(f"Stage '{args.stage}' not found in configuration.")
            logger.info(f"Available stages: {list(stages_config.keys())}")
            sys.exit(1)
        stages_to_run = [args.stage]

    success = True
    for stage in stages_to_run:
        script = stages_config.get(stage, {}).get('script')
        if not script:
             logger.warning(f"No script defined for stage {stage}. Skipping.")
             continue
             
        if not run_stage(stage, script):
            success = False
            if args.stage.lower() == 'all':
                logger.error("Pipeline aborted due to stage failure.")
                sys.exit(1)
            else:
                sys.exit(1)

    if success:
        logger.info("Pipeline execution finished successfully.")

if __name__ == "__main__":
    main()
