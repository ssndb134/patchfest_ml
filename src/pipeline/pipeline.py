import argparse
import logging 
import os
#ldir: log directory lfile: log file
ldir = "logs"
lfile = os.path.join(ldir, "pipeline.log")

os.makedirs(ldir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(lfile),
        logging.StreamHandler()
    ],
)

logger = logging.getLogger(__name__)
#info
def run_stage(stage):
    logger.info(f"Running stage: {stage}")
    # placeholder routing
    if stage=="ocr":
        logger.info("OCR stage started")
    elif stage=="preprocess":
        logger.info("preprocessor stage started")
    elif stage=="train_lstm":
        logger.info("train_lstm stage started")
    elif stage=="all":
        logger.info("full pipeline")
    else:
        #warning
        logger.warning(f"Unknown stage: {stage}")
    

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--stage", required = True)
    args = p.parse_args()
    try: #error
        run_stage(args.stage)
    except Exception as e: 
        logger.error(f"Pipeline failed: {e}", exc_info = True)
