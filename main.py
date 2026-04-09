import sys
import os
import argparse

# Ensure the root directory is in sys.path for proper imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils import setup_logging, load_config
from src.data_prep import preprocess
from src.train import train

def main():
    """Main entry point for the Industry-Grade Llama-3 Fine-Tuning Pipeline."""
    
    parser = argparse.ArgumentParser(description="Llama-3 Domain Specific Fine-Tuning Pipeline")
    parser.add_argument("--stage", type=str, choices=["preprocess", "train", "all"], default="all",
                        help="Which stage of the pipeline to run.")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to the configuration file.")
    
    args = parser.parse_args()
    
    # 1. Load Configuration
    config = load_config(args.config)
    
    # 2. Setup professional logging
    logger = setup_logging(config)
    
    logger.info("Starting Industry-Grade Llama-3 Fine-Tuning Pipeline")
    logger.info(f"Project: {config['project']['name']} v{config['project']['version']}")

    try:
        # 3. Data Preprocessing
        if args.stage in ["preprocess", "all"]:
            logger.info("Stage 1: Data Preprocessing...")
            preprocess(config)
        
        # 4. Model Training
        if args.stage in ["train", "all"]:
            logger.info("Stage 2: Model Training...")
            train(config)
        
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"? Pipeline failed: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
