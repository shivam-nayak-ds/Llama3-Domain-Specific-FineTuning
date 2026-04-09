import logging
import yaml
import os
from logging.handlers import RotatingFileHandler
from typing import Any, Dict

def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Loads a YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def setup_logging(config: Dict[str, Any] = None) -> logging.Logger:
    """Sets up a professional logging system with console and file handlers."""
    if config is None:
        try:
            config = load_config()
        except:
            config = {}

    log_settings = config.get("logging", {})
    log_level = log_settings.get("level", "INFO").upper()
    log_format = log_settings.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    log_file = log_settings.get("log_file", "logs/pipeline.log")

    # Ensure logs directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Initialize Logger
    logger = logging.getLogger("Llama3-FineTuning")
    logger.setLevel(log_level)

    # Prevent duplicate handlers
    if logger.hasHandlers():
        return logger

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)

    # Rotating File Handler (Max 10MB per file, keep 5 backups)
    file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
    file_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(file_handler)

    return logger

def get_logger():
    """Returns the existing logger or initializes a default one."""
    logger = logging.getLogger("Llama3-FineTuning")
    if not logger.hasHandlers():
        setup_logging()
    return logger

