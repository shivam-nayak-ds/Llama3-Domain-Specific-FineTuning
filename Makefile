from datasets import load_dataset
from loguru import logger
import os
from dotenv import load_dotenv
from src.utils.formatting import format_to_alpaca

load_dotenv()

def download_dataset(dataset_name, cache_dir="data"):
    logger.info(f"Downloading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, cache_dir=cache_dir)
    logger.info(f"Dataset downloaded: {dataset_name}")
    return dataset

def format_dataset(dataset, output_dir="data"):
    logger.info(f"Formatting dataset")
    formatted_dataset = dataset.map(format_to_alpaca)
    logger.info(f"Dataset formatted")
    return formatted_dataset

def save_dataset(dataset, output_dir="data"):
    logger.info(f"Saving dataset")
    dataset.save_to_disk(output_dir)
    logger.info(f"Dataset saved to {output_dir}")