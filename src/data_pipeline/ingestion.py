from datasets import load_dataset, concatenate_datasets, DatasetDict
from loguru import logger
import os
from dotenv import load_dotenv

load_dotenv()

PRIMARY_DATASET   = "gbharti/finance-alpaca"
SECONDARY_DATASET = "virattt/financial-qa-10K"
CACHE_DIR         = "data/cache" 
RAW_DATA_DIR      = "data/raw/financial_merged"

def download_primary(hf_token=None):
    logger.info(f"Downloading primary dataset: {PRIMARY_DATASET}")
    ds = load_dataset(PRIMARY_DATASET, token=hf_token or os.getenv("HF_TOKEN"), cache_dir=CACHE_DIR)
    return ds

def download_secondary(hf_token=None):
    logger.info(f"Downloading secondary dataset: {SECONDARY_DATASET}")
    ds = load_dataset(SECONDARY_DATASET, token=hf_token or os.getenv("HF_TOKEN"), cache_dir=CACHE_DIR)
    return ds

def normalize_secondary(ds):
    """
    Secondary dataset ke columns ko Primary dataset jaisa banata hai.
    question -> instruction, context -> input, answer -> output
    """
    logger.info("Normalizing secondary dataset columns...")
    
    def _reshape(example):
        return {
            "instruction": example.get("question", ""),
            "input": f"Context from 10-K filing:\n{example.get('context', '')}",
            "output": example.get("answer", ""),
        }

    # Purane columns remove karke naye format mein map kar rahe hain
    normalized_ds = ds.map(_reshape, remove_columns=ds["train"].column_names)
    logger.success("Secondary dataset normalized!")
    return normalized_ds

def merge_and_save(primary_ds, secondary_ds):
    """Dono datasets ko jodata hai, shuffle karta hai, aur disk pe save karta hai"""
    logger.info("Merging datasets...")
    
    # Dono datasets ke 'train' split ko combine karo
    merged_train = concatenate_datasets([primary_ds["train"], secondary_ds["train"]])
    
    # Data ko achhe se fenth (shuffle) lo
    merged_train = merged_train.shuffle(seed=42)
    
    # Final dataset dict banana
    final_dataset = DatasetDict({"train": merged_train})
    logger.success(f"Merged Dataset Ready: {final_dataset}")
    
    # Disk par save karna
    final_dataset.save_to_disk(RAW_DATA_DIR)
    logger.success(f"Data saved locally at: {RAW_DATA_DIR}")

# --- YAHAN SE SCRIPT RUN HOGI ---
if __name__ == "__main__":
    logger.info("Starting Data Download Phase...")
    
    primary_data = download_primary()
    secondary_data = download_secondary()
    
    # Normalization
    secondary_normalized = normalize_secondary(secondary_data)
    
    # Merging & Saving
    merge_and_save(primary_data, secondary_normalized)
    
    logger.info("🎉 Ingestion Phase Completely Done!")
