from datasets import load_from_disk, Dataset, DatasetDict
from loguru import logger
import pandas as pd
import os

INPUT_PATH = "data/raw/financial_merged"
OUTPUT_PATH = "data/raw/financial_validated"

def clean_and_validate():
    try:
        # 1. Load
        logger.info(f"Loading dataset from {INPUT_PATH}...")
        ds = load_from_disk(INPUT_PATH)
        df = ds["train"].to_pandas()
        initial_count = len(df)

        # 2. Drop Text Column (Safe check)
        if "text" in df.columns:
            df = df.drop(columns=["text"])
            logger.info("Purged legacy 'text' column.")

        # 3. Drop Nulls/Duplicates in core columns
        df = df.dropna(subset=["instruction", "output"])
        df = df.drop_duplicates(subset=["instruction", "input", "output"])

        # 4. Correct Filtering Logic
        # We only require instruction and output to be > 10 chars. 
        # Input can be empty (that's normal for general logic).
        mask = (df["instruction"].str.len() > 10) & (df["output"].str.len() > 10)
        df = df[mask]
        
        # 5. Final Stats & Save
        final_count = len(df)
        logger.success(f"Validation Complete! Kept {final_count} / {initial_count} samples.")

        if not os.path.exists(os.path.dirname(OUTPUT_PATH)):
            os.makedirs(os.path.dirname(OUTPUT_PATH))

        # Converting back to HF format for Phase 4
        clean_ds = DatasetDict({"train": Dataset.from_pandas(df.reset_index(drop=True))})
        clean_ds.save_to_disk(OUTPUT_PATH)
        logger.info(f"Dataset saved to {OUTPUT_PATH}")

    except Exception as e:
        logger.error(f"Critical error in validation pipeline: {e}")

if __name__ == "__main__":
    clean_and_validate()



        






