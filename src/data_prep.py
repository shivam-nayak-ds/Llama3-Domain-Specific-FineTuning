import os
import pandas as pd
import json
from tqdm import tqdm
from src.utils import get_logger

logger = get_logger()

def create_prompt(row):
    """
    Converts a dataset row into a natural language prompt and response for Llama-3.
    """
    # Define features for the prompt
    features = {
        "Amount": f"${row['TransactionAmt']}",
        "Product Code": row['ProductCD'],
        "Card Type": f"{row['card4']} {row['card6']}",
        "Email Domain": row['P_emaildomain'],
        "Address Code": row['addr1'],
        "Distance": row['dist1'],
        "Device Type": row['DeviceType'],
        "Device Info": row['DeviceInfo'],
        "Browser": row['id_31'],
        "Operating System": row['id_30']
    }
    
    # Filter out nulls/NaNs to keep prompt clean
    feature_str = ", ".join([f"{k}: {v}" for k, v in features.items() if pd.notnull(v)])
    
    instruction = "Analyze the following transaction details and determine if it is likely to be fraudulent (\'Fraud\') or legitimate (\'Legitimate\')."
    input_text = f"Transaction Details: {feature_str}"
    output_text = "Fraud" if row['isFraud'] == 1 else "Legitimate"
    
    # Llama-3 Instruction Format
    return {
        "instruction": instruction,
        "input": input_text,
        "output": output_text
    }

def preprocess(config: dict):
    """
    Handles data loading, merging, cleaning, sampling, and formatting for Llama-3 fine-tuning.
    """
    # Use path from config or default if missing
    raw_dir = os.path.dirname(config['paths'].get('raw_data', "data/raw/train_transaction.csv"))
    # Check if 'ieee-fraud-detection' subfolder exists (Kaggle style)
    if os.path.exists(os.path.join(raw_dir, "ieee-fraud-detection")):
        raw_dir = os.path.join(raw_dir, "ieee-fraud-detection")
    
    output_dir = config['paths']['processed_data']
    os.makedirs(output_dir, exist_ok=True)
    
    trans_path = os.path.join(raw_dir, "train_transaction.csv")
    id_path = os.path.join(raw_dir, "train_identity.csv")
    
    logger.info("Starting data preprocessing for IEEE-CIS Fraud Detection dataset...")
    
    if not os.path.exists(trans_path):
        logger.error(f"Transaction file not found: {trans_path}")
        return

    # 1. Load Data (Selecting only 10-15 key features to save memory)
    cols_trans = ['TransactionID', 'isFraud', 'TransactionAmt', 'ProductCD', 'card4', 'card6', 'P_emaildomain', 'addr1', 'dist1', 'M4', 'M6']
    cols_id = ['TransactionID', 'DeviceType', 'DeviceInfo', 'id_31', 'id_30']
    
    logger.info("Reading CSV files...")
    df_trans = pd.read_csv(trans_path, usecols=cols_trans)
    
    if os.path.exists(id_path):
        df_id = pd.read_csv(id_path, usecols=cols_id)
        # 2. Merge Transaction and Identity data
        logger.info("Merging transaction and identity datasets...")
        df = pd.merge(df_trans, df_id, on='TransactionID', how='left')
    else:
        logger.warning("Identity file not found. Preprocessing with transaction data only.")
        df = df_trans

    # 3. Sampling (20,000 to 30,000 rows as requested)
    sample_size = min(25000, len(df))
    logger.info(f"Sampling {sample_size} rows from {len(df)} total rows...")
    
    # Stratified sampling to ensure we have enough fraud cases
    fraud_df = df[df['isFraud'] == 1]
    legit_df = df[df['isFraud'] == 0]
    
    num_fraud = min(len(fraud_df), sample_size // 2)
    num_legit = sample_size - num_fraud
    
    df_sampled = pd.concat([
        fraud_df.sample(num_fraud, random_state=42),
        legit_df.sample(num_legit, random_state=42)
    ]).sample(frac=1, random_state=42) # Shuffle

    # 4. Convert to Llama-3 Instruction Format
    logger.info("Formatting data for Llama-3 fine-tuning...")
    processed_data = []
    for _, row in tqdm(df_sampled.iterrows(), total=len(df_sampled)):
        processed_data.append(create_prompt(row))

    # 5. Save Processed Data
    output_file = os.path.join(output_dir, "train_llama3.jsonl")
    logger.info(f"Saving processed data to {output_file}...")
    
    with open(output_file, 'w') as f:
        for entry in processed_data:
            f.write(json.dumps(entry) + '\n')
    
    logger.info("Preprocessing completed successfully!")
