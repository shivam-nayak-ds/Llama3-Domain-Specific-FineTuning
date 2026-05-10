"""
Formatter Module
Transforms structured JSON dataset into Llama-3 specific conversation format.
"""

import os
from typing import Dict, Any
from datasets import load_from_disk
from loguru import logger

# ── Configuration ────────────────────────────────────────────────────────────
INPUT_DATA_PATH = "data/processed/financial_masked"
OUTPUT_DATA_PATH = "data/processed/financial_formatted"
NUM_WORKERS = 4

SYSTEM_PROMPT = "You are an expert financial analyst. Provide precise, data-driven, and professional financial analysis."

class Llama3Formatter:
    """Formats raw instructions/inputs into Llama-3 prompt template."""

    def format_row(self, row: Dict[str, Any]) -> Dict[str, str]:
        """
        Takes instruction, input, and output from the row and
        wraps it in Llama-3's strict <|begin_of_text|> template.
        """
        instruction = row.get("instruction", "").strip()
        context = row.get("input", "").strip()
        response = row.get("output", "").strip()

        # If there is extra context, add it to the user's prompt
        user_message = f"{instruction}\n\n{context}" if context else instruction

        # Constructing the strict Llama-3 Template
        llama3_text = (
            f"<|begin_of_text|>"
            f"<|start_header_id|>system<|end_header_id|>\n\n"
            f"{SYSTEM_PROMPT}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_message}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{response}<|eot_id|>"
        )

        # Return as a single text string
        return {"text": llama3_text}

    def execute(self) -> None:
        logger.info(f"Loading masked data from: {INPUT_DATA_PATH}")
        if not os.path.exists(INPUT_DATA_PATH):
            logger.error(f"Input path not found: {INPUT_DATA_PATH}. Run pii_masker.py first!")
            return
            
        dataset = load_from_disk(INPUT_DATA_PATH)
        
        logger.info("Applying Llama-3 formatting template...")
        # Map template and remove old JSON columns (instruction, input, output)
        formatted_dataset = dataset.map(
            self.format_row, 
            num_proc=NUM_WORKERS,
            remove_columns=dataset["train"].column_names
        )
        
        os.makedirs(os.path.dirname(OUTPUT_DATA_PATH), exist_ok=True)
        formatted_dataset.save_to_disk(OUTPUT_DATA_PATH)
        
        logger.success(f"Formatting complete! Final training data saved to: {OUTPUT_DATA_PATH}")


if __name__ == "__main__":
    formatter = Llama3Formatter()
    formatter.execute()
