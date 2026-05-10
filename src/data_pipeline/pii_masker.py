"""
PII Masker Module
Responsible for identifying and masking Personally Identifiable Information (PII)
in the dataset before fine-tuning to ensure compliance and privacy.
"""

import os
from typing import Dict, Any
from datasets import load_from_disk
from loguru import logger
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine

# ── Configuration ────────────────────────────────────────────────────────────
INPUT_DATA_PATH = "data/raw/financial_validated"
OUTPUT_DATA_PATH = "data/processed/financial_masked"
NLP_MODEL_NAME = "en_core_web_sm"
TARGET_ENTITIES = ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"]
NUM_WORKERS = 4


class PIIMasker:
    """Handles PII detection and anonymization using Microsoft Presidio."""

    def __init__(self) -> None:
        logger.info(f"Initializing PII Masker with NLP engine: {NLP_MODEL_NAME}")
        
        nlp_configuration = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": NLP_MODEL_NAME}],
        }
        provider = NlpEngineProvider(nlp_configuration=nlp_configuration)
        nlp_engine = provider.create_engine()
        
        self.analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
        self.anonymizer = AnonymizerEngine()

    def mask_text(self, text: str) -> str:
        """
        Detects and masks PII entities in a single text string.
        """
        if not text or len(str(text)) < 5:
            return text
            
        results = self.analyzer.analyze(
            text=str(text), 
            language='en', 
            entities=TARGET_ENTITIES
        )
        anonymized = self.anonymizer.anonymize(
            text=str(text), 
            analyzer_results=results
        )
        return anonymized.text

    def process_row(self, row: Dict[str, Any]) -> Dict[str, str]:
        """
        Applies masking to all relevant text columns in a dataset row.
        """
        return {
            "instruction": self.mask_text(row.get("instruction", "")),
            "input": self.mask_text(row.get("input", "")),
            "output": self.mask_text(row.get("output", ""))
        }

    def execute(self) -> None:
        """
        Executes the full masking pipeline on the validated dataset.
        """
        logger.info(f"Loading validated data from: {INPUT_DATA_PATH}")
        if not os.path.exists(INPUT_DATA_PATH):
            logger.error(f"Input path not found: {INPUT_DATA_PATH}")
            return
            
        dataset = load_from_disk(INPUT_DATA_PATH)
        
        logger.info(f"Starting PII anonymization using {NUM_WORKERS} workers...")
        masked_dataset = dataset.map(self.process_row, num_proc=NUM_WORKERS)
        
        os.makedirs(os.path.dirname(OUTPUT_DATA_PATH), exist_ok=True)
        masked_dataset.save_to_disk(OUTPUT_DATA_PATH)
        
        logger.success(f"PII Masking complete. Artifacts saved to: {OUTPUT_DATA_PATH}")


if __name__ == "__main__":
    masker = PIIMasker()
    masker.execute()
