import os
import torch
from unsloth import FastLanguageModel
from src.utils import get_logger, load_config

logger = get_logger()

class LlamaInferrer:
    def __init__(self, model_path: str = None):
        """
        Initializes the inference engine using Unsloth for fast 4-bit inference.
        """
        self.config = load_config()
        self.model_path = model_path or self.config['paths']['model_output']
        self.max_seq_length = 2048
        self.dtype = None
        self.load_in_4bit = True

        logger.info(f"Checking for fine-tuned model at: {self.model_path}")
        
        # Verify model exists to prevent crash
        if not os.path.exists(os.path.join(self.model_path, "adapter_config.json")):
            logger.error(f"Fine-tuned model not found at {self.model_path}!")
            raise FileNotFoundError(f"Model adapter files missing in {self.model_path}. Please run training first.")

        # Load the model and tokenizer
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = self.model_path,
            max_seq_length = self.max_seq_length,
            dtype = self.dtype,
            load_in_4bit = self.load_in_4bit,
        )
        FastLanguageModel.for_inference(self.model) # Enable native 2x faster inference

    def predict_fraud(self, transaction_details: str) -> str:
        """
        Predicts if a transaction is fraudulent or legitimate.
        """
        prompt = f"### Instruction:\nAnalyze the following transaction details and determine if it is likely to be fraudulent ('Fraud') or legitimate ('Legitimate').\n\n### Input:\n{transaction_details}\n\n### Response:\n"
        
        inputs = self.tokenizer([prompt], return_tensors = "pt").to("cuda")
        
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens = 64,
            use_cache = True
        )
        
        response = self.tokenizer.batch_decode(outputs)
        # Extract response after the prompt
        result = response[0].split("### Response:\n")[-1].strip()
        return result

if __name__ == "__main__":
    # Example Test Case
    inferrer = LlamaInferrer()
    test_transaction = "Amount: $295.0, Product Code: W, Card Type: visa debit, Address Code: 315.0, Distance: 8.0"
    
    print("\n--- Fraud Detection Test ---")
    print(f"Details: {test_transaction}")
    prediction = inferrer.predict_fraud(test_transaction)
    print(f"Prediction: {prediction}")
    print("----------------------------\n")
