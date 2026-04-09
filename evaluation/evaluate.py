import json
import pandas as pd
from src.utils import get_logger

logger = get_logger()

def evaluate_model(predictions, ground_truth):
    """
    Calculates Accuracy and generates a comparison report.
    """
    correct = 0
    total = len(predictions)
    report = []

    for pred, truth in zip(predictions, ground_truth):
        is_correct = (pred.lower() == truth.lower())
        if is_correct:
            correct += 1
        
        report.append({
            "Truth": truth,
            "Prediction": pred,
            "Match": "✅" if is_correct else "❌"
        })

    accuracy = (correct / total) * 100
    
    # Generate Table
    df = pd.DataFrame(report)
    print("\n" + "="*40)
    print("?? MODEL EVALUATION REPORT ??")
    print("="*40)
    print(df.to_string(index=False))
    print("-" * 40)
    print(f"?? FINAL ACCURACY: {accuracy:.2f}%")
    print("="*40 + "\n")

if __name__ == "__main__":
    # Sample Mock Data for Demonstration
    sample_preds = ["Fraud", "Legitimate", "Fraud", "Legitimate"]
    sample_truth = ["Fraud", "Legitimate", "Legitimate", "Legitimate"]
    
    evaluate_model(sample_preds, sample_truth)
