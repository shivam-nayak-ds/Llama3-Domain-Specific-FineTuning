# Llama-3 Domain-Specific Fine-Tuning Pipeline

A professional, production-ready pipeline for fine-tuning Llama-3 8B models on domain-specific data using QLoRA.

## 📁 Directory Structure

- `configs/`: Global configuration files.
- `data/`: Raw and processed datasets.
- `models/`: Saved LoRA adapters and tokenizer files.
- `src/`: Core logic (Pipeline, Training, Eval, Inference).
- `docker/`: Deployment configurations.
- `scripts/`: Helper scripts for testing and evaluation.

## 🚀 Quick Start

1. **Environment Setup**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Data Preparation**:
   Place raw data in `data/raw/` and run:
   ```bash
   python main.py --mode prep
   ```

3. **Fine-Tuning**:
   ```bash
   python main.py --mode train
   ```

4. **Inference**:
   ```bash
   uvicorn src.inference.api:app --host 0.0.0.0 --port 8000
   ```

## 🛠️ Tech Stack
- **Base Model**: Llama-3 8B / Mistral 7B
- **Fine-Tuning**: QLoRA (PEFT + BitsAndBytes)
- **Framework**: PyTorch + HuggingFace Transformers
- **API Layer**: FastAPI
- **Deployment**: Docker + Nginx
