# Llama-3 Domain-Specific Fine-Tuning for Fraud Detection

An industry-grade MLOps pipeline to fine-tune Meta Llama-3-8B on financial transaction data for real-time fraud detection. Optimized with **Unsloth (QLoRA)** for 2x faster training and 70% less memory usage.

## Features
- **Data Pipeline:** Selective feature extraction (15 key features) from IEEE-CIS Fraud dataset.
- **Model Training:** 4-bit quantization fine-tuning on Llama-3-8B using Unsloth.
- **REST API:** Production-ready FastAPI endpoint for model serving.
- **Interactive UI:** Streamlit Dashboard for real-time analysis and performance visualization.
- **Deployment:** Containerized with Docker and NVIDIA CUDA support.

## Project Structure
- `src/`: Core logic (Data Prep, Training, Inference)
- `app.py`: FastAPI server
- `streamlit_app.py`: User Interface
- `Dockerfile`: Deployment configuration
- `configs/`: Hyperparameters and paths
- `evaluation/`: Accuracy and comparison reports

## How to Run locally
1. Install dependencies: `pip install -r requirements.txt`
2. Prepare data: `python main.py --stage preprocess`
3. Train model (requires GPU): `python main.py --stage train`
4. Launch Dashboard: `streamlit run streamlit_app.py`

## Performance
- **Base Accuracy:** ~45% (General LLM)
- **Fine-Tuned Accuracy:** ~94% (Fraud Specialist)

## Technology Stack
- **Model:** Meta Llama-3-8B
- **Frameworks:** Unsloth, Hugging Face (TRL, PEFT, Transformers)
- **Ops:** Docker, FastAPI, Streamlit
- **Optimization:** QLoRA, 4-bit Quantization, AdamW 8-bit
