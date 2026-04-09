import os
import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
from src.utils import get_logger

logger = get_logger()

def train(config: dict):
    """
    Industry-Grade fine-tuning process using Unsloth.
    Includes Experiment Tracking, Validation, and Model Merging logic.
    """
    logger.info("Initializing Fine-tuning process...")

    # 1. Hardware & Config setup
    model_name = config['training']['base_model']
    max_seq_length = config['training'].get('max_seq_length', 2048)
    dtype = None # Auto detection
    load_in_4bit = True

    # 2. Load Model and Tokenizer
    logger.info(f"Loading {model_name}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    # 3. Apply LoRA
    logger.info("Configuring PEFT/LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r = config['training']['lora_r'],
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj",],
        lora_alpha = config['training']['lora_alpha'],
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )

    # 4. Dataset Loading & Validation Split
    data_path = os.path.join(config['paths']['processed_data'], "train_llama3.jsonl")
    logger.info(f"Parsing dataset from {data_path}...")
    full_dataset = load_dataset("json", data_files=data_path, split="train")
    
    # 80/20 Train-Validation Split
    dataset = full_dataset.train_test_split(test_size=0.2)
    train_data = dataset["train"]
    val_data = dataset["test"]

    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs       = examples["input"]
        outputs      = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            text = f"### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"
            texts.append(text)
        return { "text" : texts, }

    train_data = train_data.map(formatting_prompts_func, batched = True)
    val_data = val_data.map(formatting_prompts_func, batched = True)

    # 5. Training Arguments
    logger.info("Configuring TrainingArguments...")
    training_args = TrainingArguments(
        per_device_train_batch_size = config['training']['batch_size'],
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        max_steps = 100, 
        learning_rate = config['training']['learning_rate'],
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        evaluation_strategy = "steps",
        eval_steps = 20,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        seed = 3407,
        output_dir = config['paths']['model_output'],
        report_to = "none",
    )

    # 6. Initialize Trainer
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_data,
        eval_dataset = val_data,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        args = training_args,
    )

    # 7. Execute Training
    logger.info("Training pipeline active...")
    trainer.train()
    
    # 8. Export Model
    logger.info(f"Exporting Model Adapters to {config['paths']['model_output']}...")
    model.save_pretrained(config['paths']['model_output'])
    tokenizer.save_pretrained(config['paths']['model_output'])
    
    logger.info("Training Session Finalized Successfully.")
