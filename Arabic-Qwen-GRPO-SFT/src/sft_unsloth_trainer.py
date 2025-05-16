import sys
import os

# Add the project root to sys.path
# Assumes the script is in 'Arabic-Qwen-GRPO-SFT/src/'
# Then __file__ is '.../Arabic-Qwen-GRPO-SFT/src/sft_unsloth_trainer.py'
# os.path.dirname(__file__) is '.../Arabic-Qwen-GRPO-SFT/src/'
# os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) is '.../Arabic-Qwen-GRPO-SFT/'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"DEBUG: __file__ = {__file__}")
print(f"DEBUG: Calculated project_root = {project_root}")
print(f"DEBUG: Current sys.path = {sys.path}")

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model

from src.data_loader import load_and_prepare_dataset  # Use our SFT-compatible data loader

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-0.5B"  # Use standard model, not Unsloth's 4-bit version
DATASET_NAME = "Omartificial-Intelligence-Space/Arabic_Reasoning_Dataset"
DRIVE_OUTPUT_BASE = "/content/drive/MyDrive/Arabic-Qwen-Outputs"
OUTPUT_DIR = os.path.join(DRIVE_OUTPUT_BASE, "sft_qwen2.5_0.5b_standard")
MAX_SEQ_LENGTH = 1024  # Max sequence length for model

# SFT Training Hyperparameters
SFT_EPOCHS = 3
SFT_BATCH_SIZE = 2  # Keep low for Colab
SFT_GRAD_ACCUMULATION_STEPS = 4
SFT_LEARNING_RATE = 2e-4  # Common for SFT
SFT_LOGGING_STEPS = 10
SFT_OPTIMIZER = "adamw_torch"  # Standard AdamW instead of 8bit
SFT_LR_SCHEDULER_TYPE = "cosine"
SFT_WARMUP_RATIO = 0.1
SFT_MAX_GRAD_NORM = 0.3
SFT_SAVE_STEPS = 100  # Save checkpoints less frequently

# LoRA Configuration
R_LORA = 16
LORA_ALPHA = R_LORA * 2
TARGET_MODULES_LORA = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
LORA_DROPOUT = 0.05

# Helper to check if running in Colab
IS_COLAB = "google.colab" in sys.modules

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    # 1. Load Model and Tokenizer (standard way, not using Unsloth)
    # ==================================================
    print(f"DEBUG: Loading model {MODEL_NAME} with standard transformers library")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with standard approach
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,  # Use float16 for T4 GPU
        device_map="auto"
    )
    print(f"Loaded model {MODEL_NAME}")
    
    # Add LoRA adapters
    lora_config = LoraConfig(
        r=R_LORA,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES_LORA
    )
    
    model = get_peft_model(model, lora_config)
    print("Model with LoRA adapters prepared for SFT")

    # 2. Load and Prepare Dataset
    # ==================================================
    train_dataset = load_and_prepare_dataset(
        dataset_name=DATASET_NAME, 
        split="train", 
        for_grpo=False,  # Set to False for SFT formatting
        tokenizer=tokenizer, 
        max_seq_length=MAX_SEQ_LENGTH
    )
    print(f"Loaded SFT dataset with {len(train_dataset)} examples")
    
    # Define a function to format messages to a single string and then tokenize
    def format_and_tokenize_dataset(examples):
        # examples['messages'] is a list of message lists (one for each example in the batch)
        formatted_texts = []
        for single_example_messages in examples['messages']:
            formatted_str = tokenizer.apply_chat_template(
                single_example_messages,
                tokenize=False,  # Get the string representation
                add_generation_prompt=False  # Appropriate for SFT
            )
            formatted_texts.append(formatted_str)
        
        # Tokenize the batch of formatted texts
        tokenized_outputs = tokenizer(
            formatted_texts,
            truncation=True,
            padding=False,
            max_length=MAX_SEQ_LENGTH,
        )
        
        # Create labels for causal language modeling (same as input_ids)
        tokenized_outputs["labels"] = tokenized_outputs["input_ids"].copy()
        
        return tokenized_outputs

    # Apply the formatting and tokenization
    original_columns = train_dataset.column_names
    train_dataset = train_dataset.map(
        format_and_tokenize_dataset, 
        batched=True, 
        remove_columns=[col for col in original_columns if col not in ['input_ids', 'attention_mask']]
    )
    print(f"Formatted and tokenized SFT dataset. Columns: {train_dataset.column_names}")
    
    if 'input_ids' in train_dataset.column_names and len(train_dataset) > 0:
        print(f"First SFT training example input_ids length: {len(train_dataset[0]['input_ids'])}")
    
    # 3. Set up TrainingArguments and Trainer
    # ==================================================
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=SFT_EPOCHS,
        per_device_train_batch_size=SFT_BATCH_SIZE,
        gradient_accumulation_steps=SFT_GRAD_ACCUMULATION_STEPS,
        learning_rate=SFT_LEARNING_RATE,
        logging_steps=SFT_LOGGING_STEPS,
        optim=SFT_OPTIMIZER,
        lr_scheduler_type=SFT_LR_SCHEDULER_TYPE,
        warmup_ratio=SFT_WARMUP_RATIO,
        max_grad_norm=SFT_MAX_GRAD_NORM,
        seed=42,
        fp16=True,  # Enable fp16 for T4 GPU
        bf16=False,  # Disable bf16 as T4 doesn't support it
        logging_strategy="steps",
        eval_strategy="no",  # No evaluation during SFT for now
        save_strategy="steps",
        save_steps=SFT_SAVE_STEPS,
        save_total_limit=2,
        group_by_length=False,  # Faster when False
        report_to="none"  # Disable wandb/other reporting for now
    )

    # Use standard Trainer instead of SFTTrainer to avoid Triton kernels
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100
    )
    
    # Set task to causal language modeling
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False  # Disable KV cache during training for efficiency
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    print("Trainer initialized.")

    # 4. Train the model
    # ==================================================
    print("Starting SFT training...")
    trainer.train()
    print("SFT training finished.")

    # 5. Save the model
    # ==================================================
    final_save_path = os.path.join(OUTPUT_DIR, "final_checkpoint")
    trainer.model.save_pretrained(final_save_path)  # Saves LoRA adapters
    tokenizer.save_pretrained(final_save_path)
    print(f"SFT Model adapters and tokenizer saved to {final_save_path}")

if __name__ == "__main__":
    main()