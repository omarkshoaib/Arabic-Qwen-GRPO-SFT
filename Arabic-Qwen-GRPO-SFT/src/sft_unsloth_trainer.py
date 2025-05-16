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

import torch

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template # For applying chat templates if needed explicitly

from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer

# Project-specific imports
from src.data_loader import load_and_prepare_dataset

# Configuration
# You might train SFT on a GRPO-trained model or directly on the base model.
# If training on GRPO model, MODEL_NAME would be the path to your GRPO checkpoint.
BASE_MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct" 
# Example: GRPO_OUTPUT_CHECKPOINT = "./grpo_qwen2_0.5b_arabic_unsloth/final_checkpoint"
# MODEL_TO_SFT = GRPO_OUTPUT_CHECKPOINT # Or BASE_MODEL_NAME
DATASET_NAME = "Omartificial-Intelligence-Space/Arabic_Reasoning_Dataset" # Corrected Dataset Name
DRIVE_OUTPUT_BASE = "/content/drive/MyDrive/Arabic-Qwen-Outputs" # Base for Colab outputs
OUTPUT_DIR = os.path.join(DRIVE_OUTPUT_BASE, "sft_qwen2_0.5b_arabic_unsloth")
MAX_SEQ_LENGTH = 1024  # Max sequence length for model

# LoRA configuration (Unsloth defaults)
LORA_R = 16 # Keep same as GRPO or tune
LORA_ALPHA = 32
LORA_DROPOUT = 0.0 # Changed from 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# SFT Training Hyperparameters
SFT_BATCH_SIZE = 2 # Adjust based on VRAM
SFT_GRAD_ACCUMULATION_STEPS = 4
SFT_LEARNING_RATE = 2e-4 # SFT can often use a slightly higher LR than PPO/GRPO
SFT_EPOCHS = 3 # SFT usually involves more epochs
SFT_LOGGING_STEPS = 10
SFT_WARMUP_RATIO = 0.03
SFT_LR_SCHEDULER_TYPE = "linear"
SFT_OPTIMIZER = "adamw_8bit" # Unsloth recommended for memory saving

def main():
    # 1. Load Model and Tokenizer with Unsloth
    # ==================================================
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # Auto-detect
        load_in_4bit=True,
        # token = "hf_..." # Add your Hugging Face token if loading private models or specific revisions
    )
    print(f"Loaded model {BASE_MODEL_NAME} with Unsloth.")

    # Add LoRA adapters for PEFT if training the base model or fine-tuning existing adapters.
    # If MODEL_TO_SFT is a GRPO checkpoint that already has adapters, Unsloth might load them.
    # If it's the base model, we need to add new ones.
    # For simplicity, we assume we are adding/replacing LoRA adapters here.
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=42,
        target_modules=LORA_TARGET_MODULES,
        max_seq_length=MAX_SEQ_LENGTH,
    )
    print("Unsloth model with LoRA adapters prepared for SFT.")

    # Pad token and chat template handling (similar to GRPO trainer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    # Apply chat template (Qwen specific or general chatml)
    # SFTTrainer can often apply this if dataset has 'messages' column and tokenizer has template
    # but explicitly ensuring it here is safer with Unsloth.
    tokenizer = FastLanguageModel.apply_chat_template(
        tokenizer,
        template="qwen", # Or "chatml"
        tokenize=False, # SFTTrainer will handle tokenization of the formatted text
    )
    if tokenizer.chat_template is None:
        print("Warning: Chat template not set on tokenizer for SFT. SFTTrainer might have issues.")

    # Unsloth's SFTTrainer is good at using the chat template automatically if the dataset
    # is formatted with a 'messages' column (list of dicts) or via a formatting_func.
    # We will use the 'messages' column from our data_loader.

    # 2. Load and Prepare Dataset
    # ==================================================
    train_dataset = load_and_prepare_dataset(
        dataset_name=DATASET_NAME, 
        split="train", 
        for_grpo=False, # <<< Set to False for SFT formatting
        tokenizer=tokenizer, 
        max_seq_length=MAX_SEQ_LENGTH
    )
    # train_dataset = train_dataset.select(range(200)) # For faster testing
    print(f"Loaded and prepared SFT dataset with {len(train_dataset)} examples.")
    print(f"First SFT training example messages: {train_dataset[0]['messages']}")

    # 3. Set up TrainingArguments and SFTTrainer
    # ==================================================
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=SFT_EPOCHS,
        per_device_train_batch_size=SFT_BATCH_SIZE,
        gradient_accumulation_steps=SFT_GRAD_ACCUMULATION_STEPS,
        learning_rate=SFT_LEARNING_RATE,
        logging_steps=SFT_LOGGING_STEPS,
        # optim="adamw_torch", # Standard AdamW
        optim=SFT_OPTIMIZER, # Use Unsloth's 8bit AdamW for memory efficiency
        lr_scheduler_type=SFT_LR_SCHEDULER_TYPE,
        warmup_ratio=SFT_WARMUP_RATIO,
        save_strategy="epoch",
        # bf16=True, # If not using 4-bit and have bf16 support
        # fp16=False,
        remove_unused_columns=False, # Keep `messages` column for SFTTrainer
        gradient_checkpointing=True, # Already set in get_peft_model
        report_to="tensorboard",
    )

    # SFTTrainer from TRL
    # Unsloth works seamlessly with TRL's SFTTrainer.
    # It expects a dataset where each example is a list of messages, or a formatting function.
    # Our `load_and_prepare_dataset` with `for_grpo=False` creates a 'messages' column.
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field=None,  # Not needed if dataset has 'messages' or formatting_func is used
                                  # If you had a single string column, you'd specify it here.
        formatting_func=None,     # Not needed as our dataset already has 'messages' column in Unsloth/TRL SFT format.
                                  # If formatting_func is provided, it takes a dataset example and returns a list of strings.
                                  # Unsloth examples often directly prepare a text column after applying chat template.
                                  # However, passing 'messages' column directly to SFTTrainer is also a common pattern for TRL.
        args=training_args,
        max_seq_length=MAX_SEQ_LENGTH,
        packing=True, # Packs multiple short examples into one sequence for efficiency - Unsloth recommends this.
                      # `packing=True` is generally preferred with `SFTDataCollator`.
        # dataset_kwargs={"skip_prepare_dataset" : True}, # Added because Unsloth example did so
    )
    print("SFTTrainer initialized.")

    # 4. Train the model
    # ==================================================
    print("Starting SFT training...")
    trainer.train()
    print("SFT training finished.")

    # 5. Save the model
    # ==================================================
    final_save_path = os.path.join(OUTPUT_DIR, "final_checkpoint")
    trainer.model.save_pretrained(final_save_path) # Saves LoRA adapters
    tokenizer.save_pretrained(final_save_path)
    print(f"SFT Model adapters and tokenizer saved to {final_save_path}")

    # For merging and saving the full model (optional, requires more RAM/CPU)
    # if hasattr(model, "merge_and_unload"):
    #     print("Merging adapters to save full model...")
    #     merged_model_path = os.path.join(OUTPUT_DIR, "final_merged_model")
    #     merged_model = trainer.model.merge_and_unload()
    #     merged_model.save_pretrained(merged_model_path)
    #     tokenizer.save_pretrained(merged_model_path)
    #     print(f"Merged SFT model saved to {merged_model_path}")
    # else:
    #     print("Standard PEFT model saving. For Unsloth merged model, manually load and merge if needed.")

if __name__ == "__main__":
    main() 