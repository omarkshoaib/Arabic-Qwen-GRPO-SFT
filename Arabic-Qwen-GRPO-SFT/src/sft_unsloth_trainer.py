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

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template # For applying chat templates if needed explicitly

from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig #, PeftModel, get_peft_model - Unsloth handles PEFT

from src.data_loader import load_and_prepare_dataset # Use our SFT-compatible data loader

# Configuration
# You might train SFT on a GRPO-trained model or directly on the base model.
# If training on GRPO model, MODEL_NAME would be the path to your GRPO checkpoint.
# If training on base model, MODEL_NAME would be the path to the base model.
BASE_MODEL_NAME = "unsloth/Qwen2-0.5B-Instruct-bnb-4bit" # Changed to Unsloth 4-bit model
# Example: GRPO_OUTPUT_CHECKPOINT = "./grpo_qwen2_0.5b_arabic_unsloth/final_checkpoint"
# MODEL_TO_SFT = GRPO_OUTPUT_CHECKPOINT # Or BASE_MODEL_NAME
DATASET_NAME = "Omartificial-Intelligence-Space/Arabic_Reasoning_Dataset" # Corrected Dataset Name
DRIVE_OUTPUT_BASE = "/content/drive/MyDrive/Arabic-Qwen-Outputs" # Base for Colab outputs
OUTPUT_DIR = os.path.join(DRIVE_OUTPUT_BASE, "sft_qwen2_0.5b_instruct_bnb_4bit_unsloth") # Updated output directory
MAX_SEQ_LENGTH = 1024  # Max sequence length for model

# SFT Training Hyperparameters
SFT_EPOCHS = 3
SFT_BATCH_SIZE = 2 # Keep low for Colab
SFT_GRAD_ACCUMULATION_STEPS = 4
SFT_LEARNING_RATE = 2e-4 # Common for SFT
SFT_LOGGING_STEPS = 10
SFT_OPTIMIZER = "adamw_8bit" # Unsloth recommends paged_adamw_8bit or adamw_8bit
SFT_LR_SCHEDULER_TYPE = "cosine"  # Define missing variable
SFT_WARMUP_RATIO = 0.1
SFT_MAX_GRAD_NORM = 0.3  # Define missing variable
SFT_SAVE_STEPS = 100 # Save checkpoints less frequently

# LoRA Configuration (if used with SFT directly on base model)
# These are ignored if MODEL_TO_SFT is a LoRA checkpoint that gets merged before SFT.
# If SFT is done on a base model, these PEFT settings would be applied.
# For now, assuming MODEL_TO_SFT will be the base model and SFTTrainer will handle LoRA.
# Unsloth's FastLanguageModel.get_peft_model handles this if we are to train LoRA adapters.
# SFTTrainer can also take a PeftConfig.
# Let's ensure LoRA is applied correctly.

R_LORA = 16
LORA_ALPHA = R_LORA * 2
# TARGET_MODULES_LORA = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
# More robust way to get all linear layers for Qwen2
TARGET_MODULES_LORA = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    # "embed_tokens", "lm_head", # Usually not targeted for LoRA
]
LORA_DROPOUT = 0.0 # Changed from 0.05 to 0.0 for Unsloth fast patching compatibility

# Model to SFT - either the base model or a GRPO checkpoint
# IMPORTANT: If GRPO_OUTPUT_CHECKPOINT is a LoRA model, it needs to be merged to base before SFT,
# or SFTTrainer needs to be aware it's training on top of existing adapters.
# For simplicity, let's assume SFT is on the base model, or a fully merged GRPO model.
MODEL_TO_SFT = BASE_MODEL_NAME
# MODEL_TO_SFT = "/content/drive/MyDrive/Arabic-Qwen-Outputs/grpo_qwen2_0.5b_instruct_bnb_4bit_unsloth/final_checkpoint" # Example if GRPO output is a full model

# Helper to check if running in Colab
IS_COLAB = "google.colab" in sys.modules

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)
# =====================================================================================

def main():
    # 1. Load Model and Tokenizer with Unsloth
    # ==================================================
    print(f"DEBUG: Attempting to load model {MODEL_TO_SFT} with dtype=None (auto-detect for 4-bit)") # Modified debug message
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_TO_SFT, # Use MODEL_TO_SFT here
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # Let Unsloth auto-detect for 4-bit model
        load_in_4bit=True,
        # token = "hf_..." # Add your Hugging Face token if loading private models or specific revisions
    )
    print(f"Loaded model {MODEL_TO_SFT} with Unsloth.")

    # Add LoRA adapters for PEFT if training the base model or fine-tuning existing adapters.
    # If MODEL_TO_SFT is a GRPO checkpoint that already has adapters, Unsloth might load them.
    # If it's the base model, we need to add new ones.
    # For simplicity, we assume we are adding/replacing LoRA adapters here.
    model = FastLanguageModel.get_peft_model(
        model,
        r=R_LORA,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=42,
        target_modules=TARGET_MODULES_LORA,
        max_seq_length=MAX_SEQ_LENGTH,
    )
    print("Unsloth model with LoRA adapters prepared for SFT.")

    # Ensure tokenizer's chat template is set up if not done by Unsloth
    # (Unsloth usually handles this, but good to double-check or be explicit if issues arise)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    # Apply chat template (Qwen specific or general chatml)
    # SFTTrainer can often apply this if dataset has 'messages' column and tokenizer has template
    # but explicitly ensuring it here is safer with Unsloth.
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="chatml", # Standard template, worked in GRPO
        mapping={"role": "role", "content": "content", "user": "user", "assistant": "assistant"}, # Consistent mapping
        map_eos_token=True, # Important for some models
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
    print(f"Loaded SFT dataset with {len(train_dataset)} examples (contains 'messages' column).")
    # train_dataset = train_dataset.select(range(200)) # For faster testing

    # Define a function to format messages to a single string and then tokenize
    # This prepares 'input_ids' and 'attention_mask' for SFTTrainer
    def format_and_tokenize_dataset(examples):
        # examples['messages'] is a list of message lists (one for each example in the batch)
        formatted_texts = []
        for single_example_messages in examples['messages']:
            formatted_str = tokenizer.apply_chat_template(
                single_example_messages,
                tokenize=False, # Get the string representation
                add_generation_prompt=False # Appropriate for SFT
            )
            formatted_texts.append(formatted_str)
        
        # Tokenize the batch of formatted texts
        # padding=False because SFTDataCollator with packing=True handles this.
        # truncation=True is essential.
        tokenized_outputs = tokenizer(
            formatted_texts,
            truncation=True,
            padding=False, 
            max_length=MAX_SEQ_LENGTH,
        )
        # tokenized_outputs is a BatchEncoding, which is dict-like
        # and will contain 'input_ids', 'attention_mask'.
        # SFTTrainer with packing=True will automatically create 'labels' from 'input_ids'.
        return tokenized_outputs

    # Apply the formatting and tokenization
    # Remove the original 'messages' column as it's now processed into input_ids/attention_mask
    # and any other columns from the original load that are not input_ids or attention_mask.
    # Be cautious if other columns are needed by the trainer, but typically not with this setup.
    original_columns = train_dataset.column_names
    train_dataset = train_dataset.map(
        format_and_tokenize_dataset, 
        batched=True, 
        remove_columns=[col for col in original_columns if col not in ['input_ids', 'attention_mask']] # Keep only what tokenizer produces + what map might add
    )
    print(f"Formatted and tokenized SFT dataset. Columns: {train_dataset.column_names}")
    if 'input_ids' in train_dataset.column_names and len(train_dataset) > 0:
        print(f"First SFT training example input_ids length: {len(train_dataset[0]['input_ids'])}")
    else:
        print("Warning: 'input_ids' column not found or dataset empty after tokenization. Check formatting function.")

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
        optim=SFT_OPTIMIZER,       # Use 8bit optimizers from bitsandbytes
        # weight_decay=0.01,      # From Unsloth example
        lr_scheduler_type=SFT_LR_SCHEDULER_TYPE, # From Unsloth example
        # warmup_steps=5,           # From Unsloth example
        warmup_ratio=SFT_WARMUP_RATIO, # More common
        max_grad_norm=SFT_MAX_GRAD_NORM, # From Unsloth example
        seed=42,
        fp16=True,             # Enable fp16
        bf16=False,              # Disable bf16
        logging_strategy="steps", # Ensure logging strategy is set
        eval_strategy="no", # No evaluation during SFT for now
        save_strategy="steps",
        save_steps=SFT_SAVE_STEPS,
        save_total_limit=2,
        # dataloader_num_workers=2, # Can sometimes cause issues in Colab
        group_by_length=False, # Faster when False, but True can sometimes help training on varied sequence lengths
        remove_unused_columns=False, # Important for custom tokenized datasets
        # gradient_checkpointing = not IS_COLAB, # True can save memory but slow down, False for Colab if facing issues
        # gradient_checkpointing_kwargs={"use_reentrant": False}, # Recommended with Unsloth if checkpointing
        report_to="none" # Disable wandb/other reporting for now
    )

    # SFTTrainer from TRL
    # Unsloth works seamlessly with TRL's SFTTrainer.
    # It expects a dataset where each example is a list of messages, or a formatting function.
    # Our `load_and_prepare_dataset` with `for_grpo=False` creates a 'messages' column.
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field=None,  # Crucial: dataset is now pre-tokenized with input_ids, attention_mask
        formatting_func=None,     # Dataset is already formatted and tokenized
        args=training_args,
        max_seq_length=MAX_SEQ_LENGTH,
        packing=False, # Packs multiple short examples into one sequence for efficiency - Unsloth recommends this.
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