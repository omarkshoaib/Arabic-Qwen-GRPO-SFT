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
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer
from transformers import TrainingArguments

from src.data_loader import load_and_prepare_dataset  # Use our SFT-compatible data loader

# Configuration
# TRAINING PIPELINE PATH FLOW:
# 1. SFT starts with: "unsloth/Qwen2.5-0.5B-Instruct" (instruction-tuned base model) ← THIS FILE
# 2. SFT saves to: /content/drive/MyDrive/Arabic-Qwen-Outputs/sft_qwen2.5_0.5b_instruct_unsloth/final_checkpoint ← THIS FILE
# 3. GRPO loads from: /content/drive/MyDrive/Arabic-Qwen-Outputs/sft_qwen2.5_0.5b_instruct_unsloth/final_checkpoint (SFT output)
# 4. GRPO saves to: /content/drive/MyDrive/Arabic-Qwen-Outputs/grpo_on_sft_qwen2.5_0.5b_bnb_4bit_unsloth

MODEL_NAME = "unsloth/Qwen2.5-0.5B-Instruct"  # Use Unsloth's instruction-tuned model
DATASET_NAME = "Omartificial-Intelligence-Space/Arabic_Reasoning_Dataset"
DRIVE_OUTPUT_BASE = "/content/drive/MyDrive/Arabic-Qwen-Outputs"
SFT_OUTPUT_DIR = os.path.join(DRIVE_OUTPUT_BASE, "sft_qwen2.5_0.5b_instruct_unsloth") # Updated to reflect new model
OUTPUT_DIR = SFT_OUTPUT_DIR # Use the consistent SFT output directory
MAX_SEQ_LENGTH = 1024  # Max sequence length for model

# SFT Training Hyperparameters
SFT_EPOCHS = 3
SFT_BATCH_SIZE = 2  # Keep low for Colab
SFT_GRAD_ACCUMULATION_STEPS = 4
SFT_LEARNING_RATE = 2e-4  # Common for SFT
SFT_LOGGING_STEPS = 10
SFT_OPTIMIZER = "adamw_8bit"  # Use 8bit optimizer with Unsloth
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
LORA_DROPOUT = 0.0  # Set to 0 for Unsloth

# Helper to check if running in Colab
IS_COLAB = "google.colab" in sys.modules

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    print("--- Starting SFT Training with Unsloth ---")
    
    # 1. Load Model and Tokenizer with Unsloth
    # ==================================================
    print(f"Loading model {MODEL_NAME} with Unsloth FastLanguageModel")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # Let Unsloth decide the best dtype
        load_in_4bit=True,  # Enable 4-bit quantization for memory efficiency
    )
    print(f"✅ Loaded model {MODEL_NAME} with Unsloth")
    
    # Apply chat template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="chatml",  # Use chatml template for consistency with GRPO
        mapping={"role": "role", "content": "content", "user": "user", "assistant": "assistant"},
        map_eos_token=True,
    )
    print("✅ Applied chatml chat template to tokenizer")
    
    # Enable LoRA for fine-tuning
    model = FastLanguageModel.get_peft_model(
        model,
        r=R_LORA,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES_LORA,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Use unsloth's gradient checkpointing
        random_state=3407,
        max_seq_length=MAX_SEQ_LENGTH,
    )
    print("✅ Model with LoRA adapters prepared for SFT")

    # 2. Load and Prepare Dataset
    # ==================================================
    train_dataset = load_and_prepare_dataset(
        dataset_name=DATASET_NAME, 
        split="train", 
        for_grpo=False,  # Set to False for SFT formatting
        tokenizer=tokenizer, 
        max_seq_length=MAX_SEQ_LENGTH
    )
    print(f"✅ Loaded SFT dataset with {len(train_dataset)} examples")
    
    # Check the first example to ensure proper formatting
    if len(train_dataset) > 0:
        print("Sample formatted data:")
        print(f"  First example keys: {list(train_dataset[0].keys())}")
        if 'messages' in train_dataset[0]:
            print(f"  First example messages: {train_dataset[0]['messages']}")

    # Define formatting function for Unsloth SFTTrainer
    def formatting_prompts_func(examples):
        """
        Format the conversation data for Unsloth SFTTrainer.
        This function takes a batch of examples and returns formatted text strings.
        """
        texts = []
        for messages in examples["messages"]:
            # Apply chat template to convert messages to text
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,  # Return string, not tokens
                add_generation_prompt=False  # Don't add generation prompt for SFT
            )
            texts.append(text)
        return {"text": texts}

    # Apply the formatting function to the dataset
    train_dataset = train_dataset.map(
        formatting_prompts_func,
        batched=True,
        remove_columns=train_dataset.column_names  # Remove original columns, keep only 'text'
    )
    print(f"✅ Formatted dataset for Unsloth SFTTrainer. New columns: {train_dataset.column_names}")
    
    # Show a sample of the formatted text
    if len(train_dataset) > 0:
        print("Sample formatted text:")
        print(f"  Text length: {len(train_dataset[0]['text'])}")
        print(f"  First 200 chars: {train_dataset[0]['text'][:200]}...")

    # 3. Set up TrainingArguments and SFTTrainer
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
        fp16=not torch.cuda.is_bf16_supported(),  # Use fp16 if bf16 not available
        bf16=torch.cuda.is_bf16_supported(),      # Use bf16 if available
        logging_strategy="steps",
        eval_strategy="no",  # No evaluation during SFT for now
        save_strategy="steps",
        save_steps=SFT_SAVE_STEPS,
        save_total_limit=2,
        group_by_length=True,   # Enable for efficiency with Unsloth
        report_to="none",       # Disable wandb/other reporting for now
        remove_unused_columns=False,  # Important for SFTTrainer
    )

    # Use SFTTrainer from TRL with Unsloth model
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",  # The field containing the formatted conversation text
        formatting_func=None,  # We already formatted the data, so no need for additional formatting
        max_seq_length=MAX_SEQ_LENGTH,
        packing=False,  # Set to False for better compatibility
    )
    print("✅ SFTTrainer initialized")

    # 4. Train the model
    # ==================================================
    print("Starting SFT training...")
    trainer.train()
    print("✅ SFT training finished")

    # 5. Save the model
    # ==================================================
    final_save_path = os.path.join(OUTPUT_DIR, "final_checkpoint")
    print(f"Saving SFT model to: {final_save_path}")
    
    # Ensure the directory exists
    os.makedirs(final_save_path, exist_ok=True)
    
    # Save LoRA adapters and tokenizer
    trainer.model.save_pretrained(final_save_path)  # Saves LoRA adapters
    tokenizer.save_pretrained(final_save_path)
    
    print(f"✅ SFT Model adapters and tokenizer saved to {final_save_path}")
    print(f"📁 Saved files: {os.listdir(final_save_path)}")
    print(f"🔗 This checkpoint will be loaded by GRPO trainer as: {final_save_path}")
    print(f"⚠️  IMPORTANT: Run GRPO training next to continue on this fine-tuned model!")

if __name__ == "__main__":
    main()
