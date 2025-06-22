from unsloth import FastLanguageModel
import os
import torch
# from datasets import load_dataset # Not directly used here, but load_and_prepare_dataset uses it
from transformers import TrainingArguments
from trl import GRPOTrainer, GRPOConfig # GRPOConfig is used for GRPOTrainer args
# from functools import partial # Not strictly needed if reward_fn_for_trainer is defined inline

# NOTE: PatchFastRL import removed - latest Unsloth handles GRPO patching automatically

# Now import Unsloth chat templates
from unsloth.chat_templates import get_chat_template # Crucial import

# Project-specific imports
from src.data_loader import load_and_prepare_dataset, SYSTEM_PROMPT_ARABIC_REASONING
from src.reward_functions import get_reward_config, grpo_reward_function_unsloth

# Configuration
# TRAINING PIPELINE PATH FLOW:
# 1. SFT starts with: "unsloth/Qwen2.5-0.5B-Instruct" (instruction-tuned base model)
# 2. SFT saves to: /content/drive/MyDrive/Arabic-Qwen-Outputs/sft_qwen2.5_0.5b_instruct_unsloth/final_checkpoint
# 3. GRPO loads from: /content/drive/MyDrive/Arabic-Qwen-Outputs/sft_qwen2.5_0.5b_instruct_unsloth/final_checkpoint (SFT output) ← THIS FILE
# 4. GRPO saves to: /content/drive/MyDrive/Arabic-Qwen-Outputs/grpo_on_sft_qwen2.5_0.5b_bnb_4bit_unsloth ← THIS FILE

# Base model for SFT training (instruction-tuned)
SFT_BASE_MODEL_NAME = "unsloth/Qwen2.5-0.5B-Instruct" # This is the instruction-tuned base model for SFT

# Output directory for SFT training (MUST match sft_unsloth_trainer.py)
DRIVE_OUTPUT_BASE = "/content/drive/MyDrive/Arabic-Qwen-Outputs"
SFT_OUTPUT_DIR = os.path.join(DRIVE_OUTPUT_BASE, "sft_qwen2.5_0.5b_instruct_unsloth") # Updated to match new SFT path
SFT_FINAL_CHECKPOINT_PATH = os.path.join(SFT_OUTPUT_DIR, "final_checkpoint")

# GRPO will train on the SFT-tuned model
MODEL_NAME = SFT_FINAL_CHECKPOINT_PATH  # Load from SFT checkpoint instead of base model
DATASET_NAME = "Omartificial-Intelligence-Space/Arabic_Reasoning_Dataset"
OUTPUT_DIR = os.path.join(DRIVE_OUTPUT_BASE, "grpo_on_sft_qwen2.5_0.5b_bnb_4bit_unsloth") # Updated for SFT->GRPO pipeline
MAX_SEQ_LENGTH = 1024

# LoRA configuration
LORA_R = 16
LORA_ALPHA = LORA_R * 2 # Common practice: alpha = 2 * r
LORA_DROPOUT = 0.0  # GRPO can be sensitive to dropout
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# GRPO Hyperparameters
GRPO_PER_DEVICE_TRAIN_BATCH_SIZE = 32 # Reduced for faster debugging on L4
GRPO_GRADIENT_ACCUMULATION_STEPS = 8 # Effective batch of 256
GRPO_LEARNING_RATE = 1e-5 # Or 5e-6, common for PPO/DPO/GRPO
GRPO_EPOCHS = 1
GRPO_LOGGING_STEPS = 1
GRPO_SAVE_STEPS = 50 # Save more frequently for long runs
GRPO_KL_COEFF = 0.05 # KL coefficient (beta in GRPOConfig)
GRPO_MAX_PROMPT_LENGTH = MAX_SEQ_LENGTH // 2 # Max length for prompt part (e.g., 512 if MAX_SEQ_LENGTH is 1024)
GRPO_MAX_NEW_TOKENS = MAX_SEQ_LENGTH // 2    # Max length for generated completion (e.g., 512)

# Trainer settings
# DTYPE = None # Let Unsloth decide, or set to torch.bfloat16 if available
# LOAD_IN_4BIT = True # Already specified by unsloth model name typically

def main():
    global MODEL_NAME  # Declare global at the beginning of function
    
    print("--- Starting GRPO Training ---")
    
    # NOTE: Latest Unsloth versions handle GRPO patching automatically
    # No need for manual PatchFastRL calls - fast_inference=True enables vLLM integration
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    current_dir = os.getcwd()
    print(f"Ensuring current working directory for training: {current_dir}")
    print(f"Executing GRPO trainer from: {current_dir}")

    # DEBUG: Check if SFT checkpoint exists
    print(f"Checking for SFT checkpoint at: {MODEL_NAME}")
    if os.path.exists(MODEL_NAME):
        print(f"✅ SFT checkpoint found at {MODEL_NAME}")
        # List files in checkpoint directory
        checkpoint_files = os.listdir(MODEL_NAME)
        print(f"Checkpoint contains: {checkpoint_files}")
    else:
        print(f"❌ SFT checkpoint NOT found at {MODEL_NAME}")
        print(f"Available directories in {DRIVE_OUTPUT_BASE}:")
        if os.path.exists(DRIVE_OUTPUT_BASE):
            available_dirs = os.listdir(DRIVE_OUTPUT_BASE)
            print(f"Available: {available_dirs}")
        else:
            print(f"Output base directory {DRIVE_OUTPUT_BASE} does not exist!")
        
        print("Falling back to base SFT model instead of checkpoint...")
        MODEL_NAME = SFT_BASE_MODEL_NAME

    # Load model and tokenizer from SFT checkpoint
    print(f"Loading model from: {MODEL_NAME}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        fast_inference=True,  # Enable vLLM integration for GRPO
        # dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16, # Unsloth handles this
        # load_in_4bit=True, # Unsloth handles this
        # token=os.environ.get("HF_TOKEN"), # if using gated models
    )
    print("✅ Unsloth model loaded successfully.")

    # Apply chat template
    # Using "chatml" as it's a common base for Qwen models and a Unsloth default
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="chatml", # Standard template
        mapping={"role": "role", "content": "content", "user": "user", "assistant": "assistant"}, # Corrected dictionary
        map_eos_token=True, # Important for some models
    )
    print("Applying 'chatml' chat template to tokenizer...")
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        print("Chat template applied successfully.")
    else:
        # Fallback or specific setting if get_chat_template doesn't set it as expected
        # For Qwen/ChatML, this is typically the format.
        # This part might be redundant if get_chat_template works as expected.
        tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'system' %}{{ '<|im_start|>system\\n' + message['content'] + '<|im_end|>' + '\\n' }}{% elif message['role'] == 'user' %}{{ '<|im_start|>user\\n' + message['content'] + '<|im_end|>' + '\\n' }}{% elif message['role'] == 'assistant' %}{{ '<|im_start|>assistant\\n' + message['content'] + '<|im_end|>' + '\\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
        print("Manually applied ChatML template structure to tokenizer.chat_template.")

    # PEFT setup
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        use_gradient_checkpointing="unsloth", # True or "unsloth" for memory saving
        random_state=3407,
        max_seq_length=MAX_SEQ_LENGTH,
    )
    print("✅ PEFT model configured.")
    
    # NOTE: fast_inference=True already handles vLLM setup automatically
    # Removed problematic: model = FastLanguageModel.for_inference(model)

    # Load and prepare dataset
    print("Loading and preparing GRPO dataset...")
    # Load the main training split
    train_dataset = load_and_prepare_dataset(
        dataset_name=DATASET_NAME, 
        tokenizer=tokenizer,
        max_seq_length=MAX_SEQ_LENGTH,
        for_grpo=True,
        split="train" # Explicitly load the train split
    )
    print("Dataset loaded and prepared.")
    print(f"Train dataset size: {len(train_dataset)}")

    # Debug: Check dataset format
    print("Debug: Checking dataset format...")
    print(f"Dataset columns: {train_dataset.column_names}")
    if len(train_dataset) > 0:
        first_example = train_dataset[0]
        print(f"First example keys: {list(first_example.keys())}")
        if 'prompt' in first_example:
            print(f"Prompt type: {type(first_example['prompt'])}")
            print(f"First prompt sample: {first_example['prompt'][:200]}...")
        if 'chosen' in first_example:
            print(f"Chosen type: {type(first_example['chosen'])}")
        if 'rejected' in first_example:
            print(f"Rejected type: {type(first_example['rejected'])}")

    reward_config = get_reward_config()

    # Define the reward function for the GRPOTrainer
    # It must align with how Unsloth's GRPOTrainer calls it.
    def reward_fn_for_trainer(prompts: list[str], completions: list[str], **batch_elements):
        # batch_elements will contain other items from the batch
        return grpo_reward_function_unsloth(
            completions=completions,
            tokenizer=tokenizer,
            reward_config=reward_config,
            prompts=prompts,
            **batch_elements
        )

    # GRPO Configuration - Reinstated
    grpo_training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=GRPO_EPOCHS,
        per_device_train_batch_size=GRPO_PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRPO_GRADIENT_ACCUMULATION_STEPS,
        learning_rate=GRPO_LEARNING_RATE,
        logging_steps=GRPO_LOGGING_STEPS,
        save_steps=GRPO_SAVE_STEPS,
        save_total_limit=2,
        report_to="wandb" if "WANDB_API_KEY" in os.environ else "none",
        remove_unused_columns=False, # Important for GRPO
        gradient_checkpointing=True,
        max_grad_norm=1.0, # Added for gradient clipping
        # fp16=not torch.cuda.is_bf16_supported(), # Let Unsloth handle mixed precision
        # bf16=torch.cuda.is_bf16_supported(),
        max_prompt_length=GRPO_MAX_PROMPT_LENGTH,
        max_completion_length=GRPO_MAX_NEW_TOKENS, # Corresponds to max_new_tokens for TRL GRPOTrainer
        beta=GRPO_KL_COEFF,
        seed=42,
        generation_kwargs={
            "max_tokens": GRPO_MAX_NEW_TOKENS,
            "temperature": 1.0,
            "top_p": 0.9,
            "stop_token_ids": [tokenizer.eos_token_id] if tokenizer.eos_token_id else None,
        },
        # NOTE: vLLM configuration is handled automatically by fast_inference=True
    )

    # Initialize GRPOTrainer
    print("Initializing GRPOTrainer with GRPOConfig and reward_funcs...")
    trainer = GRPOTrainer(
        model=model,
        args=grpo_training_args, # Pass the config object here
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset, # Pass eval_dataset if loaded
        reward_funcs=reward_fn_for_trainer, # Pass reward function separately
        # peft_config=lora_config, # Unsloth's get_peft_model handles this

        # Removed direct keyword arguments for config as they are now in grpo_training_args
    )
    print("✅ GRPOTrainer initialized.")

    # Start training
    print("Starting training...")
    print("Note: Based on Unsloth docs, expect rewards to increase after at least 300 steps.")
    total_steps = len(train_dataset) // (GRPO_PER_DEVICE_TRAIN_BATCH_SIZE * GRPO_GRADIENT_ACCUMULATION_STEPS) * GRPO_EPOCHS
    print(f"Current training will run for {total_steps} steps (may need more for significant improvements).")
    
    try:
        trainer.train()
        print("Training finished.")

        # Save the final model
        print(f"Saving final model to {OUTPUT_DIR}_final")
        trainer.save_model(f"{OUTPUT_DIR}_final")
        # tokenizer.save_pretrained(f"{OUTPUT_DIR}_final") # Trainer should save tokenizer too
        print("Model saved.")
        
    except Exception as e:
        print(f"Error during training: {e}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        
        # Try to save any progress
        try:
            print(f"Attempting to save checkpoint to {OUTPUT_DIR}_error_checkpoint")
            trainer.save_model(f"{OUTPUT_DIR}_error_checkpoint")
            print("Error checkpoint saved.")
        except:
            print("Could not save error checkpoint.")
        
        raise e

if __name__ == "__main__":
    main()
