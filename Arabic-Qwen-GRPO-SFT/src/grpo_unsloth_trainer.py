import os
import torch
# from datasets import load_dataset # Not directly used here, but load_and_prepare_dataset uses it
from transformers import TrainingArguments
from trl import GRPOTrainer, GRPOConfig # GRPOConfig is used for GRPOTrainer args
# from functools import partial # Not strictly needed if reward_fn_for_trainer is defined inline

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template # Crucial import

# Project-specific imports
from src.data_loader import load_and_prepare_dataset, SYSTEM_PROMPT_ARABIC_REASONING, format_for_grpo_unsloth
from src.reward_functions import get_reward_config, grpo_reward_function_unsloth

# Configuration
MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"
DATASET_NAME = "Omartificial-Intelligence-Space/Arabic_Reasoning_Dataset"
DRIVE_OUTPUT_BASE = "/content/drive/MyDrive/Arabic-Qwen-Outputs"
OUTPUT_DIR = os.path.join(DRIVE_OUTPUT_BASE, "grpo_qwen2_0.5b_arabic_unsloth")
MAX_SEQ_LENGTH = 1024

# LoRA configuration
LORA_R = 16
LORA_ALPHA = LORA_R * 2 # Common practice: alpha = 2 * r
LORA_DROPOUT = 0.0  # GRPO can be sensitive to dropout
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# GRPO Hyperparameters
GRPO_PER_DEVICE_TRAIN_BATCH_SIZE = 2 # Number of prompts to process for generation at once
GRPO_GRADIENT_ACCUMULATION_STEPS = 8 # Effective batch of prompts = BATCH_SIZE * GRAD_ACCUM_STEPS
GRPO_LEARNING_RATE = 1e-5 # Or 5e-6, common for PPO/DPO/GRPO
GRPO_EPOCHS = 1
GRPO_LOGGING_STEPS = 1
GRPO_SAVE_STEPS = 50 # Save more frequently for long runs
GRPO_KL_COEFF = 0.05 # KL coefficient (beta in GRPOConfig)
GRPO_MAX_PROMPT_LENGTH = MAX_SEQ_LENGTH // 2 # Max length for prompt part (e.g., 512 if MAX_SEQ_LENGTH is 1024)
GRPO_MAX_NEW_TOKENS = MAX_SEQ_LENGTH // 2    # Max length for generated completion (e.g., 512)

# Trainer settings
LOAD_IN_4BIT = True
DTYPE = None # None for auto detection by Unsloth
RANDOM_SEED = 42

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing=True, # "unsloth" is also an option
        random_state=RANDOM_SEED,
        target_modules=LORA_TARGET_MODULES,
        max_seq_length=MAX_SEQ_LENGTH,
    )
    print("Unsloth model with LoRA adapters loaded.")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    
    # Apply chat template to the tokenizer using Unsloth's utility
    print("Applying 'qwen' chat template to tokenizer...")
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="qwen",  # Explicitly use "qwen"
        mapping={"role": "role", "content": "content", "name": "name"}, # Standard mapping
        map_eos_token=True, # Recommended for consistency
    )
    if tokenizer.chat_template is None:
        print("CRITICAL WARNING: Chat template is still None after attempting to apply it.")
        # Potentially raise an error or fallback if this is critical
    else:
        print("Chat template applied successfully.")

    print("Loading and preparing GRPO dataset...")
    dataset = load_and_prepare_dataset(
        dataset_name=DATASET_NAME,
        tokenizer=tokenizer, # Pass tokenizer for potential use in formatting_func
        split="train", # Use a small slice like "train[:100]" or "train[:1%]"] for initial testing
        # max_seq_length=MAX_SEQ_LENGTH, # Now handled by GRPOTrainer args for prompt/completion length
        formatting_func_grpo=format_for_grpo_unsloth, # Your custom func from data_loader
        system_prompt_template=SYSTEM_PROMPT_ARABIC_REASONING,
        num_proc=4 # Number of processes for dataset mapping
    )
    print(f"Loaded and prepared GRPO dataset with {len(dataset)} examples.")
    if len(dataset) > 0 and 'prompt' in dataset.column_names and 'chosen' in dataset.column_names and 'rejected' in dataset.column_names:
        print(f"First example - Prompt: {dataset[0]['prompt'][:100]}...")
        print(f"First example - Chosen: {dataset[0]['chosen'][:100]}...")
        print(f"First example - Rejected: {dataset[0]['rejected'][:100]}...")
    else:
        print("Dataset does not seem to have 'prompt', 'chosen', 'rejected' columns, or is empty. Check data_loader.py")
        return

    reward_config = get_reward_config()
    
    def reward_fn_for_trainer(generated_completions, **batch_elements):
        # batch_elements will contain the tokenized prompts (e.g., 'prompt_input_ids')
        # Our grpo_reward_function_unsloth expects this as `batch` in its kwargs
        rewards = grpo_reward_function_unsloth(
            completions=generated_completions,
            tokenizer=tokenizer, 
            reward_config=reward_config,
            batch=batch_elements # Pass the whole batch_elements dict as 'batch'
        )
        # GRPOTrainer expects a torch tensor of rewards
        return rewards if isinstance(rewards, torch.Tensor) else torch.tensor(rewards, dtype=torch.float32)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=GRPO_EPOCHS,
        per_device_train_batch_size=GRPO_PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRPO_GRADIENT_ACCUMULATION_STEPS,
        learning_rate=GRPO_LEARNING_RATE,
        logging_steps=GRPO_LOGGING_STEPS,
        save_steps=GRPO_SAVE_STEPS,
        save_total_limit=2,
        report_to="tensorboard",
        seed=RANDOM_SEED,
        remove_unused_columns=False, # GRPOTrainer needs original columns for reward function batch
        # bf16=torch.cuda.is_bf16_supported(), # Unsloth handles dtype
        # fp16=not torch.cuda.is_bf16_supported(),
    )

    print("Initializing GRPOTrainer...")
    trainer = GRPOTrainer(
        model=model,         
        args=training_args,    # Pass TrainingArguments directly
        tokenizer=tokenizer,
        reward_function=reward_fn_for_trainer,
        train_dataset=dataset, # Dataset with 'prompt', 'chosen', 'rejected'
        # peft_config=None, # PEFT is already part of the model object
        beta=GRPO_KL_COEFF, # KL coefficient for GRPO loss
        max_length=MAX_SEQ_LENGTH,             
        max_prompt_length=GRPO_MAX_PROMPT_LENGTH,    
        max_completion_length=GRPO_MAX_NEW_TOKENS, # Max length of generated response
    )

    print("Starting GRPO training...")
    trainer.train()

    print("Training finished. Saving final LoRA adapters...")
    final_checkpoint_dir = os.path.join(OUTPUT_DIR, "final_checkpoint")
    trainer.save_model(final_checkpoint_dir) # Saves LoRA adapters
    # tokenizer.save_pretrained(final_checkpoint_dir) # Usually saved by trainer.save_model

    print(f"GRPO training complete. Model adapters saved to {final_checkpoint_dir}")

if __name__ == '__main__':
    import numpy as np # Add numpy import if used for random.seed
    main() 