from unsloth import FastLanguageModel
import os
import torch
# from datasets import load_dataset # Not directly used here, but load_and_prepare_dataset uses it
from transformers import TrainingArguments
from trl import GRPOTrainer, GRPOConfig # GRPOConfig is used for GRPOTrainer args
# from functools import partial # Not strictly needed if reward_fn_for_trainer is defined inline

from unsloth.chat_templates import get_chat_template # Crucial import

# Project-specific imports
from src.data_loader import load_and_prepare_dataset, SYSTEM_PROMPT_ARABIC_REASONING
from src.reward_functions import get_reward_config, grpo_reward_function_unsloth

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_NAME = "Omartificial-Intelligence-Space/Arabic_Reasoning_Dataset"
DRIVE_OUTPUT_BASE = "/content/drive/MyDrive/Arabic-Qwen-Outputs"
OUTPUT_DIR = os.path.join(DRIVE_OUTPUT_BASE, "grpo_qwen2.5_0.5b_arabic_unsloth")
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
    # import numpy as np # Numpy import is at the end for the if __name__ == '__main__' block
    # np.random.seed(RANDOM_SEED) # This should be done where np is imported and used

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        load_in_4bit=True, # Use 4-bit quantization
        # token=os.environ.get("HF_TOKEN"), # if using gated models
    )
    print("Unsloth model with LoRA adapters loaded.")

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
    
    print("Applying 'chatml' chat template to tokenizer...")
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="chatml",
        mapping={"role": "role", "content": "content", "user": "user", "assistant": "assistant"},
        map_eos_token=True,
    )
    if tokenizer.chat_template is None:
        print("CRITICAL WARNING: Chat template is still None after attempting to apply it.")
    else:
        print("Chat template applied successfully.")

    print("Loading and preparing GRPO dataset...")
    dataset = load_and_prepare_dataset(
        dataset_name=DATASET_NAME,
        tokenizer=tokenizer, 
        split="train", 
        # max_seq_length is not directly used by load_and_prepare_dataset itself for GRPO,
        # but the GRPOTrainer max_length/max_prompt_length will handle final truncation.
        # We set for_grpo=True to trigger the correct internal formatting.
        for_grpo=True, 
        system_prompt_template=SYSTEM_PROMPT_ARABIC_REASONING, # This will be used by the internal GRPO formatter
        num_proc=4
    )
    print(f"Loaded and prepared GRPO dataset with {len(dataset)} examples.")
    if len(dataset) > 0 and 'prompt' in dataset.column_names and 'chosen' in dataset.column_names and 'rejected' in dataset.column_names:
        print(f"First example - Prompt: {dataset[0]['prompt'][:100]}...")
        print(f"First example - Chosen: {dataset[0]['chosen'][:100]}...")
        print(f"First example - Rejected: {dataset[0]['rejected'][:100]}...")
    else:
        print("Dataset does not seem to have 'prompt', 'chosen', 'rejected' columns, or is empty. Check data_loader.py")
        return # Exit if dataset is not correctly formatted

    reward_config = get_reward_config()
    
    def reward_fn_for_trainer(generated_completions, **batch_elements):
        rewards = grpo_reward_function_unsloth(
            completions=generated_completions,
            tokenizer=tokenizer, 
            reward_config=reward_config,
            batch=batch_elements 
        )
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
        remove_unused_columns=False,
    )

    print("Initializing GRPOTrainer...")
    trainer = GRPOTrainer(
        model=model,         
        args=training_args,
        tokenizer=tokenizer,
        reward_function=reward_fn_for_trainer,
        train_dataset=dataset,
        beta=GRPO_KL_COEFF,
        max_length=MAX_SEQ_LENGTH,             
        max_prompt_length=GRPO_MAX_PROMPT_LENGTH,    
        max_completion_length=GRPO_MAX_NEW_TOKENS,
    )

    print("Starting GRPO training...")
    trainer.train()

    print("Training finished. Saving final LoRA adapters...")
    final_checkpoint_dir = os.path.join(OUTPUT_DIR, "final_checkpoint")
    trainer.save_model(final_checkpoint_dir)

    print(f"GRPO training complete. Model adapters saved to {final_checkpoint_dir}")

if __name__ == '__main__':
    import numpy as np 
    main() 