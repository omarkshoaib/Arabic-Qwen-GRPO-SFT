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
MODEL_NAME = "unsloth/Qwen2.5-0.5B-bnb-4bit"  # Updated Model Name
DATASET_NAME = "Omartificial-Intelligence-Space/Arabic_Reasoning_Dataset"
DRIVE_OUTPUT_BASE = "/content/drive/MyDrive/Arabic-Qwen-Outputs"
OUTPUT_DIR = os.path.join(DRIVE_OUTPUT_BASE, "grpo_qwen2.5_0.5b_bnb_4bit_unsloth") # Updated Output Dir to reflect bnb-4bit
MAX_SEQ_LENGTH = 1024

# LoRA configuration
LORA_R = 16
LORA_ALPHA = LORA_R * 2 # Common practice: alpha = 2 * r
LORA_DROPOUT = 0.0  # GRPO can be sensitive to dropout
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# GRPO Hyperparameters
GRPO_PER_DEVICE_TRAIN_BATCH_SIZE = 8 # Number of prompts to process for generation at once. Adjusted based on Unsloth message.
GRPO_GRADIENT_ACCUMULATION_STEPS = 8 # Effective batch of prompts = BATCH_SIZE * GRAD_ACCUM_STEPS
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
    print("--- Starting GRPO Training ---")
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    current_dir = os.getcwd()
    print(f"Ensuring current working directory for training: {current_dir}")
    print(f"Executing GRPO trainer from: {current_dir}")


    # Load model and tokenizer
    # For "unsloth/Qwen2.5-0.5B-bnb-4bit", load_in_4bit is implied by the model name.
    # Unsloth handles the dtype and 4-bit loading automatically for such models.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        # dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16, # Unsloth handles this
        # load_in_4bit=True, # Unsloth handles this
        # token=os.environ.get("HF_TOKEN"), # if using gated models
    )
    print("Unsloth model with LoRA adapters loaded.")

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
    print("PEFT model configured.")

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

    # Optional: Load a test/validation split if you want to use it with GRPOTrainer
    # eval_dataset = None
    # try:
    #     eval_dataset = load_and_prepare_dataset(
    #         dataset_name=DATASET_NAME, 
    #         tokenizer=tokenizer,
    #         max_seq_length=MAX_SEQ_LENGTH,
    #         for_grpo=True,
    #         split="test"  # Or your validation split name
    #     )
    #     print(f"Test dataset size: {len(eval_dataset)}")
    # except Exception as e:
    #     print(f"Could not load test/validation dataset: {e}. Proceeding without it.")

    reward_config = get_reward_config()

    # Define the reward function for the GRPOTrainer
    # It must take (completions: List[str], **kwargs)
    # kwargs will include the batch elements like `prompt_input_ids`
    def reward_fn_for_trainer(generated_completions_str: list[str], **batch_elements):
        return grpo_reward_function_unsloth(
            completions=generated_completions_str,
            tokenizer=tokenizer,
            reward_config=reward_config,
            **batch_elements # Pass along prompt_input_ids etc.
        )

    # GRPO Configuration is now passed directly to GRPOTrainer
    # grpo_training_args = GRPOConfig(  # ---- REMOVE THIS BLOCK START
    #     output_dir=OUTPUT_DIR,
    #     num_train_epochs=GRPO_EPOCHS,
    #     per_device_train_batch_size=GRPO_PER_DEVICE_TRAIN_BATCH_SIZE,
    #     gradient_accumulation_steps=GRPO_GRADIENT_ACCUMULATION_STEPS,
    #     learning_rate=GRPO_LEARNING_RATE,
    #     logging_steps=GRPO_LOGGING_STEPS,
    #     save_steps=GRPO_SAVE_STEPS,
    #     save_total_limit=2,
    #     report_to="wandb" if "WANDB_API_KEY" in os.environ else "none",
    #     remove_unused_columns=False,
    #     gradient_checkpointing=True,
    #     max_prompt_length=GRPO_MAX_PROMPT_LENGTH,
    #     max_completion_length=GRPO_MAX_NEW_TOKENS,
    #     beta=GRPO_KL_COEFF,
    #     seed=42,
    # ) # ---- REMOVE THIS BLOCK END

    # Initialize GRPOTrainer
    print("Initializing GRPOTrainer with direct keyword arguments for config...")
    trainer = GRPOTrainer(
        model=model,
        # ref_model=None, # Can add a reference model if desired
        # args=grpo_training_args, # REMOVED - passing args directly
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset, # Pass eval_dataset if loaded
        reward_fn=reward_fn_for_trainer,
        # peft_config=lora_config, # Unsloth's get_peft_model handles this

        # Directly pass GRPOConfig/TrainingArguments fields as kwargs:
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
        gradient_checkpointing=True, # From TrainingArguments
        # fp16=not torch.cuda.is_bf16_supported(), # Let Unsloth handle mixed precision
        # bf16=torch.cuda.is_bf16_supported(),     # or set based on support if needed by TRL directly
        
        # GRPOConfig specific fields
        beta=GRPO_KL_COEFF,
        max_prompt_length=GRPO_MAX_PROMPT_LENGTH,
        max_completion_length=GRPO_MAX_NEW_TOKENS, # Corresponds to max_new_tokens for generation
        # loss_type="sigmoid", # Default for GRPO, can be added if needed
        seed=42, # From TrainingArguments
    )
    print("GRPOTrainer initialized.")

    # Start training
    print("Starting training...")
    trainer.train()
    print("Training finished.")

    # Save the final model
    print(f"Saving final model to {OUTPUT_DIR}_final")
    trainer.save_model(f"{OUTPUT_DIR}_final")
    # tokenizer.save_pretrained(f"{OUTPUT_DIR}_final") # Trainer should save tokenizer too
    print("Model saved.")

if __name__ == "__main__":
    main() 