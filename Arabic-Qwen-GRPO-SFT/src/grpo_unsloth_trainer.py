import os
import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import GRPOTrainer, GRPOConfig
from functools import partial

from unsloth import FastLanguageModel

# Project-specific imports
from src.data_loader import load_and_prepare_dataset, SYSTEM_PROMPT_ARABIC_REASONING # Ensure SYSTEM_PROMPT is available if needed directly
from src.reward_functions import get_reward_config, grpo_reward_function_unsloth

# Configuration
MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct" 
DATASET_NAME = "omartificial/OmArtificial-Dolly-instruct-5k"
OUTPUT_DIR = "./grpo_qwen2_0.5b_arabic_unsloth"
MAX_SEQ_LENGTH = 1024 # Max sequence length for model

# LoRA configuration (Unsloth defaults)
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# GRPO Training Hyperparameters
GRPO_BATCH_SIZE = 2 # Adjust based on VRAM. This is per_device_train_batch_size.
GRPO_GRAD_ACCUMULATION_STEPS = 4 # Effective batch size = BATCH_SIZE * GRAD_ACCUMULATION_STEPS
GRPO_LEARNING_RATE = 1e-5 # Common learning rate for RLHF style training
GRPO_EPOCHS = 1
GRPO_LOGGING_STEPS = 1
GRPO_GENERATIONS_PER_PROMPT = 4 # Number of completions to generate for each prompt
GRPO_MAX_PROMPT_LENGTH = 512 # Max length of the prompt part
GRPO_MAX_GENERATION_LENGTH = 512 # Max length of the generated part
GRPO_KL_COEFF = 0.1 # KL coefficient for GRPO loss, might need tuning

def main():
    # 1. Load Model and Tokenizer with Unsloth
    # ==================================================
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # Auto-detect from model config (e.g., torch.bfloat16 if available)
        load_in_4bit=True, # Use 4-bit quantization for memory saving
    )

    # Add LoRA adapters for PEFT
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing=True, # Saves memory
        random_state=42,
        target_modules=LORA_TARGET_MODULES,
        max_seq_length=MAX_SEQ_LENGTH, 
    )
    print("Unsloth model with LoRA adapters loaded.")

    # Ensure tokenizer has a pad token, and it's set correctly for Qwen2
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token # Common practice
        model.config.pad_token_id = tokenizer.pad_token_id
    
    # Check if a chat template is set, Qwen2 should have one by default with HF transformers > 4.39
    if tokenizer.chat_template is None:
        print("Warning: Tokenizer chat template not found. Manually setting a basic one or ensure your Transformers version is up to date.")
        # Example for Qwen2 (this is simplified, usually loaded from tokenizer_config.json)
        # For Unsloth, it should automatically handle this. If not, this is a fallback.
        # tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'system' %}{{ 'system\n' + message['content'] + '\n' }}{% elif message['role'] == 'user' %}{{ 'user\n' + message['content'] + '\n' }}{% elif message['role'] == 'assistant' %}{{ 'assistant\n' + message['content'] + '\n' }}{% endif %}{% if loop.last and message['role'] != 'assistant' %}{{ 'assistant\n' }}{% endif %}{% endfor %}"
        # It's better to rely on Unsloth/HF to load this correctly.

    # 2. Load and Prepare Dataset
    # ==================================================
    # The data_loader will return a dataset with a 'messages' column
    train_dataset = load_and_prepare_dataset(
        dataset_name=DATASET_NAME, 
        split="train", 
        for_grpo=True
    )
    # train_dataset = train_dataset.select(range(100)) # For faster testing, select a subset
    print(f"Loaded and prepared GRPO dataset with {len(train_dataset)} examples.")
    print(f"First GRPO training example messages: {train_dataset[0]['messages']}")

    # The GRPOTrainer from TRL expects a text column for prompts.
    # We need to apply the chat template to our 'messages' column to create this text prompt column.
    # Unsloth's apply_chat_template can be used here.
    # Note: GRPOTrainer will tokenize this text prompt internally.
    def create_prompt_text(examples):
        # examples is a batch from the dataset
        all_prompts = []
        for i in range(len(examples['messages'])):
            # Get the messages for the current example
            current_messages = examples['messages'][i]
            # Apply chat template. Add generation prompt is True for inference/generation.
            all_prompts.append(tokenizer.apply_chat_template(current_messages, tokenize=False, add_generation_prompt=True))
        return {"prompt_text": all_prompts}

    train_dataset = train_dataset.map(create_prompt_text, batched=True, num_proc=4) # Using num_proc for speed if dataset is large
    print(f"Created 'prompt_text' column. First example: \n{train_dataset[0]['prompt_text']}")

    # 3. Define Reward Function
    # ==================================================
    reward_config = get_reward_config() # Get defaults from reward_functions.py
    # You can override parts of reward_config here if needed for this specific run
    # Example: reward_config["weights"]["length"] = 0.05
    
    # Partially apply the tokenizer and config to our grpo_reward_function_unsloth
    # so it has the signature `reward_fn(completions, **kwargs)` where kwargs contains the batch.
    # The GRPOTrainer passes the tokenized batch in kwargs.
    # Our grpo_reward_function_unsloth is designed to get the tokenizer and config directly.
    # The `completions` are generated by GRPOTrainer. `**kwargs` will include the batch.
    # GRPOTrainer will internally pass: generated_responses (list of str), and the batch.
    # Our wrapper expects (completions, tokenizer, reward_config, batch=batch)
    
    # The `GRPOTrainer` expects `reward_fn(generated_responses: List[str], **kwargs) -> List[torch.Tensor]`
    # where `kwargs` contains the batch. Our `grpo_reward_function_unsloth` is designed for this if we
    # partially apply the static tokenizer and reward_config.
    
    # `grpo_reward_function_unsloth` expects (completions, tokenizer, reward_config, **kwargs_with_batch)
    # We need to create a wrapper that GRPOTrainer can call as `func(completions, **batch_kwargs)`
    def reward_fn_for_trainer(generated_completions, **batch_elements):
        # `batch_elements` will contain the tokenized prompts etc. from the original dataset batch
        # E.g., batch_elements might be {'prompt_text': ..., 'input_ids': ..., 'attention_mask': ...}
        # Our `grpo_reward_function_unsloth` needs this as `batch` inside its kwargs.
        rewards = grpo_reward_function_unsloth(
            completions=generated_completions,
            tokenizer=tokenizer, 
            reward_config=reward_config,
            batch=batch_elements # Pass the whole batch elements dict as 'batch'
        )
        return torch.tensor(rewards, dtype=torch.float3_2)

    # 4. Set up GRPOConfig and GRPOTrainer
    # ==================================================
    grpo_config = GRPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=GRPO_EPOCHS,
        logging_steps=GRPO_LOGGING_STEPS,
        gradient_accumulation_steps=GRPO_GRAD_ACCUMULATION_STEPS,
        per_device_train_batch_size=GRPO_BATCH_SIZE, # This is batch size of prompts
        learning_rate=GRPO_LEARNING_RATE,
        remove_unused_columns=False, # Important if your dataset has other columns rewards might need
        # generation_kwargs for GRPOTrainer.generate_completions
        max_length=GRPO_MAX_PROMPT_LENGTH + GRPO_MAX_GENERATION_LENGTH, # Max length for model.generate context + generation
        max_new_tokens=GRPO_MAX_GENERATION_LENGTH, # Max tokens for the generated completion part
        max_prompt_length=GRPO_MAX_PROMPT_LENGTH, # Max length of the prompt to feed to model.generate
        num_generations=GRPO_GENERATIONS_PER_PROMPT, # Number of completions per prompt (k)
        kl_coeff=GRPO_KL_COEFF,
        # Unsloth related or Peft specific arguments might not go directly here but affect the model.
        # Check Unsloth GRPO examples if specific TRL args are needed for Unsloth integration.
        # bf16=True, # If not using 4-bit and have bf16 support
        # fp16=False, # If using bf16
        gradient_checkpointing=True, # Already set in get_peft_model
        report_to="tensorboard", # or wandb
        # query_dataset maps to the column name in your dataset that holds the prompt string.
        # Our `create_prompt_text` creates a 'prompt_text' column.
        query_dataset_mapper_column="prompt_text", 
    )

    trainer = GRPOTrainer(
        model=model,         # The PEFT model from Unsloth
        args=grpo_config,    # GRPOConfig
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        reward_fn=reward_fn_for_trainer, # Your reward function
        # data_collator=None, # GRPOTrainer has its own default collator
    )
    print("GRPOTrainer initialized.")

    # 5. Train the model
    # ==================================================
    print("Starting GRPO training...")
    trainer.train()
    print("GRPO training finished.")

    # 6. Save the model
    # ==================================================
    # Unsloth provides a convenient way to save PEFT models, 
    # which can then be loaded easily for inference or further SFT.
    final_save_path = os.path.join(OUTPUT_DIR, "final_checkpoint")
    # model.save_pretrained(final_save_path) # Standard Hugging Face PEFT save
    # tokenizer.save_pretrained(final_save_path)
    # For Unsloth, it might be better to use its saving method if it merges adapters or has specific formats.
    # Check Unsloth docs for preferred way to save LoRA model for inference.
    # Typically, you can save the LoRA adapters like this:
    trainer.model.save_pretrained(final_save_path) # This saves the adapters
    tokenizer.save_pretrained(final_save_path)
    print(f"Model adapters and tokenizer saved to {final_save_path}")
    
    # If you want to save a merged model (for easier deployment without PEFT):
    # merged_model_path = os.path.join(OUTPUT_DIR, "final_merged_model")
    # merged_model = model.merge_and_unload() # This requires enough RAM to merge
    # merged_model.save_pretrained(merged_model_path)
    # tokenizer.save_pretrained(merged_model_path)
    # print(f"Merged model saved to {merged_model_path}")

if __name__ == "__main__":
    main() 