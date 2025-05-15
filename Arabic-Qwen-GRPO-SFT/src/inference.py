import os # Ensure os is imported for path joining if used for MODEL_TO_LOAD
import torch
from unsloth import FastLanguageModel
from transformers import TextStreamer

# Project-specific imports (especially the system prompt)
from src.data_loader import SYSTEM_PROMPT_ARABIC_REASONING # Or define it here if preferred

# Configuration
DEFAULT_MODEL_PATH = "Qwen/Qwen2-0.5B-Instruct" # Base model for testing before training

# === How to set MODEL_TO_LOAD to point to a trained checkpoint on Drive ===
# 1. Identify the path to your saved adapters on Google Drive. Examples:
#    DRIVE_OUTPUT_BASE = "/content/drive/MyDrive/Arabic-Qwen-Outputs"
#    SFT_CHECKPOINT_ON_DRIVE = os.path.join(DRIVE_OUTPUT_BASE, "sft_qwen2_0.5b_arabic_unsloth", "final_checkpoint")
#    GRPO_CHECKPOINT_ON_DRIVE = os.path.join(DRIVE_OUTPUT_BASE, "grpo_qwen2_0.5b_arabic_unsloth", "final_checkpoint")
# 2. Then set MODEL_TO_LOAD directly to this path string:
#    MODEL_TO_LOAD = SFT_CHECKPOINT_ON_DRIVE 
#    or
#    MODEL_TO_LOAD = GRPO_CHECKPOINT_ON_DRIVE
# For now, defaulting to base model for template clarity:
MODEL_TO_LOAD = DEFAULT_MODEL_PATH

MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True

def run_inference(model_path, user_query, system_prompt=SYSTEM_PROMPT_ARABIC_REASONING):
    """
    Loads a Unsloth PEFT model and runs inference using chat templates.

    Args:
        model_path (str): Path to the saved LoRA adapters (or base model name).
        user_query (str): The user's question/prompt.
        system_prompt (str): The system prompt to guide the model.
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=LOAD_IN_4BIT,
    )
    print(f"Loaded model from: {model_path}")

    if hasattr(model, "active_adapters") and model.active_adapters:
        print(f"Active LoRA adapters: {model.active_adapters}")
    elif model_path != DEFAULT_MODEL_PATH and not os.path.basename(model_path).startswith("Qwen") : # Heuristic to check if it's a path and not a HF name
        print("Warning: Loaded a model from a specified path, but no active LoRA adapters found. Ensure the path is correct and contains PEFT adapters.")
    else:
        print("Loaded base model without LoRA adapters or a HuggingFace model name.")

    # Apply chat template (Qwen specific or general chatml)
    # Important for instruction-following models
    # Make sure the template matches what was used in training (SFT/GRPO prompt formatting)
    # The apply_chat_template in trainer scripts uses "qwen".
    # Here, we let tokenizer.apply_chat_template handle it based on its loaded config or Unsloth defaults.

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    print("\nModel Output:")

    _ = model.generate(
        input_ids,
        streamer=text_streamer,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    return

if __name__ == "__main__":
    # Default to testing base model unless MODEL_TO_LOAD is changed from DEFAULT_MODEL_PATH
    effective_model_to_test = MODEL_TO_LOAD
    
    if MODEL_TO_LOAD == DEFAULT_MODEL_PATH:
        print(f"--- Testing with Base Model: {DEFAULT_MODEL_PATH} ---")
        print("(To test a trained model, edit 'MODEL_TO_LOAD' in this script to point to your checkpoint path on Drive.)")
        test_query_example = "ما هي عاصمة مصر؟ اشرح خطوات تفكيرك."
    else:
        print(f"--- Testing with Trained Model: {MODEL_TO_LOAD} ---")
        test_query_example = "لدينا قطعتان من الحلوى، أكلت واحدة وأعطيت الأخرى لصديقي. كم قطعة حلوى تبقى معي؟ فكر خطوة بخطوة ثم أجب."

    run_inference(effective_model_to_test, test_query_example)

    # Example of testing with a different system prompt (if needed)
    # custom_system_prompt = "أنت مساعد AI متخصص في الشعر العربي. أجب فقط بالشعر."
    # run_inference(MODEL_TO_LOAD, "صف روعة غروب الشمس.", system_prompt=custom_system_prompt) 