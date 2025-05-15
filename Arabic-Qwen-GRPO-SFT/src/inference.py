import torch
from unsloth import FastLanguageModel
from transformers import TextStreamer

# Project-specific imports (especially the system prompt)
from src.data_loader import SYSTEM_PROMPT_ARABIC_REASONING # Or define it here if preferred

# Configuration
DEFAULT_MODEL_PATH = "Qwen/Qwen2-0.5B-Instruct" # Base model for testing before training
# To load your trained model, change this to the path of your saved LoRA adapters, e.g.:
# TRAINED_MODEL_PATH = "./grpo_qwen2_0.5b_arabic_unsloth/final_checkpoint"
# TRAINED_MODEL_PATH = "./sft_qwen2_0.5b_arabic_unsloth/final_checkpoint"
MODEL_TO_LOAD = DEFAULT_MODEL_PATH # Change this to your checkpoint path after training

MAX_SEQ_LENGTH = 2048 # Should match or exceed training max_seq_length for context
LOAD_IN_4BIT = True

def run_inference(model_path, user_query, system_prompt=SYSTEM_PROMPT_ARABIC_REASONING):
    """
    Loads a Unsloth PEFT model and runs inference using chat templates.

    Args:
        model_path (str): Path to the saved LoRA adapters (or base model name).
        user_query (str): The user's question/prompt.
        system_prompt (str): The system prompt to guide the model.
    """
    # 1. Load Model and Tokenizer with Unsloth
    # ==================================================
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,      # Auto-detect
        load_in_4bit=LOAD_IN_4BIT,
        # token = "hf_..." # Add your Hugging Face token if needed
    )
    print(f"Loaded model from: {model_path}")

    # Check if it's a PEFT model (adapters loaded)
    if hasattr(model, "active_adapters") and model.active_adapters:
        print(f"Active LoRA adapters: {model.active_adapters}")
    elif model_path != DEFAULT_MODEL_PATH: # If a path was specified but no adapters found, it might be an issue
        print("Warning: Loaded a model from a specified path, but no active LoRA adapters found. Ensure the path is correct and contains PEFT adapters.")
    else:
        print("Loaded base model without LoRA adapters.")
    
    # No need to explicitly call get_peft_model if loading a saved PEFT model (Unsloth handles it)
    # FastLanguageModel.for_inference(model) # Prepares model for fast inference (optional, Unsloth does this automatically)

    # 2. Prepare inputs using Chat Template
    # ==================================================
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]

    # `apply_chat_template` tokenizes and formats the input.
    # `add_generation_prompt=True` is crucial for instruction-following models to correctly format the prompt
    # such that the model knows it needs to generate a response.
    input_ids = tokenizer.apply_chat_template(
        messages, 
        tokenize=True, 
        add_generation_prompt=True, # Important for Qwen and other chat models
        return_tensors="pt"
    ).to(model.device)

    # 3. Generate Response
    # ==================================================
    text_streamer = TextStreamer(tokenizer, skip_prompt=True) # To stream output word by word
    print("\nModel Output:")
    
    # Generation parameters
    # These can be tuned. For <think><answer>, you might need a higher max_new_tokens.
    generated_outputs = model.generate(
        input_ids,
        streamer=text_streamer,
        max_new_tokens=512,       # Max tokens to generate
        do_sample=True,           # Whether to sample; False uses greedy decoding
        temperature=0.6,          # Controls randomness. Lower is more deterministic.
        top_p=0.9,                # Nucleus sampling: cumulative probability threshold
        # top_k=50,               # Top-K sampling: number of highest probability tokens to consider
        # repetition_penalty=1.1, # Penalizes repeated tokens
        # eos_token_id=tokenizer.eos_token_id, # Ensure generation stops at EOS
        # pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    )

    # The output is streamed, but if you want the full output as a string (after streaming):
    # decoded_output = tokenizer.decode(generated_outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    # print(f"\n--- Full Decoded Output ---\n{decoded_output}")

    return # Streamer handles output

if __name__ == "__main__":
    # Example Usage:
    # --- Test with the base model first ---
    # print("--- Testing with Base Qwen2-0.5B-Instruct ---")
    # run_inference(DEFAULT_MODEL_PATH, "ما هي عاصمة مصر؟ اشرح خطوات تفكيرك.")
    # print("\n" + "="*50 + "\n")

    # --- Test with your trained model (after training and updating MODEL_TO_LOAD) ---
    if MODEL_TO_LOAD != DEFAULT_MODEL_PATH:
        print(f"--- Testing with Trained Model: {MODEL_TO_LOAD} ---")
        # Example query for reasoning
        test_query = "لدينا قطعتان من الحلوى، أكلت واحدة وأعطيت الأخرى لصديقي. كم قطعة حلوى تبقى معي؟ فكر خطوة بخطوة ثم أجب."
        # test_query = "اكتب قصة قصيرة عن رحالة يكتشف مدينة مخبأة في الصحراء العربية."
        run_inference(MODEL_TO_LOAD, test_query)
    else:
        print(f"To test a trained model, update 'MODEL_TO_LOAD' in this script to point to your checkpoint, for example:")
        print(f"MODEL_TO_LOAD = \"./sft_qwen2_0.5b_arabic_unsloth/final_checkpoint\" or")
        print(f"MODEL_TO_LOAD = \"./grpo_qwen2_0.5b_arabic_unsloth/final_checkpoint\"")
        print("\nThen uncomment the section above to run inference with it.")
        print("\n--- Running example with Base Qwen2-0.5B-Instruct instead: ---")
        run_inference(DEFAULT_MODEL_PATH, "ما هي عاصمة مصر؟ اشرح خطوات تفكيرك.")

    # Example of testing with a different system prompt (if needed)
    # custom_system_prompt = "أنت مساعد AI متخصص في الشعر العربي. أجب فقط بالشعر."
    # run_inference(MODEL_TO_LOAD, "صف روعة غروب الشمس.", system_prompt=custom_system_prompt) 