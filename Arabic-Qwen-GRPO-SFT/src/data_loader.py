from datasets import load_dataset
import re

# System prompt (consistent with your notebooks and reward functions)
# This prompt guides the model to use <think> and <answer> tags and respond in Arabic.
SYSTEM_PROMPT_ARABIC_REASONING = (
    "محادثة بين المستخدم والمساعد. يطرح المستخدم سؤالاً، ويقوم المساعد بحله. "
    "يفكر المساعد أولاً في عملية التفكير ثم يقدم الإجابة للمستخدم. "
    "يتم وضع عملية التفكير والإجابة داخل علامتي <think> </think> و <answer> </answer> على التوالي، أي: "
    "<think>عملية التفكير هنا</think><answer>الإجابة هنا</answer>. "
    "يجب أن تكون الإجابة باللغة العربية فقط وأن تكون مفصلة وواضحة."
)

def normalize_arabic_numbers(text):
    """Converts English ASCII numbers in a string to Arabic-Indic numerals."""
    if not text: return ""
    number_map = {
        '0': '٠', '1': '١', '2': '٢', '3': '٣', '4': '٤',
        '5': '٥', '6': '٦', '7': '٧', '8': '٨', '9': '٩'
    }
    return "".join([number_map.get(char, char) for char in str(text)])

def format_example_for_grpo(example, tokenizer):
    """
    Formats a single example from the omartificial dataset for GRPO training with Unsloth.
    Outputs a dictionary containing a 'prompt' string (from applying chat template to messages)
    and the original 'messages' list.
    """
    instruction = example.get("instruction", "")
    context = example.get("context", "")

    # Combine context and instruction if context exists
    user_query = instruction
    if context and context.strip():
        user_query = f"{context}\\n\\n{instruction}" # Context first, then instruction
    
    user_query = normalize_arabic_numbers(user_query) # Normalize numbers in the query

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_ARABIC_REASONING},
        {"role": "user", "content": user_query.strip()},
    ]

    # Apply chat template to create the prompt string
    # add_generation_prompt=True is crucial for the model to know it needs to generate a response.
    prompt_string = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True # Ensures the template ends with assistant's turn to generate
    )
    
    return {
        "prompt": prompt_string, # This is what the trainer seems to want
        "messages": messages,    # Keep original messages for potential later use or clarity
        # GRPO datasets also often have 'chosen' and 'rejected' for *preference pairs*
        # but this initial dataset is for prompts. The trainer generates completions.
        # Adding empty placeholders for chosen/rejected in case the trainer expects them,
        # though for the prompt dataset, they might not be strictly necessary until pairs are formed.
        "chosen": "", 
        "rejected": ""
    }

def format_example_for_sft(example, tokenizer, max_seq_length):
    """
    Formats a single example for Supervised Fine-Tuning (SFT) using Unsloth.
    This typically involves creating a single text string that includes the prompt and the answer,
    formatted according to the model's chat template.
    """
    instruction = example.get("instruction", "")
    context = example.get("context", "")
    answer_text = example.get("answer", "")

    user_query = instruction
    if context and context.strip():
        user_query = f"{context}\\n\\n{instruction}"
        
    user_query = normalize_arabic_numbers(user_query)
    answer_text = normalize_arabic_numbers(answer_text)
    
    # Simple formatting for SFT: try to extract reasoning and final answer if possible,
    # otherwise use the whole answer as assistant's response.
    # This is a placeholder; more sophisticated parsing of the 'answer' field might be needed.
    # The R1 paper and DeepSeek use <think>...</think><answer>...</answer> structure.
    # If your `omartificial` `answer` field already has this, great.
    # Otherwise, you might need to heuristically create it or just use the raw answer.

    # Heuristic to separate reasoning and final answer from your notebook's make_conversation logic
    # (This was specific to `إذن،` prefix, adapt if needed)
    match = re.search(r"إذن،(.*)", answer_text, re.DOTALL)
    final_answer_part = ""
    reasoning_part = ""

    if match:
        final_answer_part = match.group(1).strip()
        reasoning_part = answer_text[:match.start()].strip()
    else:
        reasoning_part = answer_text # Default to whole answer as reasoning if no clear separator

    # Construct the assistant's response with <think> and <answer> tags
    # If reasoning_part is empty but final_answer_part exists, just use answer.
    # If both are empty, the response will be empty (tokenizer should handle this).
    assistant_response = ""
    if reasoning_part and final_answer_part:
        assistant_response = f"<think>{reasoning_part}</think><answer>{final_answer_part}</answer>"
    elif final_answer_part: # Only final answer found
        assistant_response = f"<think>لم يتم تقديم خطوات تفكير منفصلة.</think><answer>{final_answer_part}</answer>"
    elif reasoning_part: # Only reasoning found (treat as answer)
        assistant_response = f"<think>{reasoning_part}</think><answer>{reasoning_part}</answer>" 
    # else: both empty, assistant_response remains empty string

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_ARABIC_REASONING},
        {"role": "user", "content": user_query.strip()},
        {"role": "assistant", "content": assistant_response.strip()} # Ground truth completion for SFT
    ]
    
    # Unsloth's SFTTrainer expects the dataset to be tokenized and often packed.
    # The apply_chat_template usually happens *before* tokenization for SFT if doing it manually.
    # Or, SFTTrainer can take a formatting_func that returns a list of strings.
    # Here, we return the messages; Unsloth's SFTDataCollator will tokenize and format.
    return {"messages": messages} # SFTTrainer will process this with apply_chat_template


def load_and_prepare_dataset(dataset_name, split="train", for_grpo=True, tokenizer=None, max_seq_length=None):
    """
    Loads the specified dataset and prepares it for GRPO or SFT.

    Args:
        dataset_name (str): Name of the dataset on Hugging Face Hub (e.g., "omartificial/OmArtificial-Dolly-instruct-5k").
        split (str): Dataset split to load (e.g., "train", "test").
        for_grpo (bool): If True, formats for GRPO. Otherwise, for SFT.
        tokenizer: Tokenizer, needed if for_sft=True for applying chat template (though Unsloth SFT might do it internally).
        max_seq_length: Max sequence length, needed if for_sft=True for SFT packing.

    Returns:
        Dataset: The processed Hugging Face Dataset.
    """
    print(f"Loading dataset: {dataset_name}, split: {split}")
    dataset = load_dataset(dataset_name, split=split)
    print(f"Loaded {len(dataset)} examples.")

    if for_grpo:
        # For GRPO, GRPOTrainer will take the 'messages' and use tokenizer.apply_chat_template internally
        # to create the 'prompt' that the model generates from.
        # It expects a column like 'prompt' or you can specify query_dataset_mapper in GRPOConfig.
        # We will map to create a 'messages' column, which then Unsloth can process.
        
        # MODIFICATION: format_example_for_grpo now needs the tokenizer.
        if tokenizer is None:
            raise ValueError("Tokenizer is required for GRPO formatting to create 'prompt' string.")
            
        dataset = dataset.map(lambda x: format_example_for_grpo(x, tokenizer), batched=False)
        # GRPOTrainer will need a 'prompt' string. Unsloth might handle conversion from 'messages'.
        # If not, we might need another step here to apply_chat_template to 'messages' -> 'prompt_text_column'
        # For now, let's assume Unsloth handles 'messages' for GRPO input.
        print("Formatted dataset for GRPO (now includes 'prompt' string).")
    else:
        if tokenizer is None or max_seq_length is None:
            raise ValueError("Tokenizer and max_seq_length are required for SFT formatting.")
        # For SFT, SFTTrainer will take 'messages' and process it.
        dataset = dataset.map(lambda x: format_example_for_sft(x, tokenizer, max_seq_length), batched=False)
        print("Formatted dataset for SFT.")
        # SFTTrainer typically expects the text to be in a single column after formatting (e.g. 'text').
        # Unsloth's SFT process handles this well if provided with a formatting_func or a dataset
        # with a column of message lists.

    # Remove original columns if they are not needed anymore to save memory, 
    # but GRPOTrainer/SFTTrainer with remove_unused_columns=False might need them or handle it.
    # For now, keep them and let the trainer manage columns.
    # cols_to_remove = [col for col in dataset.column_names if col not in ['messages', 'prompt', 'text']]
    # dataset = dataset.remove_columns(cols_to_remove)
    
    return dataset

if __name__ == '__main__':
    # This is a placeholder for a tokenizer. In a real script, you'd load it.
    class MockTokenizer:
        def __init__(self):
            self.chat_template = None # Or a simple template for testing
            self.eos_token = "<|endoftext|>"
            self.model_max_length = 2048

        def apply_chat_template(self, conversation, tokenize=True, add_generation_prompt=False, **kwargs):
            # Simplified mock: just join content for demonstration
            text = ""
            for msg in conversation:
                text += f"{msg['role']}: {msg['content']}\n"
            if add_generation_prompt and conversation[-1]['role'] == 'user':
                 text += "assistant: " # typical addition for generation
            return text if not tokenize else text.split() # mock tokenization
    
    mock_tokenizer = MockTokenizer()

    print("--- Testing GRPO Data Formatting ---")
    # Create a dummy dataset for testing
    dummy_grpo_data = {
        "instruction": ["اشرح قانون الجاذبية", "ما هي عاصمة السعودية؟"],
        "context": ["السياق الأول", ""],
        "answer": ["الجاذبية هي... إذن، الجواب النهائي.", "الرياض هي عاصمة المملكة."]
    }
    from datasets import Dataset
    dummy_dataset_grpo = Dataset.from_dict(dummy_grpo_data)
    
    processed_grpo_dataset = dummy_dataset_grpo.map(lambda x: format_example_for_grpo(x, mock_tokenizer))
    print("Sample GRPO formatted example:")
    print(processed_grpo_dataset[0])
    # Example of how Unsloth might convert messages to prompt string:
    # (This step might be internal to Unsloth's GRPOTrainer or data processing)
    if 'messages' in processed_grpo_dataset.column_names and 'prompt' in processed_grpo_dataset.column_names:
        print("\nGRPO example now contains 'prompt' and 'messages':")
        print(f"Prompt string:\n{processed_grpo_dataset[0]['prompt']}")
        # print(f"Messages list: {processed_grpo_dataset[0]['messages']}") # Optionally print messages
        # templated_prompt = mock_tokenizer.apply_chat_template(processed_grpo_dataset[0]['messages'], tokenize=False, add_generation_prompt=True)
        # print(templated_prompt) # This is now done inside format_example_for_grpo

    print("\n--- Testing SFT Data Formatting ---")
    dummy_sft_data = {
        "instruction": ["ترجم 'hello' إلى العربية", "لخص هذا النص: ..."],
        "context": ["", "الذكاء الاصطناعي هو مجال واسع."],
        "answer": ["مرحباً", "إذن، الملخص هو أن الذكاء الاصطناعي مهم."]
    }
    dummy_dataset_sft = Dataset.from_dict(dummy_sft_data)
    processed_sft_dataset = dummy_dataset_sft.map(lambda x: format_example_for_sft(x, mock_tokenizer, 1024))
    print("Sample SFT formatted example (messages format for Unsloth SFTTrainer):")
    print(processed_sft_dataset[0]['messages'])
    # Unsloth SFT would take these messages and tokenize/pack them.
    # Example of applying chat template for SFT (Unsloth might do this internally)
    templated_sft_text = mock_tokenizer.apply_chat_template(processed_sft_dataset[0]['messages'], tokenize=False)
    print("\nApplying mock chat template to SFT messages:")
    print(templated_sft_text)

    print("\nNote: For actual Unsloth training, ensure the tokenizer is correctly loaded and has a chat template.")
    print("The 'messages' output format is generally preferred by Unsloth for both GRPO (as input that gets templated) and SFT.") 