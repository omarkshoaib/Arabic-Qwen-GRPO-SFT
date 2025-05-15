import re
import numpy as np

# Basic Arabic character detection (you might want a more robust library for this)
def is_arabic_char(char):
    return '\\u0600' <= char <= '\\u06FF' or \
           '\\u0750' <= char <= '\\u077F' or \
           '\\uFB50' <= char <= '\\uFDFF' or \
           '\\uFE70' <= char <= '\\uFEFF'

def reward_arabic_only(completions, penalty_factor=10.0):
    """
    Rewards completions that are entirely in Arabic.
    Penalizes completions with non-Arabic characters.
    """
    scores = []
    for completion in completions:
        if not completion:
            scores.append(-penalty_factor) # Penalize empty completions
            continue
        
        arabic_chars = sum(1 for char in completion if is_arabic_char(char))
        total_chars = len(completion)
        
        if total_chars == 0: # Should be caught by 'if not completion'
             scores.append(-penalty_factor)
             continue

        ratio_arabic = arabic_chars / total_chars
        
        if ratio_arabic == 1.0:
            scores.append(1.0)  # Max reward for fully Arabic
        else:
            # Penalize proportionally to non-Arabic content, up to penalty_factor
            # A completion with 0% Arabic gets -penalty_factor.
            # A completion with 50% Arabic gets -(penalty_factor / 2) if we want linear.
            # Or simply a large penalty if not 100% Arabic.
            # For simplicity here: heavy penalty if not pure Arabic.
            scores.append(-penalty_factor * (1.0 - ratio_arabic))
            # Alternatively, a simpler binary penalty:
            # scores.append(-penalty_factor if ratio_arabic < 1.0 else 1.0)
    return scores

def reward_length(completions, target_length=50, penalty_per_char=0.1):
    """
    Rewards completions for being close to a target length.
    Penalizes based on the absolute difference from the target length.
    Returns scores that are 0 or negative. Max score is 0 (perfect length).
    """
    scores = []
    for completion in completions:
        if not completion:
            scores.append(-target_length * penalty_per_char * 2) # Heavy penalty for empty
            continue
        diff = abs(len(completion) - target_length)
        scores.append(-diff * penalty_per_char)
    return scores

# Example keywords - these should be tailored to your specific reasoning tasks
KEYWORDS_TO_REWARD_AR = [
    "لأن", "بسبب", "وبالتالي", "إذاً", "إذن", "نتيجة لذلك", 
    "بما أن", "حيث أن", "وفقًا لـ", "يشير إلى", "يدل على", "يعني أن",
    "الخطوة الأولى", "الخطوة التالية", "أخيراً", "في الختام" # Reasoning/structure words
]
KEYWORDS_TO_PENALIZE_AR = [
    "أنا آسف", "لا أعرف", "لست متأكدا", "غير قادر على", "ليس لدي معلومات",
    "مرحباً", "أهلاً" # Generic, non-task-focused greetings if the task is specific
] 
# Arabic question words (common ones)
ARABIC_QUESTION_WORDS = [
    "هل", "ماذا", "ما", "لماذا", "متى", "أين", "كيف", "كم", "من", "أي"
]


def reward_contains_keywords(completions, keywords, reward_value=0.5):
    """
    Rewards completions that contain any of the specified keywords.
    """
    scores = []
    for completion in completions:
        if not completion:
            scores.append(0) # No reward for empty string
            continue
        if any(keyword in completion for keyword in keywords):
            scores.append(reward_value)
        else:
            scores.append(0)
    return scores

def reward_not_contains_forbidden_keywords(completions, forbidden_keywords, penalty_value=1.0):
    """
    Penalizes completions that contain any of the forbidden keywords.
    """
    scores = []
    for completion in completions:
        if not completion:
            scores.append(0) # No penalty for empty
            continue
        if any(keyword in completion for keyword in forbidden_keywords):
            scores.append(-penalty_value)
        else:
            scores.append(0) # No penalty if no forbidden keywords
    return scores

def reward_question_words_not_in_answer(completions, prompts, question_words, penalty_value=0.5):
    """
    Penalizes if question words from the prompt (or general question words)
    are repeated in the completion, unless the completion itself is a question.
    This version is simplified to check against a general list of question words.
    A more advanced version would extract question words from the specific prompt.
    """
    scores = []
    for i, completion in enumerate(completions):
        if not completion:
            scores.append(0)
            continue
            
        # Simple check: if completion ends with '؟', it's a question, so don't penalize
        if completion.strip().endswith("؟"):
            scores.append(0)
            continue

        penalty = 0
        for qw in question_words:
            if qw in completion:
                penalty = -penalty_value
                break 
        scores.append(penalty)
    return scores


# --- Combined Reward Function for GRPO ---
# This is where you'll combine the above functions.
# The `batch` argument in GRPO's reward function typically contains tokenized prompts and other info.
# You'll need to decode prompts if you need their text content.

def get_reward_config():
    """Returns a default configuration for reward weights and keywords."""
    return {
        "weights": {
            "length": 0.1,
            "arabic_only": 0.4,
            "contains_keywords": 0.2,
            "not_contains_forbidden": 0.2,
            "question_words_not_in_answer": 0.1,
        },
        "target_length": 70, # Adjusted target length
        "length_penalty_per_char": 0.05, # Reduced penalty per char
        "arabic_penalty_factor": 10.0,
        "reward_keywords_value": 1.0, # Increased reward for good keywords
        "forbidden_penalty_value": 2.0, # Increased penalty for bad keywords
        "q_words_penalty_value": 1.0, # Increased penalty for repeating q-words
        "keywords_to_reward": KEYWORDS_TO_REWARD_AR,
        "keywords_to_penalize": KEYWORDS_TO_PENALIZE_AR,
        "arabic_question_words": ARABIC_QUESTION_WORDS,
        "clamp_rewards": {"min": -5.0, "max": 5.0} # Optional clamping for total reward
    }

def combined_reward_pipeline(completions, prompts_text, reward_config):
    """
    Calculates a combined reward for a list of completions based on a list of text prompts.
    
    Args:
        completions (list[str]): A list of generated text completions.
        prompts_text (list[str]): A list of corresponding text prompts.
        reward_config (dict): Configuration for reward functions and weights.

    Returns:
        list[float]: A list of reward scores for each completion.
    """
    final_rewards = []

    # Ensure prompts_text is a list of the same effective batch size as completions might be (due to num_generations)
    # If num_generations_per_prompt = N, completions will be N * original_batch_size
    # prompts_text should be [prompt1, prompt1, ..., prompt2, prompt2, ...]
    # This needs to be handled by the caller or GRPOTrainer's batching.
    # For now, assume prompts_text is already correctly tiled or matched.

    num_completions = len(completions)
    num_prompts = len(prompts_text)

    if num_completions == 0:
        return []

    # Basic check: if completions is a multiple of prompts, assume generations per prompt
    # This is a simplification; the GRPOTrainer usually handles this alignment.
    if num_prompts > 0 and num_completions % num_prompts == 0:
        generations_per_prompt = num_completions // num_prompts
        expanded_prompts = [p for p in prompts_text for _ in range(generations_per_prompt)]
    else:
        # Fallback or error: if alignment is unclear, pair one-to-one or raise error
        # For simplicity, let's assume they are meant to align if counts differ but not by exact multiple
        expanded_prompts = prompts_text * (num_completions // num_prompts + 1) if num_prompts > 0 else [""] * num_completions
        expanded_prompts = expanded_prompts[:num_completions]


    cfg = reward_config
    w = cfg["weights"]

    r_len_scores = reward_length(completions, cfg["target_length"], cfg["length_penalty_per_char"])
    r_arabic_scores = reward_arabic_only(completions, cfg["arabic_penalty_factor"])
    r_keywords_scores = reward_contains_keywords(completions, cfg["keywords_to_reward"], cfg["reward_keywords_value"])
    r_forbidden_scores = reward_not_contains_forbidden_keywords(completions, cfg["keywords_to_penalize"], cfg["forbidden_penalty_value"])
    # For reward_question_words_not_in_answer, it needs prompts.
    r_q_words_scores = reward_question_words_not_in_answer(completions, expanded_prompts, cfg["arabic_question_words"], cfg["q_words_penalty_value"])
    
    for i in range(num_completions):
        total_reward = (
            w["length"] * r_len_scores[i] +
            w["arabic_only"] * r_arabic_scores[i] +
            w["contains_keywords"] * r_keywords_scores[i] +
            w["not_contains_forbidden"] * r_forbidden_scores[i] +
            w["question_words_not_in_answer"] * r_q_words_scores[i]
        )
        
        if "clamp_rewards" in cfg and cfg["clamp_rewards"]:
            total_reward = np.clip(total_reward, cfg["clamp_rewards"]["min"], cfg["clamp_rewards"]["max"])
            
        final_rewards.append(total_reward)
        
    return final_rewards

# Wrapper for GRPOTrainer
# The GRPOTrainer expects a function that takes `completions` (list of str), 
# and `**kwargs` which will contain the tokenized `batch`.
# We need to decode the prompts from the batch.
def grpo_reward_function_unsloth(completions, tokenizer, reward_config, **kwargs):
    """
    Wrapper for the combined_reward_pipeline to be used with Unsloth GRPOTrainer.
    It decodes prompts from the batch.
    
    Args:
        completions (list[str]): List of generated texts.
        tokenizer: The tokenizer used for the model.
        reward_config (dict): Configuration for reward calculation.
        **kwargs: Should contain the batch from the GRPOTrainer,
                  which includes 'prompt_input_ids' and 'prompt_attention_mask'.

    Returns:
        list[float]: A list of reward scores.
    """
    if 'batch' not in kwargs:
        raise ValueError("The 'batch' containing tokenized prompts was not found in kwargs.")
    
    batch = kwargs['batch']
    
    # Assuming 'prompt_input_ids' is the key for tokenized prompts in the batch
    # This key might vary based on how the dataset is prepared for GRPOTrainer
    if "prompt_input_ids" not in batch:
         # Fallback for datasets that might just have 'input_ids' for the prompt part
        if "input_ids" in batch and "query_input_ids" not in batch : # if query_input_ids exists then it is the prompt
            # this is the case for datasets like trl-lib/tldr where the dataset only has a prompt column
            # and the GRPOTrainer formats it into input_ids, attention_mask and adds a query column for original prompt
            prompt_token_ids = batch["input_ids"]
        elif "query_input_ids" in batch: # Unsloth's SFTTrainer might prepare it this way for GRPO
            prompt_token_ids = batch["query_input_ids"]
        else:
            raise ValueError("Could not find 'prompt_input_ids' or 'input_ids' or 'query_input_ids' in the batch for decoding prompts.")
    else:
        prompt_token_ids = batch["prompt_input_ids"]

    # Decode prompts
    # Need to handle padding tokens if they are not skipped by default during decoding.
    prompts_text = tokenizer.batch_decode(prompt_token_ids, skip_special_tokens=True)
    
    return combined_reward_pipeline(completions, prompts_text, reward_config)


if __name__ == '__main__':
    # Example Usage and Testing
    sample_completions = [
        "الإجابة هي اثنان لأن واحد زائد واحد يساوي اثنان.", # Good
        "I don't know the answer.", # Bad - English
        "الجواب هو ثلاثة. لماذا تسأل؟", # Okay, but repeats question word
        "ما هو الجواب؟", # Bad - just a question
        "", # Bad - empty
        "بسبب الأمطار الغزيرة، تأخر القطار. وبالتالي، يجب أن ننتظر.", # Good reasoning example
        "أنا آسف، لا يمكنني المساعدة في هذا.", # Bad - forbidden keyword
        "هذه جملة عربية طويلة جدا جدا جدا تمتد لأكثر من خمسين حرفا لكي نختبر طول النص وكيف يتم تقييمه.", # Length test
        "قطة." # Short, Arabic
    ]
    
    # Mock prompts (in a real scenario, these come from the batch)
    sample_prompts = [
        "ما هو ناتج واحد زائد واحد؟",
        "Why is the sky blue?",
        "لماذا تأخر القطار؟",
        "What is the capital of France?",
        "Test prompt for empty completion.",
        "اشرح سبب تأخر القطار.",
        "هل يمكنك مساعدتي؟",
        "اكتب جملة طويلة.",
        "اكتب كلمة واحدة."
    ]

    # Ensure sample_prompts aligns with sample_completions if combined_reward_pipeline expects it
    # If GRPOTrainer handles generation (e.g. 4 generations per prompt), then completions list is longer.
    # For this test, let's assume 1 completion per prompt for simplicity of testing combined_reward_pipeline directly
    if len(sample_completions) != len(sample_prompts):
        print(f"Warning: Mismatch in length of sample completions ({len(sample_completions)}) and prompts ({len(sample_prompts)}). Adjusting prompts for test.")
        # Simple adjustment: repeat prompts or truncate. For this test, let's tile prompts.
        num_gens_per_prompt_test = len(sample_completions) // len(sample_prompts) if len(sample_prompts) > 0 and len(sample_completions) % len(sample_prompts) == 0 else 1
        if len(sample_prompts) > 0 :
            sample_prompts_expanded = [p for p in sample_prompts for _ in range(num_gens_per_prompt_test)]
            sample_prompts_expanded = sample_prompts_expanded[:len(sample_completions)]
        else:
            sample_prompts_expanded = [""] * len(sample_completions)

    else:
        sample_prompts_expanded = sample_prompts


    default_config = get_reward_config()
    print("Default Reward Config:", default_config)

    print("\\n--- Testing individual reward functions ---")
    print("Reward Length:", reward_length(sample_completions, default_config["target_length"], default_config["length_penalty_per_char"]))
    print("Reward Arabic Only:", reward_arabic_only(sample_completions, default_config["arabic_penalty_factor"]))
    print("Reward Contains Keywords:", reward_contains_keywords(sample_completions, default_config["keywords_to_reward"], default_config["reward_keywords_value"]))
    print("Reward Not Contains Forbidden:", reward_not_contains_forbidden_keywords(sample_completions, default_config["keywords_to_penalize"], default_config["forbidden_penalty_value"]))
    print("Reward Question Words Not In Answer:", reward_question_words_not_in_answer(sample_completions, sample_prompts_expanded, default_config["arabic_question_words"], default_config["q_words_penalty_value"]))

    print("\\n--- Testing combined_reward_pipeline ---")
    combined_scores = combined_reward_pipeline(sample_completions, sample_prompts_expanded, default_config)
    print("Combined Rewards:", combined_scores)
    for i, comp in enumerate(sample_completions):
        print(f"Prompt: {sample_prompts_expanded[i][:50]}... | Completion: {comp[:50]}... | Score: {combined_scores[i]:.2f}")

    # Mock tokenizer and batch for grpo_reward_function_unsloth
    class MockTokenizer:
        def batch_decode(self, token_ids, skip_special_tokens=True):
            # This is a very simplified mock. In reality, it would convert IDs to strings.
            return [f"decoded_prompt_{i}" for i in range(len(token_ids))] 

    mock_tokenizer = MockTokenizer()
    mock_batch_v1 = {
        # Assuming GRPOTrainer provides 'prompt_input_ids' after its processing
        "prompt_input_ids": [[101, 7592, 2026, 2003, 102], [101, 2054, 2003, 102]] # Mock token IDs
    }
    # If completions has 4 items, and prompt_input_ids has 2, it implies 2 generations per prompt
    sample_completions_for_grpo_test = sample_completions[:4]
    mock_batch_for_grpo_test = { "prompt_input_ids": [[1],[2]] } # 2 prompts
    
    # Adjust prompts_text for the grpo_reward_function_unsloth test case
    # It expects the raw batch and decodes prompts itself.
    # The number of completions should be num_prompts * num_generations.
    # Let's say num_generations is 2 for this test, so 2 completions per prompt in mock_batch.
    
    # Test case 1: prompt_input_ids exists
    print("\\n--- Testing grpo_reward_function_unsloth (with prompt_input_ids) ---")
    try:
        grpo_rewards = grpo_reward_function_unsloth(
            completions=sample_completions_for_grpo_test, # 4 completions
            tokenizer=mock_tokenizer,
            reward_config=default_config,
            batch=mock_batch_for_grpo_test # 2 prompts => 2 gens per prompt
        )
        print("GRPO Rewards (prompt_input_ids):", grpo_rewards)
    except Exception as e:
        print(f"Error in GRPO reward function test (prompt_input_ids): {e}")

    # Test case 2: input_ids exists (e.g. tldr dataset style)
    print("\\n--- Testing grpo_reward_function_unsloth (with input_ids) ---")
    mock_batch_v2 = { "input_ids": [[3],[4]] }
    try:
        grpo_rewards_v2 = grpo_reward_function_unsloth(
            completions=sample_completions_for_grpo_test, # 4 completions
            tokenizer=mock_tokenizer,
            reward_config=default_config,
            batch=mock_batch_v2 # 2 prompts => 2 gens per prompt
        )
        print("GRPO Rewards (input_ids):", grpo_rewards_v2)
    except Exception as e:
        print(f"Error in GRPO reward function test (input_ids): {e}")

    # Test case 3: query_input_ids (Unsloth SFTTrainer style for GRPO)
    print("\\n--- Testing grpo_reward_function_unsloth (with query_input_ids) ---")
    mock_batch_v3 = { "query_input_ids": [[5],[6]] }
    try:
        grpo_rewards_v3 = grpo_reward_function_unsloth(
            completions=sample_completions_for_grpo_test, # 4 completions
            tokenizer=mock_tokenizer,
            reward_config=default_config,
            batch=mock_batch_v3 # 2 prompts => 2 gens per prompt
        )
        print("GRPO Rewards (query_input_ids):", grpo_rewards_v3)
    except Exception as e:
        print(f"Error in GRPO reward function test (query_input_ids): {e}")

    print("\\nConsiderations for reward function design:")
    print("1. Balance: Ensure positive rewards are achievable and penalties are not overly harsh or frequent.")
    print("2. Scaling/Normalization: If reward components have vastly different scales, normalize them before weighting.")
    print("3. Target Behavior: Rewards should clearly guide the model towards desired Arabic reasoning and conversational properties.")
    print("4. Iteration: Expect to iterate on weights and logic based on training behavior (e.g., if mean rewards are consistently negative).") 