import re
import numpy as np
import torch
from langdetect import detect # For language consistency reward
import math # For cosine scaled reward

# Project-specific import for number normalization
from src.data_loader import normalize_arabic_numbers # Used by accuracy_reward

# ==============================================================================
# DeepSeek R1-Zero Inspired Reward Functions (adapted for Arabic)
# ==============================================================================

def accuracy_reward(completions: list[str], solutions: list[str], **kwargs) -> list[float]:
    """
    Reward function to check if the model's *extracted answer* from its completion
    matches the *extracted answer* from the ground truth solution.
    This uses string comparison for Arabic numbers, not mathematical equivalence.
    Expected format in `solutions` is "<think>...</think><answer>...</answer>".
    """
    rewards = []

    for content, sol in zip(completions, solutions):
        # Extract predicted answer from model's completion
        answer_match_pred = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
        predicted_answer = normalize_arabic_numbers(answer_match_pred.group(1).strip()) if answer_match_pred else ""
        
        # Extract true answer from ground truth solution
        answer_match_true = re.search(r"<answer>(.*?)</answer>", sol, re.DOTALL)
        # FIX: Changed 'sol_match' to 'answer_match_true'
        true_answer = normalize_arabic_numbers(answer_match_true.group(1).strip()) if answer_match_true else ""
        
        # Determine reward based on strict string equality
        reward = 1.0 if predicted_answer == true_answer and true_answer != "" else 0.0 # Also penalize empty true_answer
        rewards.append(reward)
    return rewards

def format_reward(completions: list[str], **kwargs) -> list[float]:
    """
    Reward function to check if the completion has the correct format:
    <think>...</think><answer>...</answer>.
    Rewards for correct presence and order of tags. Penalizes malformed or missing tags.
    """
    scores = []
    think_open, think_close = "<think>", "</think>"
    answer_open, answer_close = "<answer>", "</answer>"

    for comp in completions:
        score = 0.0
        # Check for presence and completeness of tags
        think_starts = [m.start() for m in re.finditer(re.escape(think_open), comp)]
        think_ends = [m.start() for m in re.finditer(re.escape(think_close), comp)]
        answer_starts = [m.start() for m in re.finditer(re.escape(answer_open), comp)]
        answer_ends = [m.start() for m in re.finditer(re.escape(answer_close), comp)]

        has_think_pair = len(think_starts) == 1 and len(think_ends) == 1 and think_starts[0] < think_ends[0]
        has_answer_pair = len(answer_starts) == 1 and len(answer_ends) == 1 and answer_starts[0] < answer_ends[0]
        
        if has_think_pair and has_answer_pair:
            # Check correct order: <think> before <answer>
            if think_starts[0] < answer_starts[0]:
                score = 1.0 # Perfect format
            else:
                score = -0.5 # Incorrect order penalty
        elif has_think_pair or has_answer_pair:
            score = -0.25 # Partial presence, implies incomplete format
        else:
            score = -1.0 # No recognized tags or completely malformed
        
        # Additional small penalty for multiple occurrences of tags
        if len(think_starts) > 1 or len(think_ends) > 1 or \
           len(answer_starts) > 1 or len(answer_ends) > 1:
            score -= 0.1 # Small penalty for ambiguity

        scores.append(score)
    return scores

def get_cosine_scaled_reward(
    min_value_wrong: float = -0.5,
    max_value_wrong: float = -0.1,
    min_value_correct: float = 0.8,
    max_value_correct: float = 1.0,
    max_len: int = 1000,
):
    """
    Returns a cosine scaled reward function. This function scales the accuracy reward
    based on completion length. Shorter correct solutions get higher rewards,
    longer incorrect solutions get less penalty.
    """
    def cosine_scaled_reward(completions: list[str], solutions: list[str], accuracy_rewards: list[float], **kwargs) -> list[float]:
        rewards = []

        for content, sol, acc_reward in zip(completions, solutions, accuracy_rewards):
            gen_len = len(content.split())  # Length by words, not characters for better robustness
            
            # Clamp progress to [0, 1] to avoid math domain errors or unexpected behavior
            progress = min(1.0, gen_len / max_len) 
            
            cosine = math.cos(progress * math.pi) # Cosine value based on progress

            if acc_reward > 0.5: # Assuming accuracy_reward gives ~1.0 for correct answers
                min_val = min_value_correct
                max_val = max_value_correct
            else: # Incorrect answer
                min_val = max_value_wrong  # Note the swap!
                max_val = min_value_wrong

            # Cosine scaling formula: scales a value from [0, 1] to [min_val, max_val]
            # When progress is 0 (short), cosine is 1, reward is max_val
            # When progress is 1 (long), cosine is -1, reward is min_val
            reward = min_val + 0.5 * (max_val - min_val) * (1.0 + cosine)
            rewards.append(float(reward))
        return rewards
    return cosine_scaled_reward

def get_repetition_penalty_reward(ngram_size: int = 3, max_penalty: float = -0.1) -> callable:
    """
    Returns a repetition penalty reward function. Penalizes repetitions of n-grams
    in the generated text.
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    def repetition_penalty_reward(completions: list[str], **kwargs) -> list[float]:
        rewards = []
        for completion in completions:
            if completion == "":
                rewards.append(0.0) # No penalty for empty completions
                continue
            
            words = completion.lower().split() # Lowercase and split into words
            if len(words) < ngram_size: # No penalty for short completions that can't form n-grams
                rewards.append(0.0)
                continue

            ngrams = set() # Use a set to store unique n-grams
            total_ngrams_count = 0
            
            # Generate n-grams
            for i in range(len(words) - ngram_size + 1):
                ng = tuple(words[i : i + ngram_size])
                ngrams.add(ng)
                total_ngrams_count += 1

            if total_ngrams_count == 0: # Avoid division by zero for very short texts
                rewards.append(0.0)
                continue

            # Calculate scaling factor: more repetition -> higher scaling (closer to 1)
            # less repetition -> lower scaling (closer to 0)
            scaling = 1 - (len(ngrams) / total_ngrams_count)
            reward = scaling * max_penalty # Apply penalty based on scaling
            rewards.append(reward)
        return rewards
    return repetition_penalty_reward # Corrected: return the outer function itself

# Added from your previous full code for Arabic specific system prompt
def language_consistency_reward(completions: list[str], **kwargs) -> list[float]:
    """
    Reward function to ensure responses are predominantly in Arabic.
    Uses langdetect library.
    """
    rewards = []
    for content in completions:
        try:
            # Short contents might cause langdetect.errors.LangDetectException
            if len(content.strip()) < 5: # Min length for reliable detection
                reward = 0.5 # Neutral for very short or empty strings
            else:
                lang = detect(content)
                # Max reward for Arabic, harsh penalty if not Arabic
                reward = 1.0 if lang == "ar" else 0.0 
        except Exception:
            # If detection fails (e.g., very short string, special characters), assume neutral
            reward = 0.5 
        rewards.append(reward)
    return rewards

# ==============================================================================
# Combined Reward Functions
# ==============================================================================

def get_reward_config():
    """
    Returns a default configuration for reward weights and parameters.
    These weights are for the combined reward pipeline.
    """
    # Calculate weights based on user requirements:
    # Group 1 (2/3 total): Accuracy, Format, Language Consistency (3 functions)
    # Each = (2/3) / 3 = 2/9 approx 0.2222...
    WEIGHT_GROUP1 = 2/9 
    
    # Group 2 (1/3 total): Cosine Scaled, Repetition Penalty (2 functions)
    # Each = (1/3) / 2 = 1/6 approx 0.1666...
    WEIGHT_GROUP2 = 1/6

    return {
        "weights": {
            "accuracy": WEIGHT_GROUP1,
            "format": WEIGHT_GROUP1,
            "language_consistency": WEIGHT_GROUP1,
            "cosine_scaled": WEIGHT_GROUP2,
            "repetition_penalty": WEIGHT_GROUP2,
            # "reasoning_steps": 0.0, # Removed as per new requirements
        },
        # Parameters for specific reward functions
        "cosine_min_value_wrong": -0.5,
        "cosine_max_value_wrong": -0.1,
        "cosine_min_value_correct": 0.8,
        "cosine_max_value_correct": 1.0,
        "cosine_max_len": 500, # Max words for scaling. Adjust based on expected output length.
        "repetition_ngram_size": 3,
        "repetition_max_penalty": -0.1,
        "clamp_rewards": {"min": -2.0, "max": 2.0}, # Clamp total rewards
    }

def combined_reward_pipeline(completions: list[str], prompts_text: list[str], solutions: list[str], reward_config: dict) -> list[float]:
    """
    Calculates a combined reward for a list of completions based on specified criteria.
    
    Args:
        completions (list[str]): A list of generated text completions.
        prompts_text (list[str]): A list of corresponding text prompts (for context).
        solutions (list[str]): A list of ground truth solutions (for accuracy).
        reward_config (dict): Configuration for reward functions and weights.

    Returns:
        list[float]: A list of total reward scores for each completion.
    """
    final_rewards = []
    num_completions = len(completions)
    cfg = reward_config
    w = cfg["weights"]

    # Calculate individual reward components
    acc_scores = accuracy_reward(completions, solutions)
    fmt_scores = format_reward(completions)
    lang_scores = language_consistency_reward(completions) # No prompts needed here

    # Cosine scaled reward needs accuracy_rewards and solutions
    cosine_fn = get_cosine_scaled_reward(
        min_value_wrong=cfg["cosine_min_value_wrong"],
        max_value_wrong=cfg["cosine_max_value_wrong"],
        min_value_correct=cfg["cosine_min_value_correct"],
        max_value_correct=cfg["cosine_max_value_correct"],
        max_len=cfg["cosine_max_len"],
    )
    cos_scores = cosine_fn(completions, solutions, acc_scores)

    # Repetition penalty reward
    rep_fn = get_repetition_penalty_reward(
        ngram_size=cfg["repetition_ngram_size"],
        max_penalty=cfg["repetition_max_penalty"],
    )
    rep_scores = rep_fn(completions)

    for i in range(num_completions):
        total_reward = (
            w["accuracy"] * acc_scores[i] +
            w["format"] * fmt_scores[i] +
            w["language_consistency"] * lang_scores[i] +
            w["cosine_scaled"] * cos_scores[i] +
            w["repetition_penalty"] * rep_scores[i]
        )
        
        if "clamp_rewards" in cfg and cfg["clamp_rewards"]:
            total_reward = np.clip(total_reward, cfg["clamp_rewards"]["min"], cfg["clamp_rewards"]["max"])
            
        final_rewards.append(total_reward)
        
        # Detailed print for debugging and understanding
        print(f"\n--- Completion {i+1} Analysis ---")
        print(f"Completion: {completions[i][:150]}...")
        print(f"Prompt: {prompts_text[i][:100]}...")
        print(f"Solution: {solutions[i][:100]}...")
        print(f"  Accuracy Reward: {acc_scores[i]:.4f} (Weight: {w['accuracy']:.4f})")
        print(f"  Format Reward: {fmt_scores[i]:.4f} (Weight: {w['format']:.4f})")
        print(f"  Language Consistency Reward: {lang_scores[i]:.4f} (Weight: {w['language_consistency']:.4f})")
        print(f"  Cosine Scaled Reward: {cos_scores[i]:.4f} (Weight: {w['cosine_scaled']:.4f})")
        print(f"  Repetition Penalty Reward: {rep_scores[i]:.4f} (Weight: {w['repetition_penalty']:.4f})")
        print(f"  Total Reward: {final_rewards[i]:.4f}")

    return final_rewards

def grpo_reward_function_unsloth(completions: list[str], tokenizer, reward_config: dict, **kwargs) -> torch.Tensor:
    """
    Adapter function for GRPOTrainer.
    `completions` here are the generated sequences (list of strings).
    `kwargs` will contain `prompt_input_ids` or `query_input_ids` (representing prompts),
             and `generated_input_ids` (representing completions from the policy model).
             It might also contain `reference_generated_input_ids` (from reference model, if used).
    The GRPOTrainer expects a torch.Tensor of rewards, one for each generated sequence.
    """
    generated_texts = completions

    # Extract prompt texts
    prompt_token_ids = None
    if "prompt_input_ids" in kwargs:
        prompt_token_ids = kwargs["prompt_input_ids"]
    elif "query_input_ids" in kwargs: 
        prompt_token_ids = kwargs["query_input_ids"]
    elif "input_ids" in kwargs: 
        prompt_token_ids = kwargs["input_ids"]

    prompts_text = [""] * len(generated_texts) # Default to empty prompts if not found
    if prompt_token_ids is not None:
        if isinstance(prompt_token_ids, torch.Tensor):
            prompt_token_ids_list = prompt_token_ids.tolist()
        else:
            prompt_token_ids_list = prompt_token_ids

        decoded_prompts = tokenizer.batch_decode(prompt_token_ids_list, skip_special_tokens=True)
        
        num_prompts_in_batch = len(prompt_token_ids_list)
        num_generations_per_prompt = len(generated_texts) // num_prompts_in_batch if num_prompts_in_batch > 0 else 1
        
        aligned_prompts_text = []
        for p_text in decoded_prompts:
            aligned_prompts_text.extend([p_text] * num_generations_per_prompt)
        prompts_text = aligned_prompts_text
    
    # Extract solutions from the batch if available
    solutions = kwargs.get("solutions", []) 

    # If solutions array is shorter than completions, expand it
    if len(solutions) > 0 and len(generated_texts) > len(solutions):
        aligned_solutions = []
        num_solutions_in_batch = len(solutions)
        num_generations_per_solution = len(generated_texts) // num_solutions_in_batch if num_solutions_in_batch > 0 else 1
        for sol_text in solutions:
            aligned_solutions.extend([sol_text] * num_generations_per_solution)
        solutions = aligned_solutions
    elif not solutions: # If no solutions provided, use empty strings
        solutions = [""] * len(generated_texts)

    if not generated_texts:
        return torch.tensor([], dtype=torch.float32) # Return empty tensor if no completions

    # Call the existing combined reward pipeline
    rewards = combined_reward_pipeline(generated_texts, prompts_text, solutions, reward_config)
    
    # Convert to tensor for GRPOTrainer
    return torch.tensor(rewards, dtype=torch.float32)


if __name__ == '__main__':
    # This block is for testing the reward functions independently
    # It simulates input to the reward functions.

    print("--- Testing Reward Functions (Arabic Adaptation) ---")

    # Mock tokenizer (for `grpo_reward_function_unsloth`)
    class MockTokenizer:
        def batch_decode(self, token_ids, skip_special_tokens=True):
            # A very simplified mock for decoding token IDs
            decoded_texts = []
            for ids in token_ids:
                if isinstance(ids, list): # Handle lists of lists of ints
                    text = f"Decoded_Prompt_{hash(tuple(ids)) % 1000}" # Just a placeholder
                else: # Handle single list of ints (e.g. from single prompt)
                    text = f"Decoded_Prompt_{hash(tuple(ids)) % 1000}" # Just a placeholder
                decoded_texts.append(text)
            return decoded_texts

    mock_tokenizer_for_tests = MockTokenizer()
    reward_cfg = get_reward_config()

    # --- Sample data for testing (All lists are now length 9) ---
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

    sample_solutions = [
        "<think>جمع 1 و 1.</think><answer>إذن، الناتج هو ٢.</answer>", # Corresponds to "one plus one"
        "<think>السماء زرقاء بسبب تشتت رايلي للضوء.</think><answer>إذن، اللون الأزرق بسبب التشتت.</answer>", # Corresponds to "Why is the sky blue?"
        "<think>تأخر القطار بسبب الأمطار الغزيرة.</think><answer>إذن، يجب أن ننتظر.</answer>", # Corresponds to "لماذا تأخر القطار؟"
        "<think>عاصمة فرنسا هي باريس.</think><answer>إذن، عاصمة فرنسا هي باريس.</answer>", # Corresponds to "What is the capital of France?"
        "<think>هذا للتحقق من الإجابة الفارغة.</think><answer>هذه إجابة فارغة.</answer>", # Corresponds to "empty completion"
        "<think>سبب تأخر القطار هو ظروف جوية سيئة.</think><answer>إذن، تأخر القطار.</answer>", # Corresponds to "اشرح سبب تأخر القطار."
        "<think>المساعدة ممكنة في هذا السياق.</think><answer>بالتأكيد يمكنني المساعدة.</answer>", # Corresponds to "هل يمكنك مساعدتي؟"
        "<think>هذا هو مثال لجملة طويلة.</think><answer>هذه جملة طويلة.</answer>", # Corresponds to "اكتب جملة طويلة."
        "<think>الإجابة هي كلمة واحدة فقط.</think><answer>كلمة.</answer>" # Corresponds to "اكتب كلمة واحدة."
    ]

    sample_completions = [
        "<think>لحل 1 + 1، نقوم بجمع العددين: 1 + 1 = 2.</think><answer>إذن، الناتج هو ٢.</answer>", # Good: Format, Arabic, Accurate (for 1+1=2)
        "I don't know the answer.", # Bad: English, forbidden keyword, no tags.
        "الجواب هو ثلاثة. لماذا تسأل؟", # Okay: Repeats question word, no tags, not accurate.
        "<think>هذا سؤال بسيط.</think><answer>الجواب هو ما هو الجواب؟</answer>", # Bad: repeats question, bad format.
        "", # Bad: Empty.
        "<think>تأخر القطار بسبب الأمطار الغزيرة. وبالتالي، يجب أن ننتظر.</think><answer>إذن، تأخر القطار.</answer>", # Good: Reasoning, Arabic, Format
        "أنا آسف، لا يمكنني المساعدة في هذا.", # Bad: Forbidden keyword, no tags, no reasoning.
        "<think>هذه جملة عربية طويلة جدا جدا جدا تمتد لأكثر من خمسين حرفا لكي نختبر طول النص وكيف يتم تقييمه.</think><answer>النص طويل.</answer>", # Length test
        "<think>قطة.</think><answer>قطة.</answer>" # Short, Arabic, Repetition.
    ]
    
    # Ensure all sample lists have the exact same length
    min_len = min(len(sample_completions), len(sample_prompts), len(sample_solutions))
    sample_completions = sample_completions[:min_len]
    sample_prompts = sample_prompts[:min_len]
    sample_solutions = sample_solutions[:min_len]

    print(f"\n--- Running combined_reward_pipeline on samples (showing detailed breakdown) ---")
    combined_scores = combined_reward_pipeline(sample_completions, sample_prompts, sample_solutions, reward_cfg)
    print("\n--- Final Combined Rewards ---")
    for i, score in enumerate(combined_scores):
        print(f"Sample {i+1}: Total Reward = {score:.4f}")

    # --- Testing grpo_reward_function_unsloth (simulating GRPOTrainer call) ---
    print("\n--- Testing grpo_reward_function_unsloth (simulating GRPOTrainer call) ---")
    
    # Simulate a batch with 2 prompts, and 2 completions per prompt (total 4 completions)
    simulated_prompts_input_ids = [
        [1, 2, 3, 4, 5], # Mock token IDs for prompt 1
        [6, 7, 8, 9, 10] # Mock token IDs for prompt 2
    ]
    simulated_prompts_text_decoded = [
        "Prompt 1: ما هو الناتج؟",
        "Prompt 2: اشرح السبب؟"
    ]
    simulated_solutions_from_dataset = [
        "<think>حل ١.</think><answer>الناتج هو ١٠.</answer>",
        "<think>سبب ٢.</think><answer>السبب هو كذا.</answer>"
    ]

    # Assume 2 generations per prompt
    simulated_completions_for_grpo = [
        "<think>الحل هو كذا. الخطوة الأولى.</think><answer>الناتج هو ١٠.</answer>", # Gen for Prompt 1 (correct)
        "<think>الحل هو كذا. ولكن بالإنجليزية. Step 1.</think><answer>The answer is 10.</answer>", # Gen for Prompt 1 (lang mix)
        "<answer>السبب كذا.</answer>", # Gen for Prompt 2 (missing think)
        "هذا تكرار، تكرار، تكرار. هذا تكرار، تكرار، تكرار. هذا تكرار." # Gen for Prompt 2 (repetition)
    ]

    # Manually align prompts_text and solutions based on num_generations_per_prompt for the mock call
    aligned_simulated_prompts = []
    aligned_simulated_solutions = []
    num_gens_per_prompt = len(simulated_completions_for_grpo) // len(simulated_prompts_input_ids)
    
    for i in range(len(simulated_prompts_input_ids)):
        aligned_simulated_prompts.extend([simulated_prompts_text_decoded[i]] * num_gens_per_prompt)
        aligned_simulated_solutions.extend([simulated_solutions_from_dataset[i]] * num_gens_per_prompt)

    mock_kwargs_for_grpo_trainer = {
        "prompt_input_ids": torch.tensor(simulated_prompts_input_ids), # GRPOTrainer provides token IDs
        "solutions": simulated_solutions_from_dataset # Assuming 'solutions' column is passed
    }

    print("\n--- Running grpo_reward_function_unsloth with simulated batch ---")
    # The `grpo_reward_function_unsloth` takes `completions` (list of strings) directly
    # and decodes prompts/aligns solutions internally from `kwargs`.
    grpo_calculated_rewards = grpo_reward_function_unsloth(
        completions=simulated_completions_for_grpo,
        tokenizer=mock_tokenizer_for_tests,
        reward_config=reward_cfg,
        **mock_kwargs_for_grpo_trainer
    )
    print("\nFinal Rewards from grpo_reward_function_unsloth (as torch.Tensor):")
    print(grpo_calculated_rewards)