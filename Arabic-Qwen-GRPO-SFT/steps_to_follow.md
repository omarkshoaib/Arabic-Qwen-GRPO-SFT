
Okay, let's break down the steps to test and run your project successfully. This will involve setting up your environment, testing individual components, running the training pipelines, and finally, performing inference.

**Phase 1: Environment Setup & Initial Git**

1.  **Initialize Git and Connect to GitHub:**
    *   Open your terminal in the `Arabic-Qwen-GRPO-SFT` directory.
    *   Run `git init` to initialize a local Git repository.
    *   Go to GitHub and create a new repository (e.g., `Arabic-Qwen-GRPO-SFT`). **Do not** initialize it with a README, .gitignore, or license if you've already created these locally.
    *   Copy the commands GitHub provides to link your local repository and push the initial commit. It will look something like:
        ```bash
        git remote add origin https://github.com/YOUR_USERNAME/Arabic-Qwen-GRPO-SFT.git
        git branch -M main
        git add .
        git commit -m "Initial project structure and scripts"
        git push -u origin main
        ```

2.  **Create a Python Virtual Environment:**
    *   It's highly recommended to use a virtual environment to manage project dependencies.
    *   In your project directory (`Arabic-Qwen-GRPO-SFT`):
        ```bash
        python3 -m venv .venv 
        source .venv/bin/activate 
        ```
        (On Windows, activation is `.venv\Scripts\activate`)

3.  **Install Dependencies:**
    *   Ensure your virtual environment is activated.
    *   Install all required packages:
        ```bash
        pip install --upgrade pip
        pip install -r requirements.txt
        ```
    *   For the CI workflow and local linting, also install `ruff`:
        ```bash
        pip install ruff
        ```

**Phase 2: Component Testing & Sanity Checks (Crucial!)**

Before launching full training runs, test the core components:

1.  **Test `src/data_loader.py`:**
    *   **How to test:** You might want to add a temporary `if __name__ == "__main__":` block to `src/data_loader.py` to test its functionality.
        ```python
        # Add this at the end of src/data_loader.py for testing
        if __name__ == "__main__":
            from transformers import AutoTokenizer
            
            # Test GRPO data loading
            print("--- Testing GRPO Data Loader ---")
            tokenizer_grpo = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct") # Or your model
            if tokenizer_grpo.pad_token is None:
                tokenizer_grpo.pad_token = tokenizer_grpo.eos_token
            
            grpo_dataset = load_and_prepare_dataset(
                dataset_name="omartificial/OmArtificial-Dolly-instruct-5k",
                split="train", # Or a small slice like "train[:10]"
                for_grpo=True,
                tokenizer=tokenizer_grpo, # Pass tokenizer for GRPO template application
                max_seq_length=512
            )
            print(f"GRPO dataset size: {len(grpo_dataset)}")
            if len(grpo_dataset) > 0:
                print("First GRPO example 'prompt_text':")
                print(grpo_dataset[0]['prompt_text'])
                print("First GRPO example 'original_messages':") # To see the source
                print(grpo_dataset[0]['original_messages'])


            print("\n--- Testing SFT Data Loader ---")
            # For SFT, the tokenizer is mainly for checking length if not applying template here
            tokenizer_sft = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
            sft_dataset = load_and_prepare_dataset(
                dataset_name="omartificial/OmArtificial-Dolly-instruct-5k",
                split="train", # Or "train[:10]"
                for_grpo=False,
                tokenizer=tokenizer_sft, # For SFT, tokenizer is used by the trainer
                max_seq_length=512
            )
            print(f"SFT dataset size: {len(sft_dataset)}")
            if len(sft_dataset) > 0:
                print("First SFT example 'messages':")
                print(sft_dataset[0]['messages'])
        ```
    *   **Run it:** `python src/data_loader.py`
    *   **What to check:**
        *   Does it download/load the dataset without errors?
        *   **For GRPO (`for_grpo=True`):** Is the `prompt_text` column created? Does it correctly combine the system prompt and the user query from the dataset? Does it look like a ready-to-use prompt?
        *   **For SFT (`for_grpo=False`):** Is the `messages` column present? Does it contain a list of dictionaries with "role" and "content", correctly structured with the system prompt first, then user, then assistant content?
        *   Are Arabic characters handled correctly?

2.  **Test `src/reward_functions.py`:**
    *   **How to test:** Add a temporary `if __name__ == "__main__":` block.
        ```python
        # Add this at the end of src/reward_functions.py for testing
        if __name__ == "__main__":
            import torch
            from transformers import AutoTokenizer

            # Test individual reward functions
            print("--- Testing individual reward functions ---")
            sample_completions_good = ["<think>أفكر باللغة العربية</think><answer>هذه إجابة عربية.</answer>"]
            sample_completions_bad_lang = ["<think>Thinking in English</think><answer>English answer.</answer>"]
            sample_completions_no_tags = ["مجرد نص عربي بدون علامات."]
            sample_completions_short = ["<think>قصير</think><answer>جدا</answer>"]
            sample_completions_empty = [""]

            print(f"Arabic only (good): {reward_arabic_only(sample_completions_good)}")
            print(f"Arabic only (bad lang): {reward_arabic_only(sample_completions_bad_lang)}")
            print(f"Think/Answer tags (good): {reward_think_answer_tags(sample_completions_good)}")
            print(f"Think/Answer tags (no tags): {reward_think_answer_tags(sample_completions_no_tags)}")
            print(f"Length (good): {reward_length(sample_completions_good, min_len=10, max_len=200)}")
            print(f"Length (short): {reward_length(sample_completions_short, min_len=10, max_len=200)}")
            
            # Test combined GRPO reward function
            print("\n--- Testing combined GRPO reward function ---")
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct") # Or your model
            config = get_reward_config() # Use default config
            
            # Simulate a batch from GRPOTrainer
            # GRPOTrainer passes `generated_responses` (list of strings) and `**batch`
            # where `batch` contains original inputs. Here we only need completions.
            # Our wrapper `grpo_reward_function_unsloth` needs `completions`, `tokenizer`, `reward_config`, and optional `batch_elements`
            
            # Example batch elements (simplified, GRPOTrainer provides more)
            # These aren't directly used by our current reward_fn_for_trainer wrapper's core logic
            # but good to simulate if your rewards get more complex
            simulated_batch_elements = {
                # 'prompt_text': ["Some prompt text 1", "Some prompt text 2"], 
                # 'log_probs': [torch.tensor([-0.1, -0.2]), torch.tensor([-0.3, -0.4])] 
            }

            all_completions = sample_completions_good + sample_completions_bad_lang + sample_completions_no_tags
            
            rewards_tensor, detailed_rewards_log = grpo_reward_function_unsloth(
                completions=all_completions,
                tokenizer=tokenizer, # Not strictly used by current rewards but good for consistency
                reward_config=config,
                batch_elements=simulated_batch_elements # Pass the simulated batch
            )
            print(f"Combined rewards (tensor): {rewards_tensor}")
            print(f"Detailed rewards log: {detailed_rewards_log}")
            assert isinstance(rewards_tensor, torch.Tensor), "Rewards must be a torch.Tensor"
            assert rewards_tensor.ndim == 1 and rewards_tensor.size(0) == len(all_completions), "Rewards tensor shape incorrect"

        ```
    *   **Run it:** `python src/reward_functions.py`
    *   **What to check:**
        *   Do individual functions give expected scores for good and bad examples?
        *   Does `grpo_reward_function_unsloth` combine them as expected according to weights in `reward_config`?
        *   **Crucially**: Is the final output a `torch.Tensor` with the correct shape (1D tensor, length equal to the number of completions)? This is required by TRL.
        *   Are the `detailed_rewards` logged correctly?

3.  **Local Linting:**
    *   Run `ruff check .` in your project root. Fix any reported issues.
    *   You can also try `ruff format .` to auto-format your code (commit any changes).

**Phase 3: Running GRPO Training (`grpo_unsloth_trainer.py`)**

1.  **Pre-requisites:**
    *   `data_loader.py` and `reward_functions.py` are tested and working.
    *   You have sufficient GPU VRAM. For Qwen2-0.5B 4-bit + LoRA, Unsloth is very efficient. Start with the default batch sizes in the script.
    *   Ensure Unsloth is installed correctly for your CUDA version.

2.  **Modify the script (if needed):**
    *   You might want to reduce `GRPO_EPOCHS` to 1 or even fewer steps (`max_steps` in `GRPOConfig`) for an initial test run.
    *   Consider using a very small slice of the dataset for the first run, e.g., by modifying `load_and_prepare_dataset` call: `train_dataset = train_dataset.select(range(100))` (after loading, before `create_prompt_text`).

3.  **Run the Training:**
    *   `python src/grpo_unsloth_trainer.py`

4.  **Monitor During Training:**
    *   **Console Output:**
        *   Watch for Unsloth's messages about model loading, 4-bit quantization, LoRA adapter injection.
        *   Look for dataset loading messages.
        *   TRL/Transformers trainer will output logs for each `logging_steps`. Pay close attention to `rewards/mean`. This is your primary indicator for GRPO. **You want this to be positive and ideally increasing.**
        *   No Python errors or CUDA errors.
    *   **GPU Utilization:** Use `watch -n 1 nvidia-smi` in another terminal to see if your GPU is being utilized.
    *   **TensorBoard (Optional but Recommended):**
        *   The `TrainingArguments` are set to report to TensorBoard.
        *   Run `tensorboard --logdir ./grpo_qwen2_0.5b_arabic_unsloth` (or your output directory) in another terminal.
        *   Open the provided URL in your browser to see live graphs of rewards and other metrics.

5.  **Expected Output:**
    *   The script will create an output directory (e.g., `grpo_qwen2_0.5b_arabic_unsloth`).
    *   Inside, you'll find checkpoints, including a `final_checkpoint` (if it completes) containing the LoRA adapter files (`adapter_model.safetensors`, `adapter_config.json`) and tokenizer files.

6.  **Troubleshooting GRPO:**
    *   **Negative `rewards/mean`:** This was your previous issue.
        *   **Most likely cause:** Your `reward_functions.py` (weights in `reward_config`, logic of individual functions, penalty vs. reward scaling) needs adjustment.
        *   The model might be generating poor quality responses initially that get heavily penalized.
        *   Ensure `grpo_reward_function_unsloth` correctly returns a scaled reward tensor.
    *   **CUDA Out of Memory:**
        *   Reduce `GRPO_BATCH_SIZE` (in `grpo_unsloth_trainer.py`).
        *   Reduce `GRPO_GENERATIONS_PER_PROMPT`.
        *   Reduce `MAX_SEQ_LENGTH` (but ensure it's appropriate for your data).
        *   Unsloth is usually very good, but GRPO generates multiple completions, which can be memory-intensive.
    *   **Very Slow Training:** Ensure Unsloth's optimizations are active (they should be by default). Check GPU utilization.

**Phase 4: Running SFT Training (`sft_unsloth_trainer.py`)**

1.  **Decision Point:**
    *   **Option A (SFT from base model):** Keep `MODEL_TO_SFT = BASE_MODEL_NAME` in `sft_unsloth_trainer.py`.
    *   **Option B (SFT from GRPO output):** Change `MODEL_TO_SFT` to the path of your GRPO checkpoint, e.g.:
        `MODEL_TO_SFT = "./grpo_qwen2_0.5b_arabic_unsloth/final_checkpoint"` (or whichever checkpoint you want to use). This allows SFT to build upon the reasoning alignment from GRPO.

2.  **Pre-requisites:**
    *   `data_loader.py` is tested for SFT format.
    *   Sufficient VRAM.

3.  **Modify the script (if needed):**
    *   Adjust `SFT_EPOCHS` (e.g., to 1 for a test).
    *   Use a small slice of the dataset for the first run: `train_dataset = train_dataset.select(range(100))` in `sft_unsloth_trainer.py`.

4.  **Run the Training:**
    *   `python src/sft_unsloth_trainer.py`

5.  **Monitor During Training:**
    *   **Console Output:** Similar to GRPO, look for model/dataset loading. For SFT, the key metric is `loss`. It should generally decrease.
    *   **GPU Utilization:** Check with `nvidia-smi`.
    *   **TensorBoard:** `tensorboard --logdir ./sft_qwen2_0.5b_arabic_unsloth` (or your SFT output dir).

6.  **Expected Output:**
    *   An SFT output directory (e.g., `sft_qwen2_0.5b_arabic_unsloth`) with checkpoints containing LoRA adapters.

7.  **Troubleshooting SFT:**
    *   **CUDA Out of Memory:** Reduce `SFT_BATCH_SIZE`, `MAX_SEQ_LENGTH`. Increase `SFT_GRAD_ACCUMULATION_STEPS`.
    *   **Loss Not Decreasing / Stagnant / NaN:**
        *   Learning rate (`SFT_LEARNING_RATE`) might be too high or too low.
        *   Data quality issues or incorrect formatting.
        *   If using 4-bit, sometimes stability can be an issue with certain optimizers or learning rates. The `adamw_8bit` from Unsloth is usually stable.

**Phase 5: Running Inference (`inference.py`)**

1.  **Configure Model Path:**
    *   Open `src/inference.py`.
    *   Change the `MODEL_TO_LOAD` variable to point to the checkpoint directory of the model you want to test (e.g., from GRPO or SFT):
        ```python
        # MODEL_TO_LOAD = DEFAULT_MODEL_PATH 
        MODEL_TO_LOAD = "./grpo_qwen2_0.5b_arabic_unsloth/final_checkpoint" 
        # OR
        # MODEL_TO_LOAD = "./sft_qwen2_0.5b_arabic_unsloth/final_checkpoint"
        ```

2.  **Modify Queries (Optional):**
    *   In the `if __name__ == "__main__":` block of `inference.py`, change `test_query` to ask questions relevant to your goals (Arabic reasoning).

3.  **Run Inference:**
    *   `python src/inference.py`

4.  **Evaluate Output:**
    *   Does the model respond in Arabic only?
    *   Does it use the `<think> </think><answer> </answer>` tags correctly?
    *   Is the reasoning sound? Is the answer accurate?
    *   Is the language fluent?
    *   How well does it follow instructions given in the prompt?

**Phase 6: Iteration and Improvement**

*   Based on GRPO training metrics (especially `rewards/mean`) and inference results:
    *   **Adjust `reward_functions.py`**: Modify weights in `reward_config`, change the logic of individual reward functions, add new ones, or adjust scaling. This is key for GRPO.
    *   **Refine `data_loader.py`**: Improve system prompts, ensure data quality.
    *   **Tune Hyperparameters**: Experiment with learning rates, batch sizes, epochs for both GRPO and SFT.
    *   **Dataset Augmentation/Filtering**: If your dataset isn't ideal, you might need more/better data.
*   Commit your changes to Git regularly, especially when you have a working version or are trying a new experiment (perhaps use branches for different experiments).

This is an iterative process. Don't expect perfect results on the first try. Systematically test, monitor, and adjust. Good luck!
