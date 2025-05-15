# Arabic-Qwen-GRPO-SFT

This project aims to train a Qwen 2.5 0.5B Instruct model to reason and communicate effectively in Arabic. The training process involves:
1.  **Group Relative Policy Optimization (GRPO)** using Unsloth with custom Arabic-focused reward functions.
2.  **Supervised Fine-Tuning (SFT)** with Arabic instruction datasets to further enhance performance and ensure Arabic-only output.

The project is inspired by the methodologies used in DeepSeek's R1 models, adapted for Arabic language tasks.

## Project Goal

To develop a Qwen 2.5 0.5B model that:
-   Exhibits reasoning capabilities.
-   Communicates fluently and accurately *exclusively* in Arabic.
-   Can follow Arabic instructions effectively.

## Directory Structure

```
Arabic-Qwen-GRPO-SFT/
├── .github/
│   └── workflows/
│       └── python-ci.yml
├── .gitignore
├── README.md
├── requirements.txt
├── data/
│   ├── omartificial-Intelligence/ # Raw dataset (e.g., omartificial/OmArtificial-Dolly-instruct-5k)
│   └── processed/                 # Processed and tokenized datasets
├── src/
│   ├── data_loader.py           # Scripts for loading and preprocessing data
│   ├── reward_functions.py      # Arabic-specific reward functions for GRPO
│   ├── grpo_unsloth_trainer.py  # Script for GRPO training using Unsloth
│   ├── sft_unsloth_trainer.py   # Script for SFT training using Unsloth
│   ├── inference.py             # Script for model inference and testing
│   └── utils.py                 # Utility functions (tokenization, chat templates, etc.)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_reward_testing.ipynb
│   ├── 03_grpo_training_run.ipynb
│   ├── 04_sft_training_run.ipynb
└── configs/
    ├── grpo_config.json         # Configuration for GRPO training
    └── sft_config.json          # Configuration for SFT
```

## Setup

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/Arabic-Qwen-GRPO-SFT.git
    cd Arabic-Qwen-GRPO-SFT
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Training Workflow

1.  **Data Preparation**:
    *   Download the Arabic dataset (e.g., `omartificial/OmArtificial-Dolly-instruct-5k`) into the `data/omartificial-Intelligence/` directory.
    *   Run preprocessing scripts (via `src/data_loader.py` or notebooks) to format and tokenize the data, saving results to `data/processed/`.
2.  **Reward Function Development**:
    *   Define and test reward functions in `src/reward_functions.py`. Use `notebooks/02_reward_testing.ipynb` for iterative development.
3.  **GRPO Training**:
    *   Configure GRPO parameters in `configs/grpo_config.json`.
    *   Run GRPO training using `src/grpo_unsloth_trainer.py` or `notebooks/03_grpo_training_run.ipynb`.
4.  **SFT Training**:
    *   Configure SFT parameters in `configs/sft_config.json`.
    *   Run SFT training using `src/sft_unsloth_trainer.py` or `notebooks/04_sft_training_run.ipynb` on the GRPO-trained model or a base model.
5.  **Inference and Evaluation**:
    *   Use `src/inference.py` to test the final model.

## Key Technologies
-   Qwen 2.5 0.5B Instruct
-   Unsloth
-   TRL (Transformer Reinforcement Learning)
-   Hugging Face Transformers, Datasets, Accelerate
-   PyTorch 