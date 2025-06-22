# 🚀 Quick Start: ALLaM-Style Evaluation

## 📱 Copy-Paste Colab Cells

### Cell 1: Setup
```python
# 📦 Install and Setup
!pip install datasets transformers torch pandas tqdm

from google.colab import drive
drive.mount('/content/drive')

import os
os.chdir('/content/drive/MyDrive/Arabic-Qwen-GRPO-SFT')

import sys
sys.path.append('/content/drive/MyDrive/Arabic-Qwen-GRPO-SFT/src')
```

### Cell 2: Quick Evaluation (2-3 minutes)
```python
# 🏃‍♂️ Quick Test
from src.quick_evaluation import run_colab_evaluation

BASE_MODEL = "unsloth/Qwen2.5-0.5B-Instruct"
SFT_CHECKPOINT = "/content/drive/MyDrive/Arabic-Qwen-Outputs/sft_qwen2.5_0.5b_instruct_unsloth/final_checkpoint"
GRPO_CHECKPOINT = "/content/drive/MyDrive/Arabic-Qwen-Outputs/grpo_on_sft_qwen2.5_0.5b_bnb_4bit_unsloth_final"

# Quick test with 10 samples
results = run_colab_evaluation(
    config='quick_test',
    base_model=BASE_MODEL,
    sft_path=SFT_CHECKPOINT,
    grpo_path=GRPO_CHECKPOINT
)
```

### Cell 3: Core Evaluation (10-15 minutes)
```python
# 📊 Core Evaluation
results = run_colab_evaluation(
    config='core_evaluation',
    base_model=BASE_MODEL,
    sft_path=SFT_CHECKPOINT,
    grpo_path=GRPO_CHECKPOINT
)
```

## 🎛️ Configuration Options

| Config | Speed | Samples | Benchmarks | Use Case |
|--------|-------|---------|------------|----------|
| `quick_test` | 2-3 min | 10 | Arabic Reasoning QA | Smoke test |
| `core_evaluation` | 10-15 min | 50 | Arabic Reasoning QA | Domain assessment |
| `comprehensive` | 30-60 min | 100 | MMLU + ACVA + Reasoning | Full comparison |

## 📊 Expected Output

```
🚀 Starting Quick Arabic LLM Evaluation
📊 Benchmarks: ['arabic_reasoning_qa']
🎯 Samples per benchmark: 10

📊 Arabic LLM Training Progression Evaluation
==================================================

Training Progression Analysis:
baseline: 28.5%
sft_model: 35.1% (+6.6%)
grpo_model: 41.3% (+12.8%)
```

**Ready to evaluate? Start with Cell 1! 🎯**
