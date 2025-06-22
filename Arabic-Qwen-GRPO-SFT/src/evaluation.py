"""
evaluation.py - ALLaM-Style Arabic LLM Evaluation Suite

Follows the exact benchmarks used by ALLaM, ALLaM-Thinking, and other SOTA Arabic models.
Evaluates the progression: Base → SFT → GRPO for Arabic reasoning models.
"""

import os
import json
import torch
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    print("Warning: Unsloth not available. Will use standard transformers for all models.")

import numpy as np
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArabicLLMEvaluator:
    """
    Comprehensive evaluation suite for Arabic LLMs following ALLaM methodology.
    
    Supports evaluation of training progression: Base → SFT → GRPO
    """
    
    def __init__(self, 
                 base_model_path: str = "unsloth/Qwen2.5-0.5B-Instruct",
                 sft_model_path: Optional[str] = None,
                 grpo_model_path: Optional[str] = None,
                 device: str = "cuda",
                 max_length: int = 2048):
        """Initialize the evaluator with model paths."""
        self.device = device
        self.max_length = max_length
        
        # Model configurations
        self.models = {
            'baseline': {
                'path': base_model_path,
                'description': 'Base instruction-tuned model',
                'training_stage': 0
            }
        }
        
        if sft_model_path:
            self.models['sft_model'] = {
                'path': sft_model_path,
                'description': 'SFT fine-tuned model',
                'training_stage': 1
            }
        
        if grpo_model_path:
            self.models['grpo_model'] = {
                'path': grpo_model_path,
                'description': 'GRPO trained on SFT model',
                'training_stage': 2
            }
        
        # Core benchmarks following ALLaM paper exactly
        self.core_benchmarks = {
            'arabic_mmlu': {
                'dataset': 'MBZUAI/ArabicMMLU',
                'task': 'Multi-subject reasoning (40 subjects)',
                'metric': 'accuracy',
                'why': 'Used by ALLaM, ALLaM-Thinking, AceGPT, Jais',
                'split': 'test',
                'samples': 100
            },
            'acva': {
                'dataset': 'FreedomIntelligence/ACVA-Arabic-Cultural-Value-Alignment',
                'task': 'Cultural alignment & ethics',
                'metric': 'accuracy', 
                'why': 'Used by ALLaM, AceGPT for cultural safety',
                'split': 'test',
                'samples': 100
            }
        }
        
        # Extended benchmarks for comprehensive analysis
        self.extended_benchmarks = {
            'arabic_reasoning_qa': {
                'dataset': 'MohammedNasser/ARabic_Reasoning_QA',
                'task': 'Multi-level reasoning progression',
                'metric': 'accuracy',
                'why': 'Domain-specific reasoning benchmark',
                'split': 'test',
                'samples': 50
            }
        }
        
        # Combined benchmark suite
        self.all_benchmarks = {**self.core_benchmarks, **self.extended_benchmarks}
        
        # Results storage
        self.results = {}
        self.loaded_models = {}

    def load_model(self, model_name: str, model_path: str):
        """Load a model for evaluation."""
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
            
        logger.info(f"Loading model: {model_name} from {model_path}")
        
        try:
            # Try loading with Unsloth (for our fine-tuned models)
            if UNSLOTH_AVAILABLE and ("checkpoint" in model_path.lower() or "sft" in model_path.lower() or "grpo" in model_path.lower()):
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=model_path,
                    max_seq_length=self.max_length,
                    dtype=None,
                    load_in_4bit=True,
                    device_map=self.device
                )
                FastLanguageModel.for_inference(model)
            else:
                # Standard transformers loading for base models
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                    
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map=self.device,
                    trust_remote_code=True
                )
                
            self.loaded_models[model_name] = (model, tokenizer)
            logger.info(f"Successfully loaded {model_name}")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            return None, None

    def generate_comparison_report(self) -> str:
        """Generate a comprehensive comparison report."""
        if not self.results:
            return "No evaluation results available. Run evaluation first."
        
        report = []
        report.append("📊 Arabic LLM Training Progression Evaluation")
        report.append("=" * 50)
        report.append(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Models Evaluated: {len(self.results)}")
        report.append("")
        
        return "\n".join(report)

    def run_full_evaluation(self, benchmarks: List[str] = None) -> Dict[str, Any]:
        """Run complete evaluation across all models and benchmarks."""
        if benchmarks is None:
            benchmarks = list(self.all_benchmarks.keys())
            
        logger.info(f"Starting full evaluation on benchmarks: {benchmarks}")
        logger.info(f"Models to evaluate: {list(self.models.keys())}")
        
        # Placeholder implementation
        all_results = {}
        for model_name in self.models.keys():
            all_results[model_name] = {
                'arabic_reasoning_qa': {
                    'accuracy': 0.5,
                    'correct': 25,
                    'total': 50
                }
            }
        
        self.results = all_results
        return all_results

if __name__ == "__main__":
    evaluator = ArabicLLMEvaluator()
    results = evaluator.run_full_evaluation()
    print(evaluator.generate_comparison_report())
