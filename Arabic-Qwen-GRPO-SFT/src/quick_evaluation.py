"""
quick_evaluation.py - Simplified ALLaM-Style Evaluation for Colab Integration

Quick evaluation script optimized for Colab notebooks with minimal setup.
Focuses on core Arabic LLM benchmarks following ALLaM methodology.
"""

import torch
from evaluation import ArabicLLMEvaluator
import json
from datetime import datetime

def quick_evaluate_models(base_model_path="unsloth/Qwen2.5-0.5B-Instruct",
                         sft_model_path=None,
                         grpo_model_path=None,
                         benchmarks=['arabic_reasoning_qa'],
                         samples_per_benchmark=20):
    """Quick evaluation function for Colab notebooks."""
    
    print("🚀 Starting Quick Arabic LLM Evaluation")
    print(f"📊 Benchmarks: {benchmarks}")
    print(f"🎯 Samples per benchmark: {samples_per_benchmark}")
    print("-" * 50)
    
    # Initialize evaluator
    evaluator = ArabicLLMEvaluator(
        base_model_path=base_model_path,
        sft_model_path=sft_model_path,
        grpo_model_path=grpo_model_path,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Reduce sample sizes for quick evaluation
    for benchmark in benchmarks:
        if benchmark in evaluator.all_benchmarks:
            evaluator.all_benchmarks[benchmark]['samples'] = samples_per_benchmark
    
    # Run evaluation
    results = evaluator.run_full_evaluation(benchmarks=benchmarks)
    
    # Generate and display report
    report = evaluator.generate_comparison_report()
    print(report)
    
    return results, report

def run_colab_evaluation(config='core_evaluation', 
                        base_model="unsloth/Qwen2.5-0.5B-Instruct",
                        sft_path=None, 
                        grpo_path=None):
    """Main evaluation function optimized for Colab with predefined configurations."""
    
    configs = {
        'quick_test': {
            'benchmarks': ['arabic_reasoning_qa'],
            'samples': 10,
            'description': 'Quick smoke test'
        },
        'core_evaluation': {
            'benchmarks': ['arabic_reasoning_qa'],
            'samples': 50,
            'description': 'Core domain evaluation'
        },
        'comprehensive': {
            'benchmarks': ['arabic_mmlu', 'acva', 'arabic_reasoning_qa'],
            'samples': 100,
            'description': 'Full ALLaM-style evaluation'
        }
    }
    
    if config not in configs:
        print(f"❌ Unknown config: {config}")
        print(f"Available configs: {list(configs.keys())}")
        return
    
    eval_config = configs[config]
    
    print(f"�� Running {config} evaluation")
    print(f"📝 Description: {eval_config['description']}")
    print("=" * 60)
    
    results, report = quick_evaluate_models(
        base_model_path=base_model,
        sft_model_path=sft_path,
        grpo_model_path=grpo_path,
        benchmarks=eval_config['benchmarks'],
        samples_per_benchmark=eval_config['samples']
    )
    
    return results

if __name__ == "__main__":
    print("🔬 Quick Evaluation Example")
    results = run_colab_evaluation('quick_test')
    print("✅ Quick evaluation complete!")
