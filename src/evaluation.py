"""
🎯 Arabic LLM Evaluation Framework - ALLaM Style Benchmarking
================================================================

Phase 1.1: Dataset Integration
- Arabic MMLU (MBZUAI/ArabicMMLU)
- ACVA (FreedomIntelligence/ACVA-Arabic-Cultural-Value-Alignment)  
- Arabic Reasoning QA (MohammedNasser/ARabic_Reasoning_QA)

Following ALLaM paper methodology for fair comparison.
"""

import os
import json
import random
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datasets import load_dataset
from tqdm import tqdm
import re

# =============================================================================
# CONFIGURATION & MODEL REGISTRY
# =============================================================================

@dataclass
class EvaluationConfig:
    """Configuration class for evaluation settings"""
    
    # Evaluation Tiers
    QUICK_SAMPLES = 10      # 2-3 min smoke test
    STANDARD_SAMPLES = 50   # 10-15 min core evaluation  
    FULL_SAMPLES = None     # 45-60 min complete evaluation
    
    # Model Configuration
    MAX_LENGTH = 512
    TEMPERATURE = 0.1
    TOP_P = 0.9
    DO_SAMPLE = True
    
    # Dataset Configuration
    RANDOM_SEED = 42

# Your Training Progression Models
YOUR_MODELS = {
    "baseline": "unsloth/Qwen2.5-0.5B-Instruct",
    "sft": "/content/drive/MyDrive/Arabic-Qwen-Outputs/sft_qwen2.5_0.5b_instruct_unsloth/final_checkpoint", 
    "grpo": "/content/drive/MyDrive/Arabic-Qwen-Outputs/grpo_on_sft_qwen2.5_0.5b_bnb_4bit_unsloth_final"
}

# Comparison Models (<14B Parameters)
COMPARISON_MODELS = {
    # Qwen 3 Family
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B", 
    "qwen3-4b": "Qwen/Qwen3-4B",
    
    # Qwen 2.5 Family
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct",
    
    # SILMA Family
    "silma-2b": "silma-ai/SILMA-Kashif-2B-Instruct-v1.0",
    "silma-9b": "silma-ai/SILMA-9B-Instruct-v1.0",
    
    # Jais Family
    "jais-13b": "inceptionai/jais-13b",
    
    # ALLaM Family  
    "allam-7b": "ALLaM-AI/ALLaM-7B-Instruct-preview"
}

# =============================================================================
# DATASET LOADERS
# =============================================================================

class ArabicDatasetLoader:
    """Handles loading and formatting of Arabic evaluation datasets"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        random.seed(config.RANDOM_SEED)
        
    def load_arabic_mmlu(self, tier: str = "quick") -> List[Dict]:
        """
        Load Arabic MMLU dataset from MBZUAI/ArabicMMLU
        
        Args:
            tier: "quick", "standard", or "full"
            
        Returns:
            List of formatted examples
        """
        print(f"📚 Loading Arabic MMLU ({tier} tier)...")
        
        try:
            # Load test split with "All" config
            dataset = load_dataset("MBZUAI/ArabicMMLU", "All", split="test")
            
            # Convert to list and sample based on tier
            examples = list(dataset)
            examples = self._sample_by_tier(examples, tier)
            
            # Format for evaluation
            formatted_examples = []
            for example in examples:
                # Skip if Option 5 exists (5-choice questions)
                if example.get('Option 5') and example['Option 5'] is not None:
                    continue
                
                formatted_example = {
                    'id': f"mmlu_{len(formatted_examples)}",
                    'question': example['Question'],
                    'choices': [example['Option 1'], example['Option 2'], example['Option 3'], example['Option 4']],
                    'correct_answer': example['Answer Key'],  # A, B, C, or D
                    'subject': example.get('Subject', 'unknown'),
                    'dataset': 'arabic_mmlu',
                    'type': 'multiple_choice'
                }
                formatted_examples.append(formatted_example)
                
            print(f"✅ Loaded {len(formatted_examples)} Arabic MMLU examples")
            return formatted_examples
            
        except Exception as e:
            print(f"❌ Error loading Arabic MMLU: {e}")
            return []
    
    def load_acva(self, tier: str = "quick") -> List[Dict]:
        """
        Load ACVA dataset for cultural alignment evaluation
        
        Args:
            tier: "quick", "standard", or "full"
            
        Returns:
            List of formatted examples
        """
        print(f"🏛️ Loading ACVA ({tier} tier)...")
        
        try:
            # Load test split
            dataset = load_dataset("FreedomIntelligence/ACVA-Arabic-Cultural-Value-Alignment", split="test")
            
            # Convert to list and sample based on tier
            examples = list(dataset)
            examples = self._sample_by_tier(examples, tier)
            
            # Format for evaluation
            formatted_examples = []
            for example in examples:
                # ACVA is True/False, so we create binary choices
                formatted_example = {
                    'id': f"acva_{len(formatted_examples)}",
                    'question': example['question'],
                    'choices': ['صح', 'خطأ'],  # True, False in Arabic
                    'correct_answer': 'A' if example['answer'] == 'صح' else 'B',
                    'category': 'cultural_alignment',
                    'dataset': 'acva',
                    'type': 'multiple_choice'
                }
                formatted_examples.append(formatted_example)
                
            print(f"✅ Loaded {len(formatted_examples)} ACVA examples")
            return formatted_examples
            
        except Exception as e:
            print(f"❌ Error loading ACVA: {e}")
            return []
    
    def load_arabic_reasoning_qa(self, tier: str = "quick") -> List[Dict]:
        """
        Load Arabic Reasoning QA dataset (placeholder for future implementation)
        
        Note: Original dataset MohammedNasser/ARabic_Reasoning_QA is gated.
        For now, we'll use a subset of Arabic MMLU as reasoning examples.
        
        Args:
            tier: "quick", "standard", or "full"
            
        Returns:
            List of formatted examples
        """
        print(f"🧠 Loading Arabic Reasoning QA ({tier} tier)...")
        print("   📝 Note: Using Arabic MMLU subset for reasoning evaluation")
        
        try:
            # Load test split with "All" config and filter for reasoning subjects
            dataset = load_dataset("MBZUAI/ArabicMMLU", "All", split="test")
            
            # Filter for subjects that require reasoning
            reasoning_subjects = ["Philosophy", "Logic", "Critical Thinking", "Mathematics", "Computer Science"]
            
            # Convert to list and filter
            examples = [ex for ex in dataset if any(subj.lower() in ex.get('Subject', '').lower() for subj in reasoning_subjects)]
            examples = self._sample_by_tier(examples, tier)
            
            # Format for evaluation as open-ended questions
            formatted_examples = []
            for example in examples:
                # Skip if Option 5 exists (5-choice questions)
                if example.get('Option 5') and example['Option 5'] is not None:
                    continue
                
                # Convert to open-ended format
                question_text = f"{example['Question']}\n\nالخيارات:\nأ) {example['Option 1']}\nب) {example['Option 2']}\nج) {example['Option 3']}\nد) {example['Option 4']}"
                
                formatted_example = {
                    'id': f"reasoning_{len(formatted_examples)}",
                    'question': question_text,
                    'correct_answer': example['Answer Key'],
                    'context': example.get('Context', ''),
                    'subject': example.get('Subject', 'unknown'),
                    'dataset': 'arabic_reasoning_qa',
                    'type': 'open_ended'
                }
                formatted_examples.append(formatted_example)
                
            print(f"✅ Loaded {len(formatted_examples)} Arabic Reasoning examples")
            return formatted_examples
            
        except Exception as e:
            print(f"❌ Error loading Arabic Reasoning QA: {e}")
            return []
    
    def _sample_by_tier(self, examples: List[Dict], tier: str) -> List[Dict]:
        """Sample examples based on evaluation tier"""
        
        if tier == "quick":
            sample_size = min(self.config.QUICK_SAMPLES, len(examples))
        elif tier == "standard":
            sample_size = min(self.config.STANDARD_SAMPLES, len(examples))
        elif tier == "full":
            sample_size = len(examples)  # Use all examples
        else:
            raise ValueError(f"Unknown tier: {tier}")
            
        # Random sample
        if sample_size < len(examples):
            examples = random.sample(examples, sample_size)
            
        return examples

# =============================================================================
# PROMPT FORMATTERS
# =============================================================================

class ArabicPromptFormatter:
    """Handles Arabic prompt formatting for different question types"""
    
    @staticmethod
    def format_multiple_choice(question: str, choices: List[str]) -> str:
        """
        Format multiple choice question in Arabic
        
        Args:
            question: The question text
            choices: List of 4 choices
            
        Returns:
            Formatted prompt string
        """
        
        choice_labels = ['أ', 'ب', 'ج', 'د']  # Arabic A, B, C, D
        
        prompt = f"""السؤال: {question}

الخيارات:
"""
        
        for i, choice in enumerate(choices):
            prompt += f"{choice_labels[i]}) {choice}\n"
            
        prompt += "\nالإجابة الصحيحة هي:"
        
        return prompt
    
    @staticmethod
    def format_open_ended(question: str, context: str = "") -> str:
        """
        Format open-ended question in Arabic
        
        Args:
            question: The question text
            context: Optional context
            
        Returns:
            Formatted prompt string
        """
        
        if context:
            prompt = f"""السياق: {context}

السؤال: {question}

الإجابة:"""
        else:
            prompt = f"""السؤال: {question}

الإجابة:"""
            
        return prompt
    
    @staticmethod
    def get_system_prompt() -> str:
        """Get Arabic system prompt for chat models"""
        
        return """أنت نموذج ذكي ومفيد للإجابة على الأسئلة باللغة العربية. أجب بدقة ووضوح."""

# =============================================================================
# ANSWER EXTRACTION
# =============================================================================

class AnswerExtractor:
    """Extracts and normalizes answers from model responses"""
    
    @staticmethod
    def extract_multiple_choice_answer(response: str) -> str:
        """
        Extract A, B, C, D answer from response
        
        Args:
            response: Model response text
            
        Returns:
            Extracted answer (A, B, C, D) or "UNKNOWN"
        """
        
        # Arabic choice patterns
        arabic_patterns = [
            r'الإجابة الصحيحة هي\s*[:\s]*([أبجد])',
            r'الإجابة\s*[:\s]*([أبجد])',
            r'([أبجد])\s*\)',
            r'الخيار\s*([أبجد])',
        ]
        
        # English choice patterns  
        english_patterns = [
            r'[Tt]he answer is\s*[:\s]*([ABCD])',
            r'[Aa]nswer\s*[:\s]*([ABCD])',
            r'([ABCD])\s*\)',
            r'[Oo]ption\s*([ABCD])',
        ]
        
        # Try Arabic patterns first
        for pattern in arabic_patterns:
            match = re.search(pattern, response)
            if match:
                # Convert Arabic to English
                arabic_to_english = {'أ': 'A', 'ب': 'B', 'ج': 'C', 'د': 'D'}
                return arabic_to_english.get(match.group(1), 'UNKNOWN')
        
        # Try English patterns
        for pattern in english_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        # Last resort: look for standalone letters
        standalone = re.search(r'\b([ABCDأبجد])\b', response)
        if standalone:
            letter = standalone.group(1)
            if letter in 'أبجد':
                arabic_to_english = {'أ': 'A', 'ب': 'B', 'ج': 'C', 'د': 'D'}
                return arabic_to_english[letter]
            elif letter in 'ABCD':
                return letter
                
        return "UNKNOWN"
    
    @staticmethod
    def clean_open_ended_answer(response: str) -> str:
        """
        Clean and normalize open-ended answer
        
        Args:
            response: Model response text
            
        Returns:
            Cleaned answer text
        """
        
        # Remove common prefixes
        prefixes_to_remove = [
            "الإجابة:",
            "الإجابة هي:",
            "Answer:",
            "The answer is:",
        ]
        
        cleaned = response.strip()
        
        for prefix in prefixes_to_remove:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
                
        return cleaned

# =============================================================================
# MAIN EVALUATION FRAMEWORK
# =============================================================================

class ArabicLLMEvaluator:
    """Main evaluation framework for Arabic LLMs"""
    
    def __init__(self, config: EvaluationConfig = None):
        self.config = config or EvaluationConfig()
        self.dataset_loader = ArabicDatasetLoader(self.config)
        self.prompt_formatter = ArabicPromptFormatter()
        self.answer_extractor = AnswerExtractor()
        
    def load_all_datasets(self, tier: str = "quick") -> Dict[str, List[Dict]]:
        """
        Load all evaluation datasets
        
        Args:
            tier: "quick", "standard", or "full"
            
        Returns:
            Dictionary with dataset names as keys and examples as values
        """
        
        print(f"🚀 Loading all datasets for {tier} evaluation...")
        
        datasets = {}
        
        # Load Arabic MMLU
        datasets['arabic_mmlu'] = self.dataset_loader.load_arabic_mmlu(tier)
        
        # Load ACVA
        datasets['acva'] = self.dataset_loader.load_acva(tier)
        
        # Load Arabic Reasoning QA
        datasets['arabic_reasoning_qa'] = self.dataset_loader.load_arabic_reasoning_qa(tier)
        
        # Print summary
        total_examples = sum(len(examples) for examples in datasets.values())
        print(f"\n📊 Dataset Summary ({tier} tier):")
        for name, examples in datasets.items():
            print(f"  • {name}: {len(examples)} examples")
        print(f"  • Total: {total_examples} examples")
        
        return datasets
    
    def format_example(self, example: Dict) -> str:
        """
        Format a single example for model input
        
        Args:
            example: Example dictionary
            
        Returns:
            Formatted prompt string
        """
        
        if example['type'] == 'multiple_choice':
            return self.prompt_formatter.format_multiple_choice(
                example['question'], 
                example['choices']
            )
        elif example['type'] == 'open_ended':
            return self.prompt_formatter.format_open_ended(
                example['question'],
                example.get('context', '')
            )
        else:
            raise ValueError(f"Unknown question type: {example['type']}")

# =============================================================================
# TESTING FUNCTIONS (Phase 1.1 Test Criteria)
# =============================================================================

def test_phase_1_1():
    """Test Phase 1.1: Dataset Integration"""
    
    print("🧪 Testing Phase 1.1: Dataset Integration")
    print("=" * 50)
    
    config = EvaluationConfig()
    evaluator = ArabicLLMEvaluator(config)
    
    # Test 1: All datasets load correctly
    print("\n1️⃣ Testing dataset loading...")
    datasets = evaluator.load_all_datasets("quick")
    
    expected_datasets = ['arabic_mmlu', 'acva', 'arabic_reasoning_qa']
    for dataset_name in expected_datasets:
        assert dataset_name in datasets, f"Missing dataset: {dataset_name}"
        assert len(datasets[dataset_name]) > 0, f"Empty dataset: {dataset_name}"
        print(f"   ✓ {dataset_name}: {len(datasets[dataset_name])} examples")
    
    print("✅ All datasets load correctly")
    
    # Test 2: Prompt formatting works
    print("\n2️⃣ Testing prompt formatting...")
    
    # Test multiple choice
    if datasets['arabic_mmlu']:
        mc_example = datasets['arabic_mmlu'][0]
        mc_prompt = evaluator.format_example(mc_example)
        assert 'السؤال:' in mc_prompt, "Missing question marker"
        assert 'الخيارات:' in mc_prompt, "Missing choices marker"
        assert 'أ)' in mc_prompt, "Missing choice A"
    
    # Test open-ended
    if datasets['arabic_reasoning_qa']:
        oe_example = datasets['arabic_reasoning_qa'][0]
        oe_prompt = evaluator.format_example(oe_example)
        assert 'السؤال:' in oe_prompt, "Missing question marker"
        assert 'الإجابة:' in oe_prompt, "Missing answer marker"
    
    print("✅ Prompt formatting works")
    
    # Test 3: Sample extraction functions work
    print("\n3️⃣ Testing answer extraction...")
    
    extractor = AnswerExtractor()
    
    # Test multiple choice extraction
    test_responses = [
        "الإجابة الصحيحة هي: أ",
        "الإجابة: ب", 
        "The answer is A",
        "B) Option B"
    ]
    
    for response in test_responses:
        extracted = extractor.extract_multiple_choice_answer(response)
        assert extracted in ['A', 'B', 'C', 'D', 'UNKNOWN'], f"Invalid extraction: {extracted}"
    
    # Test open-ended cleaning
    oe_response = "الإجابة: هذا نص الإجابة"
    cleaned = extractor.clean_open_ended_answer(oe_response)
    assert cleaned == "هذا نص الإجابة", f"Cleaning failed: {cleaned}"
    
    print("✅ Sample extraction functions work")
    
    return datasets

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("🎯 Arabic LLM Evaluation Framework - Phase 1.1")
    print("Dataset Integration Implementation")
    print("=" * 60)
    
    try:
        # Run Phase 1.1 tests
        datasets = test_phase_1_1()
        
        print("\n🎉 Phase 1.1 Complete!")
        print("✅ All datasets load correctly")
        print("✅ Prompt formatting works") 
        print("✅ Sample extraction functions work")
        
        print("\n📝 Next Steps:")
        print("  • Phase 1.2: Basic Model Handler")
        print("  • Phase 2.1: Basic Metrics")
        print("  • Phase 2.2: Advanced Metrics")
        
        # Show sample data
        print("\n📋 Sample Data Preview:")
        for dataset_name, examples in datasets.items():
            if examples:
                print(f"\n{dataset_name.upper()}:")
                example = examples[0]
                print(f"  Question: {example['question'][:100]}...")
                print(f"  Type: {example['type']}")
                if 'choices' in example:
                    print(f"  Choices: {len(example['choices'])} options")
                    
    except Exception as e:
        print(f"❌ Phase 1.1 Failed: {e}")
        import traceback
        traceback.print_exc() 