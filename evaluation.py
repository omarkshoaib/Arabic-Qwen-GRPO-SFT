#!/usr/bin/env python3
"""
🎯 Arabic LLM Evaluation Framework - Phase 1.1
Dataset Integration for ALLaM-style Benchmarking
"""

import random
from typing import Dict, List
from dataclasses import dataclass
from datasets import load_dataset

@dataclass
class EvaluationConfig:
    """Configuration for evaluation"""
    QUICK_SAMPLES = 5  # Small sample for testing
    STANDARD_SAMPLES = 50
    FULL_SAMPLES = None
    RANDOM_SEED = 42

class ArabicDatasetLoader:
    """Loads Arabic evaluation datasets"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        random.seed(config.RANDOM_SEED)
    
    def load_arabic_mmlu(self, tier: str = "quick") -> List[Dict]:
        """Load Arabic MMLU dataset"""
        print(f"📚 Loading Arabic MMLU ({tier} tier)...")
        
        try:
            dataset = load_dataset("MBZUAI/ArabicMMLU", "All", split="test")
            examples = list(dataset)
            
            # Sample based on tier
            if tier == "quick":
                sample_size = min(self.config.QUICK_SAMPLES, len(examples))
            elif tier == "standard":
                sample_size = min(self.config.STANDARD_SAMPLES, len(examples))
            else:
                sample_size = len(examples)
            
            if sample_size < len(examples):
                examples = random.sample(examples, sample_size)
            
            # Format examples
            formatted = []
            for example in examples:
                # Skip 5-choice questions
                if example.get('Option 5') and example['Option 5'] is not None:
                    continue
                
                formatted.append({
                    'id': f"mmlu_{len(formatted)}",
                    'question': example['Question'],
                    'choices': [example['Option 1'], example['Option 2'], 
                               example['Option 3'], example['Option 4']],
                    'correct_answer': example['Answer Key'],
                    'subject': example.get('Subject', 'unknown'),
                    'dataset': 'arabic_mmlu',
                    'type': 'multiple_choice'
                })
                
            print(f"✅ Loaded {len(formatted)} Arabic MMLU examples")
            return formatted
            
        except Exception as e:
            print(f"❌ Error loading Arabic MMLU: {e}")
            return []
    
    def load_acva(self, tier: str = "quick") -> List[Dict]:
        """Load ACVA dataset"""
        print(f"🏛️ Loading ACVA ({tier} tier)...")
        
        try:
            dataset = load_dataset("FreedomIntelligence/ACVA-Arabic-Cultural-Value-Alignment", split="test")
            examples = list(dataset)
            
            # Sample based on tier
            if tier == "quick":
                sample_size = min(self.config.QUICK_SAMPLES, len(examples))
            elif tier == "standard":
                sample_size = min(self.config.STANDARD_SAMPLES, len(examples))
            else:
                sample_size = len(examples)
            
            if sample_size < len(examples):
                examples = random.sample(examples, sample_size)
            
            # Format examples
            formatted = []
            for example in examples:
                formatted.append({
                    'id': f"acva_{len(formatted)}",
                    'question': example['question'],
                    'choices': ['صح', 'خطأ'],  # True, False
                    'correct_answer': 'A' if example['answer'] == 'صح' else 'B',
                    'category': 'cultural_alignment',
                    'dataset': 'acva',
                    'type': 'multiple_choice'
                })
                
            print(f"✅ Loaded {len(formatted)} ACVA examples")
            return formatted
            
        except Exception as e:
            print(f"❌ Error loading ACVA: {e}")
            return []

def test_phase_1_1():
    """Test Phase 1.1 implementation"""
    print("🎯 Testing Phase 1.1: Dataset Integration")
    print("=" * 50)
    
    config = EvaluationConfig()
    loader = ArabicDatasetLoader(config)
    
    # Test Arabic MMLU
    print("\n1️⃣ Testing Arabic MMLU...")
    mmlu_data = loader.load_arabic_mmlu("quick")
    
    if mmlu_data:
        print(f"✅ Arabic MMLU: {len(mmlu_data)} examples loaded")
        example = mmlu_data[0]
        print(f"   Sample question: {example['question'][:50]}...")
        print(f"   Choices: {len(example['choices'])}")
        print(f"   Correct answer: {example['correct_answer']}")
        print(f"   Subject: {example['subject']}")
    else:
        print("❌ No Arabic MMLU data loaded")
        return False
    
    # Test ACVA
    print("\n2️⃣ Testing ACVA...")
    acva_data = loader.load_acva("quick")
    
    if acva_data:
        print(f"✅ ACVA: {len(acva_data)} examples loaded")
        example = acva_data[0]
        print(f"   Sample question: {example['question'][:50]}...")
        print(f"   Choices: {example['choices']}")
        print(f"   Correct answer: {example['correct_answer']}")
    else:
        print("❌ No ACVA data loaded")
        return False
    
    print("\n🎉 Phase 1.1 Dataset Integration - SUCCESS!")
    print("✅ All datasets load correctly")
    print("✅ Data formatting works correctly")
    
    return True

if __name__ == "__main__":
    success = test_phase_1_1()
    
    if success:
        print("\n📝 Next Steps:")
        print("  • Phase 1.2: Basic Model Handler")
        print("  • Phase 2.1: Basic Metrics Implementation")
        print("  • Phase 2.2: Advanced Metrics")
    else:
        print("\n❌ Phase 1.1 failed. Please check the issues above.") 