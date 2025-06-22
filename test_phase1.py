#!/usr/bin/env python3
"""
Simple test for Phase 1.1: Dataset Integration
"""

import sys
import time

print("🎯 Phase 1.1 Test - Arabic Dataset Loading")
print("=" * 50)

# Test imports
print("📦 Testing imports...")
try:
    from evaluation import EvaluationConfig, ArabicDatasetLoader, ArabicLLMEvaluator
    print("✅ Imports successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test basic functionality
print("\n🔧 Testing basic functionality...")
try:
    config = EvaluationConfig()
    print(f"✅ Config: quick_samples={config.QUICK_SAMPLES}")
    
    loader = ArabicDatasetLoader(config)
    print("✅ Loader created")
    
    evaluator = ArabicLLMEvaluator(config)
    print("✅ Evaluator created")
    
except Exception as e:
    print(f"❌ Basic setup failed: {e}")
    sys.exit(1)

# Test individual dataset loading with timeout
print("\n📚 Testing individual datasets...")

def test_with_timeout(func, timeout=30):
    """Test function with timeout"""
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {timeout}s")
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    
    try:
        result = func()
        signal.alarm(0)  # Cancel alarm
        return result
    except TimeoutError:
        signal.alarm(0)
        raise
    except Exception as e:
        signal.alarm(0)
        raise e

# Test Arabic MMLU
print("\n1️⃣ Testing Arabic MMLU...")
try:
    config.QUICK_SAMPLES = 3  # Very small sample for testing
    loader = ArabicDatasetLoader(config)
    
    mmlu_data = test_with_timeout(lambda: loader.load_arabic_mmlu("quick"), 30)
    
    if mmlu_data:
        print(f"✅ Arabic MMLU: {len(mmlu_data)} examples")
        example = mmlu_data[0]
        print(f"   Sample: {example['question'][:50]}...")
        print(f"   Choices: {len(example['choices'])}")
        print(f"   Answer: {example['correct_answer']}")
    else:
        print("❌ Arabic MMLU: No data returned")
        
except Exception as e:
    print(f"❌ Arabic MMLU failed: {e}")

# Test ACVA
print("\n2️⃣ Testing ACVA...")
try:
    acva_data = test_with_timeout(lambda: loader.load_acva("quick"), 30)
    
    if acva_data:
        print(f"✅ ACVA: {len(acva_data)} examples")
        example = acva_data[0]
        print(f"   Sample: {example['question'][:50]}...")
        print(f"   Choices: {example['choices']}")
        print(f"   Answer: {example['correct_answer']}")
    else:
        print("❌ ACVA: No data returned")
        
except Exception as e:
    print(f"❌ ACVA failed: {e}")

# Test prompt formatting
print("\n3️⃣ Testing prompt formatting...")
try:
    if 'mmlu_data' in locals() and mmlu_data:
        from evaluation import ArabicPromptFormatter
        formatter = ArabicPromptFormatter()
        
        mc_prompt = formatter.format_multiple_choice(
            mmlu_data[0]['question'], 
            mmlu_data[0]['choices']
        )
        
        print("✅ Multiple choice formatting works")
        print(f"   Sample prompt: {mc_prompt[:100]}...")
        
        # Test answer extraction
        from evaluation import AnswerExtractor
        extractor = AnswerExtractor()
        
        test_response = "الإجابة الصحيحة هي: أ"
        extracted = extractor.extract_multiple_choice_answer(test_response)
        print(f"✅ Answer extraction: '{test_response}' -> '{extracted}'")
        
except Exception as e:
    print(f"❌ Prompt formatting failed: {e}")

print("\n🎉 Phase 1.1 Basic Test Complete!")
print("\nNext: Run full evaluation with: python evaluation.py") 