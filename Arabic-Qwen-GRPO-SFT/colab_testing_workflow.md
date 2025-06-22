# 🧪 **Colab Testing Workflow for Arabic LLM Evaluation Framework**

## 🎯 **Quick Start - Copy These Cells to Your Colab**

### **Cell 1: Environment Setup**
```python
# =============================================================================
# SETUP CELL - Run this first in every Colab session
# =============================================================================

def setup_colab_environment():
    """Setup Colab environment for Arabic LLM evaluation"""
    print("🚀 Setting up Colab environment...")
    
    # Clone repo if not exists
    import os
    if not os.path.exists('/content/Arabic-Qwen-GRPO-SFT'):
        !git clone https://github.com/YOUR_USERNAME/Arabic-Qwen-GRPO-SFT.git /content/Arabic-Qwen-GRPO-SFT
    
    # Switch to evaluation branch
    !cd /content/Arabic-Qwen-GRPO-SFT && git checkout ALLaM_Style_Evaluation_Clean
    
    # Install requirements
    !cd /content/Arabic-Qwen-GRPO-SFT && pip install -r requirements.txt
    
    # Install additional evaluation dependencies
    !pip install sacrebleu rouge-score bert-score evaluate
    
    # Mount Google Drive
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✅ Google Drive mounted")
    except Exception as e:
        print(f"⚠️ Drive mount failed: {e}")
    
    print("✅ Environment ready for evaluation")

def check_gpu_status():
    """Check GPU memory and status"""
    import torch
    if torch.cuda.is_available():
        print(f"🔥 GPU: {torch.cuda.get_device_name()}")
        print(f"💾 Memory: {torch.cuda.memory_allocated()/1e9:.1f}GB / {torch.cuda.memory_reserved()/1e9:.1f}GB")
        print(f"🆓 Free: {(torch.cuda.memory_reserved() - torch.cuda.memory_allocated())/1e9:.1f}GB")
    else:
        print("❌ No GPU available")

def pull_latest_changes():
    """Pull latest changes from GitHub"""
    !cd /content/Arabic-Qwen-GRPO-SFT && git pull origin ALLaM_Style_Evaluation_Clean
    print("✅ Latest changes pulled")

# Run setup
setup_colab_environment()
check_gpu_status()
```

### **Cell 2: Phase 1.1 Test - Dataset Integration**
```python
# =============================================================================
# PHASE 1.1 TEST: Dataset Integration
# =============================================================================
# Expected Runtime: 2-3 minutes
# Memory Usage: <1 GB GPU
# Success Criteria: All datasets load, prompt formatting works

def test_phase_1_1():
    """Test Phase 1.1: Dataset Integration"""
    print("🧪 Testing Phase 1.1: Dataset Integration")
    print("=" * 50)
    
    try:
        # Pull latest changes
        pull_latest_changes()
        
        # Import evaluation module
        import sys
        sys.path.append('/content/Arabic-Qwen-GRPO-SFT/src')
        from evaluation import load_arabic_mmlu, load_acva, load_arabic_reasoning
        
        # Test dataset loading
        print("📊 Testing Arabic MMLU loading...")
        mmlu_data = load_arabic_mmlu(tier="quick")  # 10 samples
        print(f"✅ Arabic MMLU loaded: {len(mmlu_data)} samples")
        
        print("📊 Testing ACVA loading...")
        acva_data = load_acva(tier="quick")  # 10 samples
        print(f"✅ ACVA loaded: {len(acva_data)} samples")
        
        print("📊 Testing Arabic Reasoning loading...")
        reasoning_data = load_arabic_reasoning(tier="quick")  # 10 samples
        print(f"✅ Arabic Reasoning loaded: {len(reasoning_data)} samples")
        
        # Test prompt formatting
        from evaluation import format_prompt_mc, format_prompt_arabic
        sample_question = mmlu_data[0]
        formatted_prompt = format_prompt_mc(sample_question)
        print(f"✅ Prompt formatting works")
        print(f"Sample formatted prompt: {formatted_prompt[:100]}...")
        
        print("✅ Phase 1.1 Completed Successfully")
        return True
        
    except Exception as e:
        print(f"❌ Phase 1.1 Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# Run the test
test_phase_1_1()
```

### **Cell 3: Phase 1.2 Test - Basic Model Handler** 
```python
# =============================================================================
# PHASE 1.2 TEST: Basic Model Handler
# =============================================================================
# Expected Runtime: 5-8 minutes
# Memory Usage: 2-3 GB GPU
# Success Criteria: Baseline model loads, generates responses, memory cleanup works

def test_phase_1_2():
    """Test Phase 1.2: Basic Model Handler"""
    print("🧪 Testing Phase 1.2: Basic Model Handler")
    print("=" * 50)
    
    try:
        # Pull latest changes
        pull_latest_changes()
        
        # Import evaluation module
        import sys
        sys.path.append('/content/Arabic-Qwen-GRPO-SFT/src')
        from evaluation import ModelEvaluator
        import torch
        
        # Test baseline model loading
        print("🤖 Testing baseline model loading...")
        evaluator = ModelEvaluator()
        
        baseline_path = "unsloth/Qwen2.5-0.5B-Instruct"
        model, tokenizer = evaluator.load_model(baseline_path, "baseline")
        print(f"✅ Baseline model loaded successfully")
        
        # Test generation
        print("💬 Testing response generation...")
        test_prompt = "ما هو عاصمة مصر؟"
        response = evaluator.generate_answer(model, tokenizer, test_prompt)
        print(f"✅ Generation works. Response: {response[:100]}...")
        
        # Test memory cleanup
        print("🧹 Testing memory cleanup...")
        gpu_memory_before = torch.cuda.memory_allocated()
        evaluator.cleanup_memory()
        gpu_memory_after = torch.cuda.memory_allocated()
        print(f"✅ Memory cleanup: {gpu_memory_before/1e9:.1f}GB → {gpu_memory_after/1e9:.1f}GB")
        
        print("✅ Phase 1.2 Completed Successfully")
        return True
        
    except Exception as e:
        print(f"❌ Phase 1.2 Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# Run the test
test_phase_1_2()
```

### **Cell 4: Quick Evaluation Test (Once Phase 1.2 Complete)**
```python
# =============================================================================
# QUICK EVALUATION TEST - Test complete pipeline
# =============================================================================

def run_quick_evaluation():
    """Run a quick end-to-end evaluation test"""
    print("⚡ Running Quick Evaluation Test")
    print("=" * 40)
    
    try:
        import sys
        sys.path.append('/content/Arabic-Qwen-GRPO-SFT/src')
        from evaluation import validate_phase_1_1
        
        # Run full Phase 1.1 validation
        results = validate_phase_1_1()
        
        print(f"Phase 1.1 Status: {results['status']}")
        print("\nDetailed Results:")
        for test_name, test_result in results["tests"].items():
            print(f"  {test_name}: {test_result.get('status', 'unknown')}")
        
        return results['status'] == 'success'
        
    except Exception as e:
        print(f"❌ Quick evaluation failed: {e}")
        return False

# Run quick evaluation
success = run_quick_evaluation()
print(f"\n{'✅ READY FOR NEXT PHASE' if success else '❌ NEEDS DEBUGGING'}")
```

---

## 📋 **Testing Protocol Rules**

### **Before Each Phase**:
1. ✅ Run `pull_latest_changes()` in Colab
2. ✅ Check `check_gpu_status()` for available memory
3. ✅ Clear any existing models from memory

### **During Phase Testing**:
1. ✅ Run the phase-specific test cell
2. ✅ Monitor console output for errors
3. ✅ Verify all success criteria are met
4. ✅ Check GPU memory doesn't overflow

### **After Each Phase**:
1. ✅ Save any generated outputs/results
2. ✅ Commit working code to local repository  
3. ✅ Push changes to GitHub branch
4. ✅ Update phase status in documentation

### **Error Handling**:
- ❌ If phase test fails, debug locally before proceeding
- ⚠️ If memory issues occur, restart Colab runtime
- 🔄 Re-run `setup_colab_environment()` after restart

---

## 🎯 **Phase-by-Phase Testing Schedule**

| Phase | Test Cell | Runtime | Memory | Success Criteria |
|-------|-----------|---------|--------|------------------|
| 1.1 | Dataset Integration | 2-3 min | <1 GB | Datasets load, prompts format |
| 1.2 | Model Handler | 5-8 min | 2-3 GB | Model loads, generates, cleanup |
| 2.1 | Basic Metrics | 3-5 min | 1-2 GB | Accuracy, BLEU/ROUGE, CSV export |
| 2.2 | Advanced Metrics | 8-12 min | 3-4 GB | LLM-Judge, BertScore, NLI |
| 3.1 | Multi-Model | 15-20 min | 2-3 GB | Sequential loading works |
| 3.2 | Progression Analysis | 10-15 min | 2-3 GB | Comparisons, visualizations |

---

## 🚨 **Common Issues & Solutions**

### **Issue: "Dependencies not available"**
```python
# Solution: Install missing packages
!pip install transformers datasets torch sacrebleu rouge-score bert-score
```

### **Issue: "CUDA out of memory"**
```python
# Solution: Clear memory and restart
import torch
torch.cuda.empty_cache()
# Then restart runtime if needed
```

### **Issue: "Dataset is gated"**
```python
# Solution: Use Hugging Face token
from huggingface_hub import login
login(token="your_token_here")
```

### **Issue: "Git pull fails"**
```python
# Solution: Force pull
!cd /content/Arabic-Qwen-GRPO-SFT && git reset --hard HEAD && git pull origin ALLaM_Style_Evaluation_Clean
```

---

This workflow ensures **continuous validation** and **safe progression** through each phase of the evaluation framework development. Copy the appropriate cells to your Colab notebook and follow the testing protocol for reliable implementation. 