"""
Quick verification script for Qwen3-8B setup
Run this BEFORE training to ensure everything is ready
"""

import sys

def check_imports():
    """Check if all required packages are installed."""
    print("=" * 60)
    print("Step 1: Checking package imports...")
    print("=" * 60)
    
    required_packages = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'tokenizers': 'Tokenizers',
        'peft': 'PEFT',
        'trl': 'TRL',
        'datasets': 'Datasets',
        'bitsandbytes': 'BitsAndBytes',
    }
    
    missing = []
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - NOT INSTALLED")
            missing.append(package)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("Run: pip install " + " ".join(missing))
        return False
    
    print("✅ All packages installed")
    return True


def check_versions():
    """Check if versions are compatible."""
    print("\n" + "=" * 60)
    print("Step 2: Checking versions...")
    print("=" * 60)
    
    import torch
    import transformers
    import tokenizers
    import peft
    import trl
    
    print(f"PyTorch: {torch.__version__}")
    print(f"Transformers: {transformers.__version__}")
    print(f"Tokenizers: {tokenizers.__version__}")
    print(f"PEFT: {peft.__version__}")
    print(f"TRL: {trl.__version__}")
    
    # Check critical version
    if transformers.__version__ < "4.51.0":
        print(f"\n❌ ERROR: Transformers {transformers.__version__} is too old for Qwen3")
        print("   Qwen3 requires transformers>=4.51.0")
        print("   Run: pip install --upgrade transformers")
        return False
    
    print(f"\n✅ Transformers {transformers.__version__} is compatible with Qwen3")
    return True


def check_cuda():
    """Check CUDA availability."""
    print("\n" + "=" * 60)
    print("Step 3: Checking CUDA/GPU...")
    print("=" * 60)
    
    import torch
    
    if not torch.cuda.is_available():
        print("❌ CUDA is not available!")
        print("   This script requires a GPU")
        return False
    
    print(f"✅ CUDA Available: {torch.cuda.is_available()}")
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"   GPU Memory: {total_memory:.1f} GB")
    
    if total_memory < 10:
        print("   ⚠️  Low GPU memory - training might fail")
    else:
        print("   ✅ Sufficient GPU memory for training")
    
    return True


def check_model_access():
    """Check if Qwen3-8B can be accessed."""
    print("\n" + "=" * 60)
    print("Step 4: Checking Qwen3-8B access...")
    print("=" * 60)
    
    try:
        from transformers import AutoTokenizer
        
        print("📥 Attempting to load Qwen3-8B tokenizer...")
        print("   (This will download ~500MB on first run)")
        
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-8B",
            trust_remote_code=True,
        )
        
        print("✅ Successfully loaded Qwen3-8B tokenizer!")
        print(f"   Vocab size: {len(tokenizer)}")
        print(f"   Model max length: {tokenizer.model_max_length}")
        
        # Test tokenization
        test_text = "What are the symptoms of diabetes?"
        tokens = tokenizer.encode(test_text)
        print(f"   Test tokenization: {len(tokens)} tokens")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to access Qwen3-8B: {e}")
        print("\nPossible solutions:")
        print("1. Check your internet connection")
        print("2. Upgrade transformers: pip install --upgrade transformers")
        print("3. Try: huggingface-cli login")
        print("4. Wait a moment and try again (might be temporary network issue)")
        return False


def check_dataset_access():
    """Check if dataset can be accessed."""
    print("\n" + "=" * 60)
    print("Step 5: Checking dataset access...")
    print("=" * 60)
    
    try:
        from datasets import load_dataset
        
        print("📥 Attempting to load medical reasoning dataset...")
        print("   (This will download ~58MB on first run)")
        
        dataset = load_dataset(
            "FreedomIntelligence/medical-o1-reasoning-SFT",
            "en",
            split="train[:10]"  # Just load 10 samples for testing
        )
        
        print(f"✅ Successfully loaded dataset!")
        print(f"   Sample count (test): {len(dataset)}")
        print(f"   Columns: {dataset.column_names}")
        
        # Show a sample
        sample = dataset[0]
        print(f"\n   Sample Question: {sample['Question'][:100]}...")
        print(f"   Sample CoT length: {len(sample.get('Complex_CoT', ''))} chars")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to access dataset: {e}")
        print("\nPossible solutions:")
        print("1. Check your internet connection")
        print("2. Try again in a moment")
        return False


def main():
    """Run all checks."""
    print("\n" + "=" * 60)
    print("Qwen3-8B Training Environment Verification")
    print("=" * 60 + "\n")
    
    checks = [
        ("Package Installation", check_imports),
        ("Version Compatibility", check_versions),
        ("GPU/CUDA", check_cuda),
        ("Model Access", check_model_access),
        ("Dataset Access", check_dataset_access),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n❌ Unexpected error in {name}: {e}")
            results[name] = False
    
    # Print summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(results.values())
    
    print("=" * 60)
    if all_passed:
        print("✅ All checks passed! Ready to train!")
        print("\nRun: python train_qwen3_qlora_corrected.py")
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("1. pip install --upgrade transformers tokenizers")
        print("2. Check internet connection")
        print("3. Run: ./setup_qwen3.sh")
    print("=" * 60)
    
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
