#!/bin/bash
# Qwen3-8B Fine-tuning Setup Script
# Run this script to install all required dependencies

set -e  # Exit on error

echo "=========================================="
echo "Qwen3-8B Fine-tuning Setup Script"
echo "=========================================="

# Check if conda is activated
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "❌ Error: No conda environment activated"
    echo "Please run: conda activate LLM"
    exit 1
fi

echo "✅ Conda environment: $CONDA_DEFAULT_ENV"
echo ""

# Step 1: Uninstall conflicting packages
echo "Step 1: Cleaning up old packages..."
pip uninstall -y transformers tokenizers trl peft 2>/dev/null || true
echo "✅ Cleanup complete"
echo ""

# Step 2: Install PyTorch (if not already installed)
echo "Step 2: Checking PyTorch installation..."
if python -c "import torch" 2>/dev/null; then
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
    echo "✅ PyTorch already installed: $TORCH_VERSION"
else
    echo "Installing PyTorch with CUDA 11.8 support..."
    pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    echo "✅ PyTorch installed"
fi
echo ""

# Step 3: Install transformers (MUST BE >=4.51.0 for Qwen3)
echo "Step 3: Installing transformers>=4.51.0 (REQUIRED for Qwen3)..."
pip install "transformers>=4.51.0"
echo "✅ Transformers installed"
echo ""

# Step 4: Install tokenizers
echo "Step 4: Installing tokenizers..."
pip install "tokenizers>=0.20.0"
echo "✅ Tokenizers installed"
echo ""

# Step 5: Install training dependencies
echo "Step 5: Installing training dependencies..."
pip install accelerate>=0.34.0
pip install peft>=0.13.0
pip install trl>=0.11.0
pip install datasets>=2.21.0
pip install bitsandbytes>=0.41.3
pip install scipy
pip install tensorboard
echo "✅ Training dependencies installed"
echo ""

# Step 6: Verify installation
echo "Step 6: Verifying installation..."
python << EOF
import sys

# Check imports
try:
    import torch
    import transformers
    import tokenizers
    import peft
    import trl
    import bitsandbytes
    import datasets
    print("✅ All packages imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Check versions
print("\n📦 Installed Versions:")
print(f"   PyTorch: {torch.__version__}")
print(f"   Transformers: {transformers.__version__}")
print(f"   Tokenizers: {tokenizers.__version__}")
print(f"   PEFT: {peft.__version__}")
print(f"   TRL: {trl.__version__}")
print(f"   Datasets: {datasets.__version__}")

# Check CUDA
print(f"\n🖥️  GPU Info:")
print(f"   CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("   ⚠️  CUDA not available!")

# Check transformers version for Qwen3
if transformers.__version__ < "4.51.0":
    print("\n❌ ERROR: Qwen3 requires transformers>=4.51.0")
    print(f"   Current version: {transformers.__version__}")
    print("   Please run: pip install --upgrade transformers")
    sys.exit(1)
else:
    print(f"\n✅ Transformers version {transformers.__version__} is compatible with Qwen3")
EOF

echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Run the training script:"
echo "   python train_qwen3_qlora_corrected.py"
echo ""
echo "2. Monitor training with tensorboard:"
echo "   tensorboard --logdir ./qwen3_medical_reasoning_qlora"
echo ""
echo "3. If you get a 401 error, it might be a temporary network issue."
echo "   Just run the script again - the model downloads are cached."
echo "=========================================="
