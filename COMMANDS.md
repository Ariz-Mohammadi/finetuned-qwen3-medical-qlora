# Quick Command Reference for Qwen3-8B Training

## Step-by-Step Commands

### 1. Activate Environment
```bash
conda activate LLM
```

### 2. Navigate to Your Directory
```bash
cd /cta/users/undergrad2/LLM/
```

### 3. Install/Upgrade Packages (CRITICAL!)
```bash
# Option A: Use the setup script (recommended)
chmod +x setup_qwen3.sh
./setup_qwen3.sh

# Option B: Manual installation
pip uninstall -y transformers tokenizers
pip install "transformers>=4.51.0" "tokenizers>=0.20.0"
pip install --upgrade accelerate peft trl datasets bitsandbytes
```

### 4. Verify Setup (Highly Recommended!)
```bash
python verify_qwen3_setup.py
```

If this shows "✅ All checks passed!", you're ready to train!

### 5. Start Training
```bash
python train_qwen3_qlora_corrected.py
```

### 6. Monitor Training (Optional - in a separate terminal)
```bash
conda activate LLM
cd /cta/users/undergrad2/LLM/
tensorboard --logdir ./qwen3_medical_reasoning_qlora
```
Then open: http://localhost:6006 in your browser

---

## If You Get Errors

### Error: "transformers<4.51.0"
```bash
pip install --upgrade transformers
python -c "import transformers; print(transformers.__version__)"
```

### Error: "401 Unauthorized" or "Repository Not Found"
This is usually a version issue, not authentication. Try:
```bash
# Upgrade transformers
pip install "transformers>=4.51.0" --force-reinstall

# If that doesn't work, login to HuggingFace (optional)
pip install huggingface_hub
huggingface-cli login
# Paste any token from: https://huggingface.co/settings/tokens
```

### Error: "CUDA out of memory"
Edit `train_qwen3_qlora_corrected.py` and change:
```python
per_device_train_batch_size=1,
gradient_accumulation_steps=64,  # Increase this if you reduce batch size
```

### Error: "ModuleNotFoundError"
```bash
# Install missing package
pip install <package_name>

# Or reinstall everything
./setup_qwen3.sh
```

---

## Expected Training Time

On RTX 5000 Ada (32GB):
- **Setup/Download**: 5-10 minutes (first time only)
- **Training (3 epochs)**: ~6-8 hours
- **Memory usage**: ~10-12 GB VRAM

---

## Quick Test After Training

```bash
python << EOF
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(base_model, "./qwen3_medical_reasoning_qlora")
tokenizer = AutoTokenizer.from_pretrained("./qwen3_medical_reasoning_qlora")

prompt = "What are the symptoms of Type 2 diabetes?"
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
EOF
```

---

## Files You Have Now

1. **train_qwen3_qlora_corrected.py** - Main training script
2. **setup_qwen3.sh** - Installation script
3. **verify_qwen3_setup.py** - Pre-training verification
4. **QWEN3_COMPLETE_GUIDE.md** - Detailed troubleshooting guide
5. **COMMANDS.md** - This file!

---

## Pro Tips

✅ **Always run verify_qwen3_setup.py first** - it will catch 90% of issues
✅ **Use tensorboard** - it helps you see if training is going well
✅ **Save checkpoints** - training can be resumed if interrupted
✅ **Test on small dataset first** - add `split="train[:100]"` in dataset loading
✅ **Monitor GPU usage** - run `nvidia-smi` in another terminal

---

## Getting Help

If you're stuck:
1. Check the full error message
2. Look in QWEN3_COMPLETE_GUIDE.md
3. Verify your transformers version: `python -c "import transformers; print(transformers.__version__)"`
4. Share the complete error message for better help
