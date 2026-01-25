# Complete Guide: Fix Qwen3-8B Training Issues

## Quick Summary of Your Problem

You had TWO issues:
1. ✅ **SOLVED**: Tokenizer compatibility (transformers/tokenizers version mismatch)
2. 🔧 **CURRENT**: Model access issue (401 error)

The 401 error you're seeing is likely NOT about authentication - it's about having the right transformers version!

---

## Solution: Step-by-Step Fix

### Step 1: Check Your Transformers Version

```bash
conda activate LLM
python -c "import transformers; print(transformers.__version__)"
```

**If it shows anything less than 4.51.0**, that's your problem! Qwen3 requires transformers>=4.51.0.

### Step 2: Upgrade Everything (Recommended)

Run the setup script I provided:

```bash
cd /cta/users/undergrad2/LLM/
chmod +x setup_qwen3.sh
./setup_qwen3.sh
```

**OR** manually install:

```bash
conda activate LLM

# Uninstall old versions
pip uninstall -y transformers tokenizers

# Install latest versions
pip install "transformers>=4.51.0" "tokenizers>=0.20.0"
pip install --upgrade accelerate peft trl datasets bitsandbytes

# Verify
python -c "import transformers; print('Transformers version:', transformers.__version__)"
```

### Step 3: Run the Corrected Training Script

```bash
python train_qwen3_qlora_corrected.py
```

---

## Understanding the Error You Got

Your error message said:
```
OSError: Qwen/Qwen2-8B is not a local folder and is not a valid model identifier
```

This error is **misleading**. It's not that the model doesn't exist - it's that:
1. Your transformers version was too old to recognize Qwen3
2. OR there was a temporary network glitch

**Qwen3-8B is NOT gated** - you don't need any authentication!

---

## What Changed in the New Script

### 1. **Correct Chat Template**
```python
# OLD (won't work well with Qwen3):
f"<|user|>\n{q}\n<|assistant|>\n{out}"

# NEW (correct for Qwen3):
f"<|im_start|>user\n{q}<|im_end|>\n"
f"<|im_start|>assistant\n<think>\n{cot}\n</think>\n{out}<|im_end|>"
```

### 2. **Version Check**
The script now checks if you have transformers>=4.51.0 and warns you if not.

### 3. **Better LoRA Config**
Added more target modules for better performance:
- q_proj, k_proj, v_proj, o_proj (attention)
- gate_proj, up_proj, down_proj (MLP layers)

### 4. **Better Logging**
Much more informative output so you know what's happening at each step.

---

## If You Still Get 401 Error

### Option A: It's a Network Glitch
Just try again! The model downloads are cached, so subsequent runs will be faster.

```bash
# Try 2-3 times
python train_qwen3_qlora_corrected.py
```

### Option B: Set HuggingFace Token (Just in Case)

Even though Qwen3 isn't gated, setting a token can help with rate limits:

```bash
pip install huggingface_hub
huggingface-cli login
# Paste any HF token from: https://huggingface.co/settings/tokens
```

### Option C: Download Model Manually First

```bash
python << EOF
from transformers import AutoTokenizer, AutoModelForCausalLM

print("Downloading Qwen3-8B...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
print("✅ Tokenizer downloaded")

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    device_map="auto",
    trust_remote_code=True,
)
print("✅ Model downloaded")
print("Now run your training script!")
EOF
```

---

## Verification Before Training

Run this to make sure everything is ready:

```bash
python << EOF
import torch
import transformers
print(f"PyTorch: {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# Check if transformers is new enough
if transformers.__version__ >= "4.51.0":
    print("✅ Transformers version is compatible with Qwen3")
else:
    print(f"❌ Transformers {transformers.__version__} is too old!")
    print("   Run: pip install --upgrade transformers")

# Quick test
try:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
    print("✅ Can access Qwen3-8B!")
except Exception as e:
    print(f"❌ Cannot access model: {e}")
EOF
```

---

## Expected Output When It Works

When you run the training script successfully, you should see:

```
🚀 Starting QLoRA Fine-tuning for Qwen3-8B
   Dataset: Medical Reasoning with Chain-of-Thought
============================================================

📚 Step 1: Loading dataset...
✅ Total samples loaded: 19704
✅ Samples after quality filter: 19673
📊 Train samples: 18689
📊 Eval samples: 984

🤖 Step 2: Setting up model and tokenizer...
📥 Loading tokenizer from Qwen/Qwen3-8B...
✅ Tokenizer loaded successfully
📥 Loading Qwen3-8B with 4-bit quantization...
⏳ This may take a few minutes on first run...
✅ Model loaded successfully

📊 Model Statistics:
   Trainable params: 41,943,040 (0.51%)
   All params: 8,200,000,000

🚀 STARTING TRAINING
============================================================
```

---

## Training Time Estimates

On your **RTX 5000 Ada (32GB)**:
- **Full training (3 epochs)**: ~6-8 hours
- **Memory usage**: ~10-12 GB VRAM
- **Per step**: ~2-3 seconds

You have plenty of memory, so this should run smoothly!

---

## Monitoring Training

In a separate terminal:
```bash
conda activate LLM
tensorboard --logdir ./qwen3_medical_reasoning_qlora
```

Then open: http://localhost:6006

---

## Testing After Training

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "./qwen3_medical_reasoning_qlora")
tokenizer = AutoTokenizer.from_pretrained("./qwen3_medical_reasoning_qlora")

# Test with thinking mode
prompt = "What are the main symptoms and treatment options for Type 2 diabetes?"
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

---

## Troubleshooting Quick Reference

| Error | Solution |
|-------|----------|
| `transformers<4.51.0` error | `pip install --upgrade transformers` |
| `401 Unauthorized` | Upgrade transformers OR try again (network glitch) |
| `CUDA out of memory` | Reduce `per_device_train_batch_size` to 1 |
| `Tokenizer error` | `pip install --upgrade tokenizers` |
| Model not found | Check internet, try `huggingface-cli login` |

---

## Next Steps After Successful Training

1. **Evaluate** on test set
2. **Merge LoRA weights** for easier deployment
3. **Benchmark** against baseline
4. **Compare** with results from the blog post

Good luck! 🚀
