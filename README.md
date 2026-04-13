# Fine-Tuning Qwen3-8B for Medical Reasoning with QLoRA

Chain-of-Thought learning for transparent clinical decision support.

## 📋 Project Overview

This project fine-tunes the Qwen3-8B language model on medical reasoning tasks using Parameter-Efficient Fine-Tuning (PEFT) with QLoRA (Quantized Low-Rank Adaptation). The goal is to teach the model to provide transparent, step-by-step Chain-of-Thought (CoT) reasoning for medical diagnoses.

## 🎯 Key Results

- **100% Chain-of-Thought Usage** (vs 11% baseline, 0% Mistral)
- **+18.5% Concept Coverage** improvement (75% → 89%)
- **68% MedQA-USMLE Accuracy** maintained (no knowledge degradation)
- **+13pp better** than Mistral-7B baseline
- **14% faster inference** despite adding reasoning

## 🏗️ Architecture

- **Base Model:** Qwen3-8B (8 billion parameters)
- **Quantization:** 4-bit NF4 (reduces memory to 12GB VRAM)
- **Fine-tuning:** LoRA with r=8, α=16
- **Trainable Parameters:** 3.8M (0.08% of total model)
- **Training Time:** ~10 hours on NVIDIA RTX 5000 Ada

## 📊 Dataset

- **Training:** [medical-o1-reasoning-SFT](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT)
  - 19,673 medical cases with Chain-of-Thought reasoning
  - 95% train / 5% validation split

- **Evaluation:**
  - Custom: 18 cases across 6 medical specialties
  - Benchmark: 100 questions from MedQA-USMLE

## 🚀 Quick Start

### Prerequisites
```bash
# System requirements
- NVIDIA GPU with 12GB+ VRAM
- CUDA 11.8+
- Python 3.10+

# Install dependencies
pip install -r requirements.txt
```

### Training
```bash
# Run training
python train_qwen_qlora.py

# Training will:
# 1. Download and quantize Qwen3-8B (first run only)
# 2. Load and prepare dataset
# 3. Fine-tune with QLoRA for 3 epochs
# 4. Save adapter to ./qwen3_medical_reasoning_qlora/
```

### Inference
```bash
# Run inference with fine-tuned model
python inference_qwen3.py

# For interactive mode, just run and follow prompts
```

### Using Specific GPU
```bash
# If GPU 0 is busy, use GPU 1
CUDA_VISIBLE_DEVICES=1 python train_qwen3_qlora.py
CUDA_VISIBLE_DEVICES=1 python inference_qwen3.py
```

## 📁 Project Structure
```
.
├── train_qwen3_qlora.py              # Training script
├── inference_qwen3.py                      # Inference script
├── evaluate_expanded_18cases.py      # Custom evaluation
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
├── qwen3_8b_quantized/              # Quantized base model (NOT in repo - too large)
└── qwen3_medical_reasoning_qlora/   # Fine-tuned adapter (may be included if small)
```

## 📈 Evaluation Results

### Custom Medical Cases (18 cases, 6 specialties)

| Model | Coverage | CoT Usage | Quality | Time |
|-------|----------|-----------|---------|------|
| Base Qwen3-8B | 75% | 11% | 3.1/4 | 60.9s |
| **Fine-tuned Qwen3** | **89%** | **100%** | **3.2/4** | **52.3s** |
| Mistral-7B | 92% | 0% | 2.5/4 | 24.5s |

### MedQA-USMLE Benchmark (100 questions)

| Model | Accuracy |
|-------|----------|
| Base Qwen3-8B | 68% |
| **Fine-tuned Qwen3** | **68%** |
| Mistral-7B | 55% |

**Key Finding:** Fine-tuning added Chain-of-Thought reasoning without degrading medical knowledge.

## 🔬 Technical Details

### Training Configuration
```python
# Hyperparameters
batch_size = 2
gradient_accumulation_steps = 16  # Effective batch size: 32
learning_rate = 2e-4
lr_scheduler = "cosine"
epochs = 3
optimizer = "paged_adamw_8bit"

# LoRA Configuration
lora_r = 8
lora_alpha = 16
lora_dropout = 0.05
target_modules = ["q_proj", "v_proj"]
```

### Model Format

The fine-tuned model uses structured output with `<think>` tags:
```
<|user|>
A 58-year-old male presents with chest pain...

<|think|>
The patient presents with chest pain radiating to the left arm, 
suggesting cardiac origin. ST elevation in leads II, III, and aVF 
indicates inferior wall involvement...
</think>

<|assistant|>
The diagnosis is inferior STEMI. Immediate treatment includes...
```

## 📝 Citation
```bibtex
@misc{medical-qwen-qlora-2026,
  author = {[Your Name]},
  title = {Fine-Tuning Qwen3-8B for Medical Reasoning with QLoRA},
  year = {2026},
  url = {https://github.com/[your-username]/medical-qwen-qlora}
}
```

## ⚠️ Disclaimer

This is a **research prototype** and should NOT be used for actual clinical decision-making:

- ❌ Not validated by medical professionals
- ❌ Not FDA approved
- ❌ Not intended for patient care
- ✅ For research and educational purposes only

## 🔮 Future Work

- Multi-modal support (medical images, lab results)
- Specialty-specific adapters
- Uncertainty quantification
- Clinical validation with physicians
- REST API deployment
- EHR system integration

## 📄 License

[Choose appropriate license - e.g., MIT, Apache 2.0]

## 🙏 Acknowledgments

- **Qwen Team** for the base mode

- **FreedomIntelligence** for the medical reasoning dataset
- **Hugging Face** for transformers library
- **[Your Institution]** for computational resources
