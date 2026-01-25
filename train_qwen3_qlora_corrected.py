"""
Fine-tune Qwen3-8B for Medical Reasoning using QLoRA
Corrected version with proper dependencies and chat templates
"""

import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import logging

# Force use of GPU 0 only
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MODEL_NAME = "Qwen/Qwen3-8B"
DATASET_NAME = "FreedomIntelligence/medical-o1-reasoning-SFT"
DATASET_CONFIG = "en"
OUTPUT_DIR = "./qwen3_medical_reasoning_qlora"
MAX_SEQ_LENGTH = 2048  # Qwen3 supports up to 32K natively

# Check GPU availability
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. This script requires a GPU.")

logger.info(f"CUDA available: {torch.cuda.is_available()}")
logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
logger.info(f"CUDA version: {torch.version.cuda}")
logger.info(f"PyTorch version: {torch.__version__}")

# Check transformers version
import transformers
logger.info(f"Transformers version: {transformers.__version__}")
if transformers.__version__ < "4.51.0":
    logger.warning("⚠️  Qwen3 requires transformers>=4.51.0. Please upgrade!")
    logger.warning("   Run: pip install --upgrade transformers")


def formatting_func(example):
    """
    Format dataset examples for Qwen3 training.
    Qwen3 uses a specific chat template with thinking mode support.
    """
    q = example.get("Question", "").strip()
    cot = example.get("Complex_CoT", "").strip()
    out = example.get("Response", "").strip()
    
    # Handle missing data
    if not q or not cot or not out:
        return ""
    
    # Qwen3 chat template format
    # Using thinking mode for the Complex_CoT reasoning
    text = (
        f"<|im_start|>user\n{q}<|im_end|>\n"
        f"<|im_start|>assistant\n"
        f"<think>\n{cot}\n</think>\n"
        f"{out}<|im_end|>"
    )
    
    return text


def load_and_prepare_dataset():
    """Load and prepare the medical reasoning dataset."""
    logger.info(f"Loading dataset: {DATASET_NAME}")
    
    try:
        dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train")
        logger.info(f"✅ Total samples loaded: {len(dataset)}")
    except Exception as e:
        logger.error(f"❌ Failed to load dataset: {e}")
        raise
    
    # Filter for quality (ensure Complex_CoT has sufficient content)
    def filter_quality(example):
        return (
            len(example.get("Complex_CoT", "")) > 200 and
            len(example.get("Question", "")) > 10 and
            len(example.get("Response", "")) > 10
        )
    
    dataset = dataset.filter(filter_quality)
    logger.info(f"✅ Samples after quality filter: {len(dataset)}")
    
    # Split into train/eval (95/5 split)
    dataset = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    
    logger.info(f"📊 Train samples: {len(train_dataset)}")
    logger.info(f"📊 Eval samples: {len(eval_dataset)}")
    
    return train_dataset, eval_dataset


def setup_model_and_tokenizer():
    """Setup quantized Qwen3-8B model and tokenizer for training."""
    
    logger.info("=" * 60)
    logger.info("Setting up Qwen3-8B with 4-bit quantization")
    logger.info("=" * 60)
    
    # Quantization config (4-bit NF4 with double quantization)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    # LoRA adapter config - EXACT replica of blog post configuration
    lora_config = LoraConfig(
        r=8,  # Rank (matching blog post)
        lora_alpha=16,  # Alpha scaling (matching blog post)
        target_modules=["q_proj", "v_proj"],  # Only these 2 modules (matching blog post)
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Load tokenizer
    logger.info(f"📥 Loading tokenizer from {MODEL_NAME}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, 
            trust_remote_code=True,
        )
        logger.info("✅ Tokenizer loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load tokenizer: {e}")
        logger.error("This might be a network issue. Try again or check your internet connection.")
        raise
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set pad_token to eos_token: {tokenizer.eos_token}")
    
    tokenizer.padding_side = "right"
    
    # Load model with quantization
    logger.info(f"📥 Loading Qwen3-8B with 4-bit quantization...")
    logger.info("⏳ This may take a few minutes on first run...")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            quantization_config=bnb_config,
        )
        logger.info("✅ Model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise
    
    # Prepare model for LoRA fine-tuning
    logger.info("🔧 Preparing model for k-bit training...")
    model = prepare_model_for_kbit_training(model)
    
    logger.info("🔧 Adding LoRA adapters...")
    model = get_peft_model(model, lora_config)
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    trainable_pct = 100 * trainable_params / all_params
    
    logger.info("=" * 60)
    logger.info("📊 Model Statistics:")
    logger.info(f"   Trainable params: {trainable_params:,} ({trainable_pct:.2f}%)")
    logger.info(f"   All params: {all_params:,}")
    logger.info(f"   LoRA rank: {lora_config.r}")
    logger.info(f"   LoRA alpha: {lora_config.lora_alpha}")
    logger.info("=" * 60)
    
    return model, tokenizer


def main():
    """Main training function."""
    logger.info("=" * 60)
    logger.info("🚀 Starting QLoRA Fine-tuning for Qwen3-8B")
    logger.info("   Dataset: Medical Reasoning with Chain-of-Thought")
    logger.info("=" * 60)
    
    # Step 1: Load and prepare dataset
    logger.info("\n📚 Step 1: Loading dataset...")
    train_dataset, eval_dataset = load_and_prepare_dataset()
    
    # Step 2: Setup model and tokenizer
    logger.info("\n🤖 Step 2: Setting up model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer()
    
    # Step 3: Training arguments
    logger.info("\n⚙️  Step 3: Configuring training arguments...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=32,  # Keep effective batch size at 32
        num_train_epochs=3,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        optim="paged_adamw_8bit",
        weight_decay=0.01,
        max_grad_norm=1.0,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        report_to="tensorboard",
        seed=42,
        remove_unused_columns=False,
        push_to_hub=False,
    )
    
    logger.info("✅ Training configuration:")
    logger.info(f"   Batch size (per device): {training_args.per_device_train_batch_size}")
    logger.info(f"   Gradient accumulation: {training_args.gradient_accumulation_steps}")
    logger.info(f"   Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    logger.info(f"   Learning rate: {training_args.learning_rate}")
    logger.info(f"   Epochs: {training_args.num_train_epochs}")
    logger.info(f"   Max sequence length: {MAX_SEQ_LENGTH}")
    
    # Step 4: Create trainer
    logger.info("\n🎓 Step 4: Creating trainer...")
    
    # SFTTrainer for TRL 0.27.0
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        formatting_func=formatting_func,
    )
    
    logger.info("✅ Trainer created successfully")
    
    # Step 5: Check for existing checkpoints
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        checkpoints = [
            os.path.join(training_args.output_dir, d)
            for d in os.listdir(training_args.output_dir)
            if d.startswith("checkpoint")
        ]
        if checkpoints:
            last_checkpoint = sorted(checkpoints, key=os.path.getmtime)[-1]
            logger.info(f"🔄 Found checkpoint: {last_checkpoint}")
            logger.info("   Training will resume from this checkpoint")
    
    # Step 6: Start training
    logger.info("\n" + "=" * 60)
    logger.info("🚀 STARTING TRAINING")
    logger.info("=" * 60)
    logger.info(f"📊 Monitor progress with: tensorboard --logdir {OUTPUT_DIR}")
    logger.info("=" * 60)
    
    try:
        trainer.train(resume_from_checkpoint=last_checkpoint)
        logger.info("\n✅ Training completed successfully!")
    except KeyboardInterrupt:
        logger.info("\n⚠️  Training interrupted by user")
        logger.info("💾 Saving current state...")
    except Exception as e:
        logger.error(f"\n❌ Training failed: {e}")
        raise
    
    # Step 7: Save final model
    logger.info("\n💾 Step 7: Saving final model...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    logger.info(f"✅ Model saved to: {OUTPUT_DIR}")
    
    # Step 8: Final evaluation
    logger.info("\n📊 Step 8: Final evaluation...")
    eval_results = trainer.evaluate()
    logger.info(f"✅ Final eval loss: {eval_results.get('eval_loss', 'N/A'):.4f}")
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("🎉 TRAINING COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"📁 Model saved at: {OUTPUT_DIR}")
    logger.info(f"📊 Final eval loss: {eval_results.get('eval_loss', 'N/A'):.4f}")
    logger.info("\n📝 Next steps:")
    logger.info("   1. Test your model with inference script")
    logger.info("   2. Merge LoRA weights if needed")
    logger.info("   3. Evaluate on your test set")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
