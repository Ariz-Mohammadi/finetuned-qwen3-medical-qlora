"""
Inference script for fine-tuned Qwen3-8B Medical Reasoning model
Corrected for your training setup
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import logging

# Force use of GPU 0 (same as training)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration - CORRECTED PATHS
BASE_MODEL_NAME = "Qwen/Qwen3-8B"  # Load from HuggingFace directly
ADAPTER_DIR = "./qwen3_medical_reasoning_qlora"  # my trained adapter


def load_fine_tuned_model():
    """Load the fine-tuned model with LoRA adapter."""
    
    logger.info("=" * 60)
    logger.info("Loading Fine-tuned Qwen3-8B Medical Model")
    logger.info("=" * 60)
    
    # Quantization config (same as training)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load tokenizer from adapter directory (it was saved there)
    logger.info(f"📥 Loading tokenizer from {ADAPTER_DIR}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, trust_remote_code=True)
        logger.info("✅ Tokenizer loaded from adapter directory")
    except:
        logger.info(f"⚠️  Tokenizer not in adapter dir, loading from base model...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model with quantization
    logger.info(f"📥 Loading base model {BASE_MODEL_NAME} with 4-bit quantization...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    logger.info("✅ Base model loaded")
    
    # Load LoRA adapter
    if os.path.exists(ADAPTER_DIR):
        logger.info(f"📥 Loading LoRA adapter from {ADAPTER_DIR}...")
        model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
        logger.info("✅ LoRA adapter loaded successfully")
    else:
        logger.error(f"❌ Adapter not found at {ADAPTER_DIR}")
        logger.error("   Please complete training first!")
        raise FileNotFoundError(f"Adapter directory not found: {ADAPTER_DIR}")
    
    # Set to evaluation mode
    model.eval()
    
    logger.info("=" * 60)
    logger.info("✅ Model ready for inference!")
    logger.info("=" * 60)
    
    return model, tokenizer


def generate_response(model, tokenizer, prompt, max_new_tokens=1024, temperature=0.7, top_p=0.9):
    """
    Generate response from the fine-tuned model.
    
    Args:
        model: The fine-tuned model
        tokenizer: The tokenizer
        prompt: User's medical question
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (higher = more creative)
        top_p: Nucleus sampling parameter
    """
    # Format prompt using Qwen3 chat template
    messages = [{"role": "user", "content": prompt}]
    
    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    # Tokenize
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # Generate
    logger.info("🤔 Generating response...")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode (skip the input prompt, only show generated text)
    generated_ids = output[0][len(inputs.input_ids[0]):]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return response


def main():
    """Main inference function."""
    
    print("\n" + "=" * 60)
    print("Qwen3-8B Medical Reasoning Inference")
    print("=" * 60)
    
    # Check if adapter exists
    if not os.path.exists(ADAPTER_DIR):
        print(f"\n❌ Error: Adapter not found at {ADAPTER_DIR}")
        print("Please complete training first!")
        return
    
    # Load model
    try:
        model, tokenizer = load_fine_tuned_model()
    except Exception as e:
        print(f"\n❌ Failed to load model: {e}")
        return
    
    # Example medical reasoning prompt
    print("\n" + "=" * 60)
    print("Example Medical Question:")
    print("=" * 60)
    
    example_prompt = (
        "A 58-year-old male presents with sudden chest pain radiating to the left arm, "
        "shortness of breath, and sweating. ECG shows ST-segment elevation in II, III, and aVF. "
        "What is the most likely diagnosis and explain the underlying mechanism?"
    )
    
    print(example_prompt)
    print("\n" + "=" * 60)
    print("MODEL RESPONSE:")
    print("=" * 60)
    
    response = generate_response(model, tokenizer, example_prompt)
    print(response)
    print("=" * 60)
    
    # Interactive mode
    print("\n" + "=" * 60)
    print("Interactive Mode")
    print("=" * 60)
    print("Enter your medical questions below.")
    print("Type 'quit', 'exit', or 'q' to exit.")
    print("=" * 60)
    
    while True:
        try:
            user_input = input("\n💬 Your question: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            print("\n" + "-" * 60)
            print("🤖 Response:")
            print("-" * 60)
            
            response = generate_response(model, tokenizer, user_input)
            print(response)
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            continue


if __name__ == "__main__":
    main()
