"""
MedQA-USMLE Benchmark Evaluation
Evaluates Base, Fine-tuned, and Mistral on standard medical benchmark
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset
import json
import time
from datetime import datetime
from tqdm import tqdm
import re

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Configuration
BASE_MODEL_NAME = "Qwen/Qwen3-8B"
ADAPTER_DIR = "./qwen3_medical_reasoning_qlora"
MISTRAL_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
RESULTS_DIR = "./evaluation_results"
NUM_QUESTIONS = 100  # Start with 100, can increase to 200 or full set

os.makedirs(RESULTS_DIR, exist_ok=True)


def load_medqa_dataset(num_samples=100):
    """Load MedQA-USMLE dataset."""
    print(f"\n{'='*80}")
    print(f"Loading MedQA-USMLE Dataset")
    print(f"{'='*80}")
    
    # Try primary source first
    try:
        dataset = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
        print(f"✅ Loaded {len(dataset)} total questions from GBaker/MedQA-USMLE-4-options")
        
        # Debug: check first example
        if len(dataset) > 0:
            print("\n[DEBUG] First example keys:", list(dataset[0].keys()))
            print("[DEBUG] Sample question format:")
            for key, value in dataset[0].items():
                print(f"  {key}: {str(value)[:100]}...")
        
        # Take subset
        if num_samples < len(dataset):
            dataset = dataset.shuffle(seed=42).select(range(num_samples))
            print(f"📊 Using {num_samples} questions for evaluation")
        
        return dataset
    
    except Exception as e:
        print(f"⚠️  Primary dataset failed: {e}")
        print("\nTrying alternative: bigbio/med_qa...")
        
        try:
            dataset = load_dataset("bigbio/med_qa", "med_qa_en_4options_source", split="test")
            print(f"✅ Loaded alternative dataset: {len(dataset)} questions")
            
            # Debug first example
            if len(dataset) > 0:
                print("\n[DEBUG] First example keys:", list(dataset[0].keys()))
            
            if num_samples < len(dataset):
                dataset = dataset.shuffle(seed=42).select(range(num_samples))
                print(f"📊 Using {num_samples} questions")
            
            return dataset
            
        except Exception as e2:
            print(f"❌ All datasets failed!")
            print(f"Error: {e2}")
            return None


def format_medqa_question(example):
    """Format MedQA question as multiple choice."""
    # Get question text
    question = example.get('question', example.get('Question', example.get('sent1', '')))
    
    # Get options - handle multiple formats
    options = example.get('options', example.get('Options', {}))
    
    if isinstance(options, dict):
        # Format: {"A": "text", "B": "text", ...}
        options_text = "\n".join([f"{k}. {v}" for k, v in sorted(options.items())])
    elif isinstance(options, list):
        # Format: ["text1", "text2", ...]
        labels = ['A', 'B', 'C', 'D']
        options_text = "\n".join([f"{labels[i]}. {opt}" for i, opt in enumerate(options[:4])])
    else:
        # Try to extract from answer_choices
        answer_choices = example.get('answer_choices', [])
        if answer_choices:
            labels = ['A', 'B', 'C', 'D']
            options_text = "\n".join([f"{labels[i]}. {choice}" for i, choice in enumerate(answer_choices[:4])])
        else:
            options_text = str(options)
    
    prompt = f"""Answer this medical question by choosing the correct option (A, B, C, or D).

Question: {question}

Options:
{options_text}

Answer with ONLY the letter (A, B, C, or D):"""
    
    return prompt


def get_correct_answer(example):
    """Extract correct answer from example."""
    # PRIORITIZE answer_idx 
    if 'answer_idx' in example:
        idx = example['answer_idx']
        
        # Case 1: It's already a letter string like "A", "B", "C", "D"
        if isinstance(idx, str):
            idx_upper = idx.strip().upper()
            if idx_upper in ['A', 'B', 'C', 'D']:
                return idx_upper
            # Try to parse as integer string "0", "1", "2", "3"
            try:
                idx_int = int(idx)
                if 0 <= idx_int <= 3:
                    return ['A', 'B', 'C', 'D'][idx_int]
            except:
                pass
        
        # Case 2: It's an integer 0, 1, 2, 3
        elif isinstance(idx, int) and 0 <= idx <= 3:
            return ['A', 'B', 'C', 'D'][idx]
    
    # Fallback: Try to extract from answer field
    answer = example.get('answer', example.get('Answer', example.get('correct_answer', None)))
    
    if isinstance(answer, str):
        answer_upper = answer.upper().strip()
        # Extract first letter if it's like "A. something" or "A)" or "(A)"
        match = re.match(r'^[\(\[]?([A-D])[\)\]\.:]?', answer_upper)
        if match:
            return match.group(1)
        # If just a letter
        if answer_upper in ['A', 'B', 'C', 'D']:
            return answer_upper
    elif isinstance(answer, int) and 0 <= answer <= 3:
        return ['A', 'B', 'C', 'D'][answer]
    
    return None


def load_model(model_type):
    """Load a single model."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    if model_type == "base":
        print("\nLoading Base Qwen3-8B...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            device_map="auto",
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        model.eval()
        print("✅ Base model loaded")
        return model, tokenizer
    
    elif model_type == "finetuned":
        print("\nLoading Fine-tuned Qwen3-8B...")
        tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            device_map="auto",
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
        model.eval()
        print("✅ Fine-tuned model loaded")
        return model, tokenizer
    
    elif model_type == "mistral":
        print("\nLoading Mistral-7B...")
        tokenizer = AutoTokenizer.from_pretrained(MISTRAL_MODEL_NAME, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            MISTRAL_MODEL_NAME,
            device_map="auto",
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        model.eval()
        print("✅ Mistral model loaded")
        return model, tokenizer


def unload_model(model):
    """Free GPU memory by unloading model."""
    import gc
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("🗑️  Model unloaded, GPU memory freed")


def generate_answer(model, tokenizer, question, max_tokens=512):
    """Generate answer from model."""
    inputs = tokenizer([question], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    generated_ids = outputs[0][len(inputs.input_ids[0]):]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return response


def extract_answer_letter(response):
    """Extract A/B/C/D from model response."""
    # Look for patterns like "Answer: A", "The answer is B", "A.", etc.
    patterns = [
        r'[Aa]nswer[:\s]+([A-D])',
        r'[Tt]he correct (?:answer|option) is ([A-D])',
        r'^([A-D])[.\s]',
        r'\b([A-D])\b',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            return match.group(1).upper()
    
    # If no clear pattern, take first letter A-D that appears
    for char in response.upper():
        if char in ['A', 'B', 'C', 'D']:
            return char
    
    return None


def evaluate_model_on_medqa(model, tokenizer, dataset, model_name):
    """Evaluate a single model on MedQA."""
    print(f"\n{'='*80}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*80}")
    
    correct = 0
    total = 0
    skipped = 0
    results = []
    
    # Debug: Check first few examples
    print("\n[DEBUG] Checking first 3 examples:")
    for i in range(min(3, len(dataset))):
        ex = dataset[i]
        print(f"\nExample {i}:")
        print(f"  Keys: {list(ex.keys())}")
        print(f"  answer: {ex.get('answer', 'MISSING')}")
        print(f"  answer_idx: {ex.get('answer_idx', 'MISSING')}")
        print(f"  Extracted: {get_correct_answer(ex)}")
    print()
    
    for idx, example in enumerate(tqdm(dataset, desc=f"{model_name}")):
        # Format question
        question = format_medqa_question(example)
        correct_answer = get_correct_answer(example)
        
        if correct_answer is None:
            skipped += 1
            # Debug first few skipped
            if skipped <= 3:
                print(f"\n⚠️  Skipped question {idx}:")
                print(f"   answer: {example.get('answer', 'MISSING')}")
                print(f"   answer_idx: {example.get('answer_idx', 'MISSING')}")
                print(f"   type(answer_idx): {type(example.get('answer_idx'))}")
            continue  # Skip if we can't determine correct answer
        
        # Generate response
        response = generate_answer(model, tokenizer, question)
        predicted_answer = extract_answer_letter(response)
        
        # Check correctness
        is_correct = (predicted_answer == correct_answer)
        if is_correct:
            correct += 1
        total += 1
        
        results.append({
            "question_idx": idx,
            "question": example.get('question', example.get('Question', ''))[:200],
            "correct_answer": correct_answer,
            "predicted_answer": predicted_answer,
            "is_correct": is_correct,
            "response": response[:300]  # Truncate for storage
        })
    
    accuracy = (correct / total * 100) if total > 0 else 0
    
    print(f"\n{model_name} Results:")
    print(f"  Evaluated: {total}/{len(dataset)} questions")
    print(f"  Skipped: {skipped} questions (couldn't extract answer)")
    print(f"  Correct: {correct}/{total}")
    print(f"  Accuracy: {accuracy:.2f}%")
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "skipped": skipped,
        "results": results
    }


def run_medqa_evaluation():
    """Main evaluation function."""
    print("\n" + "="*80)
    print("MedQA-USMLE BENCHMARK EVALUATION")
    print("="*80)
    print(f"Testing {NUM_QUESTIONS} questions")
    print("="*80)
    
    # Load dataset
    dataset = load_medqa_dataset(NUM_QUESTIONS)
    if dataset is None:
        print("❌ Failed to load dataset. Exiting.")
        return
    
    # Evaluate each model ONE AT A TIME (to save GPU memory)
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "num_questions": NUM_QUESTIONS,
        "models": {}
    }
    
    model_names = {
        "base": "Base Qwen3-8B",
        "finetuned": "Fine-tuned Qwen3-8B",
        "mistral": "Mistral-7B-Instruct"
    }
    
    for model_key in ["base", "finetuned", "mistral"]:
        print(f"\n{'='*80}")
        print(f"[{list(model_names.keys()).index(model_key) + 1}/3] Evaluating {model_names[model_key]}")
        print(f"{'='*80}")
        
        # Load model
        model, tokenizer = load_model(model_key)
        
        # Evaluate
        results = evaluate_model_on_medqa(
            model, tokenizer, dataset, model_names[model_key]
        )
        all_results["models"][model_key] = results
        
        # Unload model to free GPU memory
        unload_model(model)
        del tokenizer
    
    # Print final comparison
    print("\n" + "="*80)
    print("FINAL RESULTS - MedQA-USMLE Benchmark")
    print("="*80)
    print(f"{'Model':<30} {'Accuracy':<15} {'Correct/Total'}")
    print("-"*80)
    
    for model_key in ["base", "finetuned", "mistral"]:
        if model_key in all_results["models"]:
            res = all_results["models"][model_key]
            name = model_names[model_key]
            if model_key == "finetuned":
                name += " ⭐"
            acc = f"{res['accuracy']:.2f}%"
            ratio = f"{res['correct']}/{res['total']}"
            print(f"{name:<30} {acc:<15} {ratio}")
    
    # Calculate improvements
    if "base" in all_results["models"] and "finetuned" in all_results["models"]:
        base_acc = all_results["models"]["base"]["accuracy"]
        ft_acc = all_results["models"]["finetuned"]["accuracy"]
        improvement = ft_acc - base_acc
        
        print("\n" + "="*80)
        print(f"Improvement (Fine-tuned vs Base): {improvement:+.2f} percentage points")
        print("="*80)
    
    # Save results
    output_file = os.path.join(
        RESULTS_DIR, 
        f"medqa_evaluation_{NUM_QUESTIONS}q_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_file}")
    
    # Create summary text file
    summary_file = os.path.join(
        RESULTS_DIR,
        f"medqa_summary_{NUM_QUESTIONS}q_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MedQA-USMLE BENCHMARK EVALUATION RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of Questions: {NUM_QUESTIONS}\n\n")
        
        f.write("Results:\n")
        f.write("-"*80 + "\n")
        for model_key in ["base", "finetuned", "mistral"]:
            if model_key in all_results["models"]:
                res = all_results["models"][model_key]
                f.write(f"{model_names[model_key]:<30} {res['accuracy']:.2f}% ({res['correct']}/{res['total']})\n")
        
        if "base" in all_results["models"] and "finetuned" in all_results["models"]:
            f.write("\n" + "="*80 + "\n")
            f.write(f"Improvement: {improvement:+.2f} percentage points\n")
            f.write("="*80 + "\n")
    
    print(f"✅ Summary saved to: {summary_file}")
    
    return all_results


if __name__ == "__main__":
    print("\n🏥 MedQA-USMLE Benchmark Evaluation")
    print("Testing medical reasoning on standard benchmark\n")
    
    results = run_medqa_evaluation()
    print("\n✅ Evaluation complete!")
