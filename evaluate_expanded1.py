"""
Expanded Custom Medical Evaluation
18 test cases: 3 cases per specialty across 6 specialties
Provides balanced, robust qualitative assessment
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import json
import time
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Configuration
BASE_MODEL_NAME = "Qwen/Qwen3-8B"
ADAPTER_DIR = "./qwen3_medical_reasoning_qlora"
MISTRAL_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
RESULTS_DIR = "./evaluation_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# 18 test cases: 3 per specialty across 6 specialties
TEST_CASES = [
    # ENDOCRINOLOGY (3 cases)
    {
        "id": 1,
        "specialty": "Endocrinology",
        "question": "A 45-year-old obese patient presents with polyuria, polydipsia, and fatigue for the past 3 months. Fasting blood glucose is 180 mg/dL. HbA1c is 8.5%. Explain the pathophysiology and first-line treatment.",
        "expected_diagnosis": "Type 2 Diabetes Mellitus",
        "key_concepts": {
            "insulin resistance": ["insulin resistance", "resistant to insulin", "cells don't respond to insulin"],
            "hyperglycemia": ["hyperglycemia", "high blood sugar", "elevated glucose"],
            "metformin": ["metformin"],
            "lifestyle": ["lifestyle", "diet", "exercise", "weight loss"]
        }
    },
    {
        "id": 2,
        "specialty": "Endocrinology",
        "question": "A 35-year-old woman presents with heat intolerance, weight loss despite increased appetite, palpitations, and tremor. TSH is 0.1 mIU/L, Free T4 is elevated. What is the diagnosis and mechanism?",
        "expected_diagnosis": "Hyperthyroidism",
        "key_concepts": {
            "hyperthyroidism": ["hyperthyroidism", "thyrotoxicosis", "overactive thyroid"],
            "TSH": ["tsh", "thyroid stimulating hormone"],
            "graves": ["graves", "autoimmune"],
            "treatment": ["antithyroid", "methimazole", "ptu", "propylthiouracil"]
        }
    },
    {
        "id": 3,
        "specialty": "Endocrinology",
        "question": "A 28-year-old woman presents with amenorrhea, galactorrhea, and headaches. Prolactin level is 250 ng/mL. What is the most likely diagnosis and management?",
        "expected_diagnosis": "Prolactinoma",
        "key_concepts": {
            "prolactinoma": ["prolactinoma", "pituitary", "adenoma"],
            "prolactin": ["prolactin", "hyperprolactinemia"],
            "amenorrhea": ["amenorrhea"],
            "treatment": ["dopamine agonist", "cabergoline", "bromocriptine"]
        }
    },
    
    # CARDIOLOGY (3 cases)
    {
        "id": 4,
        "specialty": "Cardiology",
        "question": "A 65-year-old woman presents with sudden severe chest pain, dyspnea, and syncope. Physical exam reveals a diastolic murmur at the left sternal border. BP is 90/60 mmHg. What is the most likely diagnosis?",
        "expected_diagnosis": "Aortic Regurgitation",
        "key_concepts": {
            "chest pain": ["chest pain"],
            "hypotension": ["hypotension", "low blood pressure"],
            "diastolic murmur": ["diastolic murmur", "diastolic"],
            "emergency": ["emergency", "acute"],
            "aortic": ["aortic", "aorta"]
        }
    },
    {
        "id": 5,
        "specialty": "Cardiology",
        "question": "A 58-year-old man with crushing substernal chest pain radiating to left arm, diaphoresis. ECG shows ST elevation in II, III, aVF. Troponin elevated. What is the diagnosis and immediate management?",
        "expected_diagnosis": "Inferior STEMI",
        "key_concepts": {
            "myocardial infarction": ["myocardial infarction", "mi", "heart attack", "stemi"],
            "ST elevation": ["st elevation", "st segment"],
            "treatment": ["aspirin", "clopidogrel", "pci", "catheterization", "reperfusion"],
            "inferior": ["inferior"]
        }
    },
    {
        "id": 6,
        "specialty": "Cardiology",
        "question": "A 72-year-old man with history of hypertension presents with exertional dyspnea, orthopnea, and bilateral ankle edema. JVP elevated, S3 gallop present. BNP is 1200 pg/mL. What is the diagnosis?",
        "expected_diagnosis": "Congestive Heart Failure",
        "key_concepts": {
            "heart failure": ["heart failure", "chf", "congestive"],
            "dyspnea": ["dyspnea", "shortness of breath"],
            "edema": ["edema", "swelling"],
            "bnp": ["bnp", "b-type natriuretic peptide"]
        }
    },
    
    # PULMONOLOGY (3 cases)
    {
        "id": 7,
        "specialty": "Pulmonology",
        "question": "A 55-year-old smoker with 30 pack-year history presents with chronic productive cough and dyspnea on exertion. Spirometry shows FEV1/FVC ratio of 0.65. What is the diagnosis?",
        "expected_diagnosis": "COPD",
        "key_concepts": {
            "COPD": ["copd", "chronic obstructive"],
            "smoking": ["smok", "tobacco"],
            "obstruction": ["obstruction", "obstructive"],
            "spirometry": ["spirometry", "fev"]
        }
    },
    {
        "id": 8,
        "specialty": "Pulmonology",
        "question": "A 24-year-old woman presents with sudden onset dyspnea, pleuritic chest pain, and hemoptysis. She recently returned from a long flight. D-dimer elevated. What is the diagnosis and treatment?",
        "expected_diagnosis": "Pulmonary Embolism",
        "key_concepts": {
            "pulmonary embolism": ["pulmonary embolism", "pe"],
            "dvt": ["dvt", "deep vein thrombosis", "thrombosis"],
            "anticoagulation": ["anticoagulation", "heparin", "warfarin", "doac"],
            "d-dimer": ["d-dimer", "d dimer"]
        }
    },
    {
        "id": 9,
        "specialty": "Pulmonology",
        "question": "A 30-year-old man with history of asthma presents with acute worsening of dyspnea, wheezing, and chest tightness. Peak flow 40% of predicted. What is the diagnosis and immediate management?",
        "expected_diagnosis": "Acute Asthma Exacerbation",
        "key_concepts": {
            "asthma": ["asthma", "bronchospasm"],
            "exacerbation": ["exacerbation", "attack", "acute"],
            "bronchodilator": ["bronchodilator", "albuterol", "salbutamol"],
            "steroids": ["steroid", "corticosteroid", "prednisone"]
        }
    },
    
    # GASTROENTEROLOGY (3 cases)
    {
        "id": 10,
        "specialty": "Gastroenterology",
        "question": "A 50-year-old man presents with epigastric pain, early satiety, and melena. Endoscopy shows a gastric ulcer. H. pylori test positive. What is the pathophysiology and treatment?",
        "expected_diagnosis": "Peptic Ulcer Disease",
        "key_concepts": {
            "peptic ulcer": ["peptic ulcer", "gastric ulcer"],
            "h. pylori": ["h. pylori", "helicobacter pylori", "h pylori"],
            "ppi": ["ppi", "proton pump inhibitor", "omeprazole"],
            "triple therapy": ["triple", "antibiotics", "clarithromycin", "amoxicillin"]
        }
    },
    {
        "id": 11,
        "specialty": "Gastroenterology",
        "question": "A 65-year-old man presents with painless jaundice, weight loss, and palpable gallbladder. CA 19-9 elevated. CT shows mass in pancreatic head. What is the diagnosis?",
        "expected_diagnosis": "Pancreatic Cancer",
        "key_concepts": {
            "pancreatic cancer": ["pancreatic cancer", "pancreatic carcinoma"],
            "jaundice": ["jaundice", "icterus"],
            "courvoisier": ["courvoisier", "palpable gallbladder"],
            "ca 19-9": ["ca 19-9", "ca19-9"]
        }
    },
    {
        "id": 12,
        "specialty": "Gastroenterology",
        "question": "A 40-year-old woman with history of gallstones presents with severe epigastric pain radiating to the back, nausea, and vomiting. Lipase is 800 U/L. CT shows pancreatic edema. What is the diagnosis and management?",
        "expected_diagnosis": "Acute Pancreatitis",
        "key_concepts": {
            "pancreatitis": ["pancreatitis"],
            "lipase": ["lipase", "amylase"],
            "gallstones": ["gallstones", "biliary"],
            "supportive care": ["supportive", "fluids", "bowel rest", "pain control"]
        }
    },
    
    # NEUROLOGY (3 cases)
    {
        "id": 13,
        "specialty": "Neurology",
        "question": "A 68-year-old man with atrial fibrillation presents with sudden onset right-sided weakness and aphasia. Symptoms started 2 hours ago. CT head shows no hemorrhage. What is the diagnosis and immediate treatment?",
        "expected_diagnosis": "Ischemic Stroke",
        "key_concepts": {
            "stroke": ["stroke", "cva", "cerebrovascular"],
            "ischemic": ["ischemic", "thrombotic"],
            "tpa": ["tpa", "alteplase", "thrombolysis"],
            "time": ["time", "window", "4.5 hours"]
        }
    },
    {
        "id": 14,
        "specialty": "Neurology",
        "question": "A 25-year-old woman presents with episodic headaches, usually unilateral, throbbing, with nausea and photophobia. Attacks last 4-72 hours. What is the diagnosis and preventive treatment?",
        "expected_diagnosis": "Migraine",
        "key_concepts": {
            "migraine": ["migraine", "headache"],
            "unilateral": ["unilateral", "one-sided"],
            "preventive": ["preventive", "prophylaxis", "beta blocker", "topiramate"],
            "acute": ["acute", "triptan", "nsaid"]
        }
    },
    {
        "id": 15,
        "specialty": "Neurology",
        "question": "A 55-year-old man presents with resting tremor, bradykinesia, and rigidity. Symptoms improve with levodopa. What is the diagnosis and underlying pathophysiology?",
        "expected_diagnosis": "Parkinson's Disease",
        "key_concepts": {
            "parkinson": ["parkinson"],
            "dopamine": ["dopamine", "dopaminergic"],
            "substantia nigra": ["substantia nigra", "basal ganglia"],
            "levodopa": ["levodopa", "l-dopa"]
        }
    },
    
    # INFECTIOUS DISEASE (3 cases)
    {
        "id": 16,
        "specialty": "Infectious Disease",
        "question": "A 30-year-old man presents with fever, night sweats, weight loss, and persistent cough for 6 weeks. CXR shows upper lobe cavitation. Sputum AFB positive. What is the diagnosis and treatment?",
        "expected_diagnosis": "Tuberculosis",
        "key_concepts": {
            "tuberculosis": ["tuberculosis", "tb"],
            "mycobacterium": ["mycobacterium"],
            "rifampin": ["rifampin", "rifampicin", "ripe", "isoniazid"],
            "cavitation": ["cavitation", "cavity"]
        }
    },
    {
        "id": 17,
        "specialty": "Infectious Disease",
        "question": "A 22-year-old woman presents with fever, severe headache, neck stiffness, and photophobia. LP shows increased WBCs (1000), protein 150 mg/dL, glucose 20 mg/dL. Gram stain shows gram-negative diplococci. What is the diagnosis?",
        "expected_diagnosis": "Bacterial Meningitis (Neisseria meningitidis)",
        "key_concepts": {
            "meningitis": ["meningitis"],
            "bacterial": ["bacterial"],
            "neisseria": ["neisseria", "meningococcal", "n. meningitidis"],
            "antibiotics": ["antibiotics", "ceftriaxone", "cefotaxime"]
        }
    },
    {
        "id": 18,
        "specialty": "Infectious Disease",
        "question": "A 35-year-old man with HIV (CD4 count 50) presents with fever, cough, and dyspnea. CXR shows bilateral interstitial infiltrates. BAL shows Pneumocystis jirovecii. What is the diagnosis and treatment?",
        "expected_diagnosis": "Pneumocystis Pneumonia (PCP)",
        "key_concepts": {
            "pneumocystis": ["pneumocystis", "pcp", "jirovecii"],
            "hiv": ["hiv", "aids", "immunocompromised"],
            "trimethoprim": ["trimethoprim", "sulfamethoxazole", "tmp-smx", "bactrim"],
            "cd4": ["cd4"]
        }
    },
]


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
    """Free GPU memory."""
    import gc
    del model
    gc.collect()
    torch.cuda.empty_cache()


def generate_response(model, tokenizer, question, max_tokens=800, use_chat_template=False):
    """Generate from model."""
    if use_chat_template:
        # Use chat template with thinking enabled (for fine-tuned model)
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
    else:
        # Simple prompt for base/mistral
        prompt = f"Question: {question}\n\nAnswer:"
        inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    
    start_time = time.time()
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
    gen_time = time.time() - start_time
    
    generated_ids = outputs[0][len(inputs.input_ids[0]):]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return response, gen_time


def check_concepts_flexible(response, key_concepts):
    """Check concepts with flexible matching."""
    if not response:
        return {}, 0.0
    
    response_lower = response.lower()
    found_concepts = {}
    
    for concept_name, variations in key_concepts.items():
        found = False
        for variation in variations:
            if variation.lower() in response_lower:
                found = True
                break
        found_concepts[concept_name] = found
    
    coverage = sum(found_concepts.values()) / len(found_concepts) if found_concepts else 0
    return found_concepts, coverage


def has_chain_of_thought(response):
    """Detect CoT reasoning."""
    if not response:
        return False, "No response"
    
    response_lower = response.lower()
    
    if "<think>" in response_lower:
        return True, "Explicit <think> tags"
    
    reasoning_phrases = ["let's think", "let me think", "step by step", "let's break"]
    if any(phrase in response_lower for phrase in reasoning_phrases):
        return True, "Reasoning phrases"
    
    analytical = ["therefore", "this suggests", "this indicates", "because"]
    count = sum(1 for word in analytical if word in response_lower)
    if count >= 2:
        return True, f"Analytical ({count} markers)"
    
    return False, "No CoT"


def evaluate_quality(response):
    """Evaluate response quality (0-4)."""
    if not response:
        return 0, "Empty"
    
    score = 0
    notes = []
    
    word_count = len(response.split())
    if word_count > 150:
        score += 1
        notes.append("Good length")
    
    if '\n' in response or '. ' in response:
        score += 1
        notes.append("Structured")
    
    medical_terms = ['diagnosis', 'treatment', 'pathophysiology', 'patient', 'symptoms']
    if sum(1 for term in medical_terms if term.lower() in response.lower()) >= 3:
        score += 1
        notes.append("Medical terminology")
    
    if 'because' in response.lower() or 'due to' in response.lower():
        score += 1
        notes.append("Explains causation")
    
    return score, "; ".join(notes) if notes else "Basic"


def run_expanded_evaluation():
    """Main evaluation function."""
    print("\n" + "="*80)
    print("EXPANDED CUSTOM MEDICAL EVALUATION")
    print("="*80)
    print(f"Test cases: {len(TEST_CASES)} (3 cases × 6 specialties)")
    print("="*80)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "num_cases": len(TEST_CASES),
        "test_cases": []
    }
    
    model_names = {
        "base": "Base Qwen3-8B",
        "finetuned": "Fine-tuned Qwen3-8B",
        "mistral": "Mistral-7B-Instruct"
    }
    
    # Evaluate each model on all cases
    for model_key in ["base", "finetuned", "mistral"]:
        print(f"\n{'='*80}")
        print(f"Evaluating: {model_names[model_key]}")
        print(f"{'='*80}")
        
        model, tokenizer = load_model(model_key)
        
        for i, case in enumerate(TEST_CASES, 1):
            print(f"\n[{i}/{len(TEST_CASES)}] {case['specialty']}: {case['expected_diagnosis']}")
            
            if model_key == "base":
                case_result = {
                    "case_id": case['id'],
                    "specialty": case['specialty'],
                    "question": case['question'],
                    "expected_diagnosis": case['expected_diagnosis'],
                    "responses": {}
                }
                results["test_cases"].append(case_result)
            
            response, gen_time = generate_response(
                model, tokenizer, case['question'], 
                use_chat_template=(model_key == "finetuned")
            )
            concepts, coverage = check_concepts_flexible(response, case['key_concepts'])
            has_cot, cot_reason = has_chain_of_thought(response)
            quality, quality_notes = evaluate_quality(response)
            
            results["test_cases"][i-1]["responses"][model_key] = {
                "response": response,
                "generation_time": gen_time,
                "concepts_found": concepts,
                "concept_coverage": coverage,
                "has_chain_of_thought": has_cot,
                "cot_reason": cot_reason,
                "quality_score": quality,
                "quality_notes": quality_notes
            }
            
            print(f"  Coverage: {coverage:.0%} | CoT: {has_cot} | Quality: {quality}/4 | Time: {gen_time:.1f}s")
        
        unload_model(model)
        del tokenizer
    
    # Calculate summary
    summary = {}
    for model_key in ["base", "finetuned", "mistral"]:
        coverages = [c["responses"][model_key]["concept_coverage"] for c in results["test_cases"]]
        cot_count = sum(1 for c in results["test_cases"] if c["responses"][model_key]["has_chain_of_thought"])
        times = [c["responses"][model_key]["generation_time"] for c in results["test_cases"]]
        qualities = [c["responses"][model_key]["quality_score"] for c in results["test_cases"]]
        
        summary[model_key] = {
            "avg_concept_coverage": sum(coverages) / len(coverages),
            "chain_of_thought_count": cot_count,
            "total_cases": len(TEST_CASES),
            "cot_percentage": (cot_count / len(TEST_CASES)) * 100,
            "avg_generation_time": sum(times) / len(times),
            "avg_quality_score": sum(qualities) / len(qualities)
        }
    
    results["summary"] = summary
    
    # Print results by specialty
    print("\n" + "="*80)
    print("RESULTS BY SPECIALTY")
    print("="*80)
    
    specialties = sorted(set(c['specialty'] for c in TEST_CASES))
    for specialty in specialties:
        print(f"\n{specialty}:")
        specialty_cases = [c for c in results["test_cases"] if c["specialty"] == specialty]
        
        for model_key in ["base", "finetuned", "mistral"]:
            coverages = [c["responses"][model_key]["concept_coverage"] for c in specialty_cases]
            avg_cov = sum(coverages) / len(coverages) if coverages else 0
            cot_count = sum(1 for c in specialty_cases if c["responses"][model_key]["has_chain_of_thought"])
            print(f"  {model_names[model_key]:<25} Coverage: {avg_cov:.0%}, CoT: {cot_count}/{len(specialty_cases)}")
    
    # Print overall summary
    print("\n" + "="*80)
    print("OVERALL SUMMARY (18 Cases Across 6 Specialties)")
    print("="*80)
    print(f"{'Model':<25} {'Coverage':<12} {'CoT %':<12} {'Quality':<12} {'Time':<12}")
    print("-"*80)
    
    for model_key in ["base", "finetuned", "mistral"]:
        s = summary[model_key]
        name = model_names[model_key]
        if model_key == "finetuned":
            name += " ⭐"
        
        cov = f"{s['avg_concept_coverage']:.0%}"
        cot = f"{s['cot_percentage']:.0f}%"
        qual = f"{s['avg_quality_score']:.1f}/4"
        time_str = f"{s['avg_generation_time']:.1f}s"
        
        print(f"{name:<25} {cov:<12} {cot:<12} {qual:<12} {time_str:<12}")
    
    # Calculate improvements
    if "base" in summary and "finetuned" in summary:
        base = summary["base"]
        ft = summary["finetuned"]
        
        print("\n" + "="*80)
        print("IMPROVEMENT ANALYSIS (Fine-tuned vs Base)")
        print("="*80)
        
        cov_imp = ((ft["avg_concept_coverage"] - base["avg_concept_coverage"]) / base["avg_concept_coverage"] * 100) if base["avg_concept_coverage"] > 0 else 0
        cot_imp = ft["cot_percentage"] - base["cot_percentage"]
        qual_imp = ((ft["avg_quality_score"] - base["avg_quality_score"]) / base["avg_quality_score"] * 100) if base["avg_quality_score"] > 0 else 0
        
        print(f"Concept Coverage:      {cov_imp:+.1f}%")
        print(f"CoT Usage:            {cot_imp:+.0f} percentage points")
        print(f"Response Quality:     {qual_imp:+.1f}%")
        print("="*80)
    
    # Save results
    output_file = os.path.join(RESULTS_DIR, f"expanded_evaluation_{len(TEST_CASES)}cases_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    print("\n🏥 Expanded Custom Medical Evaluation")
    print("Balanced evaluation: 3 cases per specialty\n")
    
    results = run_expanded_evaluation()
    print("\n✅ Evaluation complete!")
