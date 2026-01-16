import json
import os
import sys
sys.path.insert(0, '.')

from needle_haystack_sweep import (
    load_ground_truth, load_edgar_samples, MODELS,
    create_needle_in_haystack, QUESTION_TEMPLATE, 
    run_inference, check_answer, extract_needle_full_section,
    get_haystack_text
)
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm

# Full context lengths up to 130K
CONTEXT_LENGTHS = [200, 500, 1000, 2000, 4000, 8000, 10000, 15000, 20000, 30000, 50000, 75000, 100000, 128000, 130000]

print("="*60)
print("LLAMA 3.1 FULL SWEEP UP TO 130K TOKENS (SHUFFLED)")
print("="*60)

print("\nLoading model: Llama 3.1 8B")
model_name = MODELS["llama31"]
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()

print("Loading samples...")
ground_truth = load_ground_truth()
samples = load_edgar_samples(30, ground_truth)
haystack_text = get_haystack_text()

results_by_length = {}
detailed_results = {}

for ctx_len in CONTEXT_LENGTHS:
    print(f"\n{'='*60}")
    print(f"Testing context length: {ctx_len} tokens (SHUFFLED)")
    print(f"{'='*60}")
    
    correct = 0
    details = []
    
    for sample in tqdm(samples, desc=f"{ctx_len} tokens"):
        # SHUFFLED = True
        needle = extract_needle_full_section(sample['section_1'], sample['ground_truth_state'], shuffle=True)
        
        context = create_needle_in_haystack(
            needle=needle,
            tokenizer=tokenizer,
            target_tokens=ctx_len,
            needle_position=0.5,
            haystack_text=haystack_text
        )
        
        prompt = QUESTION_TEMPLATE.format(context=context)
        
        try:
            answer = run_inference(model, tokenizer, prompt)
        except Exception as e:
            print(f"Error at {ctx_len}: {e}")
            answer = f"ERROR: {str(e)}"
        
        is_correct = check_answer(answer, sample['ground_truth_state'])
        if is_correct:
            correct += 1
        
        details.append({
            'filename': sample['filename'],
            'ground_truth': sample['ground_truth_state'],
            'needle': needle[:500],
            'answer': answer[:200],
            'correct': is_correct
        })
    
    acc = correct / len(samples)
    results_by_length[ctx_len] = {
        'accuracy': acc,
        'correct': correct,
        'total': len(samples)
    }
    detailed_results[ctx_len] = details
    print(f"Accuracy at {ctx_len} tokens: {acc*100:.1f}% ({correct}/{len(samples)})")

# Save results
output = {
    'model': 'llama31',
    'model_name': model_name,
    'needle_position': 0.5,
    'shuffle_needle': True,
    'max_context_tested': 130000,
    'results_by_length': {str(k): v for k, v in results_by_length.items()},
    'detailed_results': {str(k): v for k, v in detailed_results.items()}
}

# Find next filename
i = 0
while os.path.exists(f'llama31_results/llama31_context_sweep_shuffled_{i}.json'):
    i += 1
output_path = f'llama31_results/llama31_context_sweep_shuffled_{i}.json'

with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nSaved: {output_path}")
print("\n" + "="*60)
print("LLAMA 3.1 130K SHUFFLED SWEEP COMPLETE")
print("="*60)
