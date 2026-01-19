"""
Finish Llama 3.1 Shuffled Test - Continue from where we left off

The partial results have: 200-100K
Missing: 128K, 130K

This script:
1. Runs only the missing context lengths (128K, 130K) with shuffled needle
2. Combines with existing partial results
3. Generates comparison PNG (shuffled vs unshuffled)
"""

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
import matplotlib.pyplot as plt

# Only the MISSING context lengths
MISSING_CONTEXT_LENGTHS = [128000, 130000]

print("="*60)
print("FINISHING LLAMA 3.1 SHUFFLED SWEEP (128K and 130K only)")
print("="*60)

# Load existing partial results
partial_path = 'llama31_results/llama31_context_sweep_shuffled_partial.json'
with open(partial_path, 'r') as f:
    partial_results = json.load(f)

print(f"\nLoaded partial results from: {partial_path}")
print(f"Existing context lengths: {list(partial_results['results_by_length'].keys())}")
print(f"Missing: {MISSING_CONTEXT_LENGTHS}")

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

# Run only missing context lengths
new_results_by_length = {}
new_detailed_results = {}

for ctx_len in MISSING_CONTEXT_LENGTHS:
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
    new_results_by_length[str(ctx_len)] = {
        'accuracy': acc,
        'correct': correct,
        'total': len(samples)
    }
    new_detailed_results[str(ctx_len)] = details
    print(f"Accuracy at {ctx_len} tokens: {acc*100:.1f}% ({correct}/{len(samples)})")

# Combine with existing results
combined_results = partial_results.copy()
combined_results['results_by_length'].update(new_results_by_length)
combined_results['detailed_results'].update(new_detailed_results)
combined_results['max_context_tested'] = 130000
combined_results['status'] = 'COMPLETE'

# Save combined results
output_path = 'llama31_results/llama31_context_sweep_shuffled_0.json'
with open(output_path, 'w') as f:
    json.dump(combined_results, f, indent=2)
print(f"\nSaved combined results: {output_path}")

# Now generate the comparison PNG
print("\n" + "="*60)
print("Generating comparison PNG: Normal vs Shuffled")
print("="*60)

# Load normal (unshuffled) results
normal_path = 'llama31_results/llama31_context_sweep_0.json'
with open(normal_path, 'r') as f:
    normal_results = json.load(f)

# Extract data for plotting
def extract_plot_data(results):
    lengths = []
    accuracies = []
    for length_str, data in results['results_by_length'].items():
        lengths.append(int(length_str))
        accuracies.append(data['accuracy'] * 100)
    # Sort by length
    sorted_data = sorted(zip(lengths, accuracies))
    return zip(*sorted_data)

normal_lengths, normal_accs = extract_plot_data(normal_results)
shuffled_lengths, shuffled_accs = extract_plot_data(combined_results)

# Create the comparison plot
fig, ax = plt.subplots(figsize=(14, 7))

# Plot both lines
ax.plot(normal_lengths, normal_accs, 
        marker='o', color='#2ecc71', linewidth=2.5, markersize=10,
        label='Normal (unshuffled)', zorder=3)
ax.plot(shuffled_lengths, shuffled_accs, 
        marker='s', color='#e74c3c', linewidth=2.5, markersize=10,
        label='Shuffled', zorder=3)

# Fill between to show the gap
min_len = min(len(normal_lengths), len(shuffled_lengths))
ax.fill_between(list(normal_lengths)[:min_len], 
                list(normal_accs)[:min_len], 
                list(shuffled_accs)[:min_len], 
                alpha=0.2, color='gray', label='Accuracy gap')

ax.set_xlabel('Context Length (tokens)', fontsize=13)
ax.set_ylabel('Retrieval Accuracy (%)', fontsize=13)
ax.set_title('Llama 3.1 8B: Normal vs Shuffled Needle\n(Context up to 130K tokens)', fontsize=15, fontweight='bold')
ax.set_xscale('log')

# Set x-ticks to actual context lengths
all_lengths = sorted(set(normal_lengths) | set(shuffled_lengths))
ax.set_xticks(all_lengths)
ax.set_xticklabels([f'{x//1000}K' if x >= 1000 else str(x) for x in all_lengths], rotation=45, ha='right')

ax.set_ylim(0, 105)
ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Random baseline (50%)')
ax.legend(fontsize=11, loc='lower left')
ax.grid(True, alpha=0.3, zorder=1)

# Add annotations showing the drop at key points
key_points = [8000, 50000, 100000, 130000]
for kp in key_points:
    if kp in dict(zip(normal_lengths, normal_accs)) and kp in dict(zip(shuffled_lengths, shuffled_accs)):
        normal_acc = dict(zip(normal_lengths, normal_accs))[kp]
        shuffled_acc = dict(zip(shuffled_lengths, shuffled_accs))[kp]
        drop = normal_acc - shuffled_acc
        if drop > 5:
            ax.annotate(f'-{drop:.0f}%', 
                       xy=(kp, (normal_acc + shuffled_acc) / 2),
                       fontsize=9, color='red', fontweight='bold',
                       ha='center')

plt.tight_layout()
png_path = 'llama31_results/llama31_shuffled_comparison.png'
plt.savefig(png_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved comparison PNG: {png_path}")

print("\n" + "="*60)
print("LLAMA 3.1 SHUFFLED SWEEP COMPLETE!")
print("="*60)
print(f"\nFiles created:")
print(f"  - {output_path} (combined shuffled results)")
print(f"  - {png_path} (comparison visualization)")

