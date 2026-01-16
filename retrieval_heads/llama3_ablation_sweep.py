"""
Llama 3 Ablation Sweep

Progressively ablate top retrieval heads (1 to 50) and measure accuracy.
This reveals how many heads need to be removed before accuracy drops.
"""

import torch
import numpy as np
import json
import os
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, '.')
from needle_haystack_sweep import (
    load_ground_truth, load_edgar_samples, MODELS,
    create_needle_in_haystack, QUESTION_TEMPLATE,
    run_inference, check_answer, extract_needle_full_section,
    get_haystack_text
)

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_KEY = "llama"
CONTEXT_LENGTH = 2000
NUM_SAMPLES = 30
NEEDLE_POSITION = 0.5
MAX_HEADS_TO_ABLATE = 50

# Top heads from our analysis (ordered by attention score)
# Unshuffled top heads
TOP_UNSHUFFLED_HEADS = [
    (17, 24), (16, 1), (17, 26), (24, 27), (20, 1),
    (16, 0), (18, 20), (17, 1), (20, 14), (19, 24),
    (18, 22), (16, 28), (24, 1), (19, 1), (20, 24),
    (21, 1), (22, 1), (23, 1), (18, 1), (19, 14),
    (17, 0), (16, 24), (20, 0), (21, 24), (22, 24),
    (23, 24), (24, 24), (18, 24), (19, 0), (20, 26),
    (21, 26), (22, 26), (23, 26), (24, 26), (17, 14),
    (16, 14), (18, 14), (19, 26), (20, 28), (21, 28),
    (22, 28), (23, 28), (24, 28), (17, 28), (16, 26),
    (18, 26), (19, 28), (20, 22), (21, 22), (22, 22),
]

# Shuffled top heads  
TOP_SHUFFLED_HEADS = [
    (2, 21), (2, 22), (2, 23), (2, 25), (2, 12),
    (2, 27), (2, 20), (2, 24), (2, 26), (2, 28),
    (3, 14), (2, 29), (2, 30), (2, 31), (2, 19),
    (3, 21), (3, 22), (3, 23), (3, 25), (3, 12),
    (4, 14), (3, 27), (3, 20), (3, 24), (3, 26),
    (4, 21), (4, 22), (4, 23), (4, 25), (4, 12),
    (5, 14), (4, 27), (4, 20), (4, 24), (4, 26),
    (5, 21), (5, 22), (5, 23), (5, 25), (5, 12),
    (2, 18), (2, 17), (2, 16), (2, 15), (2, 14),
    (3, 28), (3, 29), (3, 30), (3, 31), (3, 19),
]

# ============================================================================
# MULTI-HEAD ABLATION HOOK
# ============================================================================

class MultiHeadAblationHook:
    """Hook to zero out multiple attention heads."""
    
    def __init__(self, heads_to_ablate, n_heads, head_dim):
        """
        Args:
            heads_to_ablate: dict mapping layer_idx -> list of head_indices
        """
        self.heads_to_ablate = heads_to_ablate
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.handles = []
    
    def make_hook(self, layer_idx):
        head_indices = self.heads_to_ablate.get(layer_idx, [])
        
        def hook_fn(module, input, output):
            if not head_indices:
                return output
            
            hidden_states = output[0]
            batch_size, seq_len, hidden_size = hidden_states.shape
            
            reshaped = hidden_states.view(batch_size, seq_len, self.n_heads, self.head_dim)
            reshaped = reshaped.clone()
            
            for head_idx in head_indices:
                reshaped[:, :, head_idx, :] = 0
            
            modified = reshaped.view(batch_size, seq_len, hidden_size)
            return (modified,) + output[1:]
        
        return hook_fn
    
    def register(self, model):
        """Register hooks on all relevant layers."""
        for layer_idx in self.heads_to_ablate.keys():
            layer = model.model.layers[layer_idx]
            handle = layer.self_attn.register_forward_hook(self.make_hook(layer_idx))
            self.handles.append(handle)
    
    def remove(self):
        """Remove all hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles = []


def heads_list_to_dict(heads_list):
    """Convert list of (layer, head) tuples to dict mapping layer -> [heads]."""
    d = {}
    for layer, head in heads_list:
        if layer not in d:
            d[layer] = []
        d[layer].append(head)
    return d


def run_accuracy_test(model, tokenizer, samples, shuffle, haystack_text):
    """Run accuracy test."""
    correct = 0
    total = 0
    
    for sample in samples:
        needle = extract_needle_full_section(
            sample['section_1'],
            sample['ground_truth_state'],
            shuffle=shuffle
        )
        
        context = create_needle_in_haystack(
            needle=needle,
            tokenizer=tokenizer,
            target_tokens=CONTEXT_LENGTH,
            needle_position=NEEDLE_POSITION,
            haystack_text=haystack_text
        )
        
        prompt = QUESTION_TEMPLATE.format(context=context)
        
        try:
            answer = run_inference(model, tokenizer, prompt)
            is_correct = check_answer(answer, sample['ground_truth_state'])
            if is_correct:
                correct += 1
            total += 1
        except Exception as e:
            continue
    
    return correct / total if total > 0 else 0


def run_ablation_sweep(model, tokenizer, samples, haystack_text, top_heads, condition_name):
    """Run sweep ablating 1, 2, 3, ... up to MAX_HEADS_TO_ABLATE heads."""
    
    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads
    
    # Ablation counts to test
    ablation_counts = [0, 1, 2, 3, 5, 10, 15, 20, 25, 30, 40, 50]
    ablation_counts = [c for c in ablation_counts if c <= len(top_heads)]
    
    results = []
    
    for num_ablate in tqdm(ablation_counts, desc=f"{condition_name} sweep"):
        if num_ablate == 0:
            # Baseline - no ablation
            acc = run_accuracy_test(model, tokenizer, samples, 
                                   shuffle=(condition_name == "shuffled"), 
                                   haystack_text=haystack_text)
        else:
            # Ablate top N heads
            heads_to_ablate = top_heads[:num_ablate]
            heads_dict = heads_list_to_dict(heads_to_ablate)
            
            hook = MultiHeadAblationHook(heads_dict, n_heads, head_dim)
            hook.register(model)
            
            try:
                acc = run_accuracy_test(model, tokenizer, samples,
                                       shuffle=(condition_name == "shuffled"),
                                       haystack_text=haystack_text)
            finally:
                hook.remove()
        
        results.append({
            'num_ablated': num_ablate,
            'accuracy': acc
        })
        print(f"  {num_ablate} heads ablated: {acc*100:.1f}%")
    
    return results


def plot_results(results, save_path):
    """Plot ablation sweep results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot unshuffled
    ax = axes[0]
    x = [r['num_ablated'] for r in results['unshuffled']]
    y = [r['accuracy'] * 100 for r in results['unshuffled']]
    ax.plot(x, y, 'b-o', linewidth=2, markersize=8)
    ax.axhline(y=y[0], color='b', linestyle='--', alpha=0.5, label=f'Baseline: {y[0]:.1f}%')
    ax.set_xlabel('Number of Heads Ablated', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Unshuffled: Ablating Top Comprehension Heads\n(L16-24 heads)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    
    # Plot shuffled
    ax = axes[1]
    x = [r['num_ablated'] for r in results['shuffled']]
    y = [r['accuracy'] * 100 for r in results['shuffled']]
    ax.plot(x, y, 'r-o', linewidth=2, markersize=8)
    ax.axhline(y=y[0], color='r', linestyle='--', alpha=0.5, label=f'Baseline: {y[0]:.1f}%')
    ax.set_xlabel('Number of Heads Ablated', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Shuffled: Ablating Top Keyword-Matching Heads\n(L2-5 heads)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot: {save_path}")
    plt.close()


def main():
    print("="*70)
    print("LLAMA 3 ABLATION SWEEP")
    print("="*70)
    print(f"Context: {CONTEXT_LENGTH} tokens")
    print(f"Samples: {NUM_SAMPLES}")
    print(f"Max heads to ablate: {MAX_HEADS_TO_ABLATE}")
    print()
    
    # Load model
    print("Loading Llama 3...")
    model_name = MODELS[MODEL_KEY]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(f"Model loaded")
    
    # Load data
    print("\nLoading samples...")
    ground_truth = load_ground_truth()
    samples = load_edgar_samples(NUM_SAMPLES, ground_truth)
    haystack_text = get_haystack_text()
    print(f"Loaded {len(samples)} samples")
    
    results = {}
    
    # =========================================================================
    # SWEEP: Ablating top UNSHUFFLED heads (test on unshuffled)
    # =========================================================================
    print("\n" + "="*70)
    print("SWEEP 1: Ablating top COMPREHENSION heads (testing UNSHUFFLED)")
    print("="*70)
    results['unshuffled'] = run_ablation_sweep(
        model, tokenizer, samples, haystack_text,
        TOP_UNSHUFFLED_HEADS, "unshuffled"
    )
    
    # =========================================================================
    # SWEEP: Ablating top SHUFFLED heads (test on shuffled)
    # =========================================================================
    print("\n" + "="*70)
    print("SWEEP 2: Ablating top KEYWORD-MATCHING heads (testing SHUFFLED)")
    print("="*70)
    results['shuffled'] = run_ablation_sweep(
        model, tokenizer, samples, haystack_text,
        TOP_SHUFFLED_HEADS, "shuffled"
    )
    
    # =========================================================================
    # SAVE AND PLOT
    # =========================================================================
    save_dir = "llama3_results/head"
    os.makedirs(save_dir, exist_ok=True)
    
    # Save JSON
    output_path = os.path.join(save_dir, "ablation_sweep.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {output_path}")
    
    # Plot
    plot_path = os.path.join(save_dir, "ablation_sweep.png")
    plot_results(results, plot_path)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print("\nUnshuffled (ablating comprehension heads):")
    for r in results['unshuffled']:
        print(f"  {r['num_ablated']:2d} heads: {r['accuracy']*100:.1f}%")
    
    print("\nShuffled (ablating keyword-matching heads):")
    for r in results['shuffled']:
        print(f"  {r['num_ablated']:2d} heads: {r['accuracy']*100:.1f}%")


if __name__ == "__main__":
    main()


