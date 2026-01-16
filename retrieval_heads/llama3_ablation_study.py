"""
Llama 3 Ablation Study

Tests if identified retrieval heads are causally responsible for retrieval
by zeroing them out and measuring accuracy drop.

Ablations:
1. Remove L16H1 (top unshuffled head) → expect drop on unshuffled
2. Remove L17H24 (top unshuffled head) → expect drop on unshuffled  
3. Remove L2H21 (top shuffled head) → expect drop on shuffled, NOT unshuffled
"""

import torch
import numpy as np
import json
import os
import sys
from tqdm import tqdm
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
NUM_SAMPLES = 30  # More samples for statistical significance
NEEDLE_POSITION = 0.5

# Heads to ablate (layer, head)
HEADS_TO_ABLATE = [
    (16, 1),   # L16H1 - top unshuffled head
    (17, 24),  # L17H24 - second top unshuffled head
    (2, 21),   # L2H21 - top shuffled head (same as Ananya)
]

# ============================================================================
# ABLATION HOOK
# ============================================================================

class HeadAblationHook:
    """Hook to zero out a specific attention head's contribution.
    
    Strategy: Hook into self_attn and zero out the target head's 
    contribution in the output hidden states.
    
    self_attn output: tuple (hidden_states, attn_weights)
    hidden_states shape: (batch, seq_len, hidden_size)
    hidden_size = n_heads * head_dim
    """
    
    def __init__(self, head_idx, n_heads, head_dim):
        self.head_idx = head_idx
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.handle = None
    
    def hook_fn(self, module, input, output):
        """Zero out the specified head's contribution in self_attn output."""
        # output is tuple: (hidden_states, attn_weights)
        hidden_states = output[0]
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Reshape to (batch, seq, n_heads, head_dim)
        reshaped = hidden_states.view(batch_size, seq_len, self.n_heads, self.head_dim)
        
        # Zero out the target head
        reshaped = reshaped.clone()
        reshaped[:, :, self.head_idx, :] = 0
        
        # Reshape back
        modified = reshaped.view(batch_size, seq_len, hidden_size)
        
        # Return modified tuple
        return (modified,) + output[1:]
    
    def register(self, layer):
        """Register hook on self_attn module."""
        self.handle = layer.self_attn.register_forward_hook(self.hook_fn)
    
    def remove(self):
        """Remove the hook."""
        if self.handle:
            self.handle.remove()
            self.handle = None


def run_accuracy_test(model, tokenizer, samples, shuffle, haystack_text, device):
    """Run accuracy test on samples."""
    correct = 0
    total = 0
    
    for sample in tqdm(samples, desc=f"{'Shuffled' if shuffle else 'Unshuffled'}"):
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
            print(f"Error: {e}")
            continue
    
    return correct / total if total > 0 else 0, correct, total


def run_ablation_experiment(model, tokenizer, samples, haystack_text, device, layer_idx, head_idx):
    """Run ablation experiment for a specific head."""
    
    # Get model config
    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads
    
    # Create and register hook
    hook = HeadAblationHook(head_idx, n_heads, head_dim)
    hook.register(model.model.layers[layer_idx])
    
    try:
        # Test on unshuffled
        acc_unshuffled, correct_unshuf, total_unshuf = run_accuracy_test(
            model, tokenizer, samples, shuffle=False, 
            haystack_text=haystack_text, device=device
        )
        
        # Test on shuffled
        acc_shuffled, correct_shuf, total_shuf = run_accuracy_test(
            model, tokenizer, samples, shuffle=True,
            haystack_text=haystack_text, device=device
        )
    finally:
        # Always remove hook
        hook.remove()
    
    return {
        'unshuffled': {'accuracy': acc_unshuffled, 'correct': correct_unshuf, 'total': total_unshuf},
        'shuffled': {'accuracy': acc_shuffled, 'correct': correct_shuf, 'total': total_shuf}
    }


def main():
    print("="*70)
    print("LLAMA 3 ABLATION STUDY")
    print("="*70)
    print(f"Context length: {CONTEXT_LENGTH} tokens")
    print(f"Samples: {NUM_SAMPLES}")
    print(f"Heads to ablate: {HEADS_TO_ABLATE}")
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
    device = next(model.parameters()).device
    print(f"Model on {device}")
    
    # Load data
    print("\nLoading samples...")
    ground_truth = load_ground_truth()
    samples = load_edgar_samples(NUM_SAMPLES, ground_truth)
    haystack_text = get_haystack_text()
    print(f"Loaded {len(samples)} samples")
    
    results = {}
    
    # =========================================================================
    # BASELINE (no ablation)
    # =========================================================================
    print("\n" + "="*70)
    print("BASELINE (No Ablation)")
    print("="*70)
    
    acc_unshuffled, correct_unshuf, total_unshuf = run_accuracy_test(
        model, tokenizer, samples, shuffle=False,
        haystack_text=haystack_text, device=device
    )
    print(f"Unshuffled: {acc_unshuffled*100:.1f}% ({correct_unshuf}/{total_unshuf})")
    
    acc_shuffled, correct_shuf, total_shuf = run_accuracy_test(
        model, tokenizer, samples, shuffle=True,
        haystack_text=haystack_text, device=device
    )
    print(f"Shuffled: {acc_shuffled*100:.1f}% ({correct_shuf}/{total_shuf})")
    
    results['baseline'] = {
        'unshuffled': {'accuracy': acc_unshuffled, 'correct': correct_unshuf, 'total': total_unshuf},
        'shuffled': {'accuracy': acc_shuffled, 'correct': correct_shuf, 'total': total_shuf}
    }
    
    # =========================================================================
    # ABLATION EXPERIMENTS
    # =========================================================================
    for layer_idx, head_idx in HEADS_TO_ABLATE:
        print("\n" + "="*70)
        print(f"ABLATING L{layer_idx}H{head_idx}")
        print("="*70)
        
        ablation_results = run_ablation_experiment(
            model, tokenizer, samples, haystack_text, device,
            layer_idx, head_idx
        )
        
        results[f'ablate_L{layer_idx}H{head_idx}'] = ablation_results
        
        # Print results
        print(f"\nResults with L{layer_idx}H{head_idx} ablated:")
        print(f"  Unshuffled: {ablation_results['unshuffled']['accuracy']*100:.1f}% "
              f"(Δ = {(ablation_results['unshuffled']['accuracy'] - results['baseline']['unshuffled']['accuracy'])*100:+.1f}%)")
        print(f"  Shuffled: {ablation_results['shuffled']['accuracy']*100:.1f}% "
              f"(Δ = {(ablation_results['shuffled']['accuracy'] - results['baseline']['shuffled']['accuracy'])*100:+.1f}%)")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"\n{'Condition':<25} {'Unshuffled':<20} {'Shuffled':<20}")
    print("-"*65)
    
    baseline_unshuf = results['baseline']['unshuffled']['accuracy'] * 100
    baseline_shuf = results['baseline']['shuffled']['accuracy'] * 100
    print(f"{'Baseline':<25} {baseline_unshuf:.1f}%{'':<14} {baseline_shuf:.1f}%")
    
    for layer_idx, head_idx in HEADS_TO_ABLATE:
        key = f'ablate_L{layer_idx}H{head_idx}'
        unshuf = results[key]['unshuffled']['accuracy'] * 100
        shuf = results[key]['shuffled']['accuracy'] * 100
        delta_unshuf = unshuf - baseline_unshuf
        delta_shuf = shuf - baseline_shuf
        
        print(f"{'Ablate L' + str(layer_idx) + 'H' + str(head_idx):<25} "
              f"{unshuf:.1f}% ({delta_unshuf:+.1f}%){'':<5} "
              f"{shuf:.1f}% ({delta_shuf:+.1f}%)")
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    save_dir = "llama3_results/head"
    os.makedirs(save_dir, exist_ok=True)
    
    output_path = os.path.join(save_dir, "ablation_study.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {output_path}")
    
    # =========================================================================
    # INTERPRETATION
    # =========================================================================
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    
    # Check L16H1/L17H24 ablation effect
    l16h1_unshuf_drop = results['baseline']['unshuffled']['accuracy'] - results['ablate_L16H1']['unshuffled']['accuracy']
    l17h24_unshuf_drop = results['baseline']['unshuffled']['accuracy'] - results['ablate_L17H24']['unshuffled']['accuracy']
    
    print(f"\nL16H1 ablation drops unshuffled by {l16h1_unshuf_drop*100:.1f}%")
    print(f"L17H24 ablation drops unshuffled by {l17h24_unshuf_drop*100:.1f}%")
    
    if l16h1_unshuf_drop > 0.05 or l17h24_unshuf_drop > 0.05:
        print("→ These late-layer heads ARE causally important for comprehension!")
    else:
        print("→ These heads may not be solely responsible (redundancy?)")
    
    # Check L2H21 ablation effect
    l2h21_shuf_drop = results['baseline']['shuffled']['accuracy'] - results['ablate_L2H21']['shuffled']['accuracy']
    l2h21_unshuf_drop = results['baseline']['unshuffled']['accuracy'] - results['ablate_L2H21']['unshuffled']['accuracy']
    
    print(f"\nL2H21 ablation drops shuffled by {l2h21_shuf_drop*100:.1f}%")
    print(f"L2H21 ablation drops unshuffled by {l2h21_unshuf_drop*100:.1f}%")
    
    if l2h21_shuf_drop > 0.05 and l2h21_unshuf_drop < 0.05:
        print("→ L2H21 IS a keyword-matching head (hurts shuffled, not unshuffled)!")
    elif l2h21_shuf_drop > 0.05:
        print("→ L2H21 IS important for shuffled retrieval")
    else:
        print("→ L2H21 may have redundant backups")


if __name__ == "__main__":
    main()

