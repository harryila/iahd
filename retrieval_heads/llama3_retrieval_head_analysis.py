"""
Llama 3 Retrieval Head Analysis

This script identifies which attention heads are responsible for retrieving
information from the context (Section 1) in the needle-in-haystack task.

Our Approach (different from paper):
- Paper: measures attention to specific answer tokens
- Ours: measures TOTAL attention to the context region (Section 1)

This captures "comprehension heads" that attend to relevant context,
not just "copy heads" that focus on single tokens.

Plan:
1. Run model with output_attentions=True
2. For each (layer, head): compute sum of attention to Section 1 region
3. Rank heads by attention score
4. Compare across conditions (short/long, shuffled/unshuffled)
5. Ablation: remove top heads and measure accuracy change
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
    check_answer, extract_needle_full_section, get_haystack_text
)

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_KEY = "llama"  # Llama 3 (8K context limit)
CONTEXT_LENGTHS = {
    "short": 2000,   # Where model works well
    "long": 8000,    # Near collapse point (before 10K death)
}
NUM_SAMPLES = 20  # Samples for head identification
NEEDLE_POSITION = 0.5

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_attention_to_context(
    model,
    tokenizer,
    prompt: str,
    context_start_text: str,
    device: str = "cuda"
) -> np.ndarray:
    """
    Run model and compute attention from last token to context region.
    
    Returns:
        attention_scores: np.ndarray of shape (n_layers, n_heads)
        Each value = sum of attention from last token to context region
    """
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    input_ids = inputs.input_ids[0]
    
    # Find context region (Section 1)
    context_ids = tokenizer.encode(context_start_text[:500], add_special_tokens=False)  # First 500 chars
    
    # Search for context start
    context_tensor = torch.tensor(context_ids[:50], device=device)  # Match first 50 tokens
    start_idx = 0
    for i in range(len(input_ids) - len(context_tensor) + 1):
        if torch.equal(input_ids[i:i+len(context_tensor)], context_tensor):
            start_idx = i
            break
    
    # Context region: from start to ~500 tokens after (or until Alice text)
    end_idx = min(start_idx + 500, len(input_ids) - 50)  # Leave room for question
    
    # Run model with attention outputs
    with torch.no_grad():
        outputs = model(
            inputs.input_ids,
            output_attentions=True,
            use_cache=False
        )
    
    # Extract attention scores
    n_layers = len(outputs.attentions)
    n_heads = outputs.attentions[0].shape[1]
    attention_scores = np.zeros((n_layers, n_heads))
    
    for layer_idx, layer_attn in enumerate(outputs.attentions):
        # layer_attn shape: [batch, heads, seq_len, seq_len]
        # Get attention from last token to all positions
        last_token_attn = layer_attn[0, :, -1, :].float().cpu().numpy()
        
        # Sum attention to context region
        context_attn = last_token_attn[:, start_idx:end_idx].sum(axis=1)
        attention_scores[layer_idx] = context_attn
    
    return attention_scores, (start_idx, end_idx, len(input_ids))


def identify_top_heads(attention_scores: np.ndarray, top_k: int = 15) -> list:
    """
    Identify top-k heads by attention score.
    
    Returns:
        List of (layer, head, score) tuples sorted by score descending
    """
    n_layers, n_heads = attention_scores.shape
    head_scores = []
    
    for layer in range(n_layers):
        for head in range(n_heads):
            head_scores.append((layer, head, attention_scores[layer, head]))
    
    # Sort by score descending
    head_scores.sort(key=lambda x: x[2], reverse=True)
    return head_scores[:top_k]


def run_head_identification(
    model,
    tokenizer,
    samples: list,
    context_length: int,
    shuffle: bool,
    haystack_text: str,
    device: str = "cuda"
) -> dict:
    """
    Run head identification across multiple samples.
    
    Returns:
        Dictionary with aggregated attention scores and top heads
    """
    print(f"\n{'='*60}")
    print(f"Identifying heads at {context_length} tokens, shuffle={shuffle}")
    print(f"{'='*60}")
    
    all_attention_scores = []
    
    for sample in tqdm(samples, desc=f"{context_length}tk {'shuffled' if shuffle else 'normal'}"):
        # Create needle
        needle = extract_needle_full_section(
            sample['section_1'],
            sample['ground_truth_state'],
            shuffle=shuffle
        )
        
        # Create context
        context = create_needle_in_haystack(
            needle=needle,
            tokenizer=tokenizer,
            target_tokens=context_length,
            needle_position=NEEDLE_POSITION,
            haystack_text=haystack_text
        )
        
        # Create prompt
        prompt = QUESTION_TEMPLATE.format(context=context)
        
        # Get attention scores
        try:
            attn_scores, indices = get_attention_to_context(
                model, tokenizer, prompt,
                sample['section_1'][:500],
                device
            )
            all_attention_scores.append(attn_scores)
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    # Aggregate scores (mean across samples)
    if all_attention_scores:
        mean_scores = np.mean(all_attention_scores, axis=0)
        std_scores = np.std(all_attention_scores, axis=0)
    else:
        return None
    
    # Get top heads
    top_heads = identify_top_heads(mean_scores, top_k=15)
    
    return {
        'context_length': context_length,
        'shuffle': shuffle,
        'n_samples': len(all_attention_scores),
        'mean_attention': mean_scores.tolist(),
        'std_attention': std_scores.tolist(),
        'top_heads': [(int(l), int(h), float(s)) for l, h, s in top_heads]
    }


def plot_head_comparison(results: dict, save_path: str):
    """
    Create visualization comparing heads across conditions.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    conditions = [
        ('short', False, 'Short (2K) - Unshuffled'),
        ('short', True, 'Short (2K) - Shuffled'),
        ('long', False, 'Long (8K) - Unshuffled'),
        ('long', True, 'Long (8K) - Shuffled'),
    ]
    
    for ax, (length_key, shuffle, title) in zip(axes.flat, conditions):
        key = f"{length_key}_{'shuffled' if shuffle else 'unshuffled'}"
        if key in results:
            data = results[key]
            mean_attn = np.array(data['mean_attention'])
            
            im = ax.imshow(mean_attn, aspect='auto', cmap='viridis')
            ax.set_xlabel('Head Index')
            ax.set_ylabel('Layer Index')
            ax.set_title(f"{title}\nTop: L{data['top_heads'][0][0]}H{data['top_heads'][0][1]}")
            plt.colorbar(im, ax=ax, label='Attention to Section 1')
    
    plt.suptitle('Llama 3: Retrieval Head Attention Patterns', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")


def print_head_comparison(results: dict):
    """
    Print comparison of top heads across conditions.
    """
    print("\n" + "="*70)
    print("TOP RETRIEVAL HEADS COMPARISON")
    print("="*70)
    
    for key, data in results.items():
        print(f"\n{key.upper()}:")
        print(f"  Context: {data['context_length']} tokens, Shuffle: {data['shuffle']}")
        print(f"  Top 10 heads:")
        for i, (layer, head, score) in enumerate(data['top_heads'][:10]):
            print(f"    {i+1}. L{layer}H{head}: {score:.4f}")
    
    # Find common heads
    print("\n" + "-"*70)
    print("HEAD OVERLAP ANALYSIS:")
    
    all_top_sets = {}
    for key, data in results.items():
        top_set = set((l, h) for l, h, s in data['top_heads'][:10])
        all_top_sets[key] = top_set
    
    if len(all_top_sets) >= 2:
        keys = list(all_top_sets.keys())
        for i, k1 in enumerate(keys):
            for k2 in keys[i+1:]:
                common = all_top_sets[k1] & all_top_sets[k2]
                print(f"\n  {k1} ∩ {k2}:")
                print(f"    Common heads ({len(common)}): {sorted(common)}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*60)
    print("LLAMA 3 RETRIEVAL HEAD ANALYSIS")
    print("="*60)
    
    # Load model
    print("\nLoading Llama 3...")
    model_name = MODELS[MODEL_KEY]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"Model loaded on {device}")
    
    # Load data
    print("\nLoading samples...")
    ground_truth = load_ground_truth()
    samples = load_edgar_samples(NUM_SAMPLES, ground_truth)
    haystack_text = get_haystack_text()
    print(f"Loaded {len(samples)} samples")
    
    # Run head identification for each condition
    results = {}
    
    for length_key, context_length in CONTEXT_LENGTHS.items():
        for shuffle in [False, True]:
            key = f"{length_key}_{'shuffled' if shuffle else 'unshuffled'}"
            
            result = run_head_identification(
                model, tokenizer, samples,
                context_length, shuffle,
                haystack_text, device
            )
            
            if result:
                results[key] = result
    
    # Save results
    save_dir = "llama3_results"
    os.makedirs(save_dir, exist_ok=True)
    
    output_path = os.path.join(save_dir, "retrieval_head_analysis.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {output_path}")
    
    # Print comparison
    print_head_comparison(results)
    
    # Plot
    plot_path = os.path.join(save_dir, "retrieval_head_heatmaps.png")
    plot_head_comparison(results, plot_path)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Review top heads for each condition")
    print("2. Run ablation study (remove top heads, measure accuracy)")
    print("3. Try patching (boost short-context heads in long context)")


if __name__ == "__main__":
    main()

