"""
Llama 3 Retrieval Head Analysis - PROPER KEY TOKEN MEASUREMENT

This script correctly measures attention to SPECIFIC KEY TOKENS (the answer),
not the entire context region. This matches the paper's methodology.

Key differences from previous scripts:
1. Finds the actual state name tokens in the context
2. Measures attention specifically to those tokens (g_h in the paper)
3. Shows which individual tokens get the most attention
4. Compares shuffled vs unshuffled with accuracy
"""

import torch
import numpy as np
import json
import os
import sys
import gc
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, '.')
from needle_haystack_sweep import (
    load_ground_truth, load_edgar_samples, MODELS,
    create_needle_in_haystack, QUESTION_TEMPLATE,
    extract_needle_full_section, get_haystack_text,
    run_inference, check_answer
)

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_KEY = "llama"  # Llama 3
CONTEXT_LENGTH = 4000  # Fixed context length for comparison
NUM_SAMPLES = 30
NEEDLE_POSITION = 0.5
TOP_K_HEADS = 20  # Number of top heads to track

# ============================================================================
# KEY TOKEN FINDING
# ============================================================================

def find_key_tokens(input_ids, tokenizer, state_name):
    """
    Find the token positions where the state name appears in the input.
    Returns list of (start_idx, end_idx) for each occurrence.
    """
    # Tokenize the state name
    state_tokens = tokenizer.encode(state_name, add_special_tokens=False)
    state_lower_tokens = tokenizer.encode(state_name.lower(), add_special_tokens=False)
    
    occurrences = []
    input_list = input_ids.tolist()
    
    # Search for exact match
    for tokens_to_find in [state_tokens, state_lower_tokens]:
        for i in range(len(input_list) - len(tokens_to_find) + 1):
            if input_list[i:i+len(tokens_to_find)] == tokens_to_find:
                occurrences.append((i, i + len(tokens_to_find)))
    
    # Also search for partial matches (state name might be part of larger token)
    state_text = state_name.lower()
    for i, token_id in enumerate(input_list):
        token_text = tokenizer.decode([token_id]).lower()
        if state_text in token_text or token_text in state_text:
            if (i, i+1) not in occurrences:
                occurrences.append((i, i+1))
    
    return occurrences


def get_attention_to_key_tokens(attentions, key_token_positions, seq_len):
    """
    Calculate attention score to key tokens (the answer) from last position.
    
    This is g_h from the paper: how much does each head attend to the key tokens
    that contain the answer?
    
    Returns: dict mapping (layer, head) -> attention_score_to_key_tokens
    """
    n_layers = len(attentions)
    n_heads = attentions[0].shape[1]
    
    head_scores = {}
    
    # Create mask for key token positions
    key_mask = torch.zeros(seq_len)
    for start, end in key_token_positions:
        if start < seq_len and end <= seq_len:
            key_mask[start:end] = 1
    
    if key_mask.sum() == 0:
        # No key tokens found - return zeros
        for l in range(n_layers):
            for h in range(n_heads):
                head_scores[(l, h)] = 0.0
        return head_scores
    
    for layer_idx, layer_attn in enumerate(attentions):
        # layer_attn: (batch, n_heads, seq_len, seq_len)
        # Get attention from last position
        last_pos_attn = layer_attn[0, :, -1, :]  # (n_heads, seq_len)
        
        # Attention to key tokens only
        key_mask_device = key_mask.to(last_pos_attn.device)
        key_attention = (last_pos_attn * key_mask_device).sum(dim=1)  # (n_heads,)
        
        for head_idx in range(n_heads):
            head_scores[(layer_idx, head_idx)] = key_attention[head_idx].item()
    
    return head_scores


def get_top_attended_tokens(attentions, input_ids, tokenizer, layer, head, top_k=10):
    """
    For a specific head, find which tokens it attends to most from the last position.
    Returns list of (token_text, token_position, attention_score).
    """
    # Get attention from last position for this head
    last_pos_attn = attentions[layer][0, head, -1, :].cpu().numpy()
    
    # Get top attended positions
    top_positions = np.argsort(last_pos_attn)[-top_k:][::-1]
    
    results = []
    for pos in top_positions:
        token_id = input_ids[pos].item()
        token_text = tokenizer.decode([token_id])
        attn_score = last_pos_attn[pos]
        results.append((token_text, int(pos), float(attn_score)))
    
    return results


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def analyze_sample(model, tokenizer, sample, context_length, shuffle, haystack_text, device):
    """
    Analyze a single sample: get attention scores and accuracy.
    
    Returns dict with:
    - head_scores: attention to key tokens per head
    - is_correct: whether the model got the right answer
    - answer: model's answer
    - key_token_positions: where the answer appears
    - top_attended_tokens: for visualization
    """
    ground_truth = sample['ground_truth_state']
    
    # Create context
    needle = extract_needle_full_section(
        sample['section_1'],
        ground_truth,
        shuffle=shuffle
    )
    
    context = create_needle_in_haystack(
        needle=needle,
        tokenizer=tokenizer,
        target_tokens=context_length,
        needle_position=NEEDLE_POSITION,
        haystack_text=haystack_text
    )
    
    prompt = QUESTION_TEMPLATE.format(context=context)
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids[0]
    seq_len = len(input_ids)
    
    # Find key tokens (where state name appears)
    key_positions = find_key_tokens(input_ids, tokenizer, ground_truth)
    
    # Run inference to get answer
    try:
        answer = run_inference(model, tokenizer, prompt)
        is_correct = check_answer(answer, ground_truth)
    except Exception as e:
        answer = f"ERROR: {e}"
        is_correct = False
    
    # Get attention patterns
    with torch.no_grad():
        outputs = model(
            inputs.input_ids,
            output_attentions=True,
            use_cache=False,
            return_dict=True
        )
        attentions = outputs.attentions
    
    # Calculate attention to key tokens
    head_scores = get_attention_to_key_tokens(attentions, key_positions, seq_len)
    
    # Get top attended tokens for the best heads (for visualization)
    sorted_heads = sorted(head_scores.items(), key=lambda x: x[1], reverse=True)
    top_head = sorted_heads[0][0] if sorted_heads else (0, 0)
    top_attended = get_top_attended_tokens(
        attentions, input_ids, tokenizer,
        top_head[0], top_head[1], top_k=15
    )
    
    # Clean up
    del attentions, outputs
    torch.cuda.empty_cache()
    
    return {
        'filename': sample['filename'],
        'ground_truth': ground_truth,
        'answer': answer,
        'is_correct': is_correct,
        'head_scores': {f"L{l}H{h}": s for (l, h), s in head_scores.items()},
        'key_positions': key_positions,
        'num_key_tokens': sum(end - start for start, end in key_positions),
        'seq_len': seq_len,
        'top_attended_tokens': top_attended,
        'top_head': f"L{top_head[0]}H{top_head[1]}"
    }


def run_condition(model, tokenizer, samples, context_length, shuffle, haystack_text, device):
    """Run analysis for a specific condition (shuffled/unshuffled)."""
    condition_name = "shuffled" if shuffle else "unshuffled"
    print(f"\n{'='*70}")
    print(f"Analyzing: {condition_name.upper()} at {context_length} tokens")
    print(f"{'='*70}")
    
    all_results = []
    all_head_scores = defaultdict(list)
    correct_count = 0
    total_count = 0
    
    for sample in tqdm(samples, desc=condition_name):
        try:
            result = analyze_sample(
                model, tokenizer, sample,
                context_length, shuffle, haystack_text, device
            )
            all_results.append(result)
            
            # Accumulate head scores
            for head_key, score in result['head_scores'].items():
                all_head_scores[head_key].append(score)
            
            # Track accuracy
            if result['is_correct'] is not None:
                total_count += 1
                if result['is_correct']:
                    correct_count += 1
                    
        except torch.cuda.OutOfMemoryError:
            print(f"\nOOM, skipping sample...")
            torch.cuda.empty_cache()
            gc.collect()
            continue
        except Exception as e:
            print(f"\nError: {e}")
            continue
    
    # Calculate accuracy
    accuracy = correct_count / total_count if total_count > 0 else 0
    
    # Calculate mean head scores
    mean_scores = {k: np.mean(v) for k, v in all_head_scores.items()}
    
    # Get top heads
    sorted_heads = sorted(mean_scores.items(), key=lambda x: x[1], reverse=True)
    top_heads = sorted_heads[:TOP_K_HEADS]
    
    print(f"\n{condition_name.upper()} Results:")
    print(f"  Accuracy: {accuracy*100:.1f}% ({correct_count}/{total_count})")
    print(f"\n  Top 10 Retrieval Heads (by attention to KEY TOKENS):")
    for i, (head, score) in enumerate(top_heads[:10]):
        print(f"    {i+1}. {head}: {score:.6f}")
    
    # Show example of what tokens the top head attends to
    if all_results:
        print(f"\n  Example: Top head's most attended tokens (from last sample):")
        for token, pos, attn in all_results[-1]['top_attended_tokens'][:5]:
            print(f"    '{token}' at pos {pos}: {attn:.4f}")
    
    return {
        'condition': condition_name,
        'context_length': context_length,
        'accuracy': accuracy,
        'correct': correct_count,
        'total': total_count,
        'top_heads': [(h, float(s)) for h, s in top_heads],
        'mean_scores': {k: float(v) for k, v in mean_scores.items()},
        'sample_results': all_results
    }


def main():
    print("="*70)
    print("LLAMA 3 RETRIEVAL HEAD ANALYSIS - PROPER KEY TOKEN MEASUREMENT")
    print("="*70)
    print("\nThis measures attention to SPECIFIC KEY TOKENS (the answer),")
    print("not the entire context region. This matches the paper's methodology.")
    
    # Load model
    print("\nLoading Llama 3...")
    model_name = MODELS[MODEL_KEY]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",  # Required for output_attentions
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"Model loaded on {device}")
    
    # Load data
    ground_truth = load_ground_truth()
    samples = load_edgar_samples(NUM_SAMPLES, ground_truth)
    haystack_text = get_haystack_text()
    print(f"Loaded {len(samples)} samples")
    
    # Run analysis for both conditions
    results = {}
    
    # Unshuffled (normal)
    results['unshuffled'] = run_condition(
        model, tokenizer, samples,
        CONTEXT_LENGTH, shuffle=False,
        haystack_text=haystack_text, device=device
    )
    
    torch.cuda.empty_cache()
    gc.collect()
    
    # Shuffled
    results['shuffled'] = run_condition(
        model, tokenizer, samples,
        CONTEXT_LENGTH, shuffle=True,
        haystack_text=haystack_text, device=device
    )
    
    # Save results
    save_dir = "llama3_results/head"
    os.makedirs(save_dir, exist_ok=True)
    
    # Save full results
    output_path = os.path.join(save_dir, "head_analysis_proper.json")
    
    # Don't save sample_results to keep file size manageable
    save_results = {}
    for k, v in results.items():
        save_results[k] = {key: val for key, val in v.items() if key != 'sample_results'}
    
    with open(output_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"\nSaved: {output_path}")
    
    # =========================================================================
    # COMPARISON SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("COMPARISON: UNSHUFFLED vs SHUFFLED")
    print("="*70)
    
    unshuf = results['unshuffled']
    shuf = results['shuffled']
    
    print(f"\n{'Metric':<30} {'Unshuffled':<15} {'Shuffled':<15} {'Difference':<15}")
    print("-"*75)
    print(f"{'Accuracy':<30} {unshuf['accuracy']*100:.1f}%{'':<10} {shuf['accuracy']*100:.1f}%{'':<10} {(unshuf['accuracy']-shuf['accuracy'])*100:+.1f}%")
    
    print(f"\n{'Top 5 Heads (Unshuffled)':<40} {'Top 5 Heads (Shuffled)':<40}")
    print("-"*80)
    for i in range(5):
        unshuf_head = unshuf['top_heads'][i] if i < len(unshuf['top_heads']) else ('N/A', 0)
        shuf_head = shuf['top_heads'][i] if i < len(shuf['top_heads']) else ('N/A', 0)
        print(f"{unshuf_head[0]}: {unshuf_head[1]:.6f}{'':<15} {shuf_head[0]}: {shuf_head[1]:.6f}")
    
    # Check overlap in top heads
    unshuf_top_set = set(h for h, s in unshuf['top_heads'][:15])
    shuf_top_set = set(h for h, s in shuf['top_heads'][:15])
    overlap = unshuf_top_set & shuf_top_set
    
    print(f"\n{'Head Overlap (top 15):':<30} {len(overlap)}/15 ({len(overlap)/15*100:.0f}%)")
    print(f"{'Shared heads:':<30} {sorted(overlap)[:8]}...")
    print(f"{'Unshuffled-only:':<30} {sorted(unshuf_top_set - shuf_top_set)[:5]}...")
    print(f"{'Shuffled-only:':<30} {sorted(shuf_top_set - unshuf_top_set)[:5]}...")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()

