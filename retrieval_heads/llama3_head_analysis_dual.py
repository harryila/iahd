"""
Llama 3 Retrieval Head Analysis - DUAL METRIC COMPARISON

Computes TWO metrics side-by-side:
1. CONTEXT ATTENTION: Total attention to the entire context/needle region (your approach)
2. KEY TOKEN ATTENTION: Attention to specific answer tokens only (paper's approach)

This lets us compare:
- Which heads each metric identifies
- How much overlap there is
- Which might be more causally relevant
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
# CONFIGURATION - Keep small for memory
# ============================================================================

MODEL_KEY = "llama"  # Llama 3
CONTEXT_LENGTH = 1000  # Small enough to fit with output_attentions
NUM_SAMPLES = 10  # Reduced to prevent OOM
NEEDLE_POSITION = 0.5
TOP_K_HEADS = 20

# ============================================================================
# TOKEN FINDING UTILITIES
# ============================================================================

def find_context_region(input_ids, tokenizer, needle_text):
    """Find the approximate start and end of the needle/context region."""
    search_text = needle_text[:80] if len(needle_text) > 80 else needle_text
    search_tokens = tokenizer.encode(search_text, add_special_tokens=False)[:15]
    
    input_list = input_ids.tolist()
    context_start = 0
    
    for i in range(len(input_list) - len(search_tokens) + 1):
        match_count = sum(1 for j, t in enumerate(search_tokens) if i+j < len(input_list) and input_list[i+j] == t)
        if match_count >= len(search_tokens) * 0.6:
            context_start = i
            break
    
    context_end = min(context_start + 300, len(input_ids) - 30)
    return context_start, context_end


def find_key_tokens(input_ids, tokenizer, state_name):
    """Find token positions where the answer (state name) appears."""
    occurrences = []
    input_list = input_ids.tolist()
    
    variations = [state_name, state_name.lower(), state_name.upper(), state_name.title()]
    
    for variant in variations:
        variant_tokens = tokenizer.encode(variant, add_special_tokens=False)
        if len(variant_tokens) == 0:
            continue
            
        for i in range(len(input_list) - len(variant_tokens) + 1):
            if input_list[i:i+len(variant_tokens)] == variant_tokens:
                if (i, i + len(variant_tokens)) not in occurrences:
                    occurrences.append((i, i + len(variant_tokens)))
    
    return occurrences


# ============================================================================
# DUAL METRIC COMPUTATION  
# ============================================================================

def compute_dual_metrics(attentions, context_start, context_end, key_positions, seq_len):
    """
    Compute BOTH metrics for all heads.
    Returns: context_scores, key_scores as dicts mapping (layer, head) -> score
    """
    n_layers = len(attentions)
    n_heads = attentions[0].shape[1]
    
    context_scores = {}
    key_scores = {}
    
    for layer_idx, layer_attn in enumerate(attentions):
        # layer_attn: (batch, n_heads, seq_len, seq_len)
        # Get attention FROM last position
        last_pos_attn = layer_attn[0, :, -1, :].float()  # (n_heads, seq_len)
        
        for head_idx in range(n_heads):
            head_attn = last_pos_attn[head_idx].cpu().numpy()
            
            # Metric 1: Context attention (sum over entire needle region)
            ctx_end = min(context_end, seq_len)
            ctx_start = max(0, context_start)
            context_attn = head_attn[ctx_start:ctx_end].sum()
            
            # Metric 2: Key token attention (sum over answer tokens only)
            key_attn = 0.0
            for start, end in key_positions:
                if start < seq_len and end <= seq_len:
                    key_attn += head_attn[start:end].sum()
            
            context_scores[(layer_idx, head_idx)] = float(context_attn)
            key_scores[(layer_idx, head_idx)] = float(key_attn)
    
    return context_scores, key_scores


# ============================================================================
# SAMPLE ANALYSIS
# ============================================================================

def analyze_sample(model, tokenizer, sample, context_length, shuffle, haystack_text, device):
    """Analyze a single sample with both metrics."""
    ground_truth = sample['ground_truth_state']
    
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
    
    # Find regions
    context_start, context_end = find_context_region(input_ids, tokenizer, needle)
    key_positions = find_key_tokens(input_ids, tokenizer, ground_truth)
    
    num_key_tokens = sum(end - start for start, end in key_positions if start < seq_len and end <= seq_len)
    
    # Run inference for accuracy (greedy, no attention needed)
    try:
        answer = run_inference(model, tokenizer, prompt)
        is_correct = check_answer(answer, ground_truth)
    except Exception as e:
        answer = f"ERROR: {e}"
        is_correct = False
    
    # Get attention patterns with single forward pass
    with torch.no_grad():
        outputs = model(
            inputs.input_ids,
            output_attentions=True,
            use_cache=False,
            return_dict=True
        )
        attentions = outputs.attentions
        
        # Compute metrics
        context_scores, key_scores = compute_dual_metrics(
            attentions, context_start, context_end, key_positions, seq_len
        )
        
        # Clean up immediately
        del attentions, outputs
    
    torch.cuda.empty_cache()
    
    return {
        'filename': sample['filename'],
        'ground_truth': ground_truth,
        'answer': answer.strip()[:100],
        'is_correct': is_correct,
        'context_scores': {f"L{l}H{h}": s for (l, h), s in context_scores.items()},
        'key_scores': {f"L{l}H{h}": s for (l, h), s in key_scores.items()},
        'context_region': (context_start, context_end),
        'num_key_tokens': num_key_tokens,
        'seq_len': seq_len
    }


# ============================================================================
# CONDITION ANALYSIS
# ============================================================================

def run_condition(model, tokenizer, samples, context_length, shuffle, haystack_text, device):
    """Run analysis for a condition, computing both metrics."""
    condition_name = "shuffled" if shuffle else "unshuffled"
    print(f"\n{'='*70}")
    print(f"Analyzing: {condition_name.upper()} at {context_length} tokens")
    print(f"{'='*70}")
    
    all_results = []
    context_scores_all = defaultdict(list)
    key_scores_all = defaultdict(list)
    correct_count = 0
    total_count = 0
    
    for sample in tqdm(samples, desc=condition_name):
        try:
            result = analyze_sample(
                model, tokenizer, sample,
                context_length, shuffle, haystack_text, device
            )
            all_results.append(result)
            
            # Accumulate scores
            for head_key, score in result['context_scores'].items():
                context_scores_all[head_key].append(score)
            for head_key, score in result['key_scores'].items():
                key_scores_all[head_key].append(score)
            
            if result['is_correct'] is not None:
                total_count += 1
                if result['is_correct']:
                    correct_count += 1
                    
        except torch.cuda.OutOfMemoryError:
            print(f"\nOOM, skipping...")
            try:
                torch.cuda.empty_cache()
            except:
                pass
            gc.collect()
        except RuntimeError as e:
            if "CUDA" in str(e):
                print(f"\nCUDA error, skipping...")
                try:
                    torch.cuda.empty_cache()
                except:
                    pass
                gc.collect()
            else:
                print(f"\nError: {e}")
        except Exception as e:
            print(f"\nError: {e}")
            continue
    
    # Calculate means
    mean_context = {k: np.mean(v) for k, v in context_scores_all.items()}
    mean_key = {k: np.mean(v) for k, v in key_scores_all.items()}
    
    # Get top heads by each metric
    top_by_context = sorted(mean_context.items(), key=lambda x: x[1], reverse=True)[:TOP_K_HEADS]
    top_by_key = sorted(mean_key.items(), key=lambda x: x[1], reverse=True)[:TOP_K_HEADS]
    
    accuracy = correct_count / total_count if total_count > 0 else 0
    
    print(f"\n{condition_name.upper()} Results:")
    print(f"  Accuracy: {accuracy*100:.1f}% ({correct_count}/{total_count})")
    
    print(f"\n  Top 10 by CONTEXT ATTENTION (your approach):")
    for i, (head, score) in enumerate(top_by_context[:10]):
        print(f"    {i+1}. {head}: {score:.4f}")
    
    print(f"\n  Top 10 by KEY TOKEN ATTENTION (paper's approach):")
    for i, (head, score) in enumerate(top_by_key[:10]):
        print(f"    {i+1}. {head}: {score:.6f}")
    
    return {
        'condition': condition_name,
        'context_length': context_length,
        'accuracy': accuracy,
        'correct': correct_count,
        'total': total_count,
        'top_by_context': [(h, float(s)) for h, s in top_by_context],
        'top_by_key': [(h, float(s)) for h, s in top_by_key],
        'mean_context_scores': {k: float(v) for k, v in mean_context.items()},
        'mean_key_scores': {k: float(v) for k, v in mean_key.items()},
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("LLAMA 3 HEAD ANALYSIS - DUAL METRIC COMPARISON")
    print("="*70)
    print("\nComputing TWO metrics:")
    print("  1. CONTEXT ATTENTION: Sum over entire needle region (your approach)")
    print("  2. KEY TOKEN ATTENTION: Sum over answer tokens only (paper's approach)")
    
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
        attn_implementation="eager",
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"Model on {device}")
    
    # Load data
    ground_truth = load_ground_truth()
    samples = load_edgar_samples(NUM_SAMPLES, ground_truth)
    haystack_text = get_haystack_text()
    print(f"Loaded {len(samples)} samples")
    
    # Run both conditions
    results = {}
    
    save_dir = "llama3_results/head"
    os.makedirs(save_dir, exist_ok=True)
    
    results['unshuffled'] = run_condition(
        model, tokenizer, samples,
        CONTEXT_LENGTH, shuffle=False,
        haystack_text=haystack_text, device=device
    )
    
    # Save intermediate
    with open(os.path.join(save_dir, "head_analysis_dual_partial.json"), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved intermediate results")
    
    # Clear memory between conditions
    torch.cuda.empty_cache()
    gc.collect()
    import time
    time.sleep(2)  # Let GPU memory settle
    
    results['shuffled'] = run_condition(
        model, tokenizer, samples,
        CONTEXT_LENGTH, shuffle=True,
        haystack_text=haystack_text, device=device
    )
    
    # =========================================================================
    # COMPARISON ANALYSIS
    # =========================================================================
    print("\n" + "="*70)
    print("COMPARISON: TWO METRICS × TWO CONDITIONS")
    print("="*70)
    
    for cond_name in ['unshuffled', 'shuffled']:
        cond = results[cond_name]
        print(f"\n### {cond_name.upper()} (Accuracy: {cond['accuracy']*100:.1f}%) ###")
        
        context_heads = set(h for h, s in cond['top_by_context'][:15])
        key_heads = set(h for h, s in cond['top_by_key'][:15])
        overlap = context_heads & key_heads
        
        print(f"\n  Metric Agreement (top 15 heads):")
        print(f"    Overlap: {len(overlap)}/15 ({len(overlap)/15*100:.0f}%)")
        print(f"    Shared: {sorted(list(overlap))[:8]}")
        print(f"    Context-only: {sorted(list(context_heads - key_heads))[:5]}")
        print(f"    Key-only: {sorted(list(key_heads - context_heads))[:5]}")
    
    # Cross-condition comparison
    print("\n" + "="*70)
    print("CROSS-CONDITION COMPARISON")
    print("="*70)
    
    unshuf_context = set(h for h, s in results['unshuffled']['top_by_context'][:15])
    shuf_context = set(h for h, s in results['shuffled']['top_by_context'][:15])
    unshuf_key = set(h for h, s in results['unshuffled']['top_by_key'][:15])
    shuf_key = set(h for h, s in results['shuffled']['top_by_key'][:15])
    
    print(f"\n  By CONTEXT metric:")
    print(f"    Unshuffled ∩ Shuffled: {len(unshuf_context & shuf_context)}/15")
    print(f"    Unshuffled-only: {sorted(list(unshuf_context - shuf_context))[:5]}")
    print(f"    Shuffled-only: {sorted(list(shuf_context - unshuf_context))[:5]}")
    
    print(f"\n  By KEY TOKEN metric:")
    print(f"    Unshuffled ∩ Shuffled: {len(unshuf_key & shuf_key)}/15")
    print(f"    Unshuffled-only: {sorted(list(unshuf_key - shuf_key))[:5]}")
    print(f"    Shuffled-only: {sorted(list(shuf_key - unshuf_key))[:5]}")
    
    # Accuracy comparison
    print(f"\n  Accuracy Comparison:")
    print(f"    Unshuffled: {results['unshuffled']['accuracy']*100:.1f}%")
    print(f"    Shuffled: {results['shuffled']['accuracy']*100:.1f}%")
    print(f"    Drop: {(results['unshuffled']['accuracy'] - results['shuffled']['accuracy'])*100:.1f}%")
    
    # Save final results
    output_path = os.path.join(save_dir, "head_analysis_dual.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {output_path}")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
