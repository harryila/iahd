"""
Llama 3 Retrieval Head Analysis - SURGICAL VERSION

Uses monkey patching to capture attention from ONE layer at a time,
avoiding OOM errors at large context lengths (8K+).

Based on Ananya's approach in attention_head_analysis.ipynb
"""

import torch
import numpy as np
import json
import os
import sys
import types
import gc
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, '.')
from needle_haystack_sweep import (
    load_ground_truth, load_edgar_samples, MODELS,
    create_needle_in_haystack, QUESTION_TEMPLATE,
    extract_needle_full_section, get_haystack_text
)

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_KEY = "llama"  # Llama 3 (8K context limit)
CONTEXT_LENGTHS = {
    "short": 2000,
    "long": 8000,
}
NUM_SAMPLES = 15
NEEDLE_POSITION = 0.5

# ============================================================================
# SURGICAL ATTENTION EXTRACTION (One layer at a time)
# ============================================================================

def get_attention_surgical(
    model,
    tokenizer,
    prompt: str,
    context_start_text: str,
    device: str = "cuda"
) -> np.ndarray:
    """
    Extract attention scores using surgical monkey patching.
    Processes ONE layer at a time to avoid OOM.
    """
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    input_ids = inputs.input_ids[0]
    
    # Find context region
    context_ids = tokenizer.encode(context_start_text[:300], add_special_tokens=False)
    context_tensor = torch.tensor(context_ids[:30], device=device)
    
    start_idx = 0
    for i in range(len(input_ids) - len(context_tensor) + 1):
        if torch.equal(input_ids[i:i+len(context_tensor)], context_tensor):
            start_idx = i
            break
    end_idx = min(start_idx + 400, len(input_ids) - 50)
    
    n_layers = len(model.model.layers)
    n_heads = model.config.num_attention_heads
    attention_scores = np.zeros((n_layers, n_heads))
    
    # Process each layer one at a time
    for layer_idx in range(n_layers):
        captured_attn = None
        target_layer = model.model.layers[layer_idx].self_attn
        original_forward = target_layer.forward
        
        def make_spy_forward(layer_idx):
            def spy_forward(self, *args, **kwargs):
                nonlocal captured_attn
                kwargs['output_attentions'] = True
                output = original_forward(*args, **kwargs)
                if output[1] is not None:
                    # Capture attention from last position to context
                    attn = output[1][0, :, -1, start_idx:end_idx].float().cpu()
                    captured_attn = attn.sum(dim=1).numpy()  # Sum over context
                # Return without attention to save memory
                return (output[0], None) + output[2:]
            return spy_forward
        
        target_layer.forward = types.MethodType(make_spy_forward(layer_idx), target_layer)
        
        try:
            with torch.no_grad():
                model(inputs.input_ids, output_attentions=False, use_cache=False)
            
            if captured_attn is not None:
                attention_scores[layer_idx] = captured_attn
        finally:
            target_layer.forward = original_forward
            torch.cuda.empty_cache()
            gc.collect()
    
    return attention_scores


def identify_top_heads(attention_scores: np.ndarray, top_k: int = 15) -> list:
    """Identify top-k heads by attention score."""
    n_layers, n_heads = attention_scores.shape
    head_scores = []
    
    for layer in range(n_layers):
        for head in range(n_heads):
            head_scores.append((layer, head, attention_scores[layer, head]))
    
    head_scores.sort(key=lambda x: x[2], reverse=True)
    return head_scores[:top_k]


def run_head_identification(
    model, tokenizer, samples, context_length, shuffle, haystack_text, device
):
    """Run head identification for a condition."""
    print(f"\n{'='*60}")
    print(f"Identifying heads at {context_length} tokens, shuffle={shuffle}")
    print(f"{'='*60}")
    
    all_attention_scores = []
    
    for sample in tqdm(samples, desc=f"{context_length}tk {'shuf' if shuffle else 'norm'}"):
        needle = extract_needle_full_section(
            sample['section_1'],
            sample['ground_truth_state'],
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
        
        try:
            attn_scores = get_attention_surgical(
                model, tokenizer, prompt,
                sample['section_1'][:300],
                device
            )
            all_attention_scores.append(attn_scores)
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    if all_attention_scores:
        mean_scores = np.mean(all_attention_scores, axis=0)
        std_scores = np.std(all_attention_scores, axis=0)
    else:
        return None
    
    top_heads = identify_top_heads(mean_scores, top_k=15)
    
    return {
        'context_length': context_length,
        'shuffle': shuffle,
        'n_samples': len(all_attention_scores),
        'mean_attention': mean_scores.tolist(),
        'std_attention': std_scores.tolist(),
        'top_heads': [(int(l), int(h), float(s)) for l, h, s in top_heads]
    }


def main():
    print("="*60)
    print("LLAMA 3 RETRIEVAL HEAD ANALYSIS (SURGICAL - NO OOM)")
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
    
    # Run for all conditions
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
                
                # Print top heads immediately
                print(f"\n{key.upper()} Top 5 heads:")
                for i, (l, h, s) in enumerate(result['top_heads'][:5]):
                    print(f"  {i+1}. L{l}H{h}: {s:.4f}")
    
    # Save results
    save_dir = "llama3_results/head"
    os.makedirs(save_dir, exist_ok=True)
    
    output_path = os.path.join(save_dir, "retrieval_head_surgical.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {output_path}")
    
    # Print comparison
    print("\n" + "="*70)
    print("COMPARISON ACROSS ALL CONDITIONS")
    print("="*70)
    
    for key, data in results.items():
        print(f"\n{key}:")
        layers = [l for l, h, s in data['top_heads'][:10]]
        print(f"  Top 10 head layers: {layers}")
        print(f"  Mean layer: {np.mean(layers):.1f}")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()


