"""
Llama 3 Ablation Study - CORRECT VERSION

Properly ablates attention heads by zeroing their attention weights
BEFORE they're used to compute the weighted sum of values.
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import os
import sys
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaAttention
import types

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

# Top heads from analysis
TOP_UNSHUFFLED_HEADS = [
    (17, 24), (16, 1), (17, 26), (24, 27), (20, 1),
    (16, 0), (18, 20), (17, 1), (20, 14), (19, 24),
]

TOP_SHUFFLED_HEADS = [
    (2, 21), (2, 22), (2, 23), (2, 25), (2, 12),
    (2, 27), (2, 20), (2, 24), (2, 26), (2, 28),
]

# ============================================================================
# PROPER ABLATION BY PATCHING FORWARD METHOD
# ============================================================================

def create_ablated_attention_forward(original_forward, layer_idx, heads_to_zero, n_heads):
    """Create a patched forward that zeros specific heads' attention."""
    
    def ablated_forward(self, hidden_states, attention_mask=None, position_ids=None, 
                       past_key_value=None, output_attentions=False, use_cache=False, 
                       cache_position=None, **kwargs):
        
        # Get the heads to zero for this layer
        heads = heads_to_zero.get(layer_idx, [])
        
        if not heads:
            # No ablation for this layer, use original
            return original_forward(
                hidden_states, attention_mask, position_ids,
                past_key_value, output_attentions, use_cache,
                cache_position, **kwargs
            )
        
        # Manual implementation with head zeroing
        bsz, q_len, _ = hidden_states.size()
        
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # Repeat KV for GQA
        key_states = repeat_kv(key_states, self.num_heads // self.num_key_value_heads)
        value_states = repeat_kv(value_states, self.num_heads // self.num_key_value_heads)
        
        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / (self.head_dim ** 0.5)
        
        # Apply causal mask
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, :key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
        
        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        # ABLATION: Zero out attention weights for specific heads
        for head_idx in heads:
            attn_weights[:, head_idx, :, :] = 0.0
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        
        if not output_attentions:
            attn_weights = None
        
        return attn_output, attn_weights, past_key_value
    
    return ablated_forward


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(hidden_states, n_rep):
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def apply_ablation(model, heads_to_zero):
    """Apply ablation by patching attention forward methods."""
    n_heads = model.config.num_attention_heads
    original_forwards = {}
    
    for layer_idx in range(len(model.model.layers)):
        layer = model.model.layers[layer_idx]
        original_forwards[layer_idx] = layer.self_attn.forward
        
        patched_forward = create_ablated_attention_forward(
            layer.self_attn.forward, layer_idx, heads_to_zero, n_heads
        )
        layer.self_attn.forward = types.MethodType(patched_forward, layer.self_attn)
    
    return original_forwards


def remove_ablation(model, original_forwards):
    """Restore original forward methods."""
    for layer_idx, orig_forward in original_forwards.items():
        model.model.layers[layer_idx].self_attn.forward = orig_forward


def heads_list_to_dict(heads_list):
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
            print(f"Error: {e}")
            continue
    
    return correct / total if total > 0 else 0, correct, total


def main():
    print("="*70)
    print("LLAMA 3 ABLATION - CORRECT VERSION")
    print("="*70)
    print("Properly zeroing attention weights BEFORE softmax application")
    print()
    
    # Load model with eager attention (required for manual attention)
    print("Loading Llama 3 (eager attention)...")
    model_name = MODELS[MODEL_KEY]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",  # Required for our patching
    )
    model.eval()
    print("Model loaded")
    
    # Load data
    print("\nLoading samples...")
    ground_truth = load_ground_truth()
    samples = load_edgar_samples(NUM_SAMPLES, ground_truth)
    haystack_text = get_haystack_text()
    print(f"Loaded {len(samples)} samples")
    
    results = {}
    
    # =========================================================================
    # BASELINE
    # =========================================================================
    print("\n" + "="*70)
    print("BASELINE (No Ablation)")
    print("="*70)
    
    acc_unshuf, c_unshuf, t_unshuf = run_accuracy_test(
        model, tokenizer, samples, shuffle=False, haystack_text=haystack_text
    )
    print(f"Unshuffled: {acc_unshuf*100:.1f}% ({c_unshuf}/{t_unshuf})")
    
    acc_shuf, c_shuf, t_shuf = run_accuracy_test(
        model, tokenizer, samples, shuffle=True, haystack_text=haystack_text
    )
    print(f"Shuffled: {acc_shuf*100:.1f}% ({c_shuf}/{t_shuf})")
    
    results['baseline'] = {
        'unshuffled': acc_unshuf,
        'shuffled': acc_shuf
    }
    
    # =========================================================================
    # ABLATION SWEEP
    # =========================================================================
    ablation_counts = [1, 3, 5, 10]
    
    for num_ablate in ablation_counts:
        print("\n" + "="*70)
        print(f"ABLATING TOP {num_ablate} HEADS")
        print("="*70)
        
        # Ablate unshuffled heads, test on unshuffled
        heads_dict = heads_list_to_dict(TOP_UNSHUFFLED_HEADS[:num_ablate])
        print(f"Ablating: {TOP_UNSHUFFLED_HEADS[:num_ablate]}")
        
        orig_forwards = apply_ablation(model, heads_dict)
        try:
            acc, c, t = run_accuracy_test(
                model, tokenizer, samples, shuffle=False, haystack_text=haystack_text
            )
            delta = acc - results['baseline']['unshuffled']
            print(f"Unshuffled with top unshuffled heads ablated: {acc*100:.1f}% (Δ={delta*100:+.1f}%)")
            results[f'ablate_{num_ablate}_unshuf_heads_on_unshuf'] = acc
        finally:
            remove_ablation(model, orig_forwards)
        
        # Ablate shuffled heads, test on shuffled
        heads_dict = heads_list_to_dict(TOP_SHUFFLED_HEADS[:num_ablate])
        print(f"Ablating: {TOP_SHUFFLED_HEADS[:num_ablate]}")
        
        orig_forwards = apply_ablation(model, heads_dict)
        try:
            acc, c, t = run_accuracy_test(
                model, tokenizer, samples, shuffle=True, haystack_text=haystack_text
            )
            delta = acc - results['baseline']['shuffled']
            print(f"Shuffled with top shuffled heads ablated: {acc*100:.1f}% (Δ={delta*100:+.1f}%)")
            results[f'ablate_{num_ablate}_shuf_heads_on_shuf'] = acc
        finally:
            remove_ablation(model, orig_forwards)
    
    # =========================================================================
    # SAVE
    # =========================================================================
    save_dir = "llama3_results/head"
    os.makedirs(save_dir, exist_ok=True)
    
    output_path = os.path.join(save_dir, "ablation_correct.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {output_path}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Baseline unshuffled: {results['baseline']['unshuffled']*100:.1f}%")
    print(f"Baseline shuffled: {results['baseline']['shuffled']*100:.1f}%")
    for key, val in results.items():
        if key != 'baseline':
            print(f"{key}: {val*100:.1f}%")


if __name__ == "__main__":
    main()


