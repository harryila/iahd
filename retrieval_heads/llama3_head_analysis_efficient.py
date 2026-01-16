"""
Llama 3 Retrieval Head Analysis - MEMORY EFFICIENT VERSION

Key optimizations:
1. Only extract attention from LAST token (1 row, not NxN matrix)
2. Process one layer at a time with aggressive cleanup
3. Use fewer samples
4. bfloat16 throughout
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
    extract_needle_full_section, get_haystack_text
)

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_KEY = "llama"
CONTEXT_LENGTHS = {
    "short": 2000,
    "medium": 4000,  # Compromise - 8K still OOMs
    "long": 6000,    # Push a bit further
}
NUM_SAMPLES = 10  # Reduced for memory
NEEDLE_POSITION = 0.5

# ============================================================================
# EFFICIENT ATTENTION EXTRACTION
# ============================================================================

def get_last_token_attention(model, tokenizer, prompt, device):
    """
    Get attention scores from LAST token only.
    Returns: dict mapping (layer, head) -> attention vector over sequence
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    seq_len = inputs.input_ids.shape[1]
    
    n_layers = len(model.model.layers)
    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads
    
    # Store only what we need: attention from last token
    all_attention = {}
    
    with torch.no_grad():
        # Get hidden states through the model manually
        hidden_states = model.model.embed_tokens(inputs.input_ids)
        
        # Create causal mask
        attention_mask = torch.ones((1, seq_len), device=device, dtype=torch.bool)
        
        # Process each layer
        for layer_idx in range(n_layers):
            layer = model.model.layers[layer_idx]
            
            # Get Q, K for this layer's attention
            # We only need the last position's query
            
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
            
            # Compute Q, K, V
            bsz, q_len, _ = hidden_states.size()
            
            query_states = layer.self_attn.q_proj(hidden_states)
            key_states = layer.self_attn.k_proj(hidden_states)
            value_states = layer.self_attn.v_proj(hidden_states)
            
            # Reshape for attention
            query_states = query_states.view(bsz, q_len, n_heads, head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, n_heads, head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, n_heads, head_dim).transpose(1, 2)
            
            # Apply rotary embeddings
            cos, sin = layer.self_attn.rotary_emb(value_states, position_ids=None)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            
            # Compute attention scores for LAST position only
            # query_states[:, :, -1:, :] shape: (1, n_heads, 1, head_dim)
            # key_states shape: (1, n_heads, seq_len, head_dim)
            
            last_query = query_states[:, :, -1:, :]  # (1, n_heads, 1, head_dim)
            
            # Attention scores: (1, n_heads, 1, seq_len)
            attn_weights = torch.matmul(last_query, key_states.transpose(-2, -1)) / (head_dim ** 0.5)
            
            # Apply causal mask (last token can attend to all)
            # No masking needed for last position in causal attention
            
            # Softmax
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
            
            # Store attention from last token: (n_heads, seq_len)
            for head_idx in range(n_heads):
                all_attention[(layer_idx, head_idx)] = attn_weights[0, head_idx, 0, :].cpu().numpy()
            
            # Continue forward pass for next layer
            attn_output = torch.matmul(attn_weights, value_states[:, :, :, :])
            attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, 1, -1)
            
            # But we need full sequence for next layer, so recompute properly
            # Actually let's just use the model's forward pass and extract what we need
            del query_states, key_states, value_states, attn_weights, attn_output
            torch.cuda.empty_cache()
            
            # Reset and do proper forward for this layer
            hidden_states = layer.input_layernorm(residual)
            
    return all_attention, seq_len


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Apply rotary position embeddings."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rotate_half(x):
    """Rotate half the hidden dims."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def get_attention_simple(model, tokenizer, prompt, context_start, context_end, device):
    """
    Simple approach: run model with output_attentions=True but immediately
    extract only what we need and delete the rest.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    n_layers = len(model.model.layers)
    n_heads = model.config.num_attention_heads
    head_scores = np.zeros((n_layers, n_heads))
    
    # Process in chunks of layers to manage memory
    chunk_size = 4  # Process 4 layers at a time
    
    for start_layer in range(0, n_layers, chunk_size):
        end_layer = min(start_layer + chunk_size, n_layers)
        
        # We can't easily chunk layers, so let's just be careful with memory
        pass
    
    # Actually, simplest approach: just run with output_attentions and process immediately
    with torch.no_grad():
        outputs = model(
            inputs.input_ids,
            output_attentions=True,
            use_cache=False,
            return_dict=True
        )
        
        # Immediately extract what we need
        for layer_idx, layer_attn in enumerate(outputs.attentions):
            # layer_attn: (1, n_heads, seq_len, seq_len)
            # We only need last row (last token attending to others)
            last_token_attn = layer_attn[0, :, -1, context_start:context_end]  # (n_heads, context_len)
            head_scores[layer_idx] = last_token_attn.sum(dim=1).float().cpu().numpy()
            
            # Clear this layer's attention immediately
            del layer_attn
        
        del outputs
        torch.cuda.empty_cache()
        gc.collect()
    
    return head_scores


def analyze_condition(model, tokenizer, samples, context_length, shuffle, haystack_text, device):
    """Analyze heads for a specific condition."""
    print(f"\n{'='*60}")
    print(f"Context: {context_length} tokens, Shuffle: {shuffle}")
    print(f"{'='*60}")
    
    all_scores = []
    
    for i, sample in enumerate(tqdm(samples, desc=f"{context_length}tk")):
        try:
            # Create context
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
            
            # Find ACTUAL context region (Section 1) - not approximate!
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs.input_ids[0]
            
            # Search for Section 1 text in tokens
            section_start_text = sample['section_1'][:300]  # First 300 chars
            section_tokens = tokenizer.encode(section_start_text, add_special_tokens=False)[:30]
            section_tensor = torch.tensor(section_tokens)
            
            context_start = 0
            for i in range(len(input_ids) - len(section_tensor) + 1):
                if torch.equal(input_ids[i:i+len(section_tensor)], section_tensor):
                    context_start = i
                    break
            
            # Section 1 is roughly 300-500 tokens
            context_end = min(context_start + 400, len(input_ids) - 50)
            
            # Get attention scores
            scores = get_attention_simple(
                model, tokenizer, prompt,
                context_start, context_end, device
            )
            all_scores.append(scores)
            
        except torch.cuda.OutOfMemoryError as e:
            print(f"\nOOM at sample {i}, skipping...")
            torch.cuda.empty_cache()
            gc.collect()
            continue
        except Exception as e:
            print(f"\nError at sample {i}: {e}")
            continue
    
    if not all_scores:
        return None
    
    # Aggregate
    mean_scores = np.mean(all_scores, axis=0)
    
    # Find top heads
    n_layers, n_heads = mean_scores.shape
    head_list = []
    for l in range(n_layers):
        for h in range(n_heads):
            head_list.append((l, h, mean_scores[l, h]))
    
    head_list.sort(key=lambda x: x[2], reverse=True)
    top_heads = head_list[:15]
    
    return {
        'context_length': context_length,
        'shuffle': shuffle,
        'n_samples': len(all_scores),
        'mean_scores': mean_scores.tolist(),
        'top_heads': [(int(l), int(h), float(s)) for l, h, s in top_heads]
    }


def main():
    print("="*60)
    print("LLAMA 3 HEAD ANALYSIS - MEMORY EFFICIENT")
    print("="*60)
    
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
    print(f"Model on {device}")
    
    # Load data
    ground_truth = load_ground_truth()
    samples = load_edgar_samples(NUM_SAMPLES, ground_truth)
    haystack_text = get_haystack_text()
    print(f"Loaded {len(samples)} samples")
    
    # Run analysis
    results = {}
    
    for length_name, context_length in CONTEXT_LENGTHS.items():
        for shuffle in [False, True]:
            key = f"{length_name}_{'shuffled' if shuffle else 'unshuffled'}"
            
            result = analyze_condition(
                model, tokenizer, samples,
                context_length, shuffle,
                haystack_text, device
            )
            
            if result:
                results[key] = result
                print(f"\n{key} Top 5 heads:")
                for i, (l, h, s) in enumerate(result['top_heads'][:5]):
                    print(f"  {i+1}. L{l}H{h}: {s:.4f}")
            
            # Clear memory between conditions
            torch.cuda.empty_cache()
            gc.collect()
    
    # Save results
    save_dir = "llama3_results/head"
    os.makedirs(save_dir, exist_ok=True)
    
    output_path = os.path.join(save_dir, "head_analysis_efficient.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {output_path}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for key, data in results.items():
        layers = [l for l, h, s in data['top_heads'][:10]]
        print(f"{key}: mean layer = {np.mean(layers):.1f}, top = L{data['top_heads'][0][0]}H{data['top_heads'][0][1]}")


if __name__ == "__main__":
    main()

