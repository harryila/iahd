# -*- coding: utf-8 -*-
"""
Simplified Retrieval Head Analysis for EDGAR State of Incorporation

This version focuses on accuracy comparison between short and long contexts,
with memory-efficient attention head tracking.

Based on: "Retrieval Head Mechanistically Explains Long-Context Factual Recall"
https://arxiv.org/pdf/2404.15574
"""

import os
import sys
import json
import random
import numpy as np
import torch
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Models to test
MODELS = {
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
}

# Context length configurations
CONTEXT_CONFIGS = {
    "short": {
        "target_tokens": 200,
        "description": "Original short context (~200 tokens)",
    },
    "long": {
        "target_tokens": 2000, 
        "description": "Degraded long context (~2000 tokens with padding)",
    },
}

# Question template
QUESTION_TEMPLATE = """Based on the following SEC 10-K filing excerpt, answer the question.

Context:
{context}

Question: In which US state was this company incorporated?
Answer with just the state name:"""

NUM_SAMPLES = 50

# =============================================================================
# US STATES FOR VALIDATION
# =============================================================================

US_STATES = {
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
    "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho",
    "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana",
    "maine", "maryland", "massachusetts", "michigan", "minnesota",
    "mississippi", "missouri", "montana", "nebraska", "nevada",
    "new hampshire", "new jersey", "new mexico", "new york",
    "north carolina", "north dakota", "ohio", "oklahoma", "oregon",
    "pennsylvania", "rhode island", "south carolina", "south dakota",
    "tennessee", "texas", "utah", "vermont", "virginia", "washington",
    "west virginia", "wisconsin", "wyoming", "district of columbia"
}

# Common abbreviations/shortcuts
STATE_ALIASES = {
    "ny": "new york", "ca": "california", "tx": "texas", "fl": "florida",
    "pa": "pennsylvania", "il": "illinois", "oh": "ohio", "ga": "georgia",
    "nc": "north carolina", "nj": "new jersey", "va": "virginia",
    "wa": "washington", "ma": "massachusetts", "az": "arizona",
    "tn": "tennessee", "in": "indiana", "mo": "missouri", "md": "maryland",
    "wi": "wisconsin", "mn": "minnesota", "co": "colorado", "sc": "south carolina",
    "al": "alabama", "la": "louisiana", "ky": "kentucky", "or": "oregon",
    "ok": "oklahoma", "ct": "connecticut", "ia": "iowa", "ut": "utah",
    "nv": "nevada", "ar": "arkansas", "ms": "mississippi", "ks": "kansas",
    "nm": "new mexico", "ne": "nebraska", "wv": "west virginia", "id": "idaho",
    "hi": "hawaii", "nh": "new hampshire", "me": "maine", "ri": "rhode island",
    "mt": "montana", "de": "delaware", "sd": "south dakota", "nd": "north dakota",
    "ak": "alaska", "vt": "vermont", "wy": "wyoming", "dc": "district of columbia",
}

# =============================================================================
# GROUND TRUTH LOADING
# =============================================================================

def load_ground_truth():
    """Load ground truth from CSV."""
    gt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                           "gt_extract", "edgar_ground_truth_llm.csv")
    if os.path.exists(gt_path):
        df = pd.read_csv(gt_path)
        gt_dict = {}
        for _, row in df.iterrows():
            filename = row['filename']
            state = row.get('state_of_incorporation', None)
            if pd.notna(state) and state and str(state).lower() not in ['nan', 'none', '']:
                gt_dict[filename] = str(state).strip()
        print(f"Loaded ground truth for {len(gt_dict)} files")
        return gt_dict
    print("Warning: Ground truth CSV not found")
    return {}

# =============================================================================
# DATA LOADING
# =============================================================================

def load_edgar_samples(num_samples, ground_truth):
    """Load EDGAR samples that have ground truth."""
    print(f"Loading EDGAR corpus...")
    
    dataset = load_dataset(
        "c3po-ai/edgar-corpus",
        "full",
        split="train",
        streaming=True,
        trust_remote_code=True
    )
    
    samples = []
    for item in dataset:
        if len(samples) >= num_samples:
            break
            
        filename = item.get('filename', '')
        section_1 = item.get('section_1', '')
        
        # Only keep samples with ground truth
        if filename not in ground_truth:
            continue
        if not section_1 or len(section_1) < 100:
            continue
            
        samples.append({
            'filename': filename,
            'section_1': section_1,
            'section_2': item.get('section_2', ''),
            'section_7': item.get('section_7', ''),
            'ground_truth_state': ground_truth[filename],
        })
    
    print(f"Loaded {len(samples)} samples with ground truth")
    return samples

# =============================================================================
# CONTEXT MANIPULATION
# =============================================================================

def truncate_to_tokens(text, tokenizer, max_tokens):
    """Truncate text to approximately max_tokens."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return text, len(tokens)
    truncated_tokens = tokens[:max_tokens]
    return tokenizer.decode(truncated_tokens, skip_special_tokens=True), max_tokens

def create_short_context(sample, tokenizer, target_tokens=200):
    """Create a short context with just section_1 truncated."""
    context, num_tokens = truncate_to_tokens(sample['section_1'], tokenizer, target_tokens)
    return context, num_tokens

def create_long_context(sample, tokenizer, target_tokens=2000):
    """
    Create a long context by placing section_1 in the middle of padding.
    This makes retrieval harder - the model must find the needle.
    """
    section_1 = sample['section_1']
    section_2 = sample.get('section_2', '') or ''
    section_7 = sample.get('section_7', '') or ''
    
    # Core context with the key fact (~400 tokens)
    core_context, core_tokens = truncate_to_tokens(section_1, tokenizer, 400)
    
    # Calculate padding needed
    remaining = target_tokens - core_tokens
    padding_before_tokens = remaining // 2
    padding_after_tokens = remaining - padding_before_tokens
    
    # Create padding from other sections or generic filler
    filler = "The company operates across multiple business segments and geographic regions. Financial performance is subject to various market conditions and regulatory requirements. Management continues to evaluate strategic opportunities and operational efficiencies. " * 20
    
    if section_7 and len(section_7) > 100:
        padding_before, _ = truncate_to_tokens(section_7, tokenizer, padding_before_tokens)
    else:
        padding_before, _ = truncate_to_tokens(filler, tokenizer, padding_before_tokens)
    
    if section_2 and len(section_2) > 100:
        padding_after, _ = truncate_to_tokens(section_2, tokenizer, padding_after_tokens)
    else:
        padding_after, _ = truncate_to_tokens(filler[::-1], tokenizer, padding_after_tokens)  # Reversed filler
    
    # Combine: padding + core + padding
    full_context = f"{padding_before}\n\n{core_context}\n\n{padding_after}"
    final_context, final_tokens = truncate_to_tokens(full_context, tokenizer, target_tokens)
    
    return final_context, final_tokens

# =============================================================================
# ACCURACY CHECKING
# =============================================================================

def normalize_state(text):
    """Normalize state name for comparison."""
    if not text:
        return None
    text = str(text).lower().strip()
    # Remove punctuation and extra words
    for char in '.!?,;:()[]"\'':
        text = text.replace(char, '')
    text = text.strip()
    # Get first word/phrase
    words = text.split()
    if not words:
        return None
    # Handle multi-word states
    if len(words) >= 2 and words[0] in ['new', 'north', 'south', 'west', 'rhode', 'district']:
        text = ' '.join(words[:2])
    else:
        text = words[0]
    # Check aliases
    if text in STATE_ALIASES:
        return STATE_ALIASES[text]
    return text

def check_accuracy(answer, ground_truth):
    """Check if answer matches ground truth."""
    answer_norm = normalize_state(answer)
    truth_norm = normalize_state(ground_truth)
    
    if not answer_norm or not truth_norm:
        return False
    
    # Direct match
    if answer_norm == truth_norm:
        return True
    # Partial match (one contains the other)
    if answer_norm in truth_norm or truth_norm in answer_norm:
        return True
    return False

# =============================================================================
# MODEL INFERENCE
# =============================================================================

def load_model(model_name):
    """Load model for inference."""
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    
    return model, tokenizer

def run_inference(model, tokenizer, prompt, max_new_tokens=15):
    """Run inference and return generated text."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Get generated tokens only
    generated_ids = outputs[0, inputs.input_ids.shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return generated_text.strip()

# =============================================================================
# ATTENTION HEAD ANALYSIS (Memory Efficient)
# =============================================================================

def analyze_attention_heads(model, tokenizer, prompt, context_start_approx, context_end_approx):
    """
    Analyze which attention heads focus on the context region.
    Memory-efficient version that processes one layer at a time.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    head_scores = {}
    
    with torch.no_grad():
        # Forward pass with attention output
        outputs = model(inputs.input_ids, output_attentions=True, use_cache=False)
        attentions = outputs.attentions
        
        num_layers = len(attentions)
        num_heads = attentions[0].shape[1]
        seq_len = attentions[0].shape[2]
        
        # Adjust context boundaries
        context_start = min(context_start_approx, seq_len - 1)
        context_end = min(context_end_approx, seq_len)
        
        for layer_idx, layer_attn in enumerate(attentions):
            # layer_attn: (batch, heads, seq, seq)
            attn = layer_attn[0].float()  # (heads, seq, seq)
            
            for head_idx in range(num_heads):
                # Attention from last position to context region
                last_pos_attn = attn[head_idx, -1, context_start:context_end]
                score = last_pos_attn.sum().item()
                head_scores[(layer_idx, head_idx)] = score
        
        # Clear memory
        del attentions
        del outputs
        torch.cuda.empty_cache()
    
    return head_scores

# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_experiment(model_key, context_type, samples, save_dir, skip_attention=False):
    """Run experiment for one model and context type."""
    model_name = MODELS[model_key]
    config = CONTEXT_CONFIGS[context_type]
    
    print(f"\n{'='*70}")
    print(f"Running: {model_key} with {context_type} context (~{config['target_tokens']} tokens)")
    print(f"{'='*70}\n")
    
    model, tokenizer = load_model(model_name)
    
    results = []
    all_head_scores = defaultdict(list)
    correct_count = 0
    total_evaluated = 0
    
    for sample in tqdm(samples, desc=f"{model_key}/{context_type}"):
        # Create context
        if context_type == "short":
            context, num_tokens = create_short_context(sample, tokenizer, config['target_tokens'])
        else:
            context, num_tokens = create_long_context(sample, tokenizer, config['target_tokens'])
        
        prompt = QUESTION_TEMPLATE.format(context=context)
        
        # Run inference
        try:
            answer = run_inference(model, tokenizer, prompt)
        except Exception as e:
            print(f"Inference error: {e}")
            continue
        
        # Check accuracy
        is_correct = check_accuracy(answer, sample['ground_truth_state'])
        total_evaluated += 1
        if is_correct:
            correct_count += 1
        
        # Analyze attention heads (skip for long context to save memory)
        head_scores = {}
        if not skip_attention and context_type == "short":
            try:
                # Estimate context position in prompt
                pre_context = "Based on the following SEC 10-K filing excerpt, answer the question.\n\nContext:\n"
                context_start = len(tokenizer.encode(pre_context, add_special_tokens=False))
                context_end = context_start + num_tokens
                
                head_scores = analyze_attention_heads(
                    model, tokenizer, prompt, context_start, context_end
                )
                
                for (layer, head), score in head_scores.items():
                    all_head_scores[(layer, head)].append(score)
            except Exception as e:
                pass  # Skip attention analysis on error
        
        results.append({
            'filename': sample['filename'],
            'answer': answer,
            'ground_truth': sample['ground_truth_state'],
            'correct': is_correct,
            'context_tokens': num_tokens,
        })
    
    # Calculate accuracy
    accuracy = correct_count / total_evaluated if total_evaluated > 0 else 0
    
    # Get top retrieval heads
    avg_head_scores = {}
    if all_head_scores:
        avg_head_scores = {
            head: np.mean(scores) for head, scores in all_head_scores.items()
        }
    top_heads = sorted(avg_head_scores.items(), key=lambda x: x[1], reverse=True)[:20]
    
    # Save results
    output = {
        'model': model_key,
        'model_name': model_name,
        'context_type': context_type,
        'target_tokens': config['target_tokens'],
        'accuracy': accuracy,
        'correct': correct_count,
        'total': total_evaluated,
        'top_retrieval_heads': [(f"L{l}H{h}", score) for (l, h), score in top_heads],
        'results': results,
    }
    
    output_path = os.path.join(save_dir, f"{model_key}_{context_type}_results.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"Results: {model_key} - {context_type}")
    print(f"{'='*50}")
    print(f"Accuracy: {accuracy:.1%} ({correct_count}/{total_evaluated})")
    if top_heads:
        print(f"Top 5 Retrieval Heads:")
        for (layer, head), score in top_heads[:5]:
            print(f"  L{layer}H{head}: {score:.4f}")
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
    
    return output

# =============================================================================
# VISUALIZATION
# =============================================================================

def create_comparison_plot(results_dir):
    """Create accuracy comparison visualization."""
    
    # Load results
    results = {}
    for model_key in MODELS.keys():
        for ctx in ['short', 'long']:
            path = os.path.join(results_dir, f"{model_key}_{ctx}_results.json")
            if os.path.exists(path):
                with open(path) as f:
                    results[f"{model_key}_{ctx}"] = json.load(f)
    
    if not results:
        print("No results to visualize")
        return
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Retrieval Head Analysis: State of Incorporation Task', fontsize=14, fontweight='bold')
    
    # Plot 1: Accuracy comparison
    ax1 = axes[0]
    
    models = list(MODELS.keys())
    x = np.arange(len(models))
    width = 0.35
    
    short_acc = []
    long_acc = []
    for m in models:
        short_acc.append(results.get(f"{m}_short", {}).get('accuracy', 0) * 100)
        long_acc.append(results.get(f"{m}_long", {}).get('accuracy', 0) * 100)
    
    bars1 = ax1.bar(x - width/2, short_acc, width, label='Short (~200 tokens)', color='#27ae60', edgecolor='black')
    bars2 = ax1.bar(x + width/2, long_acc, width, label='Long (~2000 tokens)', color='#c0392b', edgecolor='black')
    
    # Expected lines
    ax1.axhline(y=70, color='#27ae60', linestyle='--', alpha=0.5, linewidth=2)
    ax1.axhline(y=40, color='#c0392b', linestyle='--', alpha=0.5, linewidth=2)
    ax1.text(len(models)-0.1, 72, 'Expected ~70%', color='#27ae60', fontsize=9)
    ax1.text(len(models)-0.1, 42, 'Expected ~40%', color='#c0392b', fontsize=9)
    
    ax1.set_ylabel('Accuracy (%)', fontsize=11)
    ax1.set_title('Accuracy by Context Length', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels([MODELS[m].split('/')[-1] for m in models], fontsize=10)
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.0f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=11, fontweight='bold')
    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(f'{height:.0f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=11, fontweight='bold')
    
    # Plot 2: Accuracy drop
    ax2 = axes[1]
    
    drops = [s - l for s, l in zip(short_acc, long_acc)]
    colors = ['#e74c3c' if d > 0 else '#27ae60' for d in drops]
    bars = ax2.bar(x, drops, width=0.5, color=colors, edgecolor='black')
    
    ax2.set_ylabel('Accuracy Drop (%)', fontsize=11)
    ax2.set_title('Performance Degradation (Short → Long)', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels([MODELS[m].split('/')[-1] for m in models], fontsize=10)
    ax2.axhline(y=0, color='black', linewidth=1)
    ax2.axhline(y=30, color='gray', linestyle='--', alpha=0.5)
    ax2.text(len(models)-0.1, 32, 'Expected ~30% drop', color='gray', fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, drop in zip(bars, drops):
        height = bar.get_height()
        ax2.annotate(f'{drop:.0f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3 if height > 0 else -12), textcoords='offset points', 
                    ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'accuracy_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {results_dir}/accuracy_comparison.png")

def print_summary(results_dir):
    """Print summary of all results."""
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    
    for model_key in MODELS.keys():
        print(f"\n{MODELS[model_key].split('/')[-1]}:")
        
        for ctx in ['short', 'long']:
            path = os.path.join(results_dir, f"{model_key}_{ctx}_results.json")
            if os.path.exists(path):
                with open(path) as f:
                    data = json.load(f)
                print(f"  {ctx.capitalize():6} context: {data['accuracy']*100:.1f}% ({data['correct']}/{data['total']})")

# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['llama', 'qwen', 'all'], default='all')
    parser.add_argument('--context', type=str, choices=['short', 'long', 'all'], default='all')
    parser.add_argument('--samples', type=int, default=NUM_SAMPLES)
    parser.add_argument('--skip-attention', action='store_true', help='Skip attention analysis')
    parser.add_argument('--visualize-only', action='store_true')
    
    args = parser.parse_args()
    
    save_dir = os.path.dirname(os.path.abspath(__file__))
    
    if args.visualize_only:
        create_comparison_plot(save_dir)
        print_summary(save_dir)
        return
    
    # Load ground truth and samples
    ground_truth = load_ground_truth()
    samples = load_edgar_samples(args.samples, ground_truth)
    
    if len(samples) < args.samples:
        print(f"Warning: Only found {len(samples)} samples with ground truth")
    
    # Run experiments
    models_to_run = list(MODELS.keys()) if args.model == 'all' else [args.model]
    contexts_to_run = list(CONTEXT_CONFIGS.keys()) if args.context == 'all' else [args.context]
    
    for model in models_to_run:
        for context in contexts_to_run:
            run_experiment(model, context, samples, save_dir, skip_attention=args.skip_attention)
    
    # Create visualizations
    create_comparison_plot(save_dir)
    print_summary(save_dir)

if __name__ == "__main__":
    main()

