# -*- coding: utf-8 -*-
"""
The Logit Lens on Llama Activations

This script replicates the logit lens technique from nostalgebraist's work on GPT-2
(https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)
but adapted for Meta's Llama models using HuggingFace Transformers.

The logit lens technique:
- At each layer, project the hidden state back to vocabulary space
- This reveals what the model is "thinking" at intermediate layers
- We can see how predictions evolve from input to output

Key insight from the original work:
- The model immediately converts inputs into tentative predictions
- Later layers refine these predictions toward the final output
- The model "thinks" in prediction space, not input space

Requirements:
    pip install torch transformers accelerate matplotlib pandas numpy scipy tqdm colorcet

For Llama-3.1-8B, you need:
    1. HuggingFace account with access approved at https://huggingface.co/meta-llama/Llama-3.1-8B
    2. huggingface-cli login
    3. GPU with ~16GB+ VRAM (or use Lambda Labs)

Author: Adapted from nostalgebraist's GPT-2 logit lens notebook
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# Optional: for nicer colormaps
try:
    import colorcet
    HAS_COLORCET = True
except ImportError:
    HAS_COLORCET = False
    print("Note: colorcet not installed, using default colormaps")


# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_NAME = "meta-llama/Llama-3.1-8B"  # Requires HF approval (use on Lambda GPU)
# MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # For local testing
# MODEL_NAME = "microsoft/phi-2"

# Device configuration
DEVICE = None  # Will be auto-detected
DTYPE = None   # Will be auto-detected

# Max tokens to process
MAX_TOKENS = 200


# =============================================================================
# MODEL LOADING
# =============================================================================

def get_device_and_dtype():
    """Auto-detect the best device and dtype."""
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    elif torch.backends.mps.is_available():
        return torch.device("mps"), torch.float32  # MPS doesn't support bfloat16
    else:
        return torch.device("cpu"), torch.float32


def load_model_and_tokenizer(model_name):
    """Load the model and tokenizer."""
    global DEVICE, DTYPE
    DEVICE, DTYPE = get_device_and_dtype()
    
    print(f"Loading model: {model_name}")
    print(f"Device: {DEVICE}, Dtype: {DTYPE}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model with appropriate settings
    if DEVICE.type == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map={"": DEVICE},
            low_cpu_mem_usage=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=DTYPE,
            device_map="auto",
        )
    
    model.eval()
    return model, tokenizer


# Global model and tokenizer (loaded once)
model = None
tokenizer = None


def ensure_model_loaded():
    """Ensure model is loaded (lazy loading)."""
    global model, tokenizer
    if model is None:
        model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
    return model, tokenizer


# =============================================================================
# CORE LOGIT LENS IMPLEMENTATION
# =============================================================================

def get_embedding_matrix(model):
    """Get the token embedding matrix (used to project hidden states to vocab)."""
    # For Llama, the embedding matrix is in model.model.embed_tokens
    # The LM head (model.lm_head) projects back to vocab space
    # These are NOT tied in Llama, so we use lm_head for projection
    return model.lm_head.weight.detach()


def get_layer_norm(model):
    """Get the final layer norm (applied before projection to vocab)."""
    return model.model.norm


def internal_token_dists(text,
                         layer_nums=None,
                         max_tokens_to_return=MAX_TOKENS,
                         return_logits=True,
                         return_probs=True,
                         return_argmaxes=True,
                         return_activations=False):
    """
    Get token distributions at each layer (the core logit lens operation).
    
    This is the Llama equivalent of the GPT-2 `internal_token_dists` function.
    
    For each layer, we:
    1. Take the hidden state
    2. Apply layer normalization (same as the final layer norm)
    3. Project to vocabulary space using the embedding matrix
    4. Get logits, probabilities, and argmax predictions
    
    Args:
        text: Input text string
        layer_nums: Which layers to return (None = all)
        max_tokens_to_return: Truncate to this many tokens
        return_logits: Include logits in results
        return_probs: Include probabilities in results
        return_argmaxes: Include argmax token indices in results
        return_activations: Include raw activations in results
    
    Returns:
        results: Dict with 'logits', 'probs', 'argmaxes', 'activations' lists
        layer_names: List of layer names (h_in, h0_out, h1_out, ..., h_out)
        tokens: Token IDs
    """
    model, tok = ensure_model_loaded()
    
    # Tokenize
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=max_tokens_to_return)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    tokens = inputs['input_ids'][0].tolist()
    
    # Forward pass with hidden states
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True
        )
    
    # Get embedding matrix and layer norm
    embed_matrix = get_embedding_matrix(model)
    layer_norm = get_layer_norm(model)
    
    # Process each layer's hidden states
    hidden_states = outputs.hidden_states  # Tuple of (batch, seq, hidden_dim)
    num_layers = len(hidden_states) - 1  # -1 because first is embeddings
    
    results = defaultdict(list)
    layer_names = []
    
    for layer_idx, h in enumerate(hidden_states):
        # Determine which layers to include
        if layer_nums is not None:
            if layer_idx == 0:
                pass  # Always include input embeddings
            elif layer_idx == len(hidden_states) - 1:
                pass  # Always include final output
            elif (layer_idx - 1) not in layer_nums:
                continue
        
        # Layer name
        if layer_idx == 0:
            layer_name = "h_in"
        elif layer_idx == len(hidden_states) - 1:
            layer_name = "h_out"
        else:
            layer_name = f"h{layer_idx-1}_out"
        layer_names.append(layer_name)
        
        # Get hidden state for first batch element
        h_seq = h[0]  # (seq_len, hidden_dim)
        
        # Apply layer norm before projecting to vocab
        # This is crucial - the GPT-2 notebook does this too
        h_normed = layer_norm(h_seq)
        
        # Project to vocabulary space: logits = h @ W^T
        logits = torch.matmul(h_normed, embed_matrix.T)  # (seq_len, vocab_size)
        
        if return_logits:
            results['logits'].append(logits.detach().cpu().numpy())
        
        if return_probs:
            probs = F.softmax(logits, dim=-1)
            results['probs'].append(probs.detach().cpu().numpy())
        
        if return_argmaxes:
            argmaxes = torch.argmax(logits, dim=-1)
            results['argmaxes'].append(argmaxes.detach().cpu().numpy())
        
        if return_activations:
            results['activations'].append(h_seq.detach().cpu().numpy())
    
    return results, layer_names, tokens


# =============================================================================
# HELPER FUNCTIONS (matching GPT-2 notebook interface)
# =============================================================================

def token_string(token_ix):
    """Convert token index to string representation."""
    _, tok = ensure_model_loaded()
    return repr(tok.decode([token_ix]))


def kl_div(p, q, axis=-1):
    """KL divergence between distributions p and q."""
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return np.sum(p * np.log(p / q), axis=axis)


def max_logit_frame(results, layer_names):
    """DataFrame of max logit at each position for each layer."""
    return pd.DataFrame(
        [l.max(axis=-1) for l in results['logits']],
        index=layer_names
    )


def max_p_frame(results, layer_names):
    """DataFrame of max probability at each position for each layer."""
    return pd.DataFrame(
        [p.max(axis=-1) for p in results['probs']],
        index=layer_names
    )


def argmax_frame(results, layer_names):
    """DataFrame of argmax token index at each position for each layer."""
    return pd.DataFrame(
        [[amx for amx in amaxes] for amaxes in results['argmaxes']],
        index=layer_names
    )


def argmax_token_frame(results, layer_names):
    """DataFrame of argmax token string at each position for each layer."""
    return pd.DataFrame(
        [[token_string(amx) for amx in amaxes] for amaxes in results['argmaxes']],
        index=layer_names
    )


def true_token_frame(results, layer_names, tokens):
    """DataFrame showing the true next token at each position."""
    tses = [token_string(t) for t in tokens[1:]]
    return pd.DataFrame(
        [tses for _ in layer_names],
        index=layer_names
    )


def logit_of_tokens_frame(results, layer_names, tokens, next_token=True):
    """DataFrame of logit values for specific tokens at each layer."""
    if next_token:
        iter_pairs = list(zip(range(len(tokens)-1), range(1, len(tokens))))
    else:
        iter_pairs = list(zip(range(len(tokens)), range(len(tokens))))
    return pd.DataFrame(
        [[l[i, tokens[j]] for i, j in iter_pairs] for l in results['logits']],
        index=layer_names
    )


def p_of_tokens_frame(results, layer_names, tokens, next_token=True):
    """DataFrame of probability values for specific tokens at each layer."""
    if next_token:
        iter_pairs = list(zip(range(len(tokens)-1), range(1, len(tokens))))
    else:
        iter_pairs = list(zip(range(len(tokens)), range(len(tokens))))
    return pd.DataFrame(
        [[p[i, tokens[j]] for i, j in iter_pairs] for p in results['probs']],
        index=layer_names
    )


def rank_of_tokens_frame(results, layer_names, tokens, next_token=True):
    """DataFrame of rank (1=best) of specific tokens at each layer."""
    if next_token:
        iter_pairs = list(zip(range(len(tokens)-1), range(1, len(tokens))))
    else:
        iter_pairs = list(zip(range(len(tokens)), range(len(tokens))))
    return pd.DataFrame(
        [[(p[i, :] >= p[i, tokens[j]]).sum() for i, j in iter_pairs] for p in results['probs']],
        index=layer_names
    )


def rank_of_tokens_from_layer_frame(results, layer_names, layer_name):
    """Get rank of final layer's predictions at each earlier layer."""
    finals = argmax_frame(results, layer_names).loc[layer_name, :].values
    return rank_of_tokens_frame(results, layer_names, finals, next_token=False)


def compare_to_layer_frame(results, layer_names, layer_name,
                           key="probs", compare_fn=kl_div, next_token=False):
    """Compare each layer's distribution to a reference layer using compare_fn."""
    compare_to = results[key][list(layer_names).index(layer_name)]
    if next_token:
        compare_to = compare_to[1:, :]
    return pd.DataFrame(
        [[compare_fn(compare_to[i, :], entry[i, :]) for i in range(compare_to.shape[0])]
         for entry in tqdm(results[key], desc="Comparing layers")],
        index=layer_names
    )


# =============================================================================
# WRAPPER FUNCTION (like run_example in GPT-2 notebook)
# =============================================================================

def run_example(text, max_tokens=MAX_TOKENS):
    """
    Run the logit lens analysis on a text.
    
    Args:
        text: Input text string
        max_tokens: Maximum tokens to process
    
    Returns:
        tokens: List of token IDs
        results: Dict with logits, probs, argmaxes at each layer
        layer_names: List of layer names
    """
    results, layer_names, tokens = internal_token_dists(
        text,
        max_tokens_to_return=max_tokens,
        return_logits=True,
        return_probs=True,
        return_argmaxes=True,
    )
    return tokens, results, layer_names


# =============================================================================
# LAYER DECISION ANALYSIS
# =============================================================================

def numeric_layer_name(name, layer_names):
    """Convert layer name to numeric index."""
    if name == "h_in":
        return 0
    if name == "h_out":
        return len(layer_names) - 1
    return int(name.split("_")[0].lstrip("h")) + 1


def layer_where_decision_finalizes(results, layer_names):
    """Find the layer where the final prediction first becomes stable."""
    ranks = rank_of_tokens_from_layer_frame(results, layer_names, 'h_out')
    result = (ranks.diff() != 0).iloc[::-1, :].idxmax(axis=0)
    return result.apply(lambda x: numeric_layer_name(x, layer_names))


def layer_where_decision_first_made(results, layer_names):
    """Find the first layer that predicts the final token."""
    ranks = rank_of_tokens_from_layer_frame(results, layer_names, 'h_out')
    result = (ranks == 1).idxmax(axis=0)
    return result.apply(lambda x: numeric_layer_name(x, layer_names))


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_decisions(results, layer_names, start_token=None, end_token=None):
    """Plot where decisions are made across layers."""
    if start_token is not None and end_token is not None:
        ntok = end_token - start_token
    else:
        ntok = len(results['probs'][0])
    
    plt.figure(figsize=(max(8, 0.1 * ntok), 6))
    
    # Layer where decision finalizes
    to_show = layer_where_decision_finalizes(results, layer_names)
    if start_token is not None and end_token is not None:
        to_show = to_show.iloc[start_token:end_token]
    plt.plot(to_show, marker='o', label="top1 token finalized")
    
    # Layer where decision first made
    to_show = layer_where_decision_first_made(results, layer_names)
    if start_token is not None and end_token is not None:
        to_show = to_show.iloc[start_token:end_token]
    plt.plot(to_show, marker='o', label="top1 token first match")
    
    plt.ylim(0, len(layer_names))
    plt.xlabel("Position")
    plt.ylabel("Layer #")
    plt.legend()
    plt.title("Where the decision (top1 token at end) gets made")


def show_token_progress(results, layer_names, tokens,
                        start_token, end_token=None,
                        kind="prediction",
                        colors_mean="prob",
                        cell_text_is="tokens",
                        layer_step=1,
                        only_show_tokens_at_changes=True):
    """
    Create heatmap showing how token predictions evolve across layers.
    
    This is the main visualization from the logit lens paper.
    """
    if end_token is None:
        end_token = len(tokens)
    
    # Select layers to show
    def _step_through_layers(array, names):
        indices = list(range(0, len(names) - 1, layer_step)) + [len(names) - 1]
        return [array[i] for i in indices], [names[i] for i in indices]
    
    stepped_data, stepped_names = _step_through_layers(
        list(range(len(layer_names))), layer_names
    )
    
    # Get token frame and color data
    token_df = argmax_token_frame(results, layer_names)
    
    if kind == "prediction":
        if colors_mean == "prob":
            colors_df = max_p_frame(results, layer_names)
            vmin, vmax = 0, 1
            cmap = "Blues_r"
        elif colors_mean == "logit":
            colors_df = max_logit_frame(results, layer_names)
            vmin, vmax = 0, colors_df.loc['h_out'].values[start_token:end_token].max()
            cmap = "viridis"
        elif colors_mean == "rank":
            colors_df = rank_of_tokens_from_layer_frame(results, layer_names, 'h_out')
            vmin, vmax = 1, 100
            cmap = "Blues"
    elif kind == "truth":
        colors_df = rank_of_tokens_frame(results, layer_names, tokens)
        vmin, vmax = 1, 100
        cmap = "Blues"
    
    # Extract data for selected layers and tokens
    names_to_show = [layer_names[i] for i in stepped_data][::-1]
    array_to_plot = np.array([colors_df.loc[n].values[start_token:end_token] for n in names_to_show])
    array_tokens = np.array([token_df.loc[n].values[start_token:end_token] for n in names_to_show])
    
    # Token labels
    tokens_for_axis = [token_string(t) for t in tokens[start_token:end_token]]
    
    nx = len(tokens_for_axis)
    ny = len(names_to_show)
    
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(1.2 * nx, 0.4 * ny))
    
    if colors_mean == "rank":
        norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
        im = ax.imshow(array_to_plot, aspect="auto", cmap=cmap, norm=norm)
    else:
        im = ax.imshow(array_to_plot, vmin=vmin, vmax=vmax, aspect="auto", cmap=cmap)
    
    ax.set_xticks(range(len(tokens_for_axis)))
    ax.set_xticklabels(tokens_for_axis, fontsize=10, rotation=45, ha='right')
    ax.set_yticks(range(len(names_to_show)))
    ax.set_yticklabels(names_to_show, fontsize=10)
    
    # Add cell text
    for i in range(ny):
        for j in range(nx):
            if only_show_tokens_at_changes and i < ny - 1:
                if array_tokens[i, j] == array_tokens[i + 1, j]:
                    continue
            
            text = array_tokens[i, j] if cell_text_is == "tokens" else f"{array_to_plot[i, j]:.2f}"
            color = 'w' if array_to_plot[i, j] < (vmax + vmin) / 2 else 'k'
            ax.text(j, i, text, ha='center', va='center', fontsize=8, color=color)
    
    plt.colorbar(im, ax=ax, pad=0.02)
    
    title_map = {
        ("prediction", "prob"): "Model's top token and its probability",
        ("prediction", "logit"): "Model's top token and its logit",
        ("prediction", "rank"): "Rank of final prediction at each layer",
        ("truth", "rank"): "Rank of true next token at each layer",
    }
    plt.title(title_map.get((kind, colors_mean), "Token Progress"), fontsize=12)
    plt.tight_layout()


def plot_all(results, layer_names, tokens, start_token, end_token, layer_step=2):
    """Generate all standard plots for a text."""
    # Prediction probability heatmap
    show_token_progress(results, layer_names, tokens, start_token, end_token,
                        kind="prediction", colors_mean="prob", layer_step=layer_step)
    plt.show()
    
    # Prediction rank heatmap
    show_token_progress(results, layer_names, tokens, start_token, end_token,
                        kind="prediction", colors_mean="rank", layer_step=layer_step)
    plt.show()
    
    # Truth rank heatmap
    show_token_progress(results, layer_names, tokens, start_token, end_token,
                        kind="truth", colors_mean="rank", layer_step=layer_step)
    plt.show()
    
    # Decision layers plot
    plot_decisions(results, layer_names, start_token, end_token)
    plt.show()


# =============================================================================
# ACTIVATION EXTRACTION (for specific layer)
# =============================================================================

def get_activation_at_layer(text, target_layer=15):
    """
    Get the raw activation (residual stream) at a specific layer.
    
    This is the component that prints the activation at layer 15
    for the prompt "How many letters does 'cat' have?"
    
    Args:
        text: Input text
        target_layer: Which layer to extract (0-indexed transformer layers)
    
    Returns:
        activation: Tensor of shape (seq_len, hidden_dim)
        tokens: List of token strings
        token_ids: List of token IDs
    """
    model, tok = ensure_model_loaded()
    
    # Tokenize
    inputs = tok(text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    
    # hidden_states[0] = embeddings, hidden_states[i] = output of layer i-1
    # So for layer 15, we want hidden_states[16]
    num_layers = len(outputs.hidden_states) - 1
    layer_idx = min(target_layer + 1, num_layers)  # +1 because [0] is embeddings
    
    activation = outputs.hidden_states[layer_idx][0]  # (seq_len, hidden_dim)
    token_ids = inputs['input_ids'][0].tolist()
    token_strs = [tok.decode([t]) for t in token_ids]
    
    return activation.cpu(), token_strs, token_ids


def print_activation_stats(text, target_layer=15):
    """
    Print detailed statistics about the activation at a specific layer.
    
    This fulfills the requirement:
    "print the activation (residual stream) at layer 15 of Llama7b 
     for the prompt 'How many letters does 'cat' have?'"
    """
    print("=" * 70)
    print(f"ACTIVATION EXTRACTION: Layer {target_layer}")
    print("=" * 70)
    print(f"\nPrompt: {repr(text)}")
    
    activation, token_strs, token_ids = get_activation_at_layer(text, target_layer)
    
    print(f"\nTokens ({len(token_strs)}):")
    for i, (tok_str, tok_id) in enumerate(zip(token_strs, token_ids)):
        print(f"  [{i:2d}] {repr(tok_str):15s} -> {tok_id}")
    
    print(f"\nActivation shape: {activation.shape}")
    print(f"  - Sequence length: {activation.shape[0]}")
    print(f"  - Hidden dimension: {activation.shape[1]}")
    
    print(f"\nPer-token statistics:")
    print(f"{'Pos':<4} {'Token':<15} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'L2 Norm':>10}")
    print("-" * 75)
    
    for i, (tok_str, act) in enumerate(zip(token_strs, activation)):
        mean = act.mean().item()
        std = act.std().item()
        min_val = act.min().item()
        max_val = act.max().item()
        l2_norm = act.norm().item()
        print(f"{i:<4} {repr(tok_str):<15} {mean:>10.4f} {std:>10.4f} {min_val:>10.4f} {max_val:>10.4f} {l2_norm:>10.4f}")
    
    # Show raw values for the last token (the one predicting the answer)
    print(f"\nRaw activation values for last token {repr(token_strs[-1])}:")
    last_act = activation[-1]
    print(f"  First 10: {last_act[:10].tolist()}")
    print(f"  Last 10:  {last_act[-10:].tolist()}")
    
    return activation, token_strs, token_ids


# =============================================================================
# EXAMPLE TEXTS FOR ANALYSIS
# =============================================================================

# From GPT-2 notebook
gpt3_abstract = """Recent work has demonstrated substantial gains on many NLP tasks and benchmarks by pre-training
on a large corpus of text followed by fine-tuning on a specific task. While typically task-agnostic
in architecture, this method still requires task-specific fine-tuning datasets of thousands or tens of
thousands of examples. By contrast, humans can generally perform a new language task from only
a few examples or from simple instructions – something which current NLP systems still largely
struggle to do. Here we show that scaling up language models greatly improves task-agnostic,
few-shot performance, sometimes even reaching competitiveness with prior state-of-the-art finetuning approaches.""".replace("\n", " ")

plasma = """Sometimes, when people say plasma, they mean a state of matter. Other times, when people say plasma"""

plasma_repetitive = """I love plasma. I love plasma. I love plasma. I love plasma."""

# The specific prompt requested
cat_prompt = "How many letters does 'cat' have?"


# =============================================================================
# DATASET PLACEHOLDER
# =============================================================================

# TODO: Replace with your actual dataset
# The GPT-2 notebook uses example texts directly. 
# When you have a dataset, you can load it here:
#
# def load_dataset(path):
#     """Load your dataset for analysis."""
#     # Example formats:
#     # - JSON lines: [{"text": "..."}, ...]
#     # - CSV: text column
#     # - Plain text file: one example per line
#     pass
#
# DATASET_PATH = "path/to/your/dataset.jsonl"
# dataset = load_dataset(DATASET_PATH)
#
# Then iterate:
# for example in dataset:
#     tokens, results, layer_names = run_example(example["text"])
#     # ... analyze ...

DATASET = None  # Placeholder - set this to your dataset


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("THE LOGIT LENS ON LLAMA ACTIVATIONS")
    print("=" * 70)
    print(f"\nModel: {MODEL_NAME}")
    print()
    
    # Load model
    ensure_model_loaded()
    
    # =========================================================================
    # PART 1: Print activation at layer 15 for "cat" prompt
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 1: Activation at Layer 15")
    print("=" * 70)
    
    activation, token_strs, token_ids = print_activation_stats(cat_prompt, target_layer=15)
    
    # =========================================================================
    # PART 2: Full logit lens analysis on the cat prompt
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 2: Logit Lens Analysis")
    print("=" * 70)
    
    tokens, results, layer_names = run_example(cat_prompt)
    
    print(f"\nNumber of layers: {len(layer_names)}")
    print(f"Layer names: {layer_names[:5]} ... {layer_names[-3:]}")
    
    # Show top predictions at each layer for the last token
    print(f"\nTop prediction at each layer (for last token '{token_string(tokens[-1])}'):")
    print(f"{'Layer':<10} {'Top Token':<20} {'Probability':>12}")
    print("-" * 45)
    
    for i, layer_name in enumerate(layer_names):
        last_pos = len(tokens) - 1
        top_token_idx = results['argmaxes'][i][last_pos]
        top_prob = results['probs'][i][last_pos, top_token_idx]
        print(f"{layer_name:<10} {token_string(top_token_idx):<20} {top_prob:>12.4f}")
    
    # =========================================================================
    # PART 3: Visualization (if running interactively)
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 3: Generating Visualizations")
    print("=" * 70)
    
    try:
        # Plot all visualizations
        plot_all(results, layer_names, tokens, 
                 start_token=0, end_token=len(tokens)-1, layer_step=2)
        print("\nVisualizations generated successfully!")
    except Exception as e:
        print(f"\nVisualization error (may need display): {e}")
        print("Run in Jupyter/Colab for interactive plots, or save to files.")
    
    # =========================================================================
    # PART 4: Example with dataset (placeholder)
    # =========================================================================
    if DATASET is not None:
        print("\n" + "=" * 70)
        print("PART 4: Dataset Analysis")
        print("=" * 70)
        
        for i, example in enumerate(DATASET):
            if i >= 3:  # Limit for demo
                break
            print(f"\nExample {i+1}:")
            text = example if isinstance(example, str) else example.get("text", str(example))
            tokens, results, layer_names = run_example(text[:500])  # Truncate long texts
            print(f"  Tokens: {len(tokens)}")
            print(f"  Final prediction: {token_string(results['argmaxes'][-1][-1])}")
    else:
        print("\n" + "=" * 70)
        print("PART 4: Dataset Analysis (PLACEHOLDER)")
        print("=" * 70)
        print("\nNo dataset loaded. Set DATASET variable to analyze your data.")
        print("See the DATASET PLACEHOLDER section in the code for instructions.")
    
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

