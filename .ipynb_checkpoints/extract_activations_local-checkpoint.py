"""
Extract activations (residual stream) AND apply logit lens to a model.

Shows both:
1. Raw activation statistics (mean, std, L2 norm)
2. Logit lens: what tokens the model "thinks" at each layer

Works on Apple Silicon (M1/M2/M3/M4) using MPS acceleration.
Also works on Lambda GPU (CUDA) for larger models like Llama-3.1-8B.

Usage:
    pip install torch transformers accelerate
    python extract_activations_local.py
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import pretty_print as pp

# --- Configuration ---
# Choose a model (uncomment one):

# Small, fast, no approval needed:
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # ~2GB, very fast

# Medium, good quality, no approval needed:
# MODEL_NAME = "microsoft/phi-2"  # ~6GB, good for testing

# Large, best quality, requires HF approval (use on Lambda GPU):
# MODEL_NAME = "meta-llama/Llama-3.1-8B"  # ~16GB

PROMPT = "How many letters does 'cat' have?"
TARGET_LAYER = 15  # Layer to extract (adjust based on model)
TOP_K = 10  # Number of top predictions to show in logit lens

# Set to False for cleaner logit lens analysis (matches GPT-2 notebook style)
# Set to True if you want the model to generate good responses
USE_CHAT_TEMPLATE = False


def get_device_and_dtype():
    """Get the best available device and compatible dtype."""
    if torch.backends.mps.is_available():
        # MPS doesn't support bfloat16, use float32
        return torch.device("mps"), torch.float32
    elif torch.cuda.is_available():
        # CUDA supports float16 for efficiency
        return torch.device("cuda"), torch.float16
    else:
        return torch.device("cpu"), torch.float32


def get_embedding_matrix(model):
    """Extract the token embedding matrix from the model."""
    # Different models store embeddings differently
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        # Llama, TinyLlama, Mistral style
        return model.model.embed_tokens.weight
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):
        # GPT-2 style
        return model.transformer.wte.weight
    elif hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'embed_in'):
        # GPT-NeoX style
        return model.gpt_neox.embed_in.weight
    else:
        raise ValueError(f"Cannot find embedding matrix for model type: {type(model)}")


def get_final_layer_norm(model):
    """Get the final layer norm from the model."""
    if hasattr(model, 'model') and hasattr(model.model, 'norm'):
        # Llama, TinyLlama, Mistral style (RMSNorm)
        return model.model.norm
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'ln_f'):
        # GPT-2 style
        return model.transformer.ln_f
    elif hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'final_layer_norm'):
        # GPT-NeoX style
        return model.gpt_neox.final_layer_norm
    else:
        return None


def apply_layer_norm(hidden_state, layer_norm=None):
    """
    Apply layer normalization to hidden state.
    If no layer_norm module provided, use simple RMS normalization.
    """
    if layer_norm is not None:
        # Use model's actual layer norm
        return layer_norm(hidden_state)
    else:
        # Fallback: simple RMS normalization (like GPT-2's fixed_norm)
        # Normalize to mean=0, std=1
        hidden = hidden_state.float()
        mean = hidden.mean(dim=-1, keepdim=True)
        var = ((hidden - mean) ** 2).mean(dim=-1, keepdim=True)
        hidden_norm = (hidden - mean) / torch.sqrt(var + 1e-5)
        return hidden_norm


def apply_logit_lens(hidden_state, embedding_matrix, tokenizer, layer_norm=None, top_k=10):
    """
    Apply logit lens: project hidden state to vocabulary space.
    
    This mirrors the GPT-2 logit lens implementation:
    1. Apply layer normalization to the hidden state
    2. Project to vocabulary space by multiplying by embedding matrix transpose
    3. Apply softmax to get probabilities
    
    Args:
        hidden_state: [hidden_dim] tensor - activation for one token position
        embedding_matrix: [vocab_size, hidden_dim] tensor
        tokenizer: for decoding token IDs to strings
        layer_norm: optional layer norm module to apply
        top_k: number of top predictions to return
    
    Returns:
        list of (token_string, probability, logit) tuples
    """
    # Step 1: Apply layer normalization (critical for logit lens!)
    # This is what the GPT-2 implementation does with fixed_norm or ln_f
    hidden_normed = apply_layer_norm(hidden_state, layer_norm)
    
    # Step 2: Project to vocabulary space: hidden @ embedding.T
    logits = hidden_normed.float() @ embedding_matrix.T.float()
    
    # Step 3: Apply softmax to get probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Get top-k predictions
    top_probs, top_indices = torch.topk(probs, top_k)
    top_logits = logits[top_indices]
    
    results = []
    for i in range(top_k):
        token_id = top_indices[i].item()
        token_str = tokenizer.decode([token_id])
        prob = top_probs[i].item()
        logit = top_logits[i].item()
        results.append((token_str, prob, logit))
    
    return results


def main():
    device, dtype = get_device_and_dtype()
    
    # Header
    pp.header("🔬 Activation Extraction (Local)")
    
    # Load tokenizer
    pp.subheader("Loading Model")
    pp.progress("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    pp.success("Tokenizer loaded")
    
    # Load model
    pp.progress("Loading model (this may take a minute)...")
    
    # On MPS (Mac), we need to be careful about dtype
    # Some models default to bfloat16 which MPS doesn't support
    if device.type == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,  # Force float32 for MPS
            device_map={"": device},    # Explicit device mapping
            low_cpu_mem_usage=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=dtype,
            device_map="auto",
        )
    pp.success("Model loaded")
    
    # Get number of layers
    num_layers = getattr(model.config, 'num_hidden_layers', 32)
    target = min(TARGET_LAYER, num_layers - 1)
    if TARGET_LAYER >= num_layers:
        pp.warning(f"TARGET_LAYER ({TARGET_LAYER}) >= num_layers ({num_layers}), using layer {target}")
    
    # Show model info
    pp.model_info_box(
        model_name=MODEL_NAME.split("/")[-1],
        device=str(device),
        dtype=str(dtype).replace("torch.", ""),
        num_layers=num_layers,
        target_layer=target
    )
    
    # Format prompt (optionally as chat)
    if USE_CHAT_TEMPLATE and hasattr(tokenizer, 'apply_chat_template'):
        messages = [{"role": "user", "content": PROMPT}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
    else:
        formatted_prompt = PROMPT
    
    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    token_ids = inputs['input_ids'][0].tolist()
    
    pp.tokens_display(tokens, token_ids)
    
    # Forward pass
    pp.subheader("Forward Pass")
    pp.progress("Running inference...")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    pp.success(f"Captured {len(outputs.hidden_states)} hidden state tensors")
    
    # Get target layer activation
    layer_activation = outputs.hidden_states[target + 1]
    
    # Show activation info
    pp.activation_box(
        layer=target,
        shape=f"{layer_activation.shape[0]} × {layer_activation.shape[1]} × {layer_activation.shape[2]}",
        dtype=str(layer_activation.dtype).replace("torch.", "")
    )
    
    # Token statistics table
    pp.subheader("Per-Token Activation Statistics")
    pp.token_table_header()
    
    for pos in range(layer_activation.shape[1]):
        token = tokens[pos]
        activation = layer_activation[0, pos, :].float()
        pp.token_table_row(
            pos=pos,
            token=token,
            mean=activation.mean().item(),
            std=activation.std().item(),
            min_val=activation.min().item(),
            max_val=activation.max().item(),
            l2_norm=activation.norm().item(),
            highlight=(pos == len(tokens) - 1)
        )
    
    # Raw values for last token
    last_activation = layer_activation[0, -1, :].float()
    pp.raw_values_box(
        token=tokens[-1],
        first_values=last_activation[:10].tolist(),
        last_values=last_activation[-10:].tolist(),
        dim=layer_activation.shape[2]
    )
    
    # =========================================================================
    # LOGIT LENS: Decode hidden states to vocabulary predictions
    # =========================================================================
    pp.header("🔮 Logit Lens Analysis")
    
    # Get embedding matrix and layer norm (needed for proper logit lens)
    pp.progress("Getting embedding matrix and layer norm...")
    embedding_matrix = get_embedding_matrix(model)
    layer_norm = get_final_layer_norm(model)
    pp.success(f"Embedding matrix: {embedding_matrix.shape}")
    pp.success(f"Layer norm: {type(layer_norm).__name__ if layer_norm else 'RMS (fallback)'}")
    
    # Apply logit lens to last token at target layer
    pp.subheader(f"Top Predictions at Layer {target} (last token: '{tokens[-1]}')")
    
    predictions = apply_logit_lens(
        hidden_state=layer_activation[0, -1, :],
        embedding_matrix=embedding_matrix,
        tokenizer=tokenizer,
        layer_norm=layer_norm,
        top_k=TOP_K
    )
    
    pp.logit_lens_table(predictions)
    
    # Show how predictions evolve across layers
    pp.subheader("Prediction Evolution Across Layers (last token)")
    pp.progress("Analyzing all layers...")
    
    layer_predictions = []
    for layer_idx in range(num_layers + 1):
        hidden = outputs.hidden_states[layer_idx][0, -1, :]  # Last token
        preds = apply_logit_lens(hidden, embedding_matrix, tokenizer, layer_norm, top_k=1)
        top_token, top_prob, _ = preds[0]
        layer_predictions.append((layer_idx, top_token, top_prob))
    
    pp.layer_evolution_table(layer_predictions, target_layer=target)
    
    # =========================================================================
    # TRACK SPECIFIC ANSWER TOKENS (like GPT-2 notebook)
    # =========================================================================
    # Track tokens that might be the "answer" - when do they appear?
    answer_tokens_to_track = ["3", "three", "Three", " 3", " three", " Three", "The", " The"]
    
    pp.subheader("Answer Token Tracking Across Layers")
    print("  Tracking when answer-related tokens appear in predictions...")
    print()
    
    # Find token IDs for answer tokens
    answer_token_ids = {}
    for tok in answer_tokens_to_track:
        ids = tokenizer.encode(tok, add_special_tokens=False)
        if ids:
            answer_token_ids[tok] = ids[0]
    
    # Track rank of each answer token at each layer
    print(f"  {'Layer':<8}", end="")
    for tok in answer_token_ids.keys():
        print(f"{repr(tok):>10}", end="")
    print("  (rank in vocab)")
    print("  " + "─" * (8 + 10 * len(answer_token_ids) + 15))
    
    for layer_idx in range(num_layers + 1):
        hidden = outputs.hidden_states[layer_idx][0, -1, :]
        
        # Apply layer norm and get logits
        if layer_norm is not None:
            hidden_normed = layer_norm(hidden.unsqueeze(0)).squeeze(0)
        else:
            hidden_normed = hidden
        logits = torch.matmul(hidden_normed, embedding_matrix.T)
        probs = torch.softmax(logits, dim=-1)
        
        # Get ranks for each answer token
        layer_name = "embed" if layer_idx == 0 else f"L{layer_idx-1}"
        if layer_idx - 1 == target:
            print(f"  \033[96m\033[1m→ {layer_name:<6}\033[0m", end="")
        else:
            print(f"  {layer_name:<8}", end="")
        
        for tok, tok_id in answer_token_ids.items():
            # Rank = how many tokens have higher probability
            rank = (probs > probs[tok_id]).sum().item() + 1
            if rank <= 10:
                print(f"\033[92m\033[1m{rank:>10}\033[0m", end="")
            elif rank <= 100:
                print(f"\033[93m{rank:>10}\033[0m", end="")
            else:
                print(f"\033[2m{rank:>10}\033[0m", end="")
        print()
    
    print()
    print("  \033[92m\033[1mGreen\033[0m = top 10, \033[93mYellow\033[0m = top 100, \033[2mGray\033[0m = lower")
    print()
    
    # =========================================================================
    # GENERATE MODEL OUTPUT
    # =========================================================================
    pp.subheader("Model Generation")
    pp.progress("Generating response...")
    
    with torch.no_grad():
        gen_outputs = model.generate(
            inputs['input_ids'],
            attention_mask=inputs.get('attention_mask'),
            max_new_tokens=50,
            do_sample=False,  # Greedy decoding for deterministic output
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode the full output
    full_output = tokenizer.decode(gen_outputs[0], skip_special_tokens=True)
    
    # Get the new tokens (everything after the input)
    input_length = inputs['input_ids'].shape[1]
    new_tokens = gen_outputs[0][input_length:]
    response_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    
    # If empty, show the full decoded output for debugging
    if not response_text:
        response_text = full_output[len(formatted_prompt):].strip() if len(full_output) > len(formatted_prompt) else "(no new tokens generated)"
    
    pp.model_output(prompt=PROMPT, response=response_text)
    
    # Summary
    pp.final_summary(
        prompt=PROMPT,
        layer=target,
        shape=tuple(layer_activation.shape),
        response=response_text
    )
    
    return layer_activation, predictions, response_text


if __name__ == "__main__":
    activation, predictions, response = main()
