"""
Extract activations (residual stream) AND apply logit lens to Llama on Lambda GPU.

Shows both:
1. Raw activation statistics (mean, std, L2 norm)
2. Logit lens: what tokens the model "thinks" at each layer

Requirements:
- GPU with at least 16GB VRAM (A10 24GB or A100 40GB works)
- pip install transformers torch accelerate
- HuggingFace account with Llama access (https://huggingface.co/meta-llama)

Usage:
    huggingface-cli login  # First time only
    python extract_llama_activations.py
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import pretty_print as pp

# --- Configuration ---
MODEL_NAME = "meta-llama/Llama-3.1-8B"  # Requires HF access approval
PROMPT = "How many letters does 'cat' have?"
TARGET_LAYER = 15  # Layer to extract (0-indexed, Llama-3.1-8B has 32 layers)
TOP_K = 10  # Number of top predictions to show in logit lens


def get_device_and_dtype():
    """Get the best available device and compatible dtype."""
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    elif torch.backends.mps.is_available():
        return torch.device("mps"), torch.float32
    else:
        return torch.device("cpu"), torch.float32


def get_embedding_matrix(model):
    """Extract the token embedding matrix from the model."""
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        return model.model.embed_tokens.weight
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):
        return model.transformer.wte.weight
    elif hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'embed_in'):
        return model.gpt_neox.embed_in.weight
    else:
        raise ValueError(f"Cannot find embedding matrix for model type: {type(model)}")


def get_final_layer_norm(model):
    """Get the final layer norm from the model."""
    if hasattr(model, 'model') and hasattr(model.model, 'norm'):
        return model.model.norm  # Llama RMSNorm
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'ln_f'):
        return model.transformer.ln_f  # GPT-2
    elif hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'final_layer_norm'):
        return model.gpt_neox.final_layer_norm
    else:
        return None


def apply_layer_norm(hidden_state, layer_norm=None):
    """Apply layer normalization to hidden state."""
    if layer_norm is not None:
        return layer_norm(hidden_state)
    else:
        # Fallback: simple normalization
        hidden = hidden_state.float()
        mean = hidden.mean(dim=-1, keepdim=True)
        var = ((hidden - mean) ** 2).mean(dim=-1, keepdim=True)
        return (hidden - mean) / torch.sqrt(var + 1e-5)


def apply_logit_lens(hidden_state, embedding_matrix, tokenizer, layer_norm=None, top_k=10):
    """
    Apply logit lens: project hidden state to vocabulary space.
    
    Key insight from GPT-2 implementation:
    1. Apply layer normalization first (critical!)
    2. Then project to vocabulary via embedding matrix transpose
    """
    # Apply layer norm (this is what makes logit lens work!)
    hidden_normed = apply_layer_norm(hidden_state, layer_norm)
    
    # Project to vocabulary space
    logits = hidden_normed.float() @ embedding_matrix.T.float()
    probs = F.softmax(logits, dim=-1)
    
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
    
    pp.header("🦙 Llama Activation & Logit Lens")
    
    # Load model
    pp.subheader("Loading Model")
    pp.progress("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    pp.success("Tokenizer loaded")
    
    pp.progress("Loading model (this may take a few minutes)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        device_map="auto",
    )
    pp.success("Model loaded")
    
    num_layers = getattr(model.config, 'num_hidden_layers', 32)
    target = min(TARGET_LAYER, num_layers - 1)
    if TARGET_LAYER >= num_layers:
        pp.warning(f"TARGET_LAYER ({TARGET_LAYER}) >= num_layers ({num_layers}), using layer {target}")
    
    pp.model_info_box(
        model_name=MODEL_NAME.split("/")[-1],
        device=str(device),
        dtype=str(dtype).replace("torch.", ""),
        num_layers=num_layers,
        target_layer=target
    )
    
    # Tokenize
    inputs = tokenizer(PROMPT, return_tensors="pt")
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
    
    layer_activation = outputs.hidden_states[target + 1]
    
    pp.activation_box(
        layer=target,
        shape=f"{layer_activation.shape[0]} × {layer_activation.shape[1]} × {layer_activation.shape[2]}",
        dtype=str(layer_activation.dtype).replace("torch.", "")
    )
    
    # Token statistics
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
    
    # Raw values
    last_activation = layer_activation[0, -1, :].float()
    pp.raw_values_box(
        token=tokens[-1],
        first_values=last_activation[:10].tolist(),
        last_values=last_activation[-10:].tolist(),
        dim=layer_activation.shape[2]
    )
    
    # LOGIT LENS
    pp.header("🔮 Logit Lens Analysis")
    
    pp.progress("Getting embedding matrix and layer norm...")
    embedding_matrix = get_embedding_matrix(model)
    layer_norm = get_final_layer_norm(model)
    pp.success(f"Embedding matrix: {embedding_matrix.shape}")
    pp.success(f"Layer norm: {type(layer_norm).__name__ if layer_norm else 'RMS (fallback)'}")
    
    pp.subheader(f"Top Predictions at Layer {target} (last token: '{tokens[-1]}')")
    
    predictions = apply_logit_lens(
        hidden_state=layer_activation[0, -1, :],
        embedding_matrix=embedding_matrix,
        tokenizer=tokenizer,
        layer_norm=layer_norm,
        top_k=TOP_K
    )
    
    pp.logit_lens_table(predictions)
    
    # Layer evolution
    pp.subheader("Prediction Evolution Across Layers (last token)")
    pp.progress("Analyzing all layers...")
    
    layer_predictions = []
    for layer_idx in range(num_layers + 1):
        hidden = outputs.hidden_states[layer_idx][0, -1, :]
        preds = apply_logit_lens(hidden, embedding_matrix, tokenizer, layer_norm, top_k=1)
        top_token, top_prob, _ = preds[0]
        layer_predictions.append((layer_idx, top_token, top_prob))
    
    pp.layer_evolution_table(layer_predictions, target_layer=target)
    
    # Generate model output
    pp.subheader("Model Generation")
    pp.progress("Generating response...")
    
    with torch.no_grad():
        gen_outputs = model.generate(
            inputs['input_ids'],
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    generated_text = tokenizer.decode(gen_outputs[0], skip_special_tokens=True)
    prompt_text = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
    response_text = generated_text[len(prompt_text):].strip()
    
    pp.model_output(prompt=PROMPT, response=response_text)
    
    pp.final_summary(
        prompt=PROMPT,
        layer=target,
        shape=tuple(layer_activation.shape),
        response=response_text
    )
    
    return layer_activation, predictions, response_text


if __name__ == "__main__":
    activation, predictions, response = main()
