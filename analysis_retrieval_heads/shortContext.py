# -*- coding: utf-8 -*-
"""
Short-Context Agreement Analysis for EDGAR Section 5

This experiment measures agreement and stability of numeric
retrieval (number of stockholders) across models, without
using ground truth labels.

Focus:
- Numeric-only answers
- Cross-model agreement
- Attention localization (short context only)
"""

import os
import json
import random
import re
import numpy as np
import torch
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm

# ================================================================
# Configuration
# ================================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

MODELS = {
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
}

NUM_SAMPLES = 50
MAX_CONTEXT_TOKENS = 200

QUESTION_TEMPLATE = """You are given an excerpt from an SEC 10-K filing.

Context:
{context}

Question:
How many holders of stock are there?

Instructions:
- Answer with ONLY a numerical value.
- Do NOT include commas, words, or explanations.
- If the number is not explicitly stated, output UNKNOWN.

Answer:"""

# ================================================================
# Dataset loading
# ================================================================

def load_section5_samples(limit):
    """Stream EDGAR filings and collect usable Section 5 excerpts."""
    dataset = load_dataset(
        "c3po-ai/edgar-corpus",
        "full",
        split="train",
        streaming=True,
        trust_remote_code=True
    )

    collected = []
    for row in dataset:
        if len(collected) >= limit:
            break

        sec5 = row.get("section_5", "")
        if not sec5 or len(sec5) < 100:
            continue

        collected.append({
            "filename": row.get("filename", ""),
            "section_5": sec5,
        })

    print(f"Loaded {len(collected)} Section 5 samples")
    return collected

# ================================================================
# Token utilities
# ================================================================

def clip_to_tokens(text, tokenizer, max_tokens):
    """Ensure text fits within a token budget."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    ids = ids[:max_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True), len(ids)

# ================================================================
# Answer normalization
# ================================================================

def normalize_numeric_answer(text):
    """Convert model output into a canonical numeric form."""
    if not text:
        return None

    txt = text.strip().upper()

    if "UNKNOWN" in txt:
        return "UNKNOWN"

    txt = txt.replace(",", "")
    match = re.fullmatch(r"\d+", txt)
    return match.group(0) if match else None

# ================================================================
# Model loading
# ================================================================

def load_model(model_name):
    """Load model and tokenizer for inference."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation="eager",
    )

    model.eval()
    return model, tokenizer

# ================================================================
# Inference
# ================================================================

def generate_numeric_answer(model, tokenizer, prompt):
    """Generate a constrained numeric response."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    gen_ids = output[0, inputs.input_ids.shape[1]:]
    raw = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return raw, normalize_numeric_answer(raw)

# ================================================================
# Attention analysis (short context only)
# ================================================================

def compute_retrieval_attention(model, tokenizer, prompt, ctx_start, ctx_end):
    """
    Compute how strongly each attention head attends
    to the context span when generating the final token.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(inputs.input_ids, output_attentions=True)
        attentions = outputs.attentions

    head_scores = {}
    for layer_idx, layer in enumerate(attentions):
        layer = layer[0]  # (heads, seq, seq)
        for head_idx in range(layer.shape[0]):
            score = layer[head_idx, -1, ctx_start:ctx_end].sum().item()
            head_scores[(layer_idx, head_idx)] = score

    del attentions
    torch.cuda.empty_cache()
    return head_scores

# ================================================================
# Experiment runner
# ================================================================

def run_short_context(model_key, samples, save_dir):
    """Run short-context agreement experiment for one model."""
    model_name = MODELS[model_key]
    print(f"\nRunning short-context agreement: {model_name}")

    model, tokenizer = load_model(model_name)

    attention_bank = defaultdict(list)
    outputs = []

    for sample in tqdm(samples, desc=model_key):
        context, ctx_len = clip_to_tokens(
            sample["section_5"], tokenizer, MAX_CONTEXT_TOKENS
        )

        prompt = QUESTION_TEMPLATE.format(context=context)

        raw_answer, norm_answer = generate_numeric_answer(
            model, tokenizer, prompt
        )

        prefix = "You are given an excerpt from an SEC 10-K filing.\n\nContext:\n"
        ctx_start = len(tokenizer.encode(prefix, add_special_tokens=False))
        ctx_end = ctx_start + ctx_len

        head_scores = compute_retrieval_attention(
            model, tokenizer, prompt, ctx_start, ctx_end
        )

        for head, score in head_scores.items():
            attention_bank[head].append(score)

        outputs.append({
            "filename": sample["filename"],
            "raw_answer": raw_answer,
            "normalized_answer": norm_answer,
            "context_tokens": ctx_len,
        })

    avg_attention = {
        head: float(np.mean(vals))
        for head, vals in attention_bank.items()
    }

    top_heads = sorted(
        avg_attention.items(), key=lambda x: x[1], reverse=True
    )[:20]

    result_blob = {
        "model": model_key,
        "model_name": model_name,
        "context": "short",
        "num_samples": len(outputs),
        "top_retrieval_heads": [
            (f"L{l}H{h}", score) for (l, h), score in top_heads
        ],
        "results": outputs,
    }

    out_file = os.path.join(save_dir, f"{model_key}_short_section5.json")
    with open(out_file, "w") as f:
        json.dump(result_blob, f, indent=2)

    print(f"Saved → {out_file}")

    del model
    torch.cuda.empty_cache()

# ================================================================
# Main
# ================================================================

if __name__ == "__main__":
    save_dir = os.path.dirname(os.path.abspath(__file__))
    samples = load_section5_samples(NUM_SAMPLES)

    for model_key in MODELS:
        run_short_context(model_key, samples, save_dir)

    print("\nShort-context agreement experiment complete.")
