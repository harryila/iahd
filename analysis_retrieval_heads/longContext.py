# -*- coding: utf-8 -*-
"""
Long-Context Agreement Study: EDGAR Section 5

This experiment tests whether large language models can
consistently extract a simple numeric fact when it is
embedded inside long, noisy contexts.

Fact queried:
- Number of holders of stock (Section 5)

Models:
- LLaMA 3.1 8B Instruct
- Qwen 2.5 7B Instruct

Each model sees ~2000 tokens of context containing:
- Section 5 (signal)
- Randomized padding from other filing sections (noise)

No ground truth is required.
"""

import os
import json
import random
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# ================================================================
# Reproducibility
# ================================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ================================================================
# Models under evaluation
# ================================================================

MODEL_REGISTRY = {
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
}

# ================================================================
# Experiment parameters
# ================================================================

NUM_SAMPLES = 50
MAX_CONTEXT_TOKENS = 2000
SECTION5_CORE_TOKENS = 350

QUESTION_PROMPT = (
    "Read the following SEC 10-K filing excerpt.\n\n"
    "Context:\n{context}\n\n"
    "Question: How many holders of stock are there?\n"
    "Answer with ONLY a single numerical value. Do not include words, commas, or symbols."
)

# ================================================================
# Dataset loading
# ================================================================

def collect_section5_records(limit: int):
    """
    Stream EDGAR filings and extract:
    - Section 5 (signal)
    - Other sections (for distractor padding)
    """
    dataset = load_dataset(
        "c3po-ai/edgar-corpus",
        "full",
        split="train",
        streaming=True,
        trust_remote_code=True
    )

    records = []

    for entry in dataset:
        if len(records) >= limit:
            break

        sec5 = entry.get("section_5", "")
        if not isinstance(sec5, str) or len(sec5) < 100:
            continue

        noise_sections = [
            text for key, text in entry.items()
            if key.startswith("section_")
            and key != "section_5"
            and isinstance(text, str)
        ]

        records.append({
            "filename": entry.get("filename", ""),
            "section_5": sec5,
            "noise": noise_sections
        })

    print(f"Collected {len(records)} valid filings")
    return records

# ================================================================
# Token helpers
# ================================================================

def clip_to_tokens(text: str, tokenizer, budget: int) -> str:
    """
    Truncate text so it fits exactly within a token budget.
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    return tokenizer.decode(ids[:budget], skip_special_tokens=True)

# ================================================================
# Long-context construction
# ================================================================

def assemble_long_context(sample, tokenizer):
    """
    Construct a ~2000 token context by embedding Section 5
    among randomly ordered distractor sections.
    """
    # Core factual section (kept relatively intact)
    core = clip_to_tokens(sample["section_5"], tokenizer, SECTION5_CORE_TOKENS)
    core_len = len(tokenizer.encode(core, add_special_tokens=False))

    remaining_budget = MAX_CONTEXT_TOKENS - core_len
    random.shuffle(sample["noise"])

    padding_chunks = []
    used_tokens = 0

    for block in sample["noise"]:
        if used_tokens >= remaining_budget:
            break

        chunk = clip_to_tokens(block, tokenizer, remaining_budget - used_tokens)
        chunk_len = len(tokenizer.encode(chunk, add_special_tokens=False))

        if chunk_len > 0:
            padding_chunks.append(chunk)
            used_tokens += chunk_len

    # Randomize position of the factual section
    combined_blocks = padding_chunks + [core]
    random.shuffle(combined_blocks)

    merged = "\n\n".join(combined_blocks)
    return clip_to_tokens(merged, tokenizer, MAX_CONTEXT_TOKENS)

# ================================================================
# Model loading
# ================================================================

def initialize_model(model_id):
    """
    Load model and tokenizer for deterministic inference.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16
    )

    model.eval()
    return model, tokenizer

# ================================================================
# Inference
# ================================================================

def infer_numeric_answer(prompt, model, tokenizer):
    """
    Generate a short numeric-only response.
    """
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated = model.generate(
            **encoded,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    new_tokens = generated[0, encoded.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

# ================================================================
# Experiment runner
# ================================================================

def run_long_context_experiment(records, model_key, output_dir):
    """
    Run the long-context experiment for one model.
    """
    model_name = MODEL_REGISTRY[model_key]
    print(f"\nRunning long-context inference → {model_name}")

    model, tokenizer = initialize_model(model_name)
    outputs = []

    for record in tqdm(records):
        context = assemble_long_context(record, tokenizer)
        prompt = QUESTION_PROMPT.format(context=context)

        answer = infer_numeric_answer(prompt, model, tokenizer)

        outputs.append({
            "filename": record["filename"],
            "model": model_key,
            "answer": answer
        })

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{model_key}_long_context_answers.json")

    with open(save_path, "w") as f:
        json.dump(outputs, f, indent=2)

    print(f"Saved results → {save_path}")

    del model
    torch.cuda.empty_cache()

# ================================================================
# Main
# ================================================================

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    samples = collect_section5_records(NUM_SAMPLES)

    for model_key in MODEL_REGISTRY:
        run_long_context_experiment(samples, model_key, BASE_DIR)

    print("\nLong-context agreement experiment finished.")
