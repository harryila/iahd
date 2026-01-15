# ================================================================
# Section 5 — Long Context Retrieval (2000 tokens)
# ================================================================

import random
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json
import os

# ----------------------------
# Reproducibility
# ----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ----------------------------
# Config
# ----------------------------
MODELS = {
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
}

NUM_SAMPLES = 50
TOTAL_TOKENS = 2000
CORE_TOKENS = 350  # same centering idea as Section 1

QUESTION = (
    "How many holders of the company's common stock are there?\n"
    "Answer with ONLY a number. Do not include words or symbols."
)

# ----------------------------
# Dataset
# ----------------------------
def load_samples(n):
    dataset = load_dataset(
        "c3po-ai/edgar-corpus",
        "full",
        split="train",
        streaming=True,
        trust_remote_code=True
    )

    samples = []
    for row in dataset:
        if len(samples) >= n:
            break

        sec5 = row.get("section_5", "")
        if not sec5 or len(sec5) < 100:
            continue

        distractors = [
            v for k, v in row.items()
            if k.startswith("section_") and k != "section_5" and isinstance(v, str)
        ]

        samples.append({
            "filename": row.get("filename"),
            "section_5": sec5,
            "distractors": distractors
        })

    return samples

# ----------------------------
# Token utilities
# ----------------------------
def tokenize(text, tokenizer):
    return tokenizer.encode(text, add_special_tokens=False)

# ----------------------------
# Context construction (token-first)
# ----------------------------
def build_long_context(sample, tokenizer):
    core = tokenize(sample["section_5"], tokenizer)[:CORE_TOKENS]

    remaining = TOTAL_TOKENS - len(core)
    random.shuffle(sample["distractors"])

    padding = []
    used = 0

    for sec in sample["distractors"]:
        if used >= remaining:
            break
        ids = tokenize(sec, tokenizer)
        ids = ids[: remaining - used]
        if ids:
            padding.append(ids)
            used += len(ids)

    blocks = padding + [core]
    random.shuffle(blocks)

    flat = []
    for b in blocks:
        flat.extend(b)

    return flat[:TOTAL_TOKENS]

# ----------------------------
# Model loading
# ----------------------------
def load_model(name):
    tokenizer = AutoTokenizer.from_pretrained(name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        name,
        device_map="auto",
        torch_dtype=torch.float16
    )
    model.eval()
    return model, tokenizer

# ----------------------------
# Main experiment
# ----------------------------
def run_long(model_key, samples):
    model_name = MODELS[model_key]
    model, tokenizer = load_model(model_name)

    results = []

    for s in tqdm(samples, desc=f"{model_key}-long"):
        context_ids = build_long_context(s, tokenizer)
        prompt_ids = tokenize(QUESTION, tokenizer)

        input_ids = torch.tensor(
            [context_ids + prompt_ids],
            device=model.device
        )

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        gen = output[0, input_ids.shape[1]:]
        answer = tokenizer.decode(gen, skip_special_tokens=True).strip()

        results.append({
            "filename": s["filename"],
            "answer": answer,
            "context_tokens": len(context_ids)
        })

    return results

# ----------------------------
# Entry point
# ----------------------------
if __name__ == "__main__":
    samples = load_samples(NUM_SAMPLES)
    out_dir = os.path.dirname(os.path.abspath(__file__))

    for key in MODELS:
        res = run_long(key, samples)
        with open(f"{out_dir}/section5_long_{key}.json", "w") as f:
            json.dump(res, f, indent=2)

    print("Section 5 long-context run complete.")
