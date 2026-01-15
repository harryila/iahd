# ================================================================
# Section 5 — Short Context Retrieval (200 tokens)
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
# Model config
# ----------------------------
MODELS = {
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
}

NUM_SAMPLES = 50
CONTEXT_TOKENS = 200

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
        if sec5 and len(sec5) > 100:
            samples.append({
                "filename": row.get("filename"),
                "section_5": sec5
            })
    return samples

# ----------------------------
# Token utilities (critical)
# ----------------------------
def tokenize_truncate(text, tokenizer, max_tokens):
    ids = tokenizer.encode(text, add_special_tokens=False)
    ids = ids[:max_tokens]
    return ids

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
def run_short(model_key, samples):
    model_name = MODELS[model_key]
    model, tokenizer = load_model(model_name)

    results = []

    for s in tqdm(samples, desc=f"{model_key}-short"):
        context_ids = tokenize_truncate(
            s["section_5"],
            tokenizer,
            CONTEXT_TOKENS
        )

        prompt_ids = tokenizer.encode(
            QUESTION,
            add_special_tokens=False
        )

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
        res = run_short(key, samples)
        with open(f"{out_dir}/section5_short_{key}.json", "w") as f:
            json.dump(res, f, indent=2)

    print("Section 5 short-context run complete.")
