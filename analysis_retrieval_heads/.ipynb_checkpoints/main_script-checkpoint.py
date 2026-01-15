# -*- coding: utf-8 -*-
"""
GT-Based Accuracy Evaluation for EDGAR Section 5
(Short + Long Context, LLaMA + Qwen)

Aligned with Section 1 evaluation logic.
"""

import os
import json
import csv
import re
import random
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# =====================================================
# Reproducibility
# =====================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================
# Models
# =====================================================
MODELS = {
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
}

# =====================================================
# Context settings
# =====================================================
CONTEXTS = {
    "short": 200,
    "long": 2000,
}

NUM_SAMPLES = 50
GT_PATH = "edgar_gt_verified_slim.csv"

# =====================================================
# Prompt (more explicit, numeric-only)
# =====================================================
def build_prompt(context: str) -> str:
    return f"""
You are given an excerpt from an SEC 10-K filing.

Context:
{context}

Question:
What is the number of holders of record of the registrant's common stock
as explicitly stated in the document?

Instructions:
- Return ONLY the integer
- Do NOT include commas, words, or explanations
- If the number is not explicitly stated, return UNKNOWN

Answer:
"""

# =====================================================
# Ground Truth Loader (Section 1 style)
# =====================================================
def load_ground_truth(path):
    gt = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            val = row["holder_record_amount_truth"]
            gt[row["filename"]] = (
                val.replace(",", "").strip()
                if val and val.upper() != "NULL"
                else "UNKNOWN"
            )
    return gt

GT = load_ground_truth(GT_PATH)

# =====================================================
# Numeric Normalization
# =====================================================
def normalize_numeric(text):
    if not text:
        return "UNKNOWN"
    text = text.upper().replace(",", "")
    m = re.fullmatch(r"\d+", text.strip())
    return m.group(0) if m else "UNKNOWN"

# =====================================================
# Dataset Loader (NO JSON FILE)
# =====================================================
def load_section5_samples(limit):
    dataset = load_dataset(
        "c3po-ai/edgar-corpus",
        "full",
        split="train",
        streaming=True,
        trust_remote_code=True
    )

    samples = []
    for row in dataset:
        if len(samples) >= limit:
            break

        sec5 = row.get("section_5", "")
        if not isinstance(sec5, str) or len(sec5) < 100:
            continue

        samples.append({
            "filename": row.get("filename", ""),
            "section_5": sec5,
        })

    print(f"Loaded {len(samples)} Section 5 samples")
    return samples

# =====================================================
# Token Utility
# =====================================================
def clip_to_tokens(text, tokenizer, max_tokens):
    ids = tokenizer.encode(text, add_special_tokens=False)
    ids = ids[:max_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)

# =====================================================
# Main Evaluation Loop
# =====================================================
def run_evaluation(samples, model_key, model_name):
    print(f"\nLoading model → {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
    ).eval()

    for ctx_name, ctx_len in CONTEXTS.items():
        results = []
        correct = 0

        for sample in tqdm(samples, desc=f"{model_key}-{ctx_name}"):
            filename = sample["filename"]
            context = clip_to_tokens(sample["section_5"], tokenizer, ctx_len)

            prompt = build_prompt(context)
            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=8,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            gen_ids = out[0, inputs.input_ids.shape[1]:]
            raw = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            pred = normalize_numeric(raw)
            gt = GT.get(filename, "UNKNOWN")

            is_correct = pred == gt
            correct += int(is_correct)

            results.append({
                "filename": filename,
                "raw_answer": raw,
                "normalized_answer": pred,
                "ground_truth": gt,
                "correct": is_correct,
                "context_tokens": ctx_len,
            })

        blob = {
            "model": model_key,
            "model_name": model_name,
            "context": ctx_name,
            "num_samples": len(results),
            "accuracy": correct / len(results),
            "results": results,
        }

        out_file = f"{model_key}_{ctx_name}_section5_gt.json"
        with open(out_file, "w") as f:
            json.dump(blob, f, indent=2)

        print(f"Saved → {out_file}")

    del model
    torch.cuda.empty_cache()

# =====================================================
# Entry Point
# =====================================================
if __name__ == "__main__":
    samples = load_section5_samples(NUM_SAMPLES)

    for model_key, model_name in MODELS.items():
        run_evaluation(samples, model_key, model_name)

    print("\nGT-based Section 5 evaluation complete.")
