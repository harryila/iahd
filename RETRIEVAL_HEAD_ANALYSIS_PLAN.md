# Retrieval Head Analysis - Full Project Plan

## Overview

This project investigates **retrieval heads** in Llama 3 and Llama 3.1 models - specific attention heads responsible for locating and extracting factual information from long contexts.

We're testing this using a **"needle in a haystack"** task where:
- A **needle** (Section 1 from EDGAR SEC filings containing state of incorporation) is embedded in
- A **haystack** (Alice in Wonderland text) at various context lengths
- The model must answer: "What state was the company incorporated in?"

## Research Questions

1. **How does accuracy degrade** as context length increases?
2. **How does shuffling the needle** (destroying sentence structure) affect accuracy?
3. **Which attention heads are responsible** for retrieval?
4. **Do different metrics identify different heads?**

---

## Two Attention Metrics Being Compared

### Metric 1: Context Attention (Harry's Approach)
**Sum of attention over the ENTIRE Section 1 region (~200-500 tokens)**

```python
# Attention from last token to entire needle region
context_attn = last_pos_attn[section_start:section_end].sum()
```

- Measures: "Which heads are LOCATING/ATTENDING to the relevant region?"
- More robust to shuffling (region still exists even if words scrambled)
- Identifies **"locator" heads**

### Metric 2: Key Token Attention (Paper's Copy Head Approach)
**Sum of attention to SPECIFIC ANSWER TOKENS only (e.g., "Delaware" = 1-3 tokens)**

```python
# Attention from last token to only the answer tokens
key_attn = 0
for start, end in answer_token_positions:
    key_attn += last_pos_attn[start:end].sum()
```

- Measures: "Which heads are COPYING the specific answer?"
- Very sensitive to shuffling (answer tokens get scattered)
- Identifies **"copier" heads**

---

## Experiments to Run

### Experiment 1: Context Attention Sweep (Llama 3)
**Script:** `retrieval_heads/llama3_context_attention_sweep.py`

| Parameter | Value |
|-----------|-------|
| Model | meta-llama/Llama-3-8B-Instruct |
| Context Lengths | 200, 500, 1000, 2000, 4000, 8000, 10000, 12000, 15000 |
| Samples | 30 (same across all tests) |
| Conditions | Shuffled + Unshuffled |
| Needle Position | Middle (0.5) |
| Haystack | Alice in Wonderland |
| Attention Analysis | Only for contexts ≤4000 tokens (memory limit) |

**Output:** `llama3_results/head/context_attention_sweep.json`

### Experiment 2: Key Token Attention Sweep (Llama 3)
**Script:** `retrieval_heads/llama3_key_token_attention_sweep.py`

Same parameters as Experiment 1, but measures attention to answer tokens only.

**Output:** `llama3_results/head/key_token_attention_sweep.json`

---

## Key Files

### Scripts
- `retrieval_heads/llama3_context_attention_sweep.py` - Context attention metric sweep
- `retrieval_heads/llama3_key_token_attention_sweep.py` - Key token attention metric sweep
- `retrieval_heads/needle_haystack_sweep.py` - Core functions for needle/haystack creation

### Data
- `retrieval_heads/edgar_gt_verified_slim.csv` - Ground truth (filename → state) mappings
- `retrieval_heads/alice_in_wonderland.txt` - Haystack text

### Results (to be generated)
- `llama3_results/head/context_attention_sweep.json`
- `llama3_results/head/key_token_attention_sweep.json`

---

## Current Progress (as of 2026-01-19)

### Completed
- ✅ Llama 3 accuracy sweeps (shuffled vs unshuffled) - see `llama3_shuffled_comparison.png`
- ✅ Llama 3.1 accuracy sweeps - see `llama31_shuffled_comparison.png`
- ✅ Initial dual metric comparison at 1K tokens
- ✅ Created both sweep scripts with 30 samples, all context lengths
- ✅ Saved exact package versions in `requirements.txt`

### In Progress
- 🔄 Context attention sweep was running but had OOM issues for some samples

### Not Started
- ⬜ Key token attention sweep
- ⬜ Comparison visualizations
- ⬜ Ablation experiments (disabling identified heads)

---

## Memory Requirements

**Problem:** `output_attentions=True` stores all attention matrices:
- 32 layers × 32 heads × seq_len × seq_len × 2 bytes
- seq=4000 → ~32GB
- seq=8000 → ~128GB

**Solution:** Use **1x B200 (180GB)** GPU instead of H100 (80GB)

---

## How to Run

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Login to HuggingFace (needed for Llama models)
huggingface-cli login
```

### Run Context Attention Sweep
```bash
cd retrieval_heads
python3 llama3_context_attention_sweep.py
```

### Run Key Token Attention Sweep
```bash
cd retrieval_heads  
python3 llama3_key_token_attention_sweep.py
```

---

## Expected Findings

Based on initial results at 1K tokens:

1. **Shuffling hurts accuracy** by 13-20% across all context lengths
2. **Different metrics identify partially overlapping heads** (~40% overlap)
3. **Context metric is robust to shuffling** (same heads identified)
4. **Key token metric is sensitive to shuffling** (attention drops to zero when answer tokens scattered)

### Interpretation
- **Context metric** identifies heads that FIND the relevant region
- **Key token metric** identifies heads that COPY the answer
- These may be different mechanisms in the model

---

## For Ablation Experiments (Future)

Once we identify top retrieval heads from both metrics:
1. Disable identified heads (zero out their attention)
2. Re-run accuracy tests
3. Compare which metric's heads are more causally important

**Hypothesis:** Ablating "locator" heads (context metric) might break finding; ablating "copier" heads (key token metric) might break extraction.

---

## File Structure

```
iahd/
├── requirements.txt              # Exact package versions
├── RETRIEVAL_HEAD_ANALYSIS_PLAN.md  # This file
├── retrieval_heads/
│   ├── needle_haystack_sweep.py  # Core functions
│   ├── llama3_context_attention_sweep.py
│   ├── llama3_key_token_attention_sweep.py
│   ├── edgar_gt_verified_slim.csv
│   ├── alice_in_wonderland.txt
│   ├── llama3_results/
│   │   ├── head/
│   │   │   ├── context_attention_sweep.json
│   │   │   └── key_token_attention_sweep.json
│   │   ├── llama3_context_sweep_0.json
│   │   └── llama3_shuffled_comparison.png
│   └── llama31_results/
│       └── ...
```

---

## Contact

This analysis is being done by Harry, with guidance from Eric on the retrieval head methodology.

