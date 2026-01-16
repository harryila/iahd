# Harry's Experiment Updates

## Entry 1: Needle-in-Haystack Context Sweep (Jan 15, 2026)

### Experiment Setup
- **Task**: State of incorporation retrieval from SEC 10-K filings
- **Needle**: Entire Section 1 text (~600 tokens) — NOT a single sentence
- **Haystack**: Alice in Wonderland text (Project Gutenberg)
- **Needle Position**: Middle of context (position=0.5)
- **Context Lengths**: 200, 500, 1K, 2K, 4K, 8K, 10K, 12K, 15K tokens
- **Samples**: 30 per context length
- **No shuffling**: Text kept intact, no word scrambling

---

### Result 1: Meta-Llama-3-8B-Instruct (Llama 3)

**Model**: `meta-llama/Meta-Llama-3-8B-Instruct`  
**Max Context**: 8,192 tokens  
**Output**: `llama3_context_sweep_0.json`, `llama3_context_sweep_0.png`

| Context | Accuracy | Notes |
|---------|----------|-------|
| 200 | 63.3% | Baseline |
| 500 | 63.3% | Stable |
| 1000 | 63.3% | Stable |
| 2000 | 60.0% | Slight drop |
| 4000 | 53.3% | Degrading |
| 8000 | 66.7% | Near max context |
| **10000** | **0.0%** | 💀 **COLLAPSE** |
| **12000** | **0.0%** | Beyond max context |
| **15000** | **0.0%** | Beyond max context |

**Key Finding**: Complete failure at 10K+ tokens. Model's max context is 8,192 tokens — anything beyond causes total collapse. This replicates Ananya's findings.

---

### Result 2: Meta-Llama-3.1-8B-Instruct (Llama 3.1)

**Model**: `meta-llama/Llama-3.1-8B-Instruct`  
**Max Context**: 128,000 tokens  
**Output**: `llama31_context_sweep_0.json`

| Context | Accuracy | Notes |
|---------|----------|-------|
| 200 | 83.3% | Higher baseline than Llama 3 |
| 500 | 83.3% | Stable |
| 1000 | 83.3% | Stable |
| 2000 | 83.3% | Stable |
| 4000 | 76.7% | Slight drop |
| 8000 | 80.0% | Still strong |
| **10000** | **80.0%** | ✅ No collapse (Llama 3 = 0%) |
| **15000** | **76.7%** | ✅ Still working |
| **20000** | **70.0%** | ✅ Minor degradation |
| **30000** | **80.0%** | ✅ Still strong |
| **50000** | **83.3%** | ✅ 🔥 **Excellent at 50K!** |

**Key Finding**: No collapse even at 50K tokens! Llama 3.1's 128K context window handles extreme lengths with stable ~80% accuracy.

---

### Comparison Summary

| Context | Llama 3 | Llama 3.1 |
|---------|---------|-----------|
| Max trained context | 8,192 | 128,000 |
| 8K tokens | 66.7% | 80.0% |
| 10K tokens | **0%** 💀 | **80%** ✅ |
| 15K tokens | **0%** 💀 | **77%** ✅ |
| 30K tokens | N/A | **80%** ✅ |
| 50K tokens | N/A | **83%** ✅ |

**Conclusion**: The dramatic collapse Ananya observed in Llama 3 at 10K+ tokens is due to exceeding its 8K context limit. Llama 3.1's extended 128K context window completely eliminates this failure, maintaining ~80% accuracy even at 50K tokens.

---

### Files
- `llama3_context_sweep_0.json` — Full Llama 3 results
- `llama3_context_sweep_0.png` — Llama 3 accuracy plot  
- `llama31_context_sweep_0.json` — Full Llama 3.1 results (up to 50K)
- `llama3_vs_llama31_comparison.png` — **Comparison plot showing the dramatic difference**
- `needle_haystack_sweep.py` — Experiment script (saves incremental files)

