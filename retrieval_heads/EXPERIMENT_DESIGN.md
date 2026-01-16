# Needle in Haystack Experiment Design
## Replicating Ananya's Approach with Variable Needle Position

---

## 🎯 Goal
Find at what context length LLaMA and Qwen models fail to retrieve the state of incorporation when it's buried in a large chunk of text surrounded by irrelevant content.

---

## 📋 Experiment Configuration

### Models to Test
- **LLaMA**: `meta-llama/Llama-3.1-8B-Instruct`
- **Qwen**: `Qwen/Qwen2.5-7B-Instruct`

### Context Lengths (Token Counts)
```python
[200, 500, 1000, 2000, 4000, 8000, 10000, 12000, 15000]
```

### Needle Configuration
- **What is the needle?** ENTIRE Section 1 from SEC 10-K filing (hundreds of tokens)
- **Where is it placed?** Middle of context (position = 0.5)
- **Why this matters:** The answer is buried somewhere within Section 1, not in an obvious single sentence

### Haystack (Distractor Text)
- **Source:** Alice in Wonderland from Project Gutenberg
- **Purpose:** Irrelevant padding to increase context length
- **Placement:** Before and after the needle

---

## 🔬 How Each Test Works

### Example at 10,000 tokens:

```
┌─────────────────────────────────────────────────────────────┐
│                    ALICE IN WONDERLAND                       │
│  (Irrelevant padding - approximately 4,700 tokens)          │
│  "Alice was beginning to get very tired of sitting..."      │
│  "Down, down, down. Would the fall never come to an end..." │
│                                                              │
├─────────────────────────────────────────────────────────────┤
│                    NEEDLE (Section 1)                        │
│  (~600 tokens containing the incorporation state)           │
│  "Item 1. Business. XYZ Corporation develops..."            │
│  "...incorporated in Delaware in 1995..."                   │
│  "...operates in multiple segments..."                      │
├─────────────────────────────────────────────────────────────┤
│                    ALICE IN WONDERLAND                       │
│  (Irrelevant padding - approximately 4,700 tokens)          │
│  "The rabbit-hole went straight on like a tunnel..."        │
│  "Alice opened the door and found that it led..."           │
└─────────────────────────────────────────────────────────────┘
                            ↓
              Question: In which US state was 
              this company incorporated?
              Answer with just the state name:
```

### What Makes This Hard
1. **Large needle**: Not just "incorporated in Delaware" - the entire Section 1 (hundreds of tokens)
2. **Answer is buried**: The state is mentioned somewhere within Section 1
3. **Lots of distractors**: Thousands of tokens of Alice in Wonderland
4. **Long context**: Up to 15K tokens total

---

## 📊 Expected Results (Based on Ananya's Findings)

### Short Context (< 4K tokens)
- **Expected Accuracy:** 70-85%
- **Why:** Model can easily attend to Section 1 when context is short

### Medium Context (4K - 8K tokens)
- **Expected Accuracy:** 60-70%
- **Why:** Some degradation but still manageable

### Critical Threshold (~10K tokens)
- **Expected Accuracy:** 20-40% ⚠️
- **Why:** Ananya found dramatic collapse here
- **Mechanism:** Retrieval heads can't focus attention effectively

### Long Context (12K - 15K tokens)
- **Expected Accuracy:** 0-10% 💀
- **Why:** Complete failure of retrieval mechanism
- **Ananya's result:** 0% at both 10K and 15K

---

## 🔍 What We're Measuring

### For Each Context Length:
1. **Accuracy**: Does the model generate the correct state name?
2. **Sample size**: 30 samples per context length
3. **Answer validation**: Fuzzy matching (handles "Delaware" vs "delaware" vs "DE")

### Output Files:
- `llama_context_sweep.json` - Full results for LLaMA
- `qwen_context_sweep.json` - Full results for Qwen
- `context_length_sweep.png` - Visualization comparing both models

---

## 🆚 Key Differences from Your Previous Experiment

| Aspect | Previous (Easy) | New (Hard - Like Ananya) |
|--------|----------------|--------------------------|
| **Needle** | One sentence: "Company was incorporated in Delaware." | ENTIRE Section 1 (hundreds of tokens) |
| **Max context** | 8,000 tokens | 15,000 tokens |
| **Previous accuracy** | 93%+ at 8K | Expected: ~0% at 10K+ |
| **Task difficulty** | Easy pattern matching | Hard: find answer buried in large text |

---

## 🎬 How to Run

```bash
# Run context length sweep for LLaMA (middle position)
python needle_haystack_sweep.py --model llama --samples 30 --sweep length

# Run for both models
python needle_haystack_sweep.py --model all --samples 30 --sweep length

# Visualize existing results only
python needle_haystack_sweep.py --visualize-only
```

---

## ⏱️ Estimated Runtime

- **Per sample at 15K tokens:** ~20-30 seconds
- **30 samples × 9 context lengths:** ~4-5 hours per model
- **Both models:** ~8-10 hours total

---

## 🔮 Predictions

Based on Ananya's results, we should see:

### LLaMA-3.1-8B-Instruct
```
200 tokens   → ~80% ✅
500 tokens   → ~80% ✅
1000 tokens  → ~75% ✅
2000 tokens  → ~70% ✅
4000 tokens  → ~65% ⚠️
8000 tokens  → ~40% ⚠️
10000 tokens → ~10% ❌ [COLLAPSE POINT]
12000 tokens → ~5%  ❌
15000 tokens → ~0%  ❌
```

### Qwen-2.5-7B-Instruct
- Possibly better or worse - we'll find out!
- May have different failure threshold

---

## 🚨 What We're Testing

**Hypothesis:** As context length increases beyond 10K tokens, retrieval heads can't effectively attend to the needle, causing dramatic accuracy collapse.

**This replicates Ananya's finding that specific retrieval heads "die" around 10K tokens.**

Next step: After we confirm the degradation, we'll add mechanistic analysis (attention head tracking) to understand WHY it fails.

