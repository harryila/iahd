# Retrieval Head Analysis for EDGAR Factual Recall

Based on the paper: **"Retrieval Head Mechanistically Explains Long-Context Factual Recall"**  
📄 [arXiv:2404.15574](https://arxiv.org/pdf/2404.15574)

## What Are Retrieval Heads?

**Retrieval heads** are specific attention heads in transformer models that are responsible for "retrieving" information from the context. When you ask a model a factual question about something in its context, these heads:

1. **Attend from the answer position** (where the model is generating the answer)
2. **Back to the relevant context** (where the fact is stated)
3. **Copy or aggregate information** from that location

### Key Insight from the Paper

The paper found that:
- A small subset of attention heads (the "retrieval heads") do most of the work for factual recall
- These heads are **consistent across different tasks** - the same heads retrieve different types of facts
- As context length increases, these heads must "search" through more tokens, leading to degraded performance
- The retrieval heads can be identified by measuring attention from the final token position to the context region

## Our Experiment

We adapt this framework to test the **State of Incorporation** retrieval task from EDGAR SEC filings.

### Setup

| Condition | Context Length | Description | Expected Accuracy |
|-----------|---------------|-------------|-------------------|
| **Short** | ~200 tokens | Just the relevant section | ~70% |
| **Long** | ~2000 tokens | Relevant section + padding | ~40% |

### Models Tested

1. **Llama-3.1-8B-Instruct** (`meta-llama/Llama-3.1-8B-Instruct`)
2. **Qwen2.5-7B-Instruct** (`Qwen/Qwen2.5-7B-Instruct`)

### What We Measure

1. **Accuracy**: Does the model correctly identify the state of incorporation?
2. **Retrieval Head Identification**: Which (layer, head) pairs attend most strongly to the context?
3. **Head Consistency**: Are the same heads used for short vs. long contexts?
4. **Attention Degradation**: How do attention patterns change with more context?

## How Context Length Affects Retrieval

```
Short Context (~200 tokens):
┌─────────────────────────────────────────┐
│ [Relevant Info: "incorporated in Delaware"] │
└─────────────────────────────────────────┘
         ↑
    Easy to find! Retrieval heads focus attention here.

Long Context (~2000 tokens):  
┌───────────────────────────────────────────────────────────────────┐
│ [Padding...] [Relevant Info] [More padding...more padding...] │
└───────────────────────────────────────────────────────────────────┘
                    ↑
    Harder to find! Attention is diluted across more tokens.
```

### Why Performance Degrades

1. **Attention Dilution**: Softmax attention sums to 1.0, so more tokens = less attention per token
2. **Needle in Haystack**: The relevant fact becomes harder to locate
3. **Distractor Tokens**: Irrelevant content may confuse the retrieval mechanism

## Running the Experiment

```bash
# Run full experiment (both models, both context lengths)
python retrieval_head_experiment.py --samples 50

# Run just Llama with short context
python retrieval_head_experiment.py --model llama --context short --samples 50

# Only generate visualizations from existing results
python retrieval_head_experiment.py --visualize-only
```

## Output Files

- `llama_short_results.json` - Llama results with short context
- `llama_long_results.json` - Llama results with long context  
- `qwen_short_results.json` - Qwen results with short context
- `qwen_long_results.json` - Qwen results with long context
- `retrieval_heads_comparison.png` - Visualization comparing conditions

## Interpreting Results

### Accuracy Comparison
- **Short > Long**: Expected. More context = harder retrieval.
- **Large gap** (>30%): Suggests the model relies heavily on local attention
- **Small gap** (<15%): Suggests robust long-context retrieval

### Retrieval Head Overlap
- **High overlap** (>70%): Same heads handle retrieval regardless of context length
- **Low overlap** (<50%): Model uses different strategies for short vs. long context
- This tells us whether retrieval is a **fixed circuit** or **adaptive behavior**

### Top Retrieval Heads
- Heads are labeled as `L{layer}H{head}` (e.g., `L15H7` = Layer 15, Head 7)
- Higher layers typically handle more abstract retrieval
- Early layers focus on local patterns

## Connection to Original Shuffled Context Experiment

This experiment complements our shuffled context analysis:

| Experiment | What it Tests | Key Finding |
|------------|--------------|-------------|
| **Shuffled Context** | Can models extract facts from scrambled text? | Some facts (like states) are recoverable even when shuffled |
| **Retrieval Heads** | Which attention heads perform the retrieval? | Specific heads specialize in fact retrieval |

Together, these show:
- **Shuffling** disrupts word order but preserves keywords → models can still find "Delaware"
- **Long context** dilutes attention → retrieval heads can't focus on the relevant region
- Both degrade performance, but through different mechanisms

## References

- [Retrieval Head Paper](https://arxiv.org/pdf/2404.15574)
- [Original Retrieval Head Code](https://github.com/nightdessert/Retrieval_Head)
- [Needle in a Haystack Test](https://github.com/gkamradt/LLMTest_NeedleInAHaystack)


