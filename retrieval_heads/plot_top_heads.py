"""
Clean visualization: Top Retrieval Heads
"""
import json
import matplotlib.pyplot as plt

with open('llama3_results/head/context_attention_sweep.json', 'r') as f:
    data = json.load(f)

# Collect top heads at each context length (unshuffled only)
results = []
for ctx_len, res in data['by_context_length'].items():
    ctx = int(ctx_len)
    if ctx <= 8000 and res['unshuffled'].get('top_heads'):
        top5 = res['unshuffled']['top_heads'][:5]
        results.append({'ctx': ctx, 'heads': top5})

results.sort(key=lambda x: x['ctx'])

# Print table
print("Top 5 Retrieval Heads by Context Length (Unshuffled)")
print("=" * 70)
for r in results:
    heads_str = ', '.join([f"{h[0]} ({h[1]:.3f})" for h in r['heads']])
    print(f"{r['ctx']:>5} tokens: {heads_str}")

# Create bar chart of top head attention scores
fig, ax = plt.subplots(figsize=(10, 5))

ctx_labels = [f'{r["ctx"]//1000}K' if r["ctx"] >= 1000 else str(r["ctx"]) for r in results]
x = range(len(results))
width = 0.15
colors = ['#2563eb', '#3b82f6', '#60a5fa', '#93c5fd', '#bfdbfe']

for i in range(5):  # Top 5 heads
    scores = [r['heads'][i][1] for r in results]
    labels = [r['heads'][i][0] for r in results]
    bars = ax.bar([xi + i*width for xi in x], scores, width, label=f'Rank {i+1}', color=colors[i])
    
    # Add head names on bars (only for rank 1)
    if i == 0:
        for bar, label in zip(bars, labels):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, 
                   label, ha='center', va='bottom', fontsize=8, rotation=45)

ax.set_xticks([xi + 2*width for xi in x])
ax.set_xticklabels(ctx_labels)
ax.set_xlabel('Context Length', fontsize=12)
ax.set_ylabel('Attention Score', fontsize=12)
ax.set_title('Top 5 Retrieval Heads by Context Length', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('llama3_results/head/top_heads.png', dpi=150, facecolor='white')
print("\nSaved: llama3_results/head/top_heads.png")

