#!/usr/bin/env python3
"""
Generate histograms comparing shuffled vs unshuffled accuracy
for state of incorporation and headquarters location questions.
"""

import json
import matplotlib.pyplot as plt
import numpy as np

# Load results
with open('edgar_experiment_results.json') as f:
    results = json.load(f)

def get_accuracy_data(results, question_type):
    """Extract accuracy data for a question type."""
    filtered = [r for r in results if r['question_type'] == question_type]
    
    # Separate by shuffle mode
    none_results = [r for r in filtered if r['shuffle_mode'] == 'none']
    words_results = [r for r in filtered if r['shuffle_mode'] == 'words']
    
    # Filter to those with ground truth
    none_with_gt = [r for r in none_results if r.get('ground_truth')]
    words_with_gt = [r for r in words_results if r.get('ground_truth')]
    
    # Calculate accuracy
    none_correct = sum(1 for r in none_with_gt if r.get('is_correct'))
    words_correct = sum(1 for r in words_with_gt if r.get('is_correct'))
    
    none_acc = (none_correct / len(none_with_gt) * 100) if none_with_gt else 0
    words_acc = (words_correct / len(words_with_gt) * 100) if words_with_gt else 0
    
    return {
        'unshuffled': {'total': len(none_with_gt), 'correct': none_correct, 'accuracy': none_acc},
        'shuffled': {'total': len(words_with_gt), 'correct': words_correct, 'accuracy': words_acc}
    }

# Get data for both question types
state_inc_data = get_accuracy_data(results, 'state_of_incorporation')
hq_state_data = get_accuracy_data(results, 'headquarters_state')

print("=" * 60)
print("SHUFFLE TEST RESULTS")
print("=" * 60)
print(f"\nState of Incorporation (n={state_inc_data['unshuffled']['total']}):")
print(f"  Unshuffled: {state_inc_data['unshuffled']['correct']}/{state_inc_data['unshuffled']['total']} = {state_inc_data['unshuffled']['accuracy']:.1f}%")
print(f"  Shuffled:   {state_inc_data['shuffled']['correct']}/{state_inc_data['shuffled']['total']} = {state_inc_data['shuffled']['accuracy']:.1f}%")
print(f"  Δ Accuracy: {state_inc_data['shuffled']['accuracy'] - state_inc_data['unshuffled']['accuracy']:+.1f}%")

print(f"\nHeadquarters State (n={hq_state_data['unshuffled']['total']}):")
print(f"  Unshuffled: {hq_state_data['unshuffled']['correct']}/{hq_state_data['unshuffled']['total']} = {hq_state_data['unshuffled']['accuracy']:.1f}%")
print(f"  Shuffled:   {hq_state_data['shuffled']['correct']}/{hq_state_data['shuffled']['total']} = {hq_state_data['shuffled']['accuracy']:.1f}%")
print(f"  Δ Accuracy: {hq_state_data['shuffled']['accuracy'] - hq_state_data['unshuffled']['accuracy']:+.1f}%")

# Create figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Color scheme
colors = ['#2ecc71', '#e74c3c']  # Green for unshuffled, Red for shuffled
bar_width = 0.35

# --- Histogram 1: State of Incorporation ---
ax1 = axes[0]
categories = ['Unshuffled\n(Original)', 'Shuffled\n(Words)']
accuracies = [state_inc_data['unshuffled']['accuracy'], state_inc_data['shuffled']['accuracy']]
counts = [f"{state_inc_data['unshuffled']['correct']}/{state_inc_data['unshuffled']['total']}", 
          f"{state_inc_data['shuffled']['correct']}/{state_inc_data['shuffled']['total']}"]

bars1 = ax1.bar(categories, accuracies, color=colors, edgecolor='black', linewidth=1.2)

# Add value labels on bars
for bar, acc, count in zip(bars1, accuracies, counts):
    height = bar.get_height()
    ax1.annotate(f'{acc:.1f}%\n({count})',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=11, fontweight='bold')

ax1.set_ylabel('Accuracy (%)', fontsize=12)
ax1.set_title('State of Incorporation\nShuffled vs Unshuffled Accuracy', fontsize=14, fontweight='bold')
ax1.set_ylim(0, 110)
ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Random baseline')
ax1.grid(axis='y', alpha=0.3)

# --- Histogram 2: Headquarters State ---
ax2 = axes[1]
accuracies_hq = [hq_state_data['unshuffled']['accuracy'], hq_state_data['shuffled']['accuracy']]
counts_hq = [f"{hq_state_data['unshuffled']['correct']}/{hq_state_data['unshuffled']['total']}", 
             f"{hq_state_data['shuffled']['correct']}/{hq_state_data['shuffled']['total']}"]

bars2 = ax2.bar(categories, accuracies_hq, color=colors, edgecolor='black', linewidth=1.2)

# Add value labels on bars
for bar, acc, count in zip(bars2, accuracies_hq, counts_hq):
    height = bar.get_height()
    ax2.annotate(f'{acc:.1f}%\n({count})',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=11, fontweight='bold')

ax2.set_ylabel('Accuracy (%)', fontsize=12)
ax2.set_title('Headquarters Location\nShuffled vs Unshuffled Accuracy', fontsize=14, fontweight='bold')
ax2.set_ylim(0, 110)
ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Random baseline')
ax2.grid(axis='y', alpha=0.3)

# Overall title
fig.suptitle('EDGAR Corpus Factual Recall: Impact of Context Shuffling\n(Llama-3.1-8B-Instruct)', 
             fontsize=16, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('shuffle_comparison_histograms.png', dpi=150, bbox_inches='tight', facecolor='white')
print(f"\n✓ Histograms saved to: shuffle_comparison_histograms.png")

# Also show if running interactively
plt.show()

