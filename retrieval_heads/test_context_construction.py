#!/usr/bin/env python3
"""
Quick test to visualize how the context is constructed with the new approach.
This shows exactly what the model will see at different context lengths.
"""

# Simulate the context construction
def visualize_context_construction(needle_tokens=600, total_tokens=10000, needle_position=0.5):
    """Show how context is built."""
    
    haystack_tokens_needed = total_tokens - needle_tokens
    tokens_before = int(haystack_tokens_needed * needle_position)
    tokens_after = haystack_tokens_needed - tokens_before
    
    print(f"\n{'='*70}")
    print(f"CONTEXT CONSTRUCTION FOR {total_tokens:,} TOKENS")
    print(f"{'='*70}\n")
    
    print(f"📌 Needle Configuration:")
    print(f"   - Content: ENTIRE Section 1 (not just one sentence)")
    print(f"   - Size: ~{needle_tokens} tokens")
    print(f"   - Position: {needle_position:.0%} through context (middle)")
    print(f"   - Contains: Full Section 1 text with state mentioned somewhere inside\n")
    
    print(f"🌾 Haystack Configuration:")
    print(f"   - Total haystack tokens: {haystack_tokens_needed:,}")
    print(f"   - Before needle: {tokens_before:,} tokens ({tokens_before/total_tokens:.1%})")
    print(f"   - After needle: {tokens_after:,} tokens ({tokens_after/total_tokens:.1%})\n")
    
    print(f"📊 Final Context Structure:")
    print(f"┌{'─'*68}┐")
    print(f"│ {'ALICE IN WONDERLAND (BEFORE)':^66} │")
    print(f"│ {f'{tokens_before:,} tokens of irrelevant text':^66} │")
    print(f"│ {'Alice was beginning to get very tired...':^66} │")
    bar_before = '█' * int((tokens_before / total_tokens) * 60)
    print(f"│ {bar_before:<66} │")
    print(f"├{'─'*68}┤")
    print(f"│ {'⚡ NEEDLE: ENTIRE SECTION 1 (ANSWER IS BURIED HERE) ⚡':^66} │")
    print(f"│ {f'{needle_tokens} tokens':^66} │")
    print(f"│ {'Item 1. Business. XYZ Corp develops...':^66} │")
    print(f"│ {'...incorporated in Delaware in 1995...':^66} │")
    print(f"│ {'...operates in multiple segments...':^66} │")
    bar_needle = '▓' * int((needle_tokens / total_tokens) * 60)
    print(f"│ {bar_needle:^66} │")
    print(f"├{'─'*68}┤")
    print(f"│ {'ALICE IN WONDERLAND (AFTER)':^66} │")
    print(f"│ {f'{tokens_after:,} tokens of irrelevant text':^66} │")
    print(f"│ {'Down, down, down. Would the fall never...':^66} │")
    bar_after = '█' * int((tokens_after / total_tokens) * 60)
    print(f"│ {bar_after:<66} │")
    print(f"└{'─'*68}┘")
    print(f"{'↓':^70}")
    print(f"{'Question: In which US state was this company incorporated?':^70}")
    print(f"{'Answer with just the state name:':^70}\n")

# Test at different context lengths
print("\n" + "="*70)
print("EXPERIMENT VISUALIZATION - HOW CONTEXTS ARE BUILT")
print("="*70)

visualize_context_construction(needle_tokens=600, total_tokens=2000, needle_position=0.5)
print("\n" + "-"*70 + "\n")
visualize_context_construction(needle_tokens=600, total_tokens=8000, needle_position=0.5)
print("\n" + "-"*70 + "\n")
visualize_context_construction(needle_tokens=600, total_tokens=15000, needle_position=0.5)

print("\n" + "="*70)
print("KEY DIFFERENCES FROM PREVIOUS EXPERIMENT:")
print("="*70)
print("\n❌ OLD (Easy - 93% accuracy at 8K):")
print("   Needle = One sentence: 'Company was incorporated in Delaware.'")
print("   → Too obvious! Easy to pattern match.\n")
print("✅ NEW (Hard - Like Ananya):")
print("   Needle = ENTIRE Section 1 (hundreds of tokens)")
print("   → Answer is buried! Must process whole section.\n")
print("="*70)

