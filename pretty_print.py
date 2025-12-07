"""
Pretty terminal output utilities for activation extraction scripts.
Uses ANSI colors and Unicode box-drawing for nice formatting.
"""

# ANSI color codes
class Colors:
    HEADER = '\033[95m'      # Magenta
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'          # Reset

# Box drawing characters
BOX_H = '─'
BOX_V = '│'
BOX_TL = '┌'
BOX_TR = '┐'
BOX_BL = '└'
BOX_BR = '┘'
BOX_T = '┬'
BOX_B = '┴'
BOX_L = '├'
BOX_R = '┤'
BOX_X = '┼'


def c(text, color):
    """Wrap text in color codes."""
    return f"{color}{text}{Colors.END}"


def header(text, width=60):
    """Print a fancy header box."""
    inner_width = width - 2
    print()
    print(c(BOX_TL + BOX_H * inner_width + BOX_TR, Colors.CYAN))
    # Center the text, accounting for emoji width (emoji = 2 chars display)
    display_text = f" {text} "
    # Simple centering
    content = display_text.center(inner_width)
    print(c(BOX_V, Colors.CYAN) + c(content, Colors.BOLD + Colors.YELLOW) + c(BOX_V, Colors.CYAN))
    print(c(BOX_BL + BOX_H * inner_width + BOX_BR, Colors.CYAN))


def subheader(text, width=60):
    """Print a subheader."""
    print()
    print(c(f"{'─' * 3} {text} {'─' * (width - len(text) - 5)}", Colors.DIM))


def info(label, value, label_width=20):
    """Print a labeled info line."""
    print(f"  {c(label + ':', Colors.CYAN):<{label_width + 10}} {value}")


def success(text):
    """Print a success message."""
    print(c(f"  ✓ {text}", Colors.GREEN))


def warning(text):
    """Print a warning message."""
    print(c(f"  ⚠ {text}", Colors.YELLOW))


def error(text):
    """Print an error message."""
    print(c(f"  ✗ {text}", Colors.RED))


def progress(text):
    """Print a progress message."""
    print(c(f"  ⋯ {text}", Colors.DIM))


def token_table_header():
    """Print the header for the token activation table."""
    print()
    print(c(f"{'Pos':<4} {'Token':<15} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'L2 Norm':>10}", Colors.BOLD))
    print(c("─" * 75, Colors.DIM))


def token_table_row(pos, token, mean, std, min_val, max_val, l2_norm, highlight=False):
    """Print a row in the token activation table."""
    token_display = f"'{token}'"
    if len(token_display) > 13:
        token_display = token_display[:12] + "…"
    
    row_color = Colors.YELLOW + Colors.BOLD if highlight else ""
    end = Colors.END if highlight else ""
    
    token_str = c(f"{token_display:<15}", Colors.GREEN)
    print(f"{row_color}{c(f'{pos:<4}', Colors.DIM)} {token_str} "
          f"{mean:>10.4f} {std:>10.4f} {min_val:>10.4f} {max_val:>10.4f} "
          f"{c(f'{l2_norm:>10.4f}', Colors.CYAN)}{end}")


def activation_box(layer, shape, dtype):
    """Print a box showing activation info."""
    print()
    print(c(BOX_TL + BOX_H * 40 + BOX_TR, Colors.GREEN))
    print(c(BOX_V, Colors.GREEN) + c(f" Layer {layer} Activation", Colors.BOLD + Colors.GREEN) + " " * (40 - len(f" Layer {layer} Activation") - 1) + c(BOX_V, Colors.GREEN))
    print(c(BOX_L + BOX_H * 40 + BOX_R, Colors.GREEN))
    print(c(BOX_V, Colors.GREEN) + f"  Shape: {shape}" + " " * (40 - len(f"  Shape: {shape}") - 1) + c(BOX_V, Colors.GREEN))
    print(c(BOX_V, Colors.GREEN) + f"  Dtype: {dtype}" + " " * (40 - len(f"  Dtype: {dtype}") - 1) + c(BOX_V, Colors.GREEN))
    print(c(BOX_BL + BOX_H * 40 + BOX_BR, Colors.GREEN))


def raw_values_box(token, first_values, last_values, dim):
    """Print raw activation values in a nice format."""
    print()
    print(c(f"Raw values for last token '{token}' (dim={dim}):", Colors.BOLD))
    print(c("─" * 60, Colors.DIM))
    
    # Format values nicely
    print(c("  First 10:", Colors.CYAN))
    for i, v in enumerate(first_values[:10]):
        bar = "█" * int(abs(v) * 5) if abs(v) < 2 else "█" * 10
        color = Colors.GREEN if v >= 0 else Colors.RED
        print(f"    [{i:>4}] {v:>8.4f} {c(bar, color)}")
    
    print(c("  ...", Colors.DIM))
    
    print(c(f"  Last 10:", Colors.CYAN))
    start_idx = dim - 10
    for i, v in enumerate(last_values[-10:]):
        bar = "█" * int(abs(v) * 5) if abs(v) < 2 else "█" * 10
        color = Colors.GREEN if v >= 0 else Colors.RED
        print(f"    [{start_idx + i:>4}] {v:>8.4f} {c(bar, color)}")


def model_info_box(model_name, device, dtype, num_layers, target_layer):
    """Print model configuration info."""
    print()
    width = 50
    print(c(BOX_TL + BOX_H * (width - 2) + BOX_TR, Colors.BLUE))
    print(c(BOX_V, Colors.BLUE) + c(" Model Configuration", Colors.BOLD) + " " * (width - 22) + c(BOX_V, Colors.BLUE))
    print(c(BOX_L + BOX_H * (width - 2) + BOX_R, Colors.BLUE))
    
    lines = [
        f"  Model: {model_name}",
        f"  Device: {device}",
        f"  Dtype: {dtype}",
        f"  Layers: {num_layers}",
        f"  Target: Layer {target_layer}",
    ]
    for line in lines:
        padding = width - len(line) - 2
        print(c(BOX_V, Colors.BLUE) + line + " " * padding + c(BOX_V, Colors.BLUE))
    
    print(c(BOX_BL + BOX_H * (width - 2) + BOX_BR, Colors.BLUE))


def tokens_display(tokens, token_ids):
    """Display tokenization info nicely."""
    subheader("Tokenization")
    print(f"  {c('Tokens:', Colors.CYAN)} {len(tokens)}")
    print()
    for i, (tok, tid) in enumerate(zip(tokens, token_ids)):
        print(f"    {c(f'[{i}]', Colors.DIM)} {c(repr(tok), Colors.GREEN):<20} → {c(str(tid), Colors.YELLOW)}")


def model_output(prompt, response):
    """Print the model's generated output."""
    print()
    print(c(BOX_TL + BOX_H * 58 + BOX_TR, Colors.GREEN))
    print(c(BOX_V, Colors.GREEN) + c(" 💬 Model Output", Colors.BOLD + Colors.GREEN) + " " * 41 + c(BOX_V, Colors.GREEN))
    print(c(BOX_L + BOX_H * 58 + BOX_R, Colors.GREEN))
    
    # Wrap response text
    response_lines = []
    words = response.split()
    current_line = ""
    for word in words:
        if len(current_line) + len(word) + 1 <= 54:
            current_line = current_line + " " + word if current_line else word
        else:
            response_lines.append(current_line)
            current_line = word
    if current_line:
        response_lines.append(current_line)
    
    if not response_lines:
        response_lines = ["(no response generated)"]
    
    for line in response_lines:
        padding = 56 - len(line)
        print(c(BOX_V, Colors.GREEN) + f" {c(line, Colors.BOLD)}" + " " * padding + c(BOX_V, Colors.GREEN))
    
    print(c(BOX_BL + BOX_H * 58 + BOX_BR, Colors.GREEN))


def final_summary(prompt, layer, shape, response=None):
    """Print a final summary."""
    print()
    print(c("═" * 60, Colors.CYAN))
    print(c(" Summary", Colors.BOLD + Colors.CYAN))
    print(c("═" * 60, Colors.CYAN))
    print(f"  Prompt:   {c(repr(prompt), Colors.GREEN)}")
    print(f"  Layer:    {c(str(layer), Colors.YELLOW)}")
    print(f"  Shape:    {c(str(shape), Colors.CYAN)} tensor")
    if response:
        # Truncate long responses
        display_response = response[:60] + "..." if len(response) > 60 else response
        print(f"  Response: {c(repr(display_response), Colors.GREEN + Colors.BOLD)}")
    print()


def logit_lens_table(predictions):
    """Print logit lens predictions in a nice table."""
    print()
    print(c(f"{'Rank':<6} {'Token':<20} {'Probability':>12} {'Logit':>10}", Colors.BOLD))
    print(c("─" * 52, Colors.DIM))
    
    for i, (token, prob, logit) in enumerate(predictions):
        rank = i + 1
        # Create a visual probability bar
        bar_len = int(prob * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        
        # Color based on probability
        if prob > 0.5:
            prob_color = Colors.GREEN + Colors.BOLD
        elif prob > 0.1:
            prob_color = Colors.YELLOW
        else:
            prob_color = Colors.DIM
        
        token_display = repr(token) if len(token) < 15 else repr(token[:12] + "...")
        print(f"  {c(f'#{rank:<4}', Colors.CYAN)} {c(token_display, Colors.GREEN):<20} "
              f"{c(f'{prob:>6.1%}', prob_color)} {c(bar, prob_color)} {logit:>8.2f}")


def layer_evolution_table(layer_predictions, target_layer=None):
    """Show how the top prediction changes across layers."""
    print()
    print(c(f"{'Layer':<8} {'Top Prediction':<25} {'Probability':>12}", Colors.BOLD))
    print(c("─" * 48, Colors.DIM))
    
    for layer_idx, token, prob in layer_predictions:
        # Determine layer label
        if layer_idx == 0:
            label = "embed"
        else:
            label = f"L{layer_idx-1}"
        
        # Highlight target layer
        is_target = (target_layer is not None and layer_idx == target_layer + 1)
        
        # Create visual bar
        bar_len = int(prob * 15)
        bar = "█" * bar_len
        
        # Color based on probability
        if prob > 0.5:
            color = Colors.GREEN + Colors.BOLD
        elif prob > 0.1:
            color = Colors.YELLOW
        else:
            color = Colors.DIM
        
        token_display = repr(token) if len(token) < 20 else repr(token[:17] + "...")
        
        if is_target:
            print(c(f"→ {label:<6} {token_display:<25} {prob:>6.1%} {bar}", Colors.CYAN + Colors.BOLD))
        else:
            print(f"  {c(label, Colors.DIM):<14} {c(token_display, Colors.GREEN):<25} "
                  f"{c(f'{prob:>6.1%}', color)} {c(bar, color)}")

