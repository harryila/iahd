#!/bin/bash
# Setup script for the logit lens project
# Works on Mac (local) and Lambda (remote)

set -e  # Exit on error

echo "=== Logit Lens Project Setup ==="
echo ""

# Detect environment
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected - running on Lambda or GPU machine"
    ENV_TYPE="gpu"
else
    echo "No GPU detected - running locally (Mac/CPU)"
    ENV_TYPE="local"
fi

# Create virtual environment (optional but recommended)
if [ "$1" == "--venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "Virtual environment activated: venv/"
fi

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip3 install --upgrade pip

# Install requirements
echo ""
echo "Installing Python packages..."
pip3 install -r requirements.txt

# GPU-specific setup
if [ "$ENV_TYPE" == "gpu" ]; then
    echo ""
    echo "Installing TensorFlow 1.15 for GPT-2 notebook..."
    pip3 install tensorflow-gpu==1.15 2>/dev/null || pip3 install tensorflow==1.15
    
    echo ""
    echo "Cloning GPT-2 repository..."
    if [ ! -d "gpt-2" ]; then
        git clone https://github.com/openai/gpt-2.git
    else
        echo "GPT-2 repo already exists, skipping..."
    fi
    
    echo ""
    echo "Downloading GPT-2 model (this may take a while)..."
    cd gpt-2
    python3 download_model.py 1558M
    cd ..
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
if [ "$ENV_TYPE" == "gpu" ]; then
    echo "  1. Run GPT-2 logit lens:"
    echo "     cd gpt-2 && python3 ../the_logit_lens_on_gpt2_activations.py"
    echo ""
    echo "  2. Run Llama activation extraction:"
    echo "     python3 extract_llama_activations.py"
else
    echo "  1. Run local activation extraction (smaller models):"
    echo "     python3 extract_activations_local.py"
    echo ""
    echo "  2. For Llama/GPT-2 large models, connect to Lambda GPU"
fi
echo ""

