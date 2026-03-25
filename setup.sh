#!/bin/bash
set -e

echo "=================================================================================="
echo " zbed Setup: Download Model Weights"
echo "=================================================================================="
echo ""

MODEL_DIR="model"
MODEL_NAME="sentence-transformers/static-retrieval-mrl-en-v1"
TOKENIZER_URL="https://huggingface.co/${MODEL_NAME}/resolve/main/0_StaticEmbedding/tokenizer.json"

# Check for required tools
for cmd in curl python3; do
    if ! command -v "$cmd" &> /dev/null; then
        echo "Error: $cmd is required but not installed."
        exit 1
    fi
done

# Check for zig
if ! command -v zig &> /dev/null; then
    echo "Warning: zig not found. You'll need Zig 0.13+ to build zbed."
fi

echo "Tools found."
echo ""

# Create model directory
mkdir -p "$MODEL_DIR"

# Download tokenizer.json
if [ -f "$MODEL_DIR/tokenizer.json" ]; then
    echo "Tokenizer already downloaded ($MODEL_DIR/tokenizer.json)"
else
    echo "Downloading tokenizer.json..."
    curl -L -o "$MODEL_DIR/tokenizer.json" "$TOKENIZER_URL"
    echo "Tokenizer downloaded."
fi
echo ""

# Download and quantize model weights
if [ -f "$MODEL_DIR/modelint8_256dim.safetensors" ] || [ -f "$MODEL_DIR/modelint8_512dim.safetensors" ]; then
    echo "Model weights already present."
else
    echo "Downloading and quantizing model weights..."
    echo "This requires: pip install huggingface-hub safetensors torch numpy"

    # Install Python dependencies if needed
    python3 -c "import huggingface_hub, safetensors, torch, numpy" 2>/dev/null || {
        echo "Installing Python dependencies..."
        pip3 install --quiet huggingface-hub safetensors torch numpy
    }

    python3 - <<'PYEOF'
import os
import numpy as np

model_dir = "model"

try:
    from huggingface_hub import hf_hub_download
    from safetensors.numpy import load_file, save_file

    model_name = "sentence-transformers/static-retrieval-mrl-en-v1"

    # Download the full model
    print("Downloading model from HuggingFace...")
    model_path = hf_hub_download(
        repo_id=model_name,
        filename="0_StaticEmbedding/model.safetensors",
        local_dir=model_dir,
    )

    # Load the full-precision model
    print("Loading model weights...")
    tensors = load_file(model_path)

    # Find the embedding weight tensor
    weight_key = None
    for key in tensors:
        if "weight" in key or "embedding" in key:
            weight_key = key
            break

    if weight_key is None:
        print("Error: Could not find embedding weights in model")
        exit(1)

    weights = tensors[weight_key]
    print(f"  Original shape: {weights.shape}, dtype: {weights.dtype}")

    vocab_size, full_dim = weights.shape

    # Truncate to 256 dimensions for efficiency
    dim = 256
    weights_trunc = weights[:, :dim].astype(np.float32)
    print(f"  Truncated to {dim} dimensions")

    # Quantize to int8
    scales = np.abs(weights_trunc).max(axis=1).astype(np.float32)
    scales[scales == 0] = 1.0
    quantized = np.round(weights_trunc / scales[:, None] * 127).clip(-127, 127).astype(np.int8)

    # Save as safetensors
    out_path = os.path.join(model_dir, f"modelint8_{dim}dim.safetensors")
    save_file({
        "embedding.weight": quantized,
        "embedding.scales": scales,
    }, out_path)

    file_size = os.path.getsize(out_path) / (1024 * 1024)
    print(f"  Saved quantized model: {out_path} ({file_size:.1f} MB)")
    print(f"  Vocab size: {vocab_size}, Dimensions: {dim}")

except Exception as e:
    print(f"Error: {e}")
    print("If download fails, you can manually download the model:")
    print(f"  huggingface-cli download {model_name} 0_StaticEmbedding/model.safetensors")
    exit(1)
PYEOF

    echo "Model weights ready."
fi
echo ""

# Test build
if command -v zig &> /dev/null; then
    echo "Testing Zig build..."
    zig build
    echo "Build successful."
    echo ""
fi

echo "=================================================================================="
echo " Setup Complete!"
echo "=================================================================================="
echo ""
echo "Usage:"
echo "  zig build                    Build zbed"
echo "  ./zig-out/bin/zbed index .   Index current directory"
echo "  ./zig-out/bin/zbed <query>   Search for semantically similar code"
echo "  ./zig-out/bin/zbed bench     Run performance benchmark"
echo "  ./zig-out/bin/zbed status    Show index statistics"
echo ""
