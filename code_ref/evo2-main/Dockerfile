# Use an NVIDIA CUDA base image with development tools (needed for potential compilations)
from nvcr.io/nvidia/pytorch:25.04-py3 as base
run apt-get update && apt-get install -y git python3-pip python3-tomli && rm -rf /var/lib/apt/lists/*

# Install flash-attn and evo2
run pip install evo2

workdir /workdir

# --- Notes ---
# 1. Hardware Requirement: This container assumes it will be run on a host machine
#    with NVIDIA GPUs having compute capability >= 8.9 (e.g., H100) for full FP8 support. FP8 is required for accurate performance of Evo 2 40B or 1B models.
# 2. Runtime Requirement: You MUST run this container with the --gpus flag, e.g., --gpus all.
# 3. Model Downloads: Evo2 models are downloaded on first use by the library itself
#    (e.g., when you call `Evo2('evo2_7b')`). They are NOT included in the image.
#    Mounting a volume for the Hugging Face cache is recommended to avoid re-downloading.
