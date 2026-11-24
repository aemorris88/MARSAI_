#!/bin/bash
# deploy_finetuned_model.sh
#
# Run this on your LOCAL DESKTOP after downloading the fine-tuned
# model from HiPerGator. Converts the HuggingFace checkpoint to
# GGUF format and loads it into Ollama so RescueLLM can use it.
#
# Prerequisites:
#   - ollama installed and running
#   - llama.cpp installed (for conversion)
#   - Fine-tuned model downloaded from HiPerGator to:
#     ~/rescuellm_finetuned/

set -e

FINETUNED_HF_DIR=~/rescuellm_finetuned
GGUF_DIR=~/rescuellm_finetuned_gguf
MODEL_NAME=rescuellm-nano-sar

echo "=== RescueLLM Fine-Tuned Model Deployment ==="
echo "Source: $FINETUNED_HF_DIR"

# ── Step 1: Convert HF model to GGUF (Q4_K_M quantization) ───────
# Q4_K_M gives the best balance of quality and VRAM at ~4GB footprint
echo ""
echo "Step 1: Converting to GGUF (Q4_K_M)..."
mkdir -p $GGUF_DIR

python3 $(which llama-convert-hf-to-gguf 2>/dev/null || echo \
    "$(pip show llama-cpp-python 2>/dev/null | grep Location | \
    cut -d' ' -f2)/llama_cpp/convert_hf_to_gguf.py") \
    $FINETUNED_HF_DIR \
    --outfile $GGUF_DIR/rescuellm-nano-sar-f16.gguf \
    --outtype f16

# Quantize to Q4_K_M
llama-quantize \
    $GGUF_DIR/rescuellm-nano-sar-f16.gguf \
    $GGUF_DIR/rescuellm-nano-sar-q4km.gguf \
    Q4_K_M

echo "GGUF conversion complete."

# ── Step 2: Create Ollama Modelfile ───────────────────────────────
echo ""
echo "Step 2: Creating Ollama Modelfile..."

cat > $GGUF_DIR/Modelfile << 'MODELFILE'
FROM ./rescuellm-nano-sar-q4km.gguf

SYSTEM """
You are RescueLLM, an edge-deployed large action model for autonomous
search and rescue. You produce structured JSON navigation directives
for Unitree Go1 ground robots operating in post-disaster environments.
Always respond with valid JSON containing action, confidence, and
reason fields. action must be one of: move, rotate, stop, report.
confidence must be a float between 0.0 and 1.0.
"""

PARAMETER temperature 0.2
PARAMETER top_p 0.9
PARAMETER num_ctx 4096
PARAMETER stop "<|eot_id|>"
MODELFILE

echo "Modelfile created."

# ── Step 3: Load into Ollama ──────────────────────────────────────
echo ""
echo "Step 3: Loading into Ollama..."
cd $GGUF_DIR
ollama create $MODEL_NAME -f Modelfile

echo ""
echo "=== Deployment complete ==="
echo "Model available as: $MODEL_NAME"
echo ""
echo "To test:"
echo "  ollama run $MODEL_NAME"
echo ""
echo "To use in RescueLLM, update go1_node.py:"
echo "  MODEL_NAME = '$MODEL_NAME'  # was 'nemotron-mini'"
echo ""
echo "Run a quick sanity check:"
ollama run $MODEL_NAME \
    "You are stuck near target [8, 5]. \
     RAG memory shows north and east directions failed. \
     Generate a recovery action directive."
