#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# hipergator_nemotron_super.sh
# SLURM job script to launch Nemotron 3 Super via vLLM on HiPerGator
#
# Submit with:
#   sbatch hipergator_nemotron_super.sh
#
# Then note the assigned port and node hostname from the output log.
# Use that as your hipergator_endpoint in the ROS2 launch.
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=nemotron_super
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1           # 1× A100 — adjust to your allocation
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=nemotron_super_%j.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your@email.ufl.edu

# ── Environment setup ─────────────────────────────────────────────────────────
module load conda
conda activate nemotron_env         # create this env with vLLM installed

# Print the node + port so the ROS2 client knows where to connect
PORT=8080
echo "========================================"
echo "Nemotron Super running on: $(hostname):${PORT}"
echo "Endpoint: http://$(hostname):${PORT}/v1"
echo "========================================"

# ── Launch vLLM server ────────────────────────────────────────────────────────
python -m vllm.entrypoints.openai.api_server \
  --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --host 0.0.0.0 \
  --port ${PORT} \
  --served-model-name "nemotron-super" \
  --enforce-eager

# ── Notes ─────────────────────────────────────────────────────────────────────
# • If Nemotron-3-8B-Chat-4k is gated on HuggingFace, set:
#     export HUGGING_FACE_HUB_TOKEN=your_token
# • For the 4B / Nano variant on local PC, use ollama or the vllm server
#   separately with the nano_server.py script.
