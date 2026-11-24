# SmolVLA Ablation Baseline — MARSAI

SmolVLA (450M params, ~2 GB weights) is the first VLA ablation baseline for the
MARSAI paper. It replaces Nemotron Nano 8B as the edge action model on the Go1
ground robots, allowing a direct comparison of:

- **LLM-only**: Nemotron Nano 8B via Ollama (8001)
- **VLA baseline**: SmolVLA 450M via lerobot (8002)

## Hardware requirements

| Component | Spec |
|---|---|
| GPU | RTX 4060 Ti 16 GB (or any CUDA GPU with ≥ 4 GB VRAM) |
| VRAM at fp16 | ~2 GB (fits alongside Ollama's Nano at ~6 GB) |
| CPU fallback | Supported but ~10–20× slower |

## Install (once)

```bash
pip install lerobot transformers accelerate pillow torch torchvision
```

The model weights (~2 GB) download automatically on first use from HuggingFace Hub.

## File overview

```
smolvla/
├── smolvla_adapter.py    # SmolVLA inference wrapper — action generation logic
├── smolvla_server.py     # HTTP server on port 8002 (same interface as nano_server)
├── run_vla_ablation.py   # Ablation trial runner: SmolVLA vs Nano side-by-side
└── README.md             # This file
```

## Running the SmolVLA server

```bash
# From the repo root:
python3 ros2_ws/src/go1_node/smolvla/smolvla_server.py --port 8002 --device auto
```

Health check:
```bash
curl http://localhost:8002/health
# → {"status": "ok", "model": "smolvla-base", "device": "cuda"}
```

## Switching go1_node to SmolVLA

The `go1_node` connects to whatever server `nano_url` points at.
Point it at port 8002 to use SmolVLA instead of Nano:

```bash
ros2 run go1_node go1_node --ros-args \
  -p agent_id:=go1_a \
  -p nano_url:=http://localhost:8002
```

No other code changes needed — the `/generate` interface is identical.

## Running the ablation comparison

Both servers must be up:

```bash
# Terminal 1 — Nano (baseline)
python3 llm_bridge/local_nano/nano_server.py --port 8001

# Terminal 2 — SmolVLA
python3 ros2_ws/src/go1_node/smolvla/smolvla_server.py --port 8002

# Terminal 3 — ablation runner
python3 ros2_ws/src/go1_node/smolvla/run_vla_ablation.py --trials 3
```

Results are printed as a comparison table and saved to `vla_ablation_results.json`.

## Action mapping

SmolVLA outputs a 7-DOF continuous action vector designed for robot arms.
We map it to MARSAI's discrete nav directives:

| SmolVLA output | Mapping |
|---|---|
| `action[0]` (x velocity) > 0.1 | → `"move"` |
| `action[5]` (yaw rate) > 0.15 | → `"rotate"` |
| `action[6]` (gripper) | → confidence proxy (shifted to [0,1]) |
| obstacle_dist_m < 0.3 | → `"stop"` (safety override regardless of VLA) |

This mapping is intentionally simple — the ablation question is whether VLA
visual grounding changes *what* action is chosen, not whether the mapping is optimal.

## Paper context

- SmolVLA serves as the **small VLA ablation** in Table 6 (projected configs)
- Compared against: full MARSAI (Nano 8B + RAG), no-RAG variant, cloud-only Super 49B
- Key expected finding: SmolVLA adds visual grounding but lacks the SAR-domain
  language understanding that Nano acquires through the RAG memory — expect
  higher escalation rate on novel terrain states
