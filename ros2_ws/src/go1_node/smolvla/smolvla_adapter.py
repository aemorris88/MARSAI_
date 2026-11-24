#!/usr/bin/env python3
"""
smolvla_adapter.py
──────────────────────────────────────────────────────────────────────────────
SmolVLA (450M) inference adapter for MARSAI.

SmolVLA is a compact Vision-Language-Action model from HuggingFace (~450M
params, ~2GB weights). We use it as the first VLA ablation baseline, replacing
the Nemotron Nano 8B LLM backbone for action generation on the Go1 ground
robots.

Hardware target: RTX 4060 Ti 16GB (local edge, same machine as Ollama Nano).
SmolVLA at fp16: ~2GB VRAM — easily fits alongside other processes.

Installation (run once):
    pip install lerobot transformers accelerate pillow torch torchvision

Model download (auto on first use, ~2GB):
    The adapter fetches 'lerobot/smolvla-base' from HuggingFace Hub on first
    call and caches under ~/.cache/huggingface/hub/.

Action space mapping:
    SmolVLA outputs continuous 7-DOF actions designed for robot arms.
    We map its action vector to MARSAI's discrete nav directives:
        action[0]  — x velocity component   → forward/backward
        action[1]  — y velocity component   → strafe (unused for Go1)
        action[6]  — gripper/wrist norm     → repurposed as confidence proxy

    The mapping is intentionally simple for the ablation baseline — the goal
    is to measure whether VLA-style visual grounding improves task success rate
    over the pure-LLM Nemotron baseline, not to get optimal VLA performance.

Usage:
    from smolvla_adapter import SmolVLAAdapter
    adapter = SmolVLAAdapter()
    directive = adapter.predict(obs_image_path, task_instruction, context)
"""

import json
import logging
import math
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


# ── Action dimension mapping ──────────────────────────────────────────────────
# SmolVLA 7-DOF output → MARSAI nav directive
# [x_vel, y_vel, z_vel, roll, pitch, yaw_rate, gripper]
ACTION_X_IDX     = 0   # forward velocity component
ACTION_YAW_IDX   = 5   # yaw rate component
ACTION_CONF_IDX  = 6   # gripper — repurposed as confidence proxy

FORWARD_THRESHOLD  = 0.1   # action[0] > threshold → "move"
ROTATE_THRESHOLD   = 0.15  # |action[5]| > threshold → "rotate"
CONF_SCALE         = 0.5   # gripper range [-1,1] → shift to [0,1] confidence


class SmolVLAAdapter:
    """
    Wraps the SmolVLA model for MARSAI nav directive generation.

    Parameters
    ----------
    device        : 'cuda' | 'cpu' | 'auto'  (auto selects cuda if available)
    dtype         : torch.float16 (default) or torch.float32
    model_name    : HuggingFace model id (default: lerobot/smolvla-base)
    warmup        : run a dummy forward pass on init to avoid first-call latency
    """

    def __init__(
        self,
        device: str = "auto",
        dtype: torch.dtype = torch.float16,
        model_name: str = "lerobot/smolvla-base",
        warmup: bool = True,
    ):
        from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.common.policies.smolvla.configuration_smolvla import SmolVLAConfig

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.dtype  = dtype

        logger.info(f"[SmolVLA] Loading {model_name} on {device} ({dtype})…")
        t0 = time.time()

        cfg    = SmolVLAConfig.from_pretrained(model_name)
        self.policy = SmolVLAPolicy.from_pretrained(
            model_name,
            config=cfg,
            torch_dtype=dtype,
        ).to(device).eval()

        logger.info(f"[SmolVLA] Loaded in {time.time()-t0:.1f}s")

        if warmup:
            self._warmup()

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(
        self,
        image_path: Optional[str],
        task_instruction: str,
        context: dict,
    ) -> dict:
        """
        Generate a MARSAI nav directive from a camera frame + task description.

        Parameters
        ----------
        image_path        : path to current camera frame (jpg/png); if None,
                            a blank frame is used (obs-less ablation mode)
        task_instruction  : natural-language task description from orchestrator
        context           : dict with keys: current_pose, next_waypoints,
                            obstacle_dist_m, detected

        Returns
        -------
        dict with keys: action, confidence, reason, vla_raw
            action     : "move" | "rotate" | "stop" | "report"
            confidence : float 0–1
            reason     : human-readable explanation
            vla_raw    : raw SmolVLA action vector (list[float])
        """
        obs    = self._build_obs(image_path, task_instruction)
        t0     = time.time()

        with torch.no_grad():
            raw_action = self.policy.select_action(obs)

        latency_ms = (time.time() - t0) * 1000
        logger.debug(f"[SmolVLA] Inference {latency_ms:.1f} ms")

        # raw_action is a tensor of shape (7,) or (1, 7)
        action_vec = raw_action.squeeze().cpu().float().numpy()
        directive  = self._map_action(action_vec, context, latency_ms)
        return directive

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build_obs(self, image_path: Optional[str], instruction: str) -> dict:
        """Build the observation dict SmolVLA expects."""
        if image_path and Path(image_path).exists():
            img = Image.open(image_path).convert("RGB").resize((224, 224))
        else:
            # Blank frame for obs-less / LiDAR-only scenarios
            img = Image.new("RGB", (224, 224), color=(128, 128, 128))

        img_tensor = (
            torch.from_numpy(np.array(img))
            .permute(2, 0, 1)
            .float()
            .div(255.0)
            .unsqueeze(0)
            .to(self.device, self.dtype)
        )

        return {
            "observation.image":    img_tensor,
            "observation.language_instruction": [instruction],
        }

    def _map_action(
        self, action_vec: np.ndarray, context: dict, latency_ms: float
    ) -> dict:
        """
        Map SmolVLA 7-DOF continuous action to a MARSAI nav directive.

        The mapping is a simple threshold rule — intentionally minimal for
        the ablation baseline. The key ablation question is whether VLA visual
        grounding changes *what* action is chosen, not whether the mapping is
        sophisticated.
        """
        x_vel   = float(action_vec[ACTION_X_IDX])   if len(action_vec) > ACTION_X_IDX   else 0.0
        yaw_vel = float(action_vec[ACTION_YAW_IDX]) if len(action_vec) > ACTION_YAW_IDX else 0.0
        gripper = float(action_vec[ACTION_CONF_IDX]) if len(action_vec) > ACTION_CONF_IDX else 0.0

        # Confidence proxy: shift gripper from [-1,1] to [0,1]
        confidence = float(np.clip((gripper + 1.0) * CONF_SCALE, 0.0, 1.0))

        obstacle_dist = float(context.get("obstacle_dist_m", float("inf")))

        # Obstacle override — always stop if too close regardless of VLA output
        if obstacle_dist < 0.3:
            return {
                "action":     "stop",
                "confidence": confidence,
                "reason":     f"Obstacle at {obstacle_dist:.2f} m — VLA overridden by safety stop",
                "vla_raw":    action_vec.tolist(),
                "latency_ms": latency_ms,
            }

        # Map to discrete action
        if abs(yaw_vel) > ROTATE_THRESHOLD and abs(yaw_vel) > abs(x_vel):
            action = "rotate"
            reason = f"VLA yaw component {yaw_vel:.2f} > threshold {ROTATE_THRESHOLD}"
        elif x_vel > FORWARD_THRESHOLD:
            action = "move"
            reason = f"VLA x component {x_vel:.2f} > threshold {FORWARD_THRESHOLD}"
        elif x_vel < -FORWARD_THRESHOLD:
            action = "stop"
            reason = f"VLA x component {x_vel:.2f} < -{FORWARD_THRESHOLD} (back-off signal)"
        else:
            action = "stop"
            reason = f"VLA output below thresholds (x={x_vel:.2f}, yaw={yaw_vel:.2f})"

        return {
            "action":     action,
            "confidence": confidence,
            "reason":     reason,
            "vla_raw":    action_vec.tolist(),
            "latency_ms": latency_ms,
        }

    def _warmup(self):
        """Dummy forward pass to compile kernels and avoid first-call latency spike."""
        logger.info("[SmolVLA] Running warmup pass…")
        try:
            self.predict(None, "warmup", {})
            logger.info("[SmolVLA] Warmup complete.")
        except Exception as e:
            logger.warning(f"[SmolVLA] Warmup failed (non-fatal): {e}")
