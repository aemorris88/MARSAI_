"""
go1_node_finetuned_patch.py

Patch for your existing go1_node.py to support switching between
the base Nemotron Nano model and the fine-tuned RescueLLM-SAR model.

Apply this change in go1_node.py:
    Replace the MODEL_NAME constant with the block below.

Usage:
    # Use base model (original behavior)
    ros2 run rescuellm go1_node --ros-args -p use_finetuned:=false

    # Use fine-tuned model
    ros2 run rescuellm go1_node --ros-args -p use_finetuned:=true
"""

import rclpy
from rclpy.node import Node
import requests
import json
import os

# ── Model selection ───────────────────────────────────────────────
BASE_MODEL      = "nemotron-mini"           # original base model
FINETUNED_MODEL = "rescuellm-nano-sar"      # QLoRA fine-tuned model

OLLAMA_URL = "http://localhost:11434/api/generate"

class Go1Node(Node):
    def __init__(self):
        super().__init__('go1_node')

        # Declare parameter so it can be set at launch time
        self.declare_parameter('use_finetuned', False)
        use_ft = self.get_parameter('use_finetuned').value

        self.model_name = FINETUNED_MODEL if use_ft else BASE_MODEL
        self.get_logger().info(
            f"[Go1Node] Using model: {self.model_name} "
            f"({'fine-tuned' if use_ft else 'base'})"
        )

    def query_nano(self, prompt: str) -> dict:
        """
        Query the local Nano model via Ollama.
        Returns parsed JSON action directive.
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "top_p": 0.9,
                "num_ctx": 4096
            }
        }

        try:
            response = requests.post(
                OLLAMA_URL,
                json=payload,
                timeout=5.0  # delta threshold = 5.0s
            )
            response.raise_for_status()
            raw = response.json().get("response", "{}")

            # Strip any markdown code fences if present
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]

            directive = json.loads(raw.strip())

            # Validate required fields
            assert "action" in directive
            assert "confidence" in directive
            assert "reason" in directive
            assert directive["action"] in \
                ["move", "rotate", "stop", "report"]
            assert 0.0 <= float(directive["confidence"]) <= 1.0

            return directive

        except requests.exceptions.Timeout:
            self.get_logger().warn(
                "[Go1Node] Nano query timed out — escalating to Super"
            )
            return {
                "action": "stop",
                "confidence": 0.0,
                "reason": "Timeout — escalating to Super",
                "escalate": True
            }

        except (json.JSONDecodeError, AssertionError, KeyError) as e:
            self.get_logger().error(
                f"[Go1Node] Nano response parse error: {e}"
            )
            return {
                "action": "stop",
                "confidence": 0.0,
                "reason": f"Parse error: {e} — escalating to Super",
                "escalate": True
            }

    def should_escalate(self, directive: dict) -> bool:
        """
        Check all three escalation conditions.
        """
        confidence = float(directive.get("confidence", 0.0))
        explicit_flag = directive.get("escalate", False)

        # Condition 1: Low confidence
        if confidence < 0.4:
            self.get_logger().info(
                f"[Go1Node] Escalating: confidence={confidence:.2f} < 0.4"
            )
            return True

        # Condition 2: Explicit escalation flag from model
        if explicit_flag:
            self.get_logger().info(
                "[Go1Node] Escalating: model requested escalation"
            )
            return True

        # Condition 3: Timeout is already handled in query_nano()
        return False


# ── Ablation experiment helper ────────────────────────────────────
def run_ablation_trial(use_finetuned: bool, use_rag: bool):
    """
    Helper to launch a trial with a specific ablation configuration.
    Call from your trial runner script.

    use_finetuned: True = QLoRA fine-tuned Nano, False = base Nano
    use_rag:       True = per-agent RAG memory enabled, False = disabled
    """
    config = {
        "model": FINETUNED_MODEL if use_finetuned else BASE_MODEL,
        "rag_enabled": use_rag,
        "label": (
            "FT+RAG"     if use_finetuned and use_rag else
            "FT_NoRAG"   if use_finetuned and not use_rag else
            "NoFT_RAG"   if not use_finetuned and use_rag else
            "NoFT_NoRAG"
        )
    }
    print(f"[Ablation] Starting trial: {config['label']}")
    print(f"  Model : {config['model']}")
    print(f"  RAG   : {'enabled' if config['rag_enabled'] else 'disabled'}")
    return config
