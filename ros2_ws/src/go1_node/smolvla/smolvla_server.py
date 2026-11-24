#!/usr/bin/env python3
"""
smolvla_server.py
──────────────────────────────────────────────────────────────────────────────
HTTP inference server for SmolVLA — drop-in replacement for nano_server.py.

Exposes the SAME /generate endpoint interface as nano_server.py so that
go1_node.py can switch between Nano and SmolVLA by changing only the
nano_url parameter at launch time.

Endpoint
────────
POST /generate
  Request  (JSON):
    agent_id    : str   — which Go1 is asking
    prompt      : str   — task instruction from orchestrator
    context     : str   — JSON string: {current_pose, next_waypoints,
                           obstacle_dist_m, detected}
    temperature : float — ignored by VLA (kept for interface compatibility)
    max_tokens  : int   — ignored by VLA (kept for interface compatibility)
    image_path  : str   — optional path to current camera frame (jpg/png)

  Response (JSON):
    response    : str   — JSON-encoded nav directive
                          {action, confidence, reason, vla_raw, latency_ms}

GET /health
  Returns {"status": "ok", "model": "smolvla-base", "device": "<device>"}

Launch
──────
  python3 smolvla_server.py [--port 8002] [--device auto]

  Then in go1_node launch:
    ros2 run go1_node go1_node --ros-args \
      -p agent_id:=go1_a \
      -p nano_url:=http://localhost:8002
"""

import argparse
import json
import logging
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lazy-load adapter (so import errors surface clearly) ──────────────────────
_adapter = None


def get_adapter(device: str = "auto"):
    global _adapter
    if _adapter is None:
        from smolvla_adapter import SmolVLAAdapter
        _adapter = SmolVLAAdapter(device=device, warmup=True)
    return _adapter


# ── HTTP handler ──────────────────────────────────────────────────────────────

class SmolVLAHandler(BaseHTTPRequestHandler):

    # Suppress default request log spam; use our own logger
    def log_message(self, fmt, *args):
        logger.debug(fmt % args)

    def do_GET(self):
        if self.path == "/health":
            adapter = get_adapter()
            body = json.dumps({
                "status": "ok",
                "model":  "smolvla-base",
                "device": str(adapter.device),
            }).encode()
            self._respond(200, body)
        else:
            self._respond(404, b'{"error":"not found"}')

    def do_POST(self):
        if self.path != "/generate":
            self._respond(404, b'{"error":"not found"}')
            return

        length = int(self.headers.get("Content-Length", 0))
        raw    = self.rfile.read(length)

        try:
            req = json.loads(raw)
        except json.JSONDecodeError as e:
            self._respond(400, json.dumps({"error": f"bad JSON: {e}"}).encode())
            return

        agent_id    = req.get("agent_id", "unknown")
        instruction = req.get("prompt", "Navigate to next waypoint safely.")
        image_path  = req.get("image_path", None)

        # context arrives as a JSON string (matches nano_server interface)
        try:
            context = json.loads(req.get("context", "{}"))
        except json.JSONDecodeError:
            context = {}

        logger.info(f"[SmolVLA] /generate — agent={agent_id}")

        try:
            adapter   = get_adapter()
            directive = adapter.predict(image_path, instruction, context)
            response  = json.dumps(directive)
            body      = json.dumps({"response": response}).encode()
            self._respond(200, body)
        except Exception as e:
            logger.error(f"[SmolVLA] Inference error: {e}", exc_info=True)
            # Return a safe "stop + escalate" fallback matching nano_server behaviour
            fallback = json.dumps({
                "action":     "stop",
                "confidence": 0.0,
                "reason":     f"SmolVLA inference error: {e}",
                "escalate":   True,
            })
            body = json.dumps({"response": fallback}).encode()
            self._respond(200, body)  # 200 so go1_node doesn't crash

    def _respond(self, code: int, body: bytes):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SmolVLA inference server")
    parser.add_argument("--port",   type=int, default=8002,
                        help="Port to listen on (default 8002; Nano uses 8001)")
    parser.add_argument("--device", type=str, default="auto",
                        help="'cuda', 'cpu', or 'auto' (default auto)")
    args = parser.parse_args()

    logger.info(f"[SmolVLA] Pre-loading model on device={args.device}…")
    get_adapter(device=args.device)

    server = HTTPServer(("0.0.0.0", args.port), SmolVLAHandler)
    logger.info(f"[SmolVLA] Server listening on port {args.port}")
    logger.info(f"[SmolVLA] Endpoint: POST http://localhost:{args.port}/generate")
    logger.info(f"[SmolVLA] Health:   GET  http://localhost:{args.port}/health")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("[SmolVLA] Shutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
