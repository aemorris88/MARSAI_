"""
nano_server.py
──────────────────────────────────────────────────────────────────────────────
FastAPI HTTP server wrapping Nemotron 3 Nano running locally via ollama or
vLLM. Each Go1 dog node makes POST /generate requests to this server.

Run:
    # With ollama (recommended for local PC):
    ollama pull nemotron-mini  # or your Nano model tag
    python nano_server.py --backend ollama --model nemotron-mini --port 8001

    # With vLLM:
    python nano_server.py --backend vllm --model nvidia/Nemotron-3-4B-Chat --port 8001
"""

import argparse
import json
import logging
from typing import Optional

import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nano_server")

app = FastAPI(title="Nemotron Nano Local Server")


# ── Request / Response schemas ───────────────────────────────────────────────

class GenerateRequest(BaseModel):
    agent_id: str
    prompt: str
    context: Optional[str] = None     # JSON string with sensor/map context
    temperature: float = 0.1
    max_tokens: int = 512


class GenerateResponse(BaseModel):
    agent_id: str
    response: str
    model: str
    latency_ms: float


# ── Backend adapters ─────────────────────────────────────────────────────────

class OllamaBackend:
    def __init__(self, model: str, host: str = "http://localhost:11434"):
        self.model = model
        self.host  = host

    def generate(self, prompt: str, temperature: float, max_tokens: int) -> str:
        payload = {
            "model":   self.model,
            "prompt":  prompt,
            "stream":  False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        resp = requests.post(f"{self.host}/api/generate", json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()["response"]


class VLLMBackend:
    def __init__(self, model: str, host: str = "http://localhost:8000"):
        self.model = model
        self.host  = host

    def generate(self, prompt: str, temperature: float, max_tokens: int) -> str:
        payload = {
            "model":       self.model,
            "messages":    [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens":  max_tokens,
        }
        resp = requests.post(f"{self.host}/v1/chat/completions", json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


# ── Global backend instance (set at startup) ─────────────────────────────────
backend = None
MODEL_NAME = ""


# ── Endpoint ─────────────────────────────────────────────────────────────────

@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    import time

    full_prompt = _build_nano_prompt(req)
    start = time.time()
    try:
        response_text = backend.generate(full_prompt, req.temperature, req.max_tokens)
    except Exception as e:
        logger.error(f"Nano inference failed for {req.agent_id}: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    latency = (time.time() - start) * 1000

    logger.info(f"[{req.agent_id}] {latency:.0f}ms — {response_text[:60]}...")
    return GenerateResponse(
        agent_id=req.agent_id,
        response=response_text,
        model=MODEL_NAME,
        latency_ms=latency,
    )


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_nano_prompt(req: GenerateRequest) -> str:
    ctx = ""
    if req.context:
        try:
            ctx_data = json.loads(req.context)
            ctx = f"\nSENSOR CONTEXT:\n{json.dumps(ctx_data, indent=2)}"
        except json.JSONDecodeError:
            ctx = f"\nCONTEXT: {req.context}"

    return f"""You are the onboard navigation AI for a Unitree Go1 quadruped robot in a post-disaster environment.
Your task is to decide the NEXT SINGLE WAYPOINT or ACTION to take given your current situation.
Respond ONLY with a JSON object, no explanation:
{{"action": "move|rotate|stop|report", "waypoint": [x, y, z], "yaw_deg": 0, "reason": "short reason", "confidence": 0.0}}
{ctx}
MISSION INSTRUCTION: {req.prompt}
"""


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["ollama", "vllm"], default="ollama")
    parser.add_argument("--model", default="nemotron-mini")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--backend-host", default=None,
                        help="Override host for ollama/vllm backend")
    args = parser.parse_args()

    MODEL_NAME = args.model
    if args.backend == "ollama":
        bh = args.backend_host or "http://localhost:11434"
        backend = OllamaBackend(args.model, bh)
    else:
        bh = args.backend_host or "http://localhost:8000"
        backend = VLLMBackend(args.model, bh)

    logger.info(f"Starting Nano server | backend={args.backend} model={args.model} port={args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
