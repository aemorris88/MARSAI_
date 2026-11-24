"""
hipergator_client.py
──────────────────────────────────────────────────────────────────────────────
REST client for Nemotron 3 Super deployed on UF HiPerGator.

HiPerGator serves the model through a vLLM-compatible OpenAI endpoint.
Adjust `model_name` to match your HiPerGator allocation.

Usage:
    client = HiPerGatorClient(
        endpoint="http://hipergator.rc.ufl.edu:8080/v1",
        api_key="your-hipergator-token"
    )
    response = client.generate("Your prompt here")
"""

import json
import time
import logging
import requests
from typing import Optional

logger = logging.getLogger(__name__)


class HiPerGatorClient:
    """
    Thin wrapper around the vLLM OpenAI-compatible API
    hosted on UF's HiPerGator HPC cluster.
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        model_name: str = "nemotron-super",
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        self.endpoint    = endpoint.rstrip("/")
        self.api_key     = api_key
        self.model_name  = model_name
        self.timeout     = timeout
        self.max_retries = max_retries

        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
        })

    # ─────────────────────────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> str:
        """
        Send a prompt to Nemotron Super and return the raw text response.
        Retries on transient errors with exponential backoff.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model":       self.model_name,
            "messages":    messages,
            "temperature": temperature,
            "max_tokens":  max_tokens,
        }

        last_error = None
        for attempt in range(self.max_retries):
            try:
                resp = self.session.post(
                    f"{self.endpoint}/chat/completions",
                    json=payload,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]

            except requests.exceptions.Timeout:
                logger.warning(f"HiPerGator timeout (attempt {attempt+1}/{self.max_retries})")
                last_error = "Timeout"
            except requests.exceptions.HTTPError as e:
                logger.error(f"HiPerGator HTTP error: {e.response.status_code} — {e.response.text}")
                last_error = str(e)
                break  # Don't retry on 4xx
            except Exception as e:
                logger.error(f"HiPerGator unexpected error: {e}")
                last_error = str(e)

            time.sleep(2 ** attempt)  # backoff: 1s, 2s, 4s

        raise RuntimeError(f"HiPerGator call failed after {self.max_retries} attempts: {last_error}")

    def health_check(self) -> bool:
        """Ping the model endpoint to verify connectivity."""
        try:
            resp = self.session.get(f"{self.endpoint}/models", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False
