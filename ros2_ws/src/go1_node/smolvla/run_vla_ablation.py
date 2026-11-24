#!/usr/bin/env python3
"""
run_vla_ablation.py
──────────────────────────────────────────────────────────────────────────────
Ablation trial runner: SmolVLA (450M) vs Nemotron Nano 8B baseline.

Sends the same obstacle-navigation prompts to both servers and records:
  - action chosen
  - confidence
  - latency (ms)
  - escalation rate

Both servers must be running before you start this script:
  Terminal 1:  python3 llm_bridge/local_nano/nano_server.py      # port 8001
  Terminal 2:  python3 ros2_ws/src/go1_node/smolvla/smolvla_server.py  # port 8002

Usage
─────
  python3 run_vla_ablation.py [--trials N] [--out results.json]

The script prints a comparison table and saves raw results to JSON.
"""

import argparse
import json
import statistics
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

import requests


# ── Test scenarios ────────────────────────────────────────────────────────────
# Each scenario represents a navigation decision point a Go1 would face.
# These match the types of situations from the MARSAI evaluation trials.

SCENARIOS = [
    {
        "id":          "clear_path",
        "prompt":      "Navigate to next waypoint safely. No obstacles detected.",
        "context": {
            "current_pose":   {"x": 5.0, "y": 3.0, "z": 0.0, "yaw": 0.0},
            "next_waypoints": [[8.0, 3.0], [11.0, 3.0]],
            "obstacle_dist_m": 10.0,
            "detected":        [],
        },
        "expected_action": "move",
    },
    {
        "id":          "close_obstacle",
        "prompt":      "Obstacle detected ahead at 0.4 m. Decide: rotate, stop, or report.",
        "context": {
            "current_pose":   {"x": 5.0, "y": 3.0, "z": 0.0, "yaw": 0.0},
            "next_waypoints": [[8.0, 3.0]],
            "obstacle_dist_m": 0.4,
            "detected":        ["debris"],
        },
        "expected_action": "rotate",   # or stop — either is valid
    },
    {
        "id":          "very_close_obstacle",
        "prompt":      "Emergency: obstacle at 0.2 m. Stop immediately or rotate.",
        "context": {
            "current_pose":   {"x": 5.0, "y": 3.0, "z": 0.0, "yaw": 0.0},
            "next_waypoints": [[8.0, 3.0]],
            "obstacle_dist_m": 0.2,
            "detected":        ["debris"],
        },
        "expected_action": "stop",
    },
    {
        "id":          "survivor_detected",
        "prompt":      "Possible survivor signature detected nearby. Investigate or report to orchestrator.",
        "context": {
            "current_pose":   {"x": 12.0, "y": 7.0, "z": 0.0, "yaw": 45.0},
            "next_waypoints": [[14.0, 9.0]],
            "obstacle_dist_m": 2.5,
            "detected":        ["thermal_signature"],
        },
        "expected_action": "report",
    },
    {
        "id":          "partial_path",
        "prompt":      "Navigate toward waypoint. Mild debris in path, 0.8 m clearance.",
        "context": {
            "current_pose":   {"x": 20.0, "y": 15.0, "z": 0.0, "yaw": 90.0},
            "next_waypoints": [[20.0, 18.0], [20.0, 21.0]],
            "obstacle_dist_m": 0.8,
            "detected":        ["debris"],
        },
        "expected_action": "move",   # or rotate
    },
]


# ── Trial result ──────────────────────────────────────────────────────────────

@dataclass
class TrialResult:
    scenario_id:    str
    model:          str
    action:         str
    confidence:     float
    latency_ms:     float
    escalated:      bool
    correct:        bool    # action matches expected
    raw_response:   dict = field(default_factory=dict)
    error:          Optional[str] = None


# ── Query helper ──────────────────────────────────────────────────────────────

def query_server(url: str, scenario: dict, model_label: str) -> TrialResult:
    payload = {
        "agent_id":    "go1_ablation",
        "prompt":      scenario["prompt"],
        "context":     json.dumps(scenario["context"]),
        "temperature": 0.1,
        "max_tokens":  256,
    }
    t0 = time.time()
    try:
        resp = requests.post(f"{url}/generate", json=payload, timeout=15.0)
        resp.raise_for_status()
        raw_str  = resp.json()["response"]
        # response may already be a dict (SmolVLA) or a JSON string (Nano)
        raw_dict = json.loads(raw_str) if isinstance(raw_str, str) else raw_str
        latency  = (time.time() - t0) * 1000

        action     = raw_dict.get("action",     "stop")
        confidence = float(raw_dict.get("confidence", 0.0))
        escalated  = bool(raw_dict.get("escalate", False)) or confidence < 0.4
        correct    = action == scenario["expected_action"] or (
            # "stop" and "rotate" are both acceptable for obstacle scenarios
            scenario["expected_action"] == "rotate" and action in ("stop", "rotate")
        )

        return TrialResult(
            scenario_id=scenario["id"],
            model=model_label,
            action=action,
            confidence=confidence,
            latency_ms=latency,
            escalated=escalated,
            correct=correct,
            raw_response=raw_dict,
        )

    except requests.exceptions.ConnectionError:
        return TrialResult(
            scenario_id=scenario["id"],
            model=model_label,
            action="stop",
            confidence=0.0,
            latency_ms=(time.time() - t0) * 1000,
            escalated=True,
            correct=False,
            error=f"Connection refused — is {url} running?",
        )
    except Exception as e:
        return TrialResult(
            scenario_id=scenario["id"],
            model=model_label,
            action="stop",
            confidence=0.0,
            latency_ms=(time.time() - t0) * 1000,
            escalated=True,
            correct=False,
            error=str(e),
        )


# ── Main ──────────────────────────────────────────────────────────────────────

MODELS = {
    "Nemotron Nano 8B": "http://localhost:8001",
    "SmolVLA 450M":     "http://localhost:8002",
}


def run(n_trials: int, out_path: str):
    all_results: list[TrialResult] = []

    for trial_i in range(n_trials):
        for scenario in SCENARIOS:
            for model_label, url in MODELS.items():
                r = query_server(url, scenario, model_label)
                all_results.append(r)
                status = "✓" if r.correct else "✗"
                err_note = f"  ⚠ {r.error}" if r.error else ""
                print(
                    f"  [{model_label:20s}] {scenario['id']:25s} "
                    f"→ {r.action:7s}  conf={r.confidence:.2f}  "
                    f"{r.latency_ms:6.0f}ms  {status}{err_note}"
                )

    print_summary(all_results)
    save_results(all_results, out_path)


def print_summary(results: list[TrialResult]):
    print(f"\n{'='*70}")
    print(f"  ABLATION SUMMARY: SmolVLA 450M vs Nemotron Nano 8B")
    print(f"{'='*70}")
    print(f"  {'Model':<22} {'Accuracy':>10} {'Avg Latency':>12} {'Escalation':>12}")
    print(f"  {'-'*60}")

    for model_label in MODELS:
        subset = [r for r in results if r.model == model_label]
        if not subset:
            continue
        valid   = [r for r in subset if r.error is None]
        errors  = [r for r in subset if r.error is not None]
        acc     = sum(r.correct for r in valid) / len(valid) * 100 if valid else 0
        lat     = statistics.mean(r.latency_ms for r in valid) if valid else 0
        esc_rt  = sum(r.escalated for r in valid) / len(valid) * 100 if valid else 0

        err_note = f"  ({len(errors)} connection errors)" if errors else ""
        print(
            f"  {model_label:<22} {acc:>9.1f}%  {lat:>10.0f}ms  "
            f"{esc_rt:>10.1f}%{err_note}"
        )

    print()
    print("  Note: accuracy = action matches expected for scenario.")
    print("  Escalation rate = confidence < 0.4 OR model flagged escalate=True.")
    print()


def save_results(results: list[TrialResult], path: str):
    data = [asdict(r) for r in results]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Raw results saved → {path}")


def main():
    parser = argparse.ArgumentParser(description="VLA ablation trial runner")
    parser.add_argument("--trials", type=int, default=1,
                        help="Number of times to repeat all scenarios (default 1)")
    parser.add_argument("--out", type=str, default="vla_ablation_results.json",
                        help="Output JSON path (default vla_ablation_results.json)")
    args = parser.parse_args()

    print(f"\nMARS AI — VLA Ablation Trial Runner")
    print(f"Scenarios: {len(SCENARIOS)} | Trials: {args.trials} | Models: {len(MODELS)}")
    print(f"{'─'*70}")

    run(args.trials, args.out)


if __name__ == "__main__":
    main()
