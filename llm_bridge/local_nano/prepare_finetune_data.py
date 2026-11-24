"""
prepare_finetune_data.py

Converts RescueLLM trial logs and RAG memory files into the
instruction-following JSONL format expected by NeMo Megatron Bridge
for QLoRA fine-tuning of Nemotron 3 Nano.

Run this locally before uploading to HiPerGator:
    python3 prepare_finetune_data.py

Output files:
    finetune_data/train.jsonl   (~80% of examples)
    finetune_data/val.jsonl     (~20% of examples)
"""

import json
import os
import random

random.seed(42)
OUTPUT_DIR = "finetune_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

examples = []

# ── SOURCE 1: Trial decision logs ────────────────────────────────
# Load from the rs7a JSON we generated earlier.
# Adjust path to wherever your file lives.
TRIAL_LOG_PATH = "results_data/rs7a_rag_memory_trials.json"

try:
    with open(TRIAL_LOG_PATH) as f:
        trial_data = json.load(f)

    for trial in trial_data["trials"]:
        t = trial["trial"]
        complete = trial["complete"]

        # For each dog that completed in this trial
        for dog in ["dog_1", "dog_2", "dog_3"]:
            ttr_key = f"{dog}_ttr"
            dir_key = f"{dog}_failed_dirs"

            ttr = trial.get(ttr_key)
            failed_dirs = trial.get(dir_key)
            if ttr is None:
                continue  # dog did not complete this trial

            target = trial_data["agent_targets"][dog]

            # Positive example: reached target
            examples.append({
                "input": (
                    f"You are a search and rescue robot navigating a "
                    f"post-disaster environment. Your target zone is at "
                    f"coordinates {target}. You encountered {failed_dirs} "
                    f"failed recovery directions before reaching your target. "
                    f"This is Trial {t}. "
                    f"Generate a structured navigation action directive."
                ),
                "output": json.dumps({
                    "action": "move",
                    "confidence": round(min(0.95, 0.6 + (1 / (1 + failed_dirs))), 2),
                    "reason": (
                        f"Navigating toward target at {target}. "
                        f"Reached target in {ttr:.1f}s after {failed_dirs} "
                        f"obstacle recovery attempts. "
                        f"Prioritizing direct approach while avoiding "
                        f"previously failed directions."
                    )
                }),
                "source": f"trial_{t}_{dog}_success"
            })

            # Negative / recovery examples based on failed dir count
            for i in range(failed_dirs):
                examples.append({
                    "input": (
                        f"You are a search and rescue robot. You are stuck "
                        f"near coordinates {target} in Trial {t}. "
                        f"You have already attempted {i+1} recovery "
                        f"direction(s) without success. "
                        f"Generate a recovery action directive that tries "
                        f"a new direction."
                    ),
                    "output": json.dumps({
                        "action": "rotate",
                        "confidence": round(max(0.45, 0.9 - (i * 0.1)), 2),
                        "reason": (
                            f"Obstacle recovery attempt {i+1} near {target}. "
                            f"Previous directions failed. Rotating to find "
                            f"alternative path. Will back up before rotating "
                            f"to clear collision geometry."
                        )
                    }),
                    "source": f"trial_{t}_{dog}_recovery_{i+1}"
                })

    print(f"[Source 1] Trial logs: {len(examples)} examples so far")

except FileNotFoundError:
    print(f"[Source 1] WARNING: {TRIAL_LOG_PATH} not found. "
          f"Adjust path and re-run.")

# ── SOURCE 2: RescueNet waypoint generation pairs ─────────────────
# Synthetic pairs teaching Nano to generate structured JSON waypoints
# from aerial segmentation descriptions.

TERRAIN_TYPES = [
    ("Road-Clear", True, 0.95),
    ("Background", True, 0.90),
    ("Building-No-Damage", True, 0.75),
    ("Building-Medium-Damage", False, 0.35),
    ("Road-Blocked", False, 0.15),
    ("Building-Major-Damage", False, 0.10),
    ("Building-Total-Destruction", False, 0.05),
    ("Water", False, 0.05),
    ("Tree", False, 0.20),
    ("Vehicle", False, 0.15),
]

WAYPOINT_SCENARIOS = [
    {"start": [0, 0], "target": [8, 5],   "obstacle_density": "low"},
    {"start": [0, 0], "target": [-6, -8], "obstacle_density": "high"},
    {"start": [0, 0], "target": [12, -12],"obstacle_density": "low"},
    {"start": [5, 3], "target": [15, 10], "obstacle_density": "medium"},
    {"start": [-3, -5],"target": [8, 8],  "obstacle_density": "medium"},
]

for scenario in WAYPOINT_SCENARIOS:
    for terrain, traversable, confidence in TERRAIN_TYPES:
        examples.append({
            "input": (
                f"Aerial segmentation shows the following terrain near "
                f"your current position {scenario['start']}: {terrain}. "
                f"Your target is at {scenario['target']}. "
                f"Obstacle density in this zone is {scenario['obstacle_density']}. "
                f"Generate a structured navigation action directive."
            ),
            "output": json.dumps({
                "action": "move" if traversable else "rotate",
                "confidence": confidence,
                "reason": (
                    f"Terrain type {terrain} is "
                    f"{'traversable' if traversable else 'non-traversable'}. "
                    f"{'Proceeding toward' if traversable else 'Avoiding'} "
                    f"target at {scenario['target']}. "
                    f"Obstacle density is {scenario['obstacle_density']} — "
                    f"{'maintaining current heading' if traversable else 'rotating to find clear path'}."
                ),
                "waypoints": scenario["target"] if traversable else None
            }),
            "source": f"rescuenet_{terrain.replace('-','_').lower()}"
        })

print(f"[Source 2] RescueNet pairs: {len(examples)} examples so far")

# ── SOURCE 3: RAG memory recovery scenarios ───────────────────────
# Teaches Nano to correctly interpret and act on RAG context
# when it is injected into the prompt.

RAG_SCENARIOS = [
    {
        "target": [8, 5],
        "past_failures": ["north", "northeast"],
        "past_success": "east",
        "obstacle": "collapsed wall segment at (6, 4)"
    },
    {
        "target": [-6, -8],
        "past_failures": ["south", "southwest", "west"],
        "past_success": "southeast",
        "obstacle": "debris field at (-4, -6)"
    },
    {
        "target": [12, -12],
        "past_failures": [],
        "past_success": "direct",
        "obstacle": "none"
    },
    {
        "target": [8, 5],
        "past_failures": ["north", "northeast", "east", "southeast"],
        "past_success": None,
        "obstacle": "dense rubble cluster at (7, 4)"
    },
    {
        "target": [-6, -8],
        "past_failures": ["south"],
        "past_success": "southeast",
        "obstacle": "blocked road at (-5, -7)"
    },
]

for scenario in RAG_SCENARIOS:
    failures_str = (
        ", ".join(scenario["past_failures"])
        if scenario["past_failures"]
        else "none"
    )
    rag_context = (
        f"RAG memory retrieved for 5m radius near {scenario['target']}: "
        f"Previously failed recovery directions: {failures_str}. "
        f"Known obstacle: {scenario['obstacle']}. "
        f"Previously successful direction: "
        f"{scenario['past_success'] or 'not yet found'}."
    )

    if scenario["past_success"]:
        # Positive: RAG memory has a successful direction
        examples.append({
            "input": (
                f"You are stuck near your target at {scenario['target']}. "
                f"{rag_context} "
                f"Use this memory to select a recovery action."
            ),
            "output": json.dumps({
                "action": "rotate",
                "confidence": 0.82,
                "reason": (
                    f"RAG memory indicates {scenario['past_success']} "
                    f"direction succeeded previously near {scenario['target']}. "
                    f"Avoiding previously failed directions: {failures_str}. "
                    f"Rotating toward {scenario['past_success']} and backing "
                    f"up to clear obstacle at {scenario['obstacle']}."
                )
            }),
            "source": "rag_recovery_positive"
        })
    else:
        # Escalation: all directions failed, escalate to Super
        examples.append({
            "input": (
                f"You are stuck near your target at {scenario['target']}. "
                f"{rag_context} "
                f"All known recovery directions have been attempted. "
                f"Determine the appropriate action."
            ),
            "output": json.dumps({
                "action": "report",
                "confidence": 0.21,
                "reason": (
                    f"All {len(scenario['past_failures'])} recovery directions "
                    f"near {scenario['target']} have been attempted and failed. "
                    f"Obstacle at {scenario['obstacle']} blocks all known paths. "
                    f"Confidence below threshold. Escalating to Super for "
                    f"cross-agent coordination and alternative routing."
                ),
                "escalate": True
            }),
            "source": "rag_recovery_escalation"
        })

print(f"[Source 3] RAG scenarios: {len(examples)} examples so far")

# ── FORMAT AND SPLIT ──────────────────────────────────────────────
# NeMo expects each line as:
# {"input": "...", "output": "..."}
# We wrap in the Nemotron chat template format.

def to_nemo_format(example):
    return {
        "input": (
            "<|start_header_id|>system<|end_header_id|>\n"
            "You are RescueLLM, an edge-deployed large action model "
            "for autonomous search and rescue. You produce structured "
            "JSON navigation directives for Unitree Go1 ground robots "
            "operating in post-disaster environments. Always respond "
            "with valid JSON containing action, confidence, and reason "
            "fields.\n"
            "<|eot_id|>\n"
            "<|start_header_id|>user<|end_header_id|>\n"
            f"{example['input']}\n"
            "<|eot_id|>\n"
            "<|start_header_id|>assistant<|end_header_id|>"
        ),
        "output": example["output"] + "<|eot_id|>"
    }

formatted = [to_nemo_format(e) for e in examples]
random.shuffle(formatted)

split = int(len(formatted) * 0.8)
train = formatted[:split]
val   = formatted[split:]

with open(f"{OUTPUT_DIR}/train.jsonl", "w") as f:
    for ex in train:
        f.write(json.dumps(ex) + "\n")

with open(f"{OUTPUT_DIR}/val.jsonl", "w") as f:
    for ex in val:
        f.write(json.dumps(ex) + "\n")

print(f"\n✓ Done.")
print(f"  Total examples : {len(formatted)}")
print(f"  Train          : {len(train)}  → finetune_data/train.jsonl")
print(f"  Validation     : {len(val)}   → finetune_data/val.jsonl")
print(f"\nNext step: upload finetune_data/ to HiPerGator and run")
print(f"  sbatch submit_finetune.sh")
