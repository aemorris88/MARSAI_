#!/usr/bin/env python3
"""
validate_drone_coverage.py
──────────────────────────────────────────────────────────────────────────────
Standalone validation script for MARSAI drone swarm coverage.
No ROS2 or Webots required — pure geometry simulation.

Replicates the exact logic from drone_dispatcher.py:
  - _partition(): equal vertical strips along X axis
  - _lawnmower(): boustrophedon (lawnmower) pattern at cruise altitude
  - detection_radius = 0.5 m (from paper Section 5)
  - scene = 41 m × 39 m (from paper evaluation environment)

Metrics reported
────────────────
  1. Area coverage %  — fraction of 0.5m grid cells within detection radius
                        of at least one waypoint
  2. First-to-detect  — expected time (s) until first survivor is encountered
                        for each swarm config (uniform random survivor placement)
  3. Coverage overlap — % of cells covered by >1 drone (redundancy / waste)
  4. Waypoint count   — total waypoints generated per config
  5. Estimated flight time — total path length / cruise_speed (seconds)

Usage
─────
  python validate_drone_coverage.py [--trials N] [--seed S]
"""

import argparse
import math
import random
import sys
from dataclasses import dataclass, field
from typing import Optional


# ── Scene constants (from paper) ─────────────────────────────────────────────

SCENE_X_MIN  = 0.0
SCENE_X_MAX  = 41.0
SCENE_Y_MIN  = 0.0
SCENE_Y_MAX  = 39.0
DETECTION_R  = 0.5    # metres — from paper Section 5
CRUISE_ALT   = 5.0    # metres
CRUISE_SPEED = 2.0    # m/s — from drone_node.py default
GRID_STEP    = 3.0    # metres between waypoints — dispatcher default
EVAL_GRID    = 0.5    # resolution for coverage grid (metres)


# ── Dispatcher logic (copied from drone_dispatcher.py) ───────────────────────

def _frange(start: float, stop: float, step: float):
    val = start
    while val <= stop + 1e-9:
        yield val
        val += step


def partition(area: dict, n: int) -> list[dict]:
    """Split area into n equal vertical strips along X axis."""
    x_min, x_max = area["x_min"], area["x_max"]
    y_min, y_max = area["y_min"], area["y_max"]
    strip_w = (x_max - x_min) / n
    return [
        {
            "x_min": x_min + i * strip_w,
            "x_max": x_min + (i + 1) * strip_w,
            "y_min": y_min,
            "y_max": y_max,
        }
        for i in range(n)
    ]


def lawnmower(strip: dict, altitude: float, step: float) -> list[list[float]]:
    """Boustrophedon waypoint path inside a strip."""
    waypoints = []
    x = strip["x_min"] + step / 2
    col = 0
    while x <= strip["x_max"]:
        ys = list(_frange(strip["y_min"], strip["y_max"], step))
        if col % 2 == 1:
            ys = list(reversed(ys))
        for y in ys:
            waypoints.append([round(x, 2), round(y, 2), altitude])
        x += step
        col += 1
    return waypoints


def build_waypoints(n_drones: int, search_area: dict, grid_step: float = GRID_STEP):
    """Return list of per-drone waypoint lists for n_drones."""
    strips = partition(search_area, n_drones)
    return [lawnmower(s, CRUISE_ALT, grid_step) for s in strips]


# ── Coverage grid ─────────────────────────────────────────────────────────────

def build_coverage_grid(all_waypoints_per_drone: list[list[list[float]]],
                        detection_r: float = DETECTION_R,
                        grid_res: float = EVAL_GRID) -> dict:
    """
    For each grid cell centre, record which drones cover it.
    Returns a dict: (gx, gy) -> set of drone indices that cover it.
    """
    # Build grid cell centres
    xs = [SCENE_X_MIN + (i + 0.5) * grid_res
          for i in range(int((SCENE_X_MAX - SCENE_X_MIN) / grid_res))]
    ys = [SCENE_Y_MIN + (j + 0.5) * grid_res
          for j in range(int((SCENE_Y_MAX - SCENE_Y_MIN) / grid_res))]

    total_cells = len(xs) * len(ys)

    # Build spatial index: for each drone, collect all waypoint (x,y) pairs
    # Then for each cell check if any waypoint is within detection_r
    coverage: dict[tuple, set] = {}

    for drone_idx, waypoints in enumerate(all_waypoints_per_drone):
        wpt_xy = [(w[0], w[1]) for w in waypoints]
        for gx, cx in enumerate(xs):
            for gy, cy in enumerate(ys):
                for wx, wy in wpt_xy:
                    if math.hypot(cx - wx, cy - wy) <= detection_r:
                        key = (gx, gy)
                        if key not in coverage:
                            coverage[key] = set()
                        coverage[key].add(drone_idx)
                        break  # one waypoint within range is enough

    return coverage, total_cells


# ── Path length & flight time ─────────────────────────────────────────────────

def path_length(waypoints: list[list[float]]) -> float:
    """3D path length through a list of [x, y, z] waypoints."""
    if len(waypoints) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(waypoints)):
        dx = waypoints[i][0] - waypoints[i-1][0]
        dy = waypoints[i][1] - waypoints[i-1][1]
        dz = waypoints[i][2] - waypoints[i-1][2]
        total += math.sqrt(dx*dx + dy*dy + dz*dz)
    return total


# ── First-to-detect simulation ────────────────────────────────────────────────

def simulate_first_detect(all_waypoints_per_drone: list[list[list[float]]],
                           n_trials: int = 1000,
                           detection_r: float = DETECTION_R,
                           speed: float = CRUISE_SPEED,
                           seed: Optional[int] = 42) -> dict:
    """
    Monte Carlo: place one survivor at a random location in the scene.
    Each drone flies its lawnmower path at cruise_speed.
    Record the time until any drone comes within detection_r of the survivor.
    Returns mean, median, min, max, and p95 detection time (seconds).
    If survivor is never detected, record None (counted separately).
    """
    rng = random.Random(seed)
    detection_times = []
    missed = 0

    for _ in range(n_trials):
        # Random survivor in scene
        sx = rng.uniform(SCENE_X_MIN, SCENE_X_MAX)
        sy = rng.uniform(SCENE_Y_MIN, SCENE_Y_MAX)

        # For each drone, find the time it first passes within detection_r
        # Time = cumulative path length to that waypoint / speed
        # (approximate: treat path as straight segments, detection at waypoint)
        best_time = None

        for waypoints in all_waypoints_per_drone:
            cum_dist = 0.0
            prev = waypoints[0] if waypoints else None
            for wpt in waypoints:
                if prev is not None:
                    dx = wpt[0] - prev[0]
                    dy = wpt[1] - prev[1]
                    dz = wpt[2] - prev[2]
                    cum_dist += math.sqrt(dx*dx + dy*dy + dz*dz)
                # Check detection at this waypoint
                dist_to_survivor = math.hypot(wpt[0] - sx, wpt[1] - sy)
                if dist_to_survivor <= detection_r:
                    t = cum_dist / speed
                    if best_time is None or t < best_time:
                        best_time = t
                    break  # this drone found it; no need to continue its path
                prev = wpt

        if best_time is None:
            missed += 1
        else:
            detection_times.append(best_time)

    if not detection_times:
        return {"mean": None, "median": None, "min": None, "max": None,
                "p95": None, "miss_rate": 1.0, "n_trials": n_trials}

    detection_times.sort()
    n = len(detection_times)
    return {
        "mean":      sum(detection_times) / n,
        "median":    detection_times[n // 2],
        "min":       detection_times[0],
        "max":       detection_times[-1],
        "p95":       detection_times[int(0.95 * n)],
        "miss_rate": missed / n_trials,
        "n_trials":  n_trials,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def run_config(label: str, n_drones: int, trials: int, seed: int):
    search_area = {
        "x_min": SCENE_X_MIN, "x_max": SCENE_X_MAX,
        "y_min": SCENE_Y_MIN, "y_max": SCENE_Y_MAX,
    }

    per_drone_wpts = build_waypoints(n_drones, search_area)
    all_wpts_flat  = [w for drone in per_drone_wpts for w in drone]

    # Coverage grid
    coverage, total_cells = build_coverage_grid(per_drone_wpts)
    covered_cells    = len(coverage)
    overlap_cells    = sum(1 for s in coverage.values() if len(s) > 1)
    coverage_pct     = covered_cells / total_cells * 100
    overlap_pct      = overlap_cells / total_cells * 100

    # Flight stats
    per_drone_lengths = [path_length(wpts) for wpts in per_drone_wpts]
    total_path        = sum(per_drone_lengths)
    max_drone_time    = max(per_drone_lengths) / CRUISE_SPEED  # parallel flight
    total_waypoints   = sum(len(wpts) for wpts in per_drone_wpts)

    # First-to-detect
    ftd = simulate_first_detect(per_drone_wpts, n_trials=trials, seed=seed)

    print(f"\n{'='*60}")
    print(f"  Config: {label}  ({n_drones} drone{'s' if n_drones > 1 else ''})")
    print(f"{'='*60}")
    print(f"  Scene:            {SCENE_X_MAX}m × {SCENE_Y_MAX}m = "
          f"{SCENE_X_MAX * SCENE_Y_MAX:.0f} m²")
    print(f"  Detection radius: {DETECTION_R} m")
    print(f"  Grid step:        {GRID_STEP} m")
    print(f"  Total waypoints:  {total_waypoints}  "
          f"({[len(w) for w in per_drone_wpts]} per drone)")
    print()
    print(f"  ── Coverage ──────────────────────────────────")
    print(f"  Covered cells:    {covered_cells:,} / {total_cells:,}  "
          f"({coverage_pct:.1f}%)")
    print(f"  Overlap cells:    {overlap_cells:,}  ({overlap_pct:.1f}% of scene)")
    print()
    print(f"  ── Flight time (parallel) ────────────────────")
    for i, (wpts, L) in enumerate(zip(per_drone_wpts, per_drone_lengths)):
        print(f"  drone_{i+1}:  {len(wpts):3d} waypoints, "
              f"path {L:6.1f} m,  time {L/CRUISE_SPEED:.1f} s")
    print(f"  Mission time:     {max_drone_time:.1f} s  "
          f"(limited by slowest drone)")
    print(f"  Total path:       {total_path:.1f} m  (all drones combined)")
    print()
    print(f"  ── First-to-detect ({trials} trials) ────────────")
    if ftd["mean"] is None:
        print(f"  Survivor NEVER detected (miss rate = 100%)")
    else:
        print(f"  Mean detect time: {ftd['mean']:.1f} s")
        print(f"  Median:           {ftd['median']:.1f} s")
        print(f"  Best case:        {ftd['min']:.1f} s")
        print(f"  Worst detected:   {ftd['max']:.1f} s")
        print(f"  95th percentile:  {ftd['p95']:.1f} s")
        print(f"  Miss rate:        {ftd['miss_rate']*100:.1f}%  "
              f"(survivor in uncovered zone)")

    return {
        "label":          label,
        "n_drones":       n_drones,
        "coverage_pct":   coverage_pct,
        "overlap_pct":    overlap_pct,
        "total_waypoints": total_waypoints,
        "mission_time_s": max_drone_time,
        "ftd_mean_s":     ftd["mean"],
        "ftd_p95_s":      ftd["p95"],
        "miss_rate":      ftd["miss_rate"],
    }


def print_comparison(results: list[dict]):
    r1 = next(r for r in results if r["n_drones"] == 1)
    r3 = next(r for r in results if r["n_drones"] == 3)

    print(f"\n{'='*60}")
    print(f"  SWARM vs SINGLE — SUMMARY COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Metric':<30} {'1 drone':>10} {'3 drones':>10} {'Δ':>10}")
    print(f"  {'-'*60}")

    cov_delta = r3["coverage_pct"] - r1["coverage_pct"]
    print(f"  {'Coverage %':<30} {r1['coverage_pct']:>9.1f}% "
          f"{r3['coverage_pct']:>9.1f}%  {cov_delta:>+.1f}pp")

    miss_delta = r3["miss_rate"] - r1["miss_rate"]
    print(f"  {'Miss rate %':<30} {r1['miss_rate']*100:>9.1f}% "
          f"{r3['miss_rate']*100:>9.1f}%  {miss_delta*100:>+.1f}pp")

    if r1["ftd_mean_s"] and r3["ftd_mean_s"]:
        speedup = r1["ftd_mean_s"] / r3["ftd_mean_s"]
        print(f"  {'Mean first-detect time (s)':<30} {r1['ftd_mean_s']:>10.1f} "
              f"{r3['ftd_mean_s']:>10.1f}  {speedup:>+.2f}×")
        p95_speedup = r1["ftd_p95_s"] / r3["ftd_p95_s"]
        print(f"  {'P95 first-detect time (s)':<30} {r1['ftd_p95_s']:>10.1f} "
              f"{r3['ftd_p95_s']:>10.1f}  {p95_speedup:>+.2f}×")

    mt_delta = r3["mission_time_s"] - r1["mission_time_s"]
    print(f"  {'Mission time (s)':<30} {r1['mission_time_s']:>10.1f} "
          f"{r3['mission_time_s']:>10.1f}  {mt_delta:>+.1f}s")

    print(f"  {'Total waypoints':<30} {r1['total_waypoints']:>10} "
          f"{r3['total_waypoints']:>10}")
    print()
    print(f"  Note: 3-drone mission time = time for slowest strip (parallel flight).")
    print(f"        1-drone mission time = full scene sequential sweep.")
    print()


def main():
    parser = argparse.ArgumentParser(description="MARSAI drone coverage validator")
    parser.add_argument("--trials", type=int, default=5000,
                        help="Monte Carlo trials for first-to-detect (default 5000)")
    parser.add_argument("--seed",   type=int, default=42,
                        help="Random seed for reproducibility (default 42)")
    args = parser.parse_args()

    print(f"\nMARS AI — Drone Coverage Validation")
    print(f"Scene: {SCENE_X_MAX}m × {SCENE_Y_MAX}m | "
          f"Detection radius: {DETECTION_R}m | "
          f"Grid step: {GRID_STEP}m | "
          f"Trials: {args.trials}")

    results = []
    for n in [1, 3]:
        label = f"{'Swarm (3×)' if n == 3 else 'Single drone'}"
        r = run_config(label, n, trials=args.trials, seed=args.seed)
        results.append(r)

    print_comparison(results)

    # Quick sanity check assertions
    r1 = next(r for r in results if r["n_drones"] == 1)
    r3 = next(r for r in results if r["n_drones"] == 3)
    assert r3["coverage_pct"] >= r1["coverage_pct"] - 1.0, \
        "ERROR: 3-drone coverage should be >= single drone"
    assert r3["mission_time_s"] < r1["mission_time_s"], \
        "ERROR: 3-drone parallel mission should be faster than single sweep"
    if r1["ftd_mean_s"] and r3["ftd_mean_s"]:
        assert r3["ftd_mean_s"] < r1["ftd_mean_s"], \
            "ERROR: 3-drone first-detect should be faster than single"
    print("  ✓ All sanity checks passed.")


if __name__ == "__main__":
    main()
