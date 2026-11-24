"""
drone_avoidance.py
──────────────────────────────────────────────────────────────────────────────
Reactive potential-field collision avoidance for MARSAI drone swarm.

Design
------
Each DroneNode calls `AvoidanceLayer.apply(cmd, own_pose, peer_poses)` in its
nav_loop before publishing cmd_vel.  The layer computes a repulsive velocity
vector away from any peer drone closer than `influence_radius` (default 6 m)
and blends it into the commanded velocity.  Drones closer than `hard_stop`
(default 3 m) receive a full override — the drone backs away regardless of
its nav goal until separation is restored.

Peer poses are shared via the /drone_poses topic (JSON string):
  {"drone_1": {"x": ..., "y": ..., "z": ...},
   "drone_2": {...}, ...}

Each DroneNode publishes its own pose there and subscribes to update a local
peer table — see the integration snippet at the bottom.

Usage
-----
    from drone_avoidance import AvoidanceLayer

    # Once, in __init__:
    self.avoidance = AvoidanceLayer(own_id="drone_1")

    # In nav_loop, replace  self.cmd_vel_pub.publish(cmd)  with:
    cmd = self.avoidance.apply(cmd, self.pose, self.peer_poses)
    self.cmd_vel_pub.publish(cmd)
"""

import math
from geometry_msgs.msg import Twist


class AvoidanceLayer:
    """
    Potential-field repulsion layer.

    Parameters
    ----------
    own_id          : str   — this drone's agent_id (used to skip self in peer table)
    influence_radius: float — distance (m) at which repulsion begins (default 6 m)
    hard_stop_radius: float — distance (m) at which nav goal is fully overridden (default 3 m)
    repulsion_gain  : float — scales how hard the drone pushes away (default 1.5)
    max_repulsion   : float — caps repulsion speed m/s so it can't fight a strong nav cmd (default 1.2)
    blend_weight    : float — 0..1; how much repulsion mixes into nav cmd in soft zone (default 0.6)
    """

    def __init__(
        self,
        own_id: str,
        influence_radius: float = 6.0,
        hard_stop_radius: float = 3.0,
        repulsion_gain: float = 1.5,
        max_repulsion: float = 1.2,
        blend_weight: float = 0.6,
    ):
        self.own_id           = own_id
        self.influence_radius = influence_radius
        self.hard_stop_radius = hard_stop_radius
        self.repulsion_gain   = repulsion_gain
        self.max_repulsion    = max_repulsion
        self.blend_weight     = blend_weight

    # ─────────────────────────────────────────────────────────────────────────

    def apply(
        self,
        cmd: Twist,
        own_pose: dict,
        peer_poses: dict,
    ) -> Twist:
        """
        Apply avoidance and return a (possibly modified) Twist.

        Parameters
        ----------
        cmd        : geometry_msgs/Twist  — velocity from nav_loop
        own_pose   : {"x", "y", "z", "yaw"}
        peer_poses : {agent_id: {"x", "y", "z"}, ...}  — excludes self
        """
        rep_x, rep_y, rep_z = 0.0, 0.0, 0.0
        hard_override = False

        for peer_id, pp in peer_poses.items():
            if peer_id == self.own_id:
                continue

            dx = own_pose["x"] - pp["x"]
            dy = own_pose["y"] - pp["y"]
            dz = own_pose["z"] - pp["z"]
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)

            if dist < 1e-3:
                # Exact overlap — push straight up as a last resort
                rep_z += self.max_repulsion
                hard_override = True
                continue

            if dist >= self.influence_radius:
                continue  # outside influence bubble

            # Repulsion magnitude: stronger when closer
            # F = gain * (1/dist - 1/R) * (1/dist²)  — standard APF formula
            magnitude = self.repulsion_gain * (
                (1.0 / dist - 1.0 / self.influence_radius) / (dist * dist)
            )
            magnitude = min(magnitude, self.max_repulsion)

            # Unit vector pointing away from peer
            rep_x += magnitude * dx / dist
            rep_y += magnitude * dy / dist
            rep_z += magnitude * dz / dist

            if dist < self.hard_stop_radius:
                hard_override = True

        # ── Compose final command ─────────────────────────────────────────────
        result = Twist()

        if hard_override:
            # Full override: ignore nav goal, move away from all peers
            result.linear.x  = _clamp(rep_x, self.max_repulsion)
            result.linear.y  = _clamp(rep_y, self.max_repulsion)
            result.linear.z  = _clamp(rep_z, self.max_repulsion)
            result.angular.z = 0.0
        elif rep_x != 0.0 or rep_y != 0.0 or rep_z != 0.0:
            # Soft blend: weighted sum of nav cmd + repulsion
            w = self.blend_weight
            result.linear.x  = (1 - w) * cmd.linear.x  + w * _clamp(rep_x, self.max_repulsion)
            result.linear.y  = (1 - w) * cmd.linear.y  + w * _clamp(rep_y, self.max_repulsion)
            result.linear.z  = (1 - w) * cmd.linear.z  + w * _clamp(rep_z, self.max_repulsion)
            result.angular.z = cmd.angular.z
        else:
            # No nearby peers — pass through unchanged
            result = cmd

        return result

    def nearest_peer_distance(self, own_pose: dict, peer_poses: dict) -> float:
        """Returns distance (m) to closest peer; inf if no peers."""
        min_dist = float("inf")
        for peer_id, pp in peer_poses.items():
            if peer_id == self.own_id:
                continue
            dx = own_pose["x"] - pp["x"]
            dy = own_pose["y"] - pp["y"]
            dz = own_pose["z"] - pp["z"]
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            min_dist = min(min_dist, dist)
        return min_dist


# ─────────────────────────────────────────────────────────────────────────────

def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))
