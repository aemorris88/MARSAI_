#!/usr/bin/env python3
"""
drone_dispatcher.py
──────────────────────────────────────────────────────────────────────────────
Planning-agent helper: generalized N-drone task dispatch.

The orchestrator (planning agent) calls DroneDispatcher.assign() with a list
of drone IDs and a task description; the dispatcher splits the search area
into per-drone sectors, builds per-drone subtask messages, and publishes them
to /subtask.  Each DroneNode picks up only the message addressed to it.

Usage (inside your planning agent node)
─────────────────────────────────────────
    from drone_dispatcher import DroneDispatcher

    # Once, in __init__:
    self.dispatcher = DroneDispatcher(self, drone_ids=["drone_1","drone_2","drone_3"])

    # When a SAR task arrives:
    self.dispatcher.assign(
        task_type  = "scout",
        search_area = {"x_min": 0, "x_max": 20, "y_min": -18, "y_max": 15},
        altitude   = 5.0,
        grid_step  = 3.0,   # metres between waypoints
    )
"""

import json
import math
import uuid

from std_msgs.msg import String


class DroneDispatcher:
    """
    Assigns scouting tasks to a swarm of drones by partitioning the search
    area into equal vertical strips — one strip per drone.

    Parameters
    ----------
    node       : rclpy.node.Node  — the planning agent node (needs a publisher)
    drone_ids  : list[str]        — ordered list of active drone agent_ids
    """

    def __init__(self, node, drone_ids: list[str]):
        self.node      = node
        self.drone_ids = drone_ids
        self._pub      = node.create_publisher(String, "/subtask", 10)
        node.get_logger().info(
            f"DroneDispatcher ready — managing {len(drone_ids)} drones: {drone_ids}"
        )

    # ─────────────────────────────────────────────────────────────────────────

    def assign(
        self,
        task_type:   str,
        search_area: dict,
        altitude:    float = 5.0,
        grid_step:   float = 3.0,
    ) -> list[str]:
        """
        Partition the search area and send one subtask per drone.

        Parameters
        ----------
        task_type   : "scout" | "relay"
        search_area : {"x_min", "x_max", "y_min", "y_max"}  — metres
        altitude    : cruise altitude in metres
        grid_step   : waypoint spacing in metres

        Returns
        -------
        list of task_ids assigned (one per drone)
        """
        strips   = self._partition(search_area, len(self.drone_ids))
        task_ids = []

        for drone_id, strip in zip(self.drone_ids, strips):
            waypoints = self._lawnmower(strip, altitude, grid_step)
            task_id   = str(uuid.uuid4())[:8]

            payload = {
                "task_id":   task_id,
                "agent_id":  drone_id,
                "task_type": task_type,
                "waypoints": waypoints,   # list of [x, y, z]
                "sector":    strip,
            }

            msg      = String()
            msg.data = json.dumps(payload)
            self._pub.publish(msg)
            task_ids.append(task_id)

            self.node.get_logger().info(
                f"[Dispatcher] {drone_id} → {task_type} | "
                f"sector x[{strip['x_min']:.1f},{strip['x_max']:.1f}] | "
                f"{len(waypoints)} waypoints | task_id={task_id}"
            )

        return task_ids

    def reassign(self, drone_id: str, waypoints: list, task_type: str = "scout") -> str:
        """
        Send a custom waypoint list to one specific drone (e.g. after it
        reports a survivor and the orchestrator wants it to circle/relay).
        """
        task_id = str(uuid.uuid4())[:8]
        payload = {
            "task_id":   task_id,
            "agent_id":  drone_id,
            "task_type": task_type,
            "waypoints": waypoints,
        }
        msg      = String()
        msg.data = json.dumps(payload)
        self._pub.publish(msg)
        self.node.get_logger().info(
            f"[Dispatcher] reassign {drone_id} → {len(waypoints)} waypoints | task_id={task_id}"
        )
        return task_id

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _partition(area: dict, n: int) -> list[dict]:
        """
        Split the search area into n equal vertical strips (along X axis).
        Each strip covers the full Y range.
        """
        x_min, x_max = area["x_min"], area["x_max"]
        y_min, y_max = area["y_min"], area["y_max"]
        strip_w = (x_max - x_min) / n
        strips  = []
        for i in range(n):
            strips.append({
                "x_min": x_min + i * strip_w,
                "x_max": x_min + (i + 1) * strip_w,
                "y_min": y_min,
                "y_max": y_max,
            })
        return strips

    @staticmethod
    def _lawnmower(strip: dict, altitude: float, step: float) -> list[list[float]]:
        """
        Generate a boustrophedon (lawnmower) waypoint path inside a strip.
        Alternates Y direction on each column so the drone doesn't waste time
        returning to the start of each row.
        """
        waypoints = []
        x = strip["x_min"] + step / 2
        col = 0

        while x <= strip["x_max"]:
            ys = list(_frange(strip["y_min"], strip["y_max"], step))
            if col % 2 == 1:
                ys = list(reversed(ys))
            for y in ys:
                waypoints.append([round(x, 2), round(y, 2), altitude])
            x   += step
            col += 1

        return waypoints


def _frange(start: float, stop: float, step: float):
    val = start
    while val <= stop + 1e-9:
        yield val
        val += step
