#!/usr/bin/env python3
"""
go1_node.py
──────────────────────────────────────────────────────────────────────────────
ROS2 node for a Unitree Go1 quadruped robot.

Each physical (or simulated) Go1 runs one instance of this node.
The node:
  1. Subscribes to /subtask — filters tasks addressed to its agent_id
  2. Calls the local Nano inference server to decide next waypoint
  3. Publishes velocity commands to Webots via /go1_<id>/cmd_vel
  4. Publishes its state on /agent_state for the orchestrator

Webots integration:
  The Go1 is controlled via the webots_ros2 differential-drive interface.
  LiDAR scan data is consumed from /go1_<id>/scan.

Launch with:
  ros2 run go1_node go1_node --ros-args -p agent_id:=go1_a -p nano_url:=http://localhost:8001
"""

import json
import math
import time
import threading
import requests
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import String


class Go1Node(Node):
    def __init__(self):
        super().__init__("go1_node")

        # ── Parameters ──────────────────────────────────────────────────────
        self.declare_parameter("agent_id", "go1_a")
        self.declare_parameter("nano_url", "http://localhost:8001")
        self.declare_parameter("nav_loop_hz", 5.0)
        self.declare_parameter("obstacle_stop_dist", 0.5)   # metres

        self.agent_id       = self.get_parameter("agent_id").get_parameter_value().string_value
        self.nano_url       = self.get_parameter("nano_url").get_parameter_value().string_value
        self.stop_dist      = self.get_parameter("obstacle_stop_dist").get_parameter_value().double_value

        # ── State ────────────────────────────────────────────────────────────
        self.current_pose: dict    = {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0}
        self.current_task: dict | None = None
        self.waypoints: list[list] = []
        self.status                = "idle"
        self.battery               = 100.0
        self.detected_objects: list[str] = []
        self.min_scan_dist         = float("inf")
        self._lock                 = threading.Lock()

        # ── ROS2 interfaces ──────────────────────────────────────────────────
        ns = f"/{self.agent_id}"
        self.cmd_vel_pub   = self.create_publisher(Twist, f"{ns}/cmd_vel", 10)
        self.state_pub     = self.create_publisher(String, "/agent_state", 10)

        self.subtask_sub   = self.create_subscription(String, "/subtask",        self.on_subtask,  10)
        self.odom_sub      = self.create_subscription(Odometry, f"{ns}/odom",    self.on_odom,     10)
        self.scan_sub      = self.create_subscription(LaserScan, f"{ns}/scan",   self.on_scan,     10)

        # ── Timers ───────────────────────────────────────────────────────────
        hz   = self.get_parameter("nav_loop_hz").get_parameter_value().double_value
        self.nav_timer    = self.create_timer(1.0 / hz, self.nav_loop)
        self.state_timer  = self.create_timer(1.0, self.publish_state)

        self.get_logger().info(f"Go1 node [{self.agent_id}] ready. Nano: {self.nano_url}")

    # ─────────────────────────────────────────────────────────────────────────
    # Callbacks
    # ─────────────────────────────────────────────────────────────────────────

    def on_subtask(self, msg: String):
        """Receive and store a subtask if it is addressed to this agent."""
        try:
            task = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        if task.get("agent_id") != self.agent_id:
            return

        with self._lock:
            self.current_task = task
            self.waypoints    = task.get("waypoints", [])
            self.status       = "executing"
            self.get_logger().info(
                f"Accepted task {task.get('task_id')} — {len(self.waypoints)} waypoints"
            )

    def on_odom(self, msg: Odometry):
        """Update current pose from odometry."""
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        # Simplified yaw extraction from quaternion
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y ** 2 + q.z ** 2),
        )
        with self._lock:
            self.current_pose = {"x": p.x, "y": p.y, "z": p.z, "yaw": math.degrees(yaw)}

    def on_scan(self, msg: LaserScan):
        """Track minimum LiDAR distance for obstacle detection."""
        valid = [r for r in msg.ranges if msg.range_min < r < msg.range_max]
        self.min_scan_dist = min(valid) if valid else float("inf")

        # Simple object detection heuristic
        self.detected_objects = []
        if self.min_scan_dist < 0.3:
            self.detected_objects.append("debris")

    # ─────────────────────────────────────────────────────────────────────────
    # Navigation loop
    # ─────────────────────────────────────────────────────────────────────────

    def nav_loop(self):
        """Called at nav_loop_hz. Ask Nano for next action, then execute it."""
        with self._lock:
            if self.status != "executing" or not self.waypoints:
                return
            task      = self.current_task
            waypoints = list(self.waypoints)
            pose      = dict(self.current_pose)

        target = waypoints[0]
        dist   = self._dist_to(target, pose)

        # ── Waypoint reached ─────────────────────────────────────────────────
        if dist < 0.3:
            with self._lock:
                self.waypoints.pop(0)
                if not self.waypoints:
                    self.status = "completed"
                    self.get_logger().info(f"Task {task.get('task_id')} completed.")
                    self._stop()
                    return
            return

        # ── Obstacle → ask Nano ───────────────────────────────────────────────
        if self.min_scan_dist < self.stop_dist:
            self._stop()
            nano_action = self._query_nano(task, pose, waypoints)
            if nano_action:
                self._execute_nano_action(nano_action)
        else:
            # Simple proportional heading controller
            self._drive_toward(target, pose)

    def _query_nano(self, task: dict, pose: dict, waypoints: list) -> dict | None:
        """POST to local Nano server and return parsed action dict."""
        context = {
            "current_pose":    pose,
            "next_waypoints":  waypoints[:3],
            "obstacle_dist_m": round(self.min_scan_dist, 2),
            "detected":        self.detected_objects,
        }
        payload = {
            "agent_id":   self.agent_id,
            "prompt":     task.get("instructions", "Navigate to next waypoint safely."),
            "context":    json.dumps(context),
            "temperature": 0.1,
            "max_tokens":  256,
        }
        try:
            resp = requests.post(f"{self.nano_url}/generate", json=payload, timeout=5.0)
            resp.raise_for_status()
            raw = resp.json()["response"]
            # Strip markdown fences if present
            raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            return json.loads(raw)
        except Exception as e:
            self.get_logger().warn(f"Nano query failed: {e}")
            return None

    def _execute_nano_action(self, action: dict):
        cmd = Twist()
        act = action.get("action", "stop")
        if act == "move":
            cmd.linear.x = 0.3
        elif act == "rotate":
            cmd.angular.z = 0.5
        elif act == "stop":
            pass
        elif act == "report":
            self.get_logger().info(f"[Nano report] {action.get('reason')}")
            with self._lock:
                self.status = "blocked"
        self.cmd_vel_pub.publish(cmd)

    def _drive_toward(self, target: list, pose: dict):
        dx   = target[0] - pose["x"]
        dy   = target[1] - pose["y"]
        goal_yaw  = math.degrees(math.atan2(dy, dx))
        heading_err = goal_yaw - pose["yaw"]
        # Normalise to [-180, 180]
        heading_err = (heading_err + 180) % 360 - 180

        cmd = Twist()
        if abs(heading_err) > 15:
            cmd.angular.z = 0.5 * math.copysign(1, heading_err)
        else:
            cmd.linear.x  = min(0.4, 0.4 * self._dist_to(target, pose))
            cmd.angular.z = 0.01 * heading_err
        self.cmd_vel_pub.publish(cmd)

    def _stop(self):
        self.cmd_vel_pub.publish(Twist())

    @staticmethod
    def _dist_to(target: list, pose: dict) -> float:
        return math.hypot(target[0] - pose["x"], target[1] - pose["y"])

    # ─────────────────────────────────────────────────────────────────────────

    def publish_state(self):
        """Broadcast this agent's state to /agent_state."""
        with self._lock:
            state = {
                "agent_id":        self.agent_id,
                "status":          self.status,
                "current_pose":    self.current_pose,
                "battery_percent": self.battery,
                "detected_objects": self.detected_objects,
                "current_task_id":  self.current_task.get("task_id") if self.current_task else None,
                "confidence":       0.8,
            }
        msg = String()
        msg.data = json.dumps(state)
        self.state_pub.publish(msg)


# ─────────────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = Go1Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
