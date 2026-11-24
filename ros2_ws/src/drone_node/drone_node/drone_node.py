#!/usr/bin/env python3
"""
drone_node.py
──────────────────────────────────────────────────────────────────────────────
ROS2 node for a UAV drone in Webots.  Supports 1–N drones; each instance
runs with its own agent_id parameter matching the Webots robot name and the
ROS2 namespace (drone_1 / drone_2 / drone_3 …).

Changes from single-drone version
──────────────────────────────────
1. Subscribes to /drone_poses  (shared JSON peer table) so each drone knows
   where all others are.
2. Publishes its own pose to /drone_poses on every odom update.
3. AvoidanceLayer (drone_avoidance.py) is applied inside nav_loop before
   every cmd_vel publish — enforces 3 m hard-stop, 6 m soft repulsion.
4. Subtask dispatch is list-based: the /subtask message may target any
   agent_id; filtering is unchanged, but the orchestrator can now send
   tasks to any drone by name.
"""

import json
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String

from drone_avoidance import AvoidanceLayer   # same directory


class DroneNode(Node):
    def __init__(self):
        super().__init__("drone_node")

        # ── Parameters ──────────────────────────────────────────────────────
        self.declare_parameter("agent_id",        "drone_1")
        self.declare_parameter("cruise_altitude",  5.0)
        self.declare_parameter("cruise_speed",     2.0)
        self.declare_parameter("map_resolution",   0.1)

        self.agent_id     = self.get_parameter("agent_id").get_parameter_value().string_value
        self.cruise_alt   = self.get_parameter("cruise_altitude").get_parameter_value().double_value
        self.cruise_speed = self.get_parameter("cruise_speed").get_parameter_value().double_value

        # ── State ────────────────────────────────────────────────────────────
        self.pose         = {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0}
        self.waypoints    = []
        self.status       = "idle"
        self.current_task = None
        self.scanned_cells: dict[tuple, int] = {}

        # Peer pose table: {agent_id: {"x", "y", "z"}}
        # Populated from /drone_poses; self entry is excluded in avoidance.
        self.peer_poses: dict[str, dict] = {}

        # ── Avoidance layer ──────────────────────────────────────────────────
        self.avoidance = AvoidanceLayer(
            own_id           = self.agent_id,
            influence_radius = 6.0,   # soft repulsion starts at 6 m
            hard_stop_radius = 3.0,   # full override below 3 m
            repulsion_gain   = 1.5,
            max_repulsion    = 1.2,
            blend_weight     = 0.6,
        )

        # ── ROS2 interfaces ──────────────────────────────────────────────────
        ns = f"/{self.agent_id}"

        # Per-drone topics
        self.cmd_vel_pub = self.create_publisher(Twist,         f"{ns}/cmd_vel", 10)
        self.map_pub     = self.create_publisher(OccupancyGrid, "/map",          10)
        self.state_pub   = self.create_publisher(String,        "/agent_state",  10)

        # Shared swarm pose topic — all drones publish/subscribe here
        self.poses_pub   = self.create_publisher(String, "/drone_poses", 10)

        self.subtask_sub = self.create_subscription(String,      "/subtask",         self.on_subtask,    10)
        self.odom_sub    = self.create_subscription(Odometry,    f"{ns}/odom",       self.on_odom,       10)
        self.cloud_sub   = self.create_subscription(PointCloud2, f"{ns}/points",     self.on_pointcloud, 10)
        self.poses_sub   = self.create_subscription(String,      "/drone_poses",     self.on_drone_poses, 10)

        self.nav_timer   = self.create_timer(0.1, self.nav_loop)
        self.state_timer = self.create_timer(1.0, self.publish_state)
        self.map_timer   = self.create_timer(2.0, self.publish_map)

        self.get_logger().info(f"Drone node [{self.agent_id}] ready.")

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def on_subtask(self, msg: String):
        try:
            task = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        if task.get("agent_id") != self.agent_id:
            return
        if task.get("task_type") not in ("scout", "relay"):
            return

        self.current_task = task
        self.waypoints    = task.get("waypoints", [])
        self.status       = "executing"
        self.get_logger().info(
            f"[{self.agent_id}] Task accepted: {task.get('task_type')} "
            f"— {len(self.waypoints)} waypoints"
        )

    def on_odom(self, msg: Odometry):
        p   = msg.pose.pose.position
        q   = msg.pose.pose.orientation
        yaw = math.degrees(math.atan2(
            2 * (q.w * q.z + q.x * q.y),
            1 - 2 * (q.y ** 2 + q.z ** 2),
        ))
        self.pose = {"x": p.x, "y": p.y, "z": p.z, "yaw": yaw}

        # Broadcast own position so peers can update their tables
        self._publish_own_pose()

    def on_pointcloud(self, msg: PointCloud2):
        res = 0.1
        gx  = int(self.pose["x"] / res)
        gy  = int(self.pose["y"] / res)
        self.scanned_cells[(gx, gy)] = 0

    def on_drone_poses(self, msg: String):
        """Merge incoming pose table into local peer_poses (skip own entry)."""
        try:
            table = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        for drone_id, pose in table.items():
            if drone_id != self.agent_id:
                self.peer_poses[drone_id] = pose

    # ── Navigation ────────────────────────────────────────────────────────────

    def nav_loop(self):
        """Altitude-hold + waypoint following, with avoidance composited in."""
        cmd = Twist()

        if self.status == "executing" and self.waypoints:
            target  = self.waypoints[0]
            dist_xy = math.hypot(target[0] - self.pose["x"], target[1] - self.pose["y"])
            dist_z  = (
                target[2] - self.pose["z"]
                if len(target) > 2
                else self.cruise_alt - self.pose["z"]
            )

            if abs(dist_z) > 0.3:
                cmd.linear.z = 0.5 * dist_z
            elif dist_xy < 0.5:
                self.waypoints.pop(0)
                if not self.waypoints:
                    self.status = "completed"
                    self._stop()
                    return
            else:
                dx       = target[0] - self.pose["x"]
                dy       = target[1] - self.pose["y"]
                heading  = math.degrees(math.atan2(dy, dx))
                heading_err = (heading - self.pose["yaw"] + 180) % 360 - 180
                speed    = min(self.cruise_speed, 0.6 * dist_xy)
                cmd.linear.x  = speed * math.cos(math.radians(heading_err))
                cmd.linear.y  = speed * math.sin(math.radians(heading_err))
                cmd.angular.z = 0.02 * heading_err

        # ── Apply collision avoidance before publishing ─────────────────────
        cmd = self.avoidance.apply(cmd, self.pose, self.peer_poses)

        # Log a warning when avoidance overrides navigation
        nearest = self.avoidance.nearest_peer_distance(self.pose, self.peer_poses)
        if nearest < self.avoidance.hard_stop_radius:
            self.get_logger().warn(
                f"[{self.agent_id}] HARD STOP — nearest peer {nearest:.2f} m"
            )

        self.cmd_vel_pub.publish(cmd)

    def _stop(self):
        self.cmd_vel_pub.publish(Twist())

    # ── Publishers ────────────────────────────────────────────────────────────

    def _publish_own_pose(self):
        """Publish {agent_id: {x,y,z}} to /drone_poses so peers can avoid us."""
        payload = {
            self.agent_id: {
                "x": self.pose["x"],
                "y": self.pose["y"],
                "z": self.pose["z"],
            }
        }
        m      = String()
        m.data = json.dumps(payload)
        self.poses_pub.publish(m)

    def publish_map(self):
        if not self.scanned_cells:
            return
        grid             = OccupancyGrid()
        grid.header.frame_id = "map"
        grid.info.resolution = 0.1
        xs = [k[0] for k in self.scanned_cells]
        ys = [k[1] for k in self.scanned_cells]
        w  = max(xs) - min(xs) + 1
        h  = max(ys) - min(ys) + 1
        grid.info.width  = w
        grid.info.height = h
        grid.info.origin.position.x = min(xs) * 0.1
        grid.info.origin.position.y = min(ys) * 0.1
        data = [-1] * (w * h)
        for (gx, gy), occ in self.scanned_cells.items():
            idx       = (gy - min(ys)) * w + (gx - min(xs))
            data[idx] = occ
        grid.data = data
        self.map_pub.publish(grid)

    def publish_state(self):
        state = {
            "agent_id":         self.agent_id,
            "status":           self.status,
            "current_pose":     self.pose,
            "battery_percent":  100.0,
            "detected_objects": [],
            "current_task_id":  self.current_task.get("task_id") if self.current_task else None,
            "confidence":       1.0,
            "nearest_peer_m":   round(
                self.avoidance.nearest_peer_distance(self.pose, self.peer_poses), 2
            ),
        }
        m      = String()
        m.data = json.dumps(state)
        self.state_pub.publish(m)


# ─────────────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = DroneNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()