#!/usr/bin/env python3
"""
drone_controller.py
Bridges namespaced /Drone_X/cmd_vel to the MavicDriver global /cmd_vel.
Publishes drone agent state and handles autonomous scouting missions.
"""
import json
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String


class DroneController(Node):
    def __init__(self):
        super().__init__("drone_controller")

        self.declare_parameter("drone_id", "Drone_1")
        self.drone_id = self.get_parameter("drone_id").get_parameter_value().string_value

        self.gps_pos = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.status = "idle"
        self.current_task = None
        self.waypoints = []

        # Subscribe to namespaced cmd_vel from operator/orchestrator
        self.create_subscription(
            Twist, f"/{self.drone_id}/cmd_vel", self.on_cmd_vel, 10
        )

        # Subscribe to subtasks
        self.create_subscription(String, "/subtask", self.on_subtask, 10)

        # Publish to global /cmd_vel (MavicDriver listens here)
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # Publish agent state
        self.state_pub = self.create_publisher(String, "/agent_state", 10)

        # State timer
        self.create_timer(1.0, self.publish_state)

        self.get_logger().info(f"Drone controller [{self.drone_id}] ready")

    def on_cmd_vel(self, msg):
        """Forward namespaced cmd_vel to MavicDriver."""
        self.cmd_pub.publish(msg)

    def on_subtask(self, msg):
        try:
            task = json.loads(msg.data)
            if task.get("agent_id") != self.drone_id:
                return
            self.current_task = task
            self.waypoints = task.get("waypoints", [])
            self.status = "executing"
            self.get_logger().info(
                f"Accepted task {task.get('task_id')} — {len(self.waypoints)} waypoints"
            )
        except Exception as e:
            self.get_logger().warn(f"Task parse error: {e}")

    def publish_state(self):
        state = {
            "agent_id": self.drone_id,
            "status": self.status,
            "current_pose": self.gps_pos,
            "battery_percent": 100.0,
            "detected_objects": [],
            "current_task_id": self.current_task.get("task_id") if self.current_task else None,
            "confidence": 1.0,
        }
        msg = String()
        msg.data = json.dumps(state)
        self.state_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = DroneController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()