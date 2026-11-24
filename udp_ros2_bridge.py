#!/usr/bin/env python3
"""
udp_ros2_bridge.py
────────────────────────────────────────────────────────────────────────────
Runs in WSL2. Bridges between:
  - ROS2 topics (/subtask, /agent_state, /cmd_vel)
  - UDP sockets talking to the Windows webots_bridge.py instances

One instance handles all three dogs.

Usage:
    python3 udp_ros2_bridge.py --windows-ip 172.23.48.1
"""

import json
import socket
import threading
import argparse
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist

# UDP port assignments per dog
# Windows bridge listens on recv_port, WSL2 sends to it
# Windows bridge sends to send_port, WSL2 listens on it
DOG_PORTS = {
    "dog_1": {"send": 9001, "recv": 9101},  # WSL2 listens on 9001, sends to 9101
    "dog_2": {"send": 9002, "recv": 9102},
    "dog_3": {"send": 9003, "recv": 9103},
}


class UDPRosBridge(Node):
    def __init__(self, windows_ip: str):
        super().__init__("udp_ros2_bridge")
        self.windows_ip = windows_ip

        # ── ROS2 publishers / subscribers ────────────────────────────────────
        self.state_pub   = self.create_publisher(String, "/agent_state", 10)
        self.subtask_sub = self.create_subscription(String, "/subtask", self.on_subtask, 10)

        # ── UDP send sockets (WSL2 → Windows) ────────────────────────────────
        self.send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # ── UDP receive sockets (Windows → WSL2) ─────────────────────────────
        self.recv_threads = {}
        for dog, ports in DOG_PORTS.items():
            t = threading.Thread(
                target=self._recv_loop,
                args=(dog, ports["send"]),
                daemon=True
            )
            t.start()
            self.recv_threads[dog] = t

        self.get_logger().info(f"UDP-ROS2 bridge started. Windows IP: {windows_ip}")

    def on_subtask(self, msg: String):
        """Forward subtask from ROS2 to the appropriate Windows bridge."""
        try:
            task = json.loads(msg.data)
            agent = task.get("agent_id")
            if agent in DOG_PORTS:
                task["type"] = "subtask"
                data = json.dumps(task).encode()
                port = DOG_PORTS[agent]["recv"]
                self.send_sock.sendto(data, (self.windows_ip, port))
                self.get_logger().info(f"Forwarded subtask to {agent} on port {port}")
        except Exception as e:
            self.get_logger().warn(f"Subtask forward error: {e}")

    def send_cmd_vel(self, agent: str, vx: float, vz: float):
        """Send velocity command to a specific dog."""
        if agent in DOG_PORTS:
            msg = {"type": "cmd_vel", "linear_x": vx, "angular_z": vz}
            port = DOG_PORTS[agent]["recv"]
            self.send_sock.sendto(json.dumps(msg).encode(), (self.windows_ip, port))

    def _recv_loop(self, dog: str, port: int):
        """Listen for state messages from Windows bridge for a specific dog."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", port))
        sock.settimeout(1.0)
        self.get_logger().info(f"Listening for {dog} state on UDP port {port}")

        while rclpy.ok():
            try:
                data, _ = sock.recvfrom(4096)
                state = json.loads(data.decode())
                if state.get("type") == "agent_state":
                    # Publish to ROS2 /agent_state topic
                    ros_state = {
                        "agent_id":        state.get("agent_id"),
                        "status":          state.get("status", "idle"),
                        "current_pose":    state.get("pose", {"x": 0, "y": 0, "z": 0}),
                        "battery_percent": 100.0,
                        "detected_objects": [],
                        "current_task_id": state.get("task_id"),
                        "confidence":      0.9,
                    }
                    msg = String()
                    msg.data = json.dumps(ros_state)
                    self.state_pub.publish(msg)
            except socket.timeout:
                continue
            except Exception as e:
                self.get_logger().warn(f"Recv error for {dog}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows-ip", default="172.23.48.1", help="Windows host IP from WSL2")
    args = parser.parse_args()

    rclpy.init()
    node = UDPRosBridge(args.windows_ip)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
