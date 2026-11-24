#!/usr/bin/env python3
"""
orchestrator_node.py
──────────────────────────────────────────────────────────────────────────────
Main mission orchestrator. Connects to Nemotron 3 Super on HiPerGator via
REST, translates its high-level task plans into ROS2 subtask messages, and
monitors all agent states.

Topics published:
  /subtask          (shared_msgs/SubTask) — dispatched to agents
  /mission_status   (std_msgs/String)     — JSON status summary

Topics subscribed:
  /mission_goal     (std_msgs/String)     — operator mission input
  /agent_state      (shared_msgs/AgentState) — aggregated from all agents
"""

import json
import uuid
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Point

from .hipergator_client import HiPerGatorClient  # see llm_bridge/hipergator/

# Placeholder imports — replace with actual generated message imports after build
# from shared_msgs.msg import SubTask, AgentState, MissionGoal


AGENT_IDS = ["drone_1", "drone_2", "go1_a", "go1_b", "go1_c"]


class OrchestratorNode(Node):
    def __init__(self):
        super().__init__("orchestrator_node")

        # ── Parameters ─────────────────────────────────────────────────────
        self.declare_parameter("hipergator_endpoint", "http://hipergator.rc.ufl.edu:8080/v1")
        self.declare_parameter("hipergator_api_key", "")
        self.declare_parameter("plan_interval_sec", 5.0)

        endpoint = self.get_parameter("hipergator_endpoint").get_parameter_value().string_value
        api_key  = self.get_parameter("hipergator_api_key").get_parameter_value().string_value

        # ── LLM client ─────────────────────────────────────────────────────
        self.llm = HiPerGatorClient(endpoint, api_key)

        # ── Agent state cache ───────────────────────────────────────────────
        self.agent_states: dict[str, dict] = {a: {} for a in AGENT_IDS}
        self.active_mission: dict | None = None

        # ── ROS2 pub/sub ────────────────────────────────────────────────────
        self.subtask_pub     = self.create_publisher(String, "/subtask", 10)
        self.status_pub      = self.create_publisher(String, "/mission_status", 10)

        self.mission_sub = self.create_subscription(
            String, "/mission_goal", self.on_mission_goal, 10
        )
        self.agent_sub = self.create_subscription(
            String, "/agent_state", self.on_agent_state, 10
        )

        # ── Periodic replanning ─────────────────────────────────────────────
        period = self.get_parameter("plan_interval_sec").get_parameter_value().double_value
        self.plan_timer = self.create_timer(period, self.replan_callback)

        self.get_logger().info("Orchestrator node started — waiting for mission goal.")

    # ────────────────────────────────────────────────────────────────────────

    def on_mission_goal(self, msg: String):
        """Receive operator mission string and call Super for initial plan."""
        self.get_logger().info(f"New mission goal received: {msg.data}")
        self.active_mission = {"goal": msg.data, "id": str(uuid.uuid4())}
        self._request_plan(reason="initial_mission")

    def on_agent_state(self, msg: String):
        """Parse and cache incoming agent state JSON."""
        try:
            state = json.loads(msg.data)
            agent_id = state.get("agent_id")
            if agent_id in self.agent_states:
                self.agent_states[agent_id] = state
        except json.JSONDecodeError as e:
            self.get_logger().warn(f"Bad agent state JSON: {e}")

    def replan_callback(self):
        """Periodic check — replan if any agent is blocked or completed."""
        if not self.active_mission:
            return
        blocked = [a for a, s in self.agent_states.items() if s.get("status") == "blocked"]
        completed = [a for a, s in self.agent_states.items() if s.get("status") == "completed"]
        if blocked or completed:
            self.get_logger().info(f"Replanning: blocked={blocked} completed={completed}")
            self._request_plan(reason="replan", blocked=blocked, completed=completed)

    # ────────────────────────────────────────────────────────────────────────

    def _request_plan(self, reason: str, blocked: list = None, completed: list = None):
        """Build a prompt for Nemotron Super and dispatch the resulting plan."""
        if not self.active_mission:
            return

        prompt = self._build_super_prompt(reason, blocked or [], completed or [])
        try:
            plan_json = self.llm.generate(prompt)
            plan = json.loads(plan_json)
            self._dispatch_plan(plan)
        except Exception as e:
            self.get_logger().error(f"Super LLM call failed: {e}")

    def _build_super_prompt(self, reason: str, blocked: list, completed: list) -> str:
        return f"""
You are the mission orchestrator for a post-disaster search-and-rescue system.

MISSION: {self.active_mission['goal']}
REASON FOR THIS CALL: {reason}
BLOCKED AGENTS: {blocked}
COMPLETED AGENTS: {completed}

AGENT STATES:
{json.dumps(self.agent_states, indent=2)}

AVAILABLE AGENTS: drone_1, drone_2, go1_a, go1_b, go1_c
  - Drones: aerial scouting, LiDAR mapping, do NOT navigate into rubble
  - Go1 dogs: ground navigation, can traverse debris fields

Return ONLY a valid JSON array of subtasks:
[
  {{
    "agent_id": "<agent>",
    "task_type": "scout|navigate|rescue|relay",
    "waypoints": [[x, y, z], ...],
    "instructions": "<brief natural language for Nano model>",
    "priority": 1
  }},
  ...
]
Minimize agent crossings. Assign drones first for reconnaissance before committing ground units.
""".strip()

    def _dispatch_plan(self, plan: list):
        """Publish each subtask as a JSON string on /subtask."""
        for task in plan:
            task["task_id"] = str(uuid.uuid4())
            task["mission_id"] = self.active_mission["id"]
            msg = String()
            msg.data = json.dumps(task)
            self.subtask_pub.publish(msg)
            self.get_logger().info(
                f"Dispatched task {task['task_id']} → {task['agent_id']} ({task['task_type']})"
            )

        # Publish status summary
        status = {
            "mission_id": self.active_mission["id"],
            "tasks_dispatched": len(plan),
            "agent_states": {a: s.get("status", "unknown") for a, s in self.agent_states.items()},
        }
        s_msg = String()
        s_msg.data = json.dumps(status)
        self.status_pub.publish(s_msg)


# ────────────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = OrchestratorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
