# shared_msgs package — custom message definitions
# Place files in shared_msgs/msg/

# ── MissionGoal.msg ──────────────────────────────────────────────
# Sent from operator → orchestrator
string goal_description        # natural-language mission description
string priority                # "critical" | "high" | "normal"
float32[] bounding_box         # [x_min, y_min, x_max, y_max] in world coords
string[] known_hazards         # list of known hazard type strings

---
# ── SubTask.msg ─────────────────────────────────────────────────
# Dispatched from orchestrator → individual agents
string task_id
string agent_id                # "drone_1" | "drone_2" | "go1_a" | "go1_b" | "go1_c"
string task_type               # "scout" | "navigate" | "rescue" | "relay"
geometry_msgs/Point[] waypoints
string llm_context             # JSON blob with Super's reasoning context
float32 timeout_seconds

---
# ── AgentState.msg ──────────────────────────────────────────────
# Broadcast by every agent → orchestrator
string agent_id
string status                  # "idle" | "executing" | "blocked" | "completed" | "error"
geometry_msgs/Pose current_pose
float32 battery_percent
string[] detected_objects      # e.g. ["debris", "survivor", "fire"]
string current_task_id
float32 confidence             # Nano model confidence in current decision [0-1]
