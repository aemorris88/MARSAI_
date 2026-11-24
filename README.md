# Post-Disaster Scenario Path Planning System

## System Overview

A multi-agent autonomous navigation system for post-disaster environments. The architecture consists of:

| Component | Model | Deployment | Role |
|---|---|---|---|
| Main Orchestrator | Nemotron 3 Super | HiPerGator (UF) | High-level task planning, mission coordination |
| Drone 1 & 2 | — | Webots/ROS2 sim | Aerial reconnaissance, map building |
| Go1 Dog A/B/C | Nemotron 3 Nano | Local PC | Ground navigation, rescue path execution |

Simulation runs in **Webots** with a **ROS2 (Humble)** middleware layer.

---

## Repository Structure

```
disaster_nav/
├── ros2_ws/src/
│   ├── orchestrator_node/      # Nemotron Super bridge + mission planner
│   ├── drone_node/             # Drone ROS2 controller
│   ├── go1_node/               # Unitree Go1 ROS2 controller + Nano bridge
│   ├── webots_world/           # Webots .wbt world file + protos
│   └── shared_msgs/            # Custom ROS2 message definitions
├── llm_bridge/
│   ├── hipergator/             # API client for Nemotron Super on HiPerGator
│   └── local_nano/             # Local gRPC server for Nemotron Nano
├── config/                     # YAML parameters for all nodes
└── docs/                       # Architecture docs
```

---

## Prerequisites

- ROS2 Humble (Ubuntu 22.04)
- Webots R2023b+
- Python 3.10+
- `webots_ros2` package
- Access to HiPerGator with Nemotron 3 Super endpoint
- Local PC with Nemotron 3 Nano (ollama or vLLM)

---

## Quick Start

```bash
# 1. Build the ROS2 workspace
cd ros2_ws
colcon build --symlink-install
source install/setup.bash

# 2. Start local Nano inference server
cd llm_bridge/local_nano
python3 nano_server.py --model nemotron3-nano --port 8001

# 3. Launch Webots world + all nodes
ros2 launch orchestrator_node disaster_nav.launch.py

# 4. Send an initial mission goal
ros2 topic pub /mission_goal std_msgs/String "data: 'Search grid B3-F7 for survivors'"
```

---

## Communication Flow

```
HiPerGator (Super)
    │  HTTP/REST  (task plans as JSON)
    ▼
orchestrator_node  ─── ROS2 topics ──►  drone_node × 2
    │                                        │ /map, /scan
    │  ROS2 /subtask                         ▼
    ▼                               [OccupancyGrid published]
go1_node × 3
    │  HTTP  (local:8001)
    ▼
Nemotron Nano  →  waypoint decisions
```
```
# Verify neno server health:
# Should already be up — verify with:
curl http://localhost:8001/health

# Fire Go1 inference call:
curl -X POST http://localhost:8001/generate \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"go1_a","prompt":"Navigate to survivor at waypoint safely.","context":"{\"obstacle_dist_m\": 0.4, \"detected\": [\"debris\"]}","temperature":0.1,"max_tokens":256}'

# run Go1 node: 
source /opt/ros/humble/setup.bash
source ~/disaster_nav/ros2_ws/install/setup.bash
ros2 run go1_node go1_node --ros-args -p agent_id:=go1_a -p nano_url:=http://localhost:8001

#ROS2 live topic echo:
source /opt/ros/humble/setup.bash
source ~/disaster_nav/ros2_ws/install/setup.bash
ros2 topic echo /agent_state


to open webots world file directly:
webots ~/disaster_nav/ros2_ws/src/webots_world/NewRescueLAM.wbt



to run program:
run the nemotron server

in Windows Powershell,open 3 terminals and run each of these:

cd C:\Users\abbyw\Downloads
$env:WEBOTS_CONTROLLER_URL = "tcp://localhost:1234/dog_1"
python webots_bridge.py --robot dog_1 --wsl-ip 172.23.49.213 --send-port 9001 --recv-port 9101

# PowerShell window 2
cd C:\Users\abbyw\Downloads
$env:WEBOTS_CONTROLLER_URL = "tcp://localhost:1234/dog_2"
python webots_bridge.py --robot dog_2 --wsl-ip 172.23.49.213 --send-port 9002 --recv-port 9102

# PowerShell window 3
cd C:\Users\abbyw\Downloads
$env:WEBOTS_CONTROLLER_URL = "tcp://localhost:1234/dog_3"
python webots_bridge.py --robot dog_3 --wsl-ip 172.23.49.213 --send-port 9003 --recv-port 9103

then start the UDP bridge on WSL2 side:
source /opt/ros/humble/setup.bash
source ~/disaster_nav/ros2_ws/install/setup.bash
python3 ~/udp_ros2_bridge.py --windows-ip 172.23.48.1


to kill instance:
kill -9 $(pgrep -f "webots-controller") 2>/dev/null
kill -9 $(pgrep -f "ros2") 2>/dev/null

for launching drones and dogs separately:
cd ~/disaster_nav/ros2_ws && source install/setup.bash && WEBOTS_HOME=/usr/local/webots ros2 launch go1_node go1_launch.py

source ~/disaster_nav/ros2_ws/install/setup.bash && WEBOTS_HOME=/usr/local/webots ros2 launch drone_node drone_launch.py

write and rebuild aftr updating world.py file

python3 /mnt/c/Users/abbyw/Downloads/write_world.py
cd ~/disaster_nav/ros2_ws
colcon build --symlink-install
source install/setup.bash



drones:
cd ~/disaster_nav/ros2_ws
source install/setup.bash
WEBOTS_HOME=/usr/local/webots PYTHONPATH=/usr/local/webots/lib/controller/python:$PYTHONPATH \
python3 src/drone_node/drone_node/autonomous_drone_drone_1.py &

WEBOTS_HOME=/usr/local/webots PYTHONPATH=/usr/local/webots/lib/controller/python:$PYTHONPATH \
python3 src/drone_node/drone_node/autonomous_drone_drone_2.py &


source /opt/ros/humble/setup.bash
source ~/disaster_nav/ros2_ws/install/setup.bash
/opt/ros/humble/share/webots_ros2_driver/scripts/webots-controller \
  --protocol=tcp --ip-address=172.23.48.1 --port=1234 \
  --robot-name=Drone_1 \
  /home/abbyw/disaster_nav/ros2_ws/src/drone_node/drone_node/autonomous_drone_drone_1.py 2>&1

  source /opt/ros/humble/setup.bash
source ~/disaster_nav/ros2_ws/install/setup.bash
/opt/ros/humble/share/webots_ros2_driver/scripts/webots-controller \
  --protocol=tcp --ip-address=172.23.48.1 --port=1234 \
  --robot-name=Drone_1 \
  /home/abbyw/disaster_nav/ros2_ws/src/drone_node/drone_node/autonomous_drone_drone_1.py


  open QGC and run PX4 in terminal

terminal 1: (PX4 SITL + Gazebo)
kill -9 $(pgrep -f px4) 2>/dev/null
kill -9 $(pgrep -f gz) 2>/dev/null
sleep 2
rm -f /tmp/px4pipe
mkfifo /tmp/px4pipe
cd ~/PX4-Autopilot
make px4_sitl gz_x500_mono_cam_down < /tmp/px4pipe &
sleep 25
echo "mavlink start -x -u 14550 -t 172.23.48.1 -r 4000000" > /tmp/px4pipe

or 
cd ~/PX4-Autopilot
rm -f /tmp/px4pipe
mkfifo /tmp/px4pipe
make px4_sitl gz_x500 < /tmp/px4pipe &
sleep 25
echo "mavlink start -x -u 14550 -t 172.23.48.1 -r 4000000" > /tmp/px4pipe

terminal 2: (MAVROS)
unset PYTHONPATH AMENT_PREFIX_PATH COLCON_PREFIX_PATH
source /opt/ros/humble/setup.bash
ros2 launch mavros px4.launch fcu_url:="udp://:14540@172.23.48.1:14557"

terminal 3: (drone node)
unset PYTHONPATH AMENT_PREFIX_PATH COLCON_PREFIX_PATH
source /opt/ros/humble/setup.bash
source ~/disaster_nav/ros2_ws/install/setup.bash
ros2 run drone_node mavros_drone --ros-args -p drone_id:=Drone_1 -p drone_index:=1



mission goal for orchestrator: 
unset PYTHONPATH AMENT_PREFIX_PATH COLCON_PREFIX_PATH
source /opt/ros/humble/setup.bash
source ~/disaster_nav/ros2_ws/install/setup.bash
ros2 topic pub /mission_goal std_msgs/msg/String \
  "data: '{\"goal\": \"Search the disaster zone for survivors. Drone_1 is providing aerial coverage. Dispatch dogs to investigate areas the drone has identified.\"}'" --once


Current system:

Open Webots

term 1: run PX4 + Gazebo
cd ~/PX4-Autopilot
rm -f /tmp/px4pipe
mkfifo /tmp/px4pipe
make px4_sitl gz_x500 < /tmp/px4pipe &
sleep 25
echo "mavlink start -x -u 14550 -t 172.23.48.1 -r 4000000" > /tmp/px4pipe

if old session still running:
kill -9 $(pgrep -f px4) 2>/dev/null
kill -9 $(pgrep -f gz) 2>/dev/null
sleep 2
echo "Clean"

term 2: run mavros node
unset PYTHONPATH AMENT_PREFIX_PATH COLCON_PREFIX_PATH
source /opt/ros/humble/setup.bash
ros2 launch mavros px4.launch fcu_url:="udp://:14540@172.23.48.1:14557" system_id:=255 component_id:=190

term 3: drone node
unset PYTHONPATH AMENT_PREFIX_PATH COLCON_PREFIX_PATH
source /opt/ros/humble/setup.bash
source ~/disaster_nav/ros2_ws/install/setup.bash
ros2 run drone_node mavros_drone --ros-args -p drone_id:=Drone_1 -p drone_index:=1

term 4: run webots dogs
unset PYTHONPATH AMENT_PREFIX_PATH COLCON_PREFIX_PATH
source /opt/ros/humble/setup.bash
source ~/disaster_nav/ros2_ws/install/setup.bash
WEBOTS_HOME=/usr/local/webots ros2 launch go1_node go1_launch.py

term 5: Ollama
OLLAMA_KEEP_ALIVE=-1 ollama serve &

term 6: run Nano LLM
cd ~/disaster_nav/llm_bridge/local_nano
python3 nano_server.py --backend ollama --model nemotron-mini --port 8001

term 7: orchestrator node
unset PYTHONPATH AMENT_PREFIX_PATH COLCON_PREFIX_PATH
source /opt/ros/humble/setup.bash
source ~/disaster_nav/ros2_ws/install/setup.bash
ros2 run orchestrator_node orchestrator_node

term 8: subtask / mission goal
unset PYTHONPATH AMENT_PREFIX_PATH COLCON_PREFIX_PATH
source /opt/ros/humble/setup.bash
source ~/disaster_nav/ros2_ws/install/setup.bash
ros2 topic pub /mission_goal std_msgs/msg/String \
  "data: '{\"goal\": \"Search the disaster zone for survivors. Drone_1 is patrolling the grid. Dispatch dogs to investigate areas.\"}'" --once
```


6/8
updated the go1 node to call Nemotron Nano for every navigation decision, not just obstacle avoidance
and print reasoning

rebuild:
cd ~/disaster_nav/ros2_ws
colcon build --packages-select go1_node 2>&1 | tail -3


what we're testing for:
Time-to-waypoint — how long each dog takes to reach its assigned location
Task completion rate — how many tasks completed successfully
Escalation rate — how often Nano escalated to Super
LLM response latency — average Nano query time


dispatch:
python3 /mnt/c/Users/abbyw/Downloads/dispatch_tasks.py cd ~/disaster_nav/ros2_ws && colcon build --packages-select go1_node && source install/setup.bash
echo "Ready"


to see mean std for TTR and completion rate across all trials completed so far:
cat ~/disaster_nav/results/trials.json


to run data collector:
unset PYTHONPATH AMENT_PREFIX_PATH COLCON_PREFIX_PATH
source /opt/ros/humble/setup.bash
source ~/disaster_nav/ros2_ws/install/setup.bash
ros2 run orchestrator_node data_collector