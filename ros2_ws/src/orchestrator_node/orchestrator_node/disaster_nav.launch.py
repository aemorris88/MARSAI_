"""
disaster_nav.launch.py
──────────────────────────────────────────────────────────────────────────────
Master launch file. Starts:
  1. Webots with the disaster world
  2. orchestrator_node   (bridges to HiPerGator Nemotron Super)
  3. drone_node × 2
  4. go1_node × 3        (each calls local Nano server)

Usage:
  ros2 launch orchestrator_node disaster_nav.launch.py \
    hipergator_endpoint:=http://hipergator.rc.ufl.edu:8080/v1 \
    hipergator_api_key:=YOUR_KEY \
    nano_url:=http://localhost:8001
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from webots_ros2_driver.webots_launcher import WebotsLauncher


def generate_launch_description():
    # ── Launch arguments ─────────────────────────────────────────────────────
    hipergator_endpoint = LaunchConfiguration("hipergator_endpoint")
    hipergator_api_key  = LaunchConfiguration("hipergator_api_key")
    nano_url            = LaunchConfiguration("nano_url")

    declare_endpoint = DeclareLaunchArgument(
        "hipergator_endpoint",
        default_value="http://hipergator.rc.ufl.edu:8080/v1",
        description="HiPerGator vLLM endpoint for Nemotron Super",
    )
    declare_key = DeclareLaunchArgument(
        "hipergator_api_key",
        default_value="",
        description="API key / token for HiPerGator",
    )
    declare_nano = DeclareLaunchArgument(
        "nano_url",
        default_value="http://localhost:8001",
        description="Local Nano inference server URL",
    )

    # ── Webots launcher ──────────────────────────────────────────────────────
    world_path = PathJoinSubstitution([
        FindPackageShare("webots_world"),
        "disaster_world.wbt",
    ])
    webots = WebotsLauncher(world=world_path, mode="realtime")

    # ── Orchestrator ─────────────────────────────────────────────────────────
    orchestrator = Node(
        package="orchestrator_node",
        executable="orchestrator_node",
        name="orchestrator",
        output="screen",
        parameters=[{
            "hipergator_endpoint": hipergator_endpoint,
            "hipergator_api_key":  hipergator_api_key,
            "plan_interval_sec":   5.0,
        }],
    )

    # ── Drone nodes ───────────────────────────────────────────────────────────
    drone_nodes = [
        Node(
            package="drone_node",
            executable="drone_node",
            name=f"drone_{i}",
            output="screen",
            parameters=[{"agent_id": f"drone_{i}", "cruise_altitude": 5.0}],
        )
        for i in range(1, 3)
    ]

    # ── Go1 dog nodes ─────────────────────────────────────────────────────────
    go1_nodes = [
        Node(
            package="go1_node",
            executable="go1_node",
            name=f"go1_{letter}",
            output="screen",
            parameters=[{
                "agent_id": f"go1_{letter}",
                "nano_url": nano_url,
                "obstacle_stop_dist": 0.5,
            }],
        )
        for letter in ("a", "b", "c")
    ]

    return LaunchDescription([
        declare_endpoint,
        declare_key,
        declare_nano,
        webots,
        orchestrator,
        *drone_nodes,
        *go1_nodes,
    ])
