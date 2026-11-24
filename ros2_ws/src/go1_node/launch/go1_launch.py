import os
from launch import LaunchDescription
from launch_ros.actions import Node
from webots_ros2_driver.webots_controller import WebotsController
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    package_dir = get_package_share_directory("go1_node")
    urdf = os.path.join(package_dir, "resource", "go1_a.urdf")
    mavic_urdf = "/opt/ros/humble/share/webots_ros2_mavic/resource/mavic_webots.urdf"

    def make_dog_controller(name):
        return WebotsController(
            robot_name=name,
            parameters=[{"robot_description": urdf}],
            respawn=True,
            port="1234",
        )

    def make_drone_controller(name):
        return WebotsController(
            robot_name=name,
            parameters=[{"robot_description": mavic_urdf}],
            respawn=True,
            port="1234",
        )

    def make_nav_node(name):
        return Node(
            package="go1_node",
            executable="go1_node",
            name=f"nav_{name}",
            parameters=[{
                "agent_id": name,
                "nano_url": "http://localhost:8001",
            }],
        )

    def make_drone_node(name):
        return Node(
            package="drone_node",
            executable="drone_controller",
            name=f"ctrl_{name}",
            parameters=[{"drone_id": name}],
        )

    return LaunchDescription([
        make_dog_controller("dog_1"),
        make_dog_controller("dog_2"),
        make_dog_controller("dog_3"),
        make_drone_controller("Drone_1"),
        make_drone_controller("Drone_2"),
        make_nav_node("dog_1"),
        make_nav_node("dog_2"),
        make_nav_node("dog_3"),
        make_drone_node("Drone_1"),
        make_drone_node("Drone_2"),
    ])