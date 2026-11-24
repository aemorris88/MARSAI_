from setuptools import setup
import os
from glob import glob

setup(
    name='drone_node',
    version='0.1.0',
    packages=['drone_node'],
    install_requires=['setuptools'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/drone_node']),
        ('share/drone_node', ['package.xml']),
    ],
    entry_points={
        'console_scripts': [
            'drone_node = drone_node.drone_node:main',
            'drone_controller = drone_node.drone_controller:main',
        ],
    },
)