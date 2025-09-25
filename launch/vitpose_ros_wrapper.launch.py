#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='mp_wrapper_ros',
            executable='vitpose_ros_wrapper',
            name='vitpose_ros_node',
            output='screen',
            parameters=[
                # Add any parameters here if needed
            ],
            remappings=[
                # Remap topics if needed
                # ('/camera1/image_raw', '/your_camera_topic'),
            ]
        )
    ])