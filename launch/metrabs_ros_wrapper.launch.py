from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            namespace='mp_ros',
            package='mp_wrapper_ros',
            executable='metrabs_ros_wrapper',
            name='metrabs_ros_wrapper',
            output='screen',
            arguments=['--ros-args', '--log-level', 'info']
        ),
    ])
