import os
from setuptools import setup

package_name = 'mp_wrapper_ros'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), ['launch/mp_ros_wrapper.launch.py', 'launch/metrabs_ros_wrapper.launch.py', 'launch/vitpose_ros_wrapper.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='fzoric',
    maintainer_email='filip.zoric@fer.hr',
    description='The mediapipe wrapper for ROS 2',
    license='TODO',  # Replace with actual license
    tests_require=['pytest'],
    entry_points={
    'console_scripts': [
        'mp_ros_wrapper = mp_wrapper_ros.mp_ros_wrapper:main',
        'metrabs_ros_wrapper = mp_wrapper_ros.metrabs_ros_wrapper:main',
        'vitpose_ros_wrapper = mp_wrapper_ros.vitpose_ros_wrapper:main',
    ],
},
)

