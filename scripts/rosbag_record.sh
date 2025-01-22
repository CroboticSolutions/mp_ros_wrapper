#!/bin/bash

# Define the topics to record
topics=(
    "/mp_ros/loc/hpe3d"
    "/mp_ros/global/hpe3d"
)

sleep 10
# Start recording with the specified topics
echo "Starting rosbag recording for selected topics..."
rosbag record "${topics[@]}" --duration=30 -O mp_$1.bag


