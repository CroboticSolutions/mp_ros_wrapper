# mp_ros_wrapper

Wrapper for the mediapipe ML pipelines with ROS 2. 

# Docker with GPU support 

Docker with GPU suport can be found at the following [link](https://github.com/larics/docker_files/tree/master/ros2/ros2-humble/gpu). 


# Launch luxonis camera with: 
```
docker start -i depthai_humble_cont 
```

```
ros2 launch depthai_ros_driver camera.launch.py
```

# Launch mediapipe ROS2 wrapper with: 

```
ros2 launch mp_wrapper_ros mp_ros_wrapper.launch.py 
```

# Launch metrabs ROS 2 wrapper with: 
```
ros2 launch mp_ros_wrapper metrabs_ros_wrapper.launch.py 
```

# TODO: 
- [x] Check image delay
- [x] Add gesture recognition
- [x] Add launch
- [x] Add 3D points info 
- [x] Record bag and compare with the openpose
- [x] Added VitPose
- [ ] Connect with the H2AMI 
- [ ] Check 3D point estimation 
- [ ] Compare with openpose

