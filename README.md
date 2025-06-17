# mp_ros_wrapper

Wrapper for the mediapipe ML pipelines with ROS 2. 

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

# TODO: 
- [x] Check image delay
- [x] Add gesture recognition
- [x] Add launch
- [x] Add 3D points info 
- [x] Record bag and compare with the openpose
- [ ] Connect with the H2AMI 
- [ ] Check 3D point estimation 
- [ ] Compare with openpose

