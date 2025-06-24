#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import cv2
import copy

import tensorflow as tf
import tensorflow_hub as tf_hub

from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge

from hpe_ros_msgs.msg import MpGesture, MpHumanPose3D
from visualization_msgs.msg import Marker, MarkerArray

from mp_wrapper_ros.mp_utils import packMPHPE3DMsg, getMarkerArray, createMarkerArrow

QUEUE_SIZE = 1
PLOT_HPE_POSE = True
DETECT_GESTURES = False
PLOT_MARKER = True
GET_HAND_ORIENTATION = False

# Mediapipe documentation/tutorials: 
# - [ ] Add metrabs documentation link

# TODO: 
# - [ ] Add init loading for the metrabs model
# - [ ] Create launch file for this node
# - [ ] Add building of this node to the CMakeLists.txt

class MPROSWrapper(Node):
    def __init__(self):
        super().__init__('human_pose_node')
        self.bridge = CvBridge()

        self.pose_model = self.load_hpe_model()

        self.img_recv = False
        self.img_msg = None

        self._init_publishers()
        self._init_subscribers()

        self.get_logger().info("Mediapipe ROS 2 node initialized.")
        freq = 25 
        self.timer = self.create_timer(1/freq, self.timer_callback)

    def _init_publishers(self):
        self.image_pub = self.create_publisher(Image, 'human_pose_img', QUEUE_SIZE)
        self.loc_hpe3d_pub = self.create_publisher(MpHumanPose3D, 'loc/hpe3d', QUEUE_SIZE)

    def _init_subscribers(self):
        self.create_subscription(Image, '/oak/rgb/image_raw', self.img_cb, QUEUE_SIZE)

    def load_hpe_model(self, model_link='https://bit.ly/metrabs_s_256'):
        # If loaded from the web link  
        model = tf_hub.load(model_link)
        return model 

    def img_cb(self, msg):
        self.img_msg = msg
        self.img_recv = True
        self.get_logger().debug("Image received at %s stamp" % msg.header.stamp)

    def timer_callback(self):
        if self.img_recv:
            copied_img_msg = copy.deepcopy(self.img_msg)
            try:
                self.process_image(copied_img_msg)
            except Exception as e:
                self.get_logger().error(f"Error processing image: {e}")

    def process_image(self, img_msg):
        cv_img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)

        debug_proc_img = False
        if debug_proc_img: 
            self.get_logger().debug("RGB image type is %s" % type(rgb_image))
            self.get_logger().debug("RGB image shape is %s" % str(rgb_image.shape))
            self.get_logger().debug("RGB image encoding is %s" % img_msg.encoding)
            self.get_logger().debug("cv_image type is %s" % type(cv_img))
            self.get_logger().debug("cv_image shape is %s" % str(cv_img.shape))

        start_time = self.get_clock().now()
        anot_img = self.detect_pose(cv_img, mp_img, rgb_img)
        duration = (self.get_clock().now() - start_time).nanoseconds / 1e6  # Convert to milliseconds
        self.get_logger().info("Pose detection took %.2f ms" % duration)

        # Detect hands and gestures if enabled
        if DETECT_HANDS:
            start_time = self.get_clock().now()
            anot_img = self.detect_hands(mp_img, anot_img, gestures=DETECT_GESTURES)
            duration = (self.get_clock().now() - start_time).nanoseconds / 1e6  # Convert to milliseconds
            self.get_logger().info("Hand detection took %.2f ms" % duration)

        ros_image = self.bridge.cv2_to_imgmsg(anot_img, encoding='rgb8')
        ros_image.header.stamp = self.get_clock().now().to_msg()
        self.get_logger().debug("Publishing image with stamp %s" % ros_image.header.stamp)
        self.image_pub.publish(ros_image)


def main(args=None):
    rclpy.init(args=args)
    node = MPROSWrapper()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

# DEPRECATED GESTURE DETECTION CODE
#             for i, hand_info in enumerate(results_hands.handedness):
#               label = hand_info.classification[0].label
#                crop = lambda img, min_x, max_x, min_y, max_y: img[int(min_y):int(max_y), int(min_x):int(max_x)]

#                def crop_hand(img, landmarks):
#                    x_ = [a.x * w for a in landmarks.landmark]
#                    y_ = [a.y * h for a in landmarks.landmark]
#                    return crop(img, max(min(x_) - 20, 0), min(max(x_) + 20, w),
#                                     max(min(y_) - 20, 0), min(max(y_) + 20, h))
#
#                hand_img = crop_hand(rgb_img, results_hands.multi_hand_landmarks[i])
#                rgb_crop = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
#                gesture = self.detect_gesture(rgb_crop)
#
#                if gesture.gestures:
#                    gest_msg = self.create_gesture_msg(label.lower(), gesture)
#                    if label == "Left":
#                        self.r_gest_pub.publish(gest_msg)
#                    else:
#                        self.l_gest_pub.publish(gest_msg)
