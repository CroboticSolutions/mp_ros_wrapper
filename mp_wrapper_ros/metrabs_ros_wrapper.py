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

from mp_wrapper_ros.mp_utils import packMPHPE3DMsg, getMarkerArray, createMarkerArrow, draw_bounding_box

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle



QUEUE_SIZE = 1
PLOT_HPE_POSE = True
DETECT_GESTURES = False
PLOT_MARKER = True
GET_HAND_ORIENTATION = False

# Mediapipe documentation/tutorials: 
# - [ ] Add metrabs documentation link

# TODO: 
# - [x] Add init loading for the metrabs model
# - [x] Create launch file for this node
# - [x] Add building of this node to the CMakeLists.txt
# - [ ] Add a subscriber to the image topic
# - [ ] Add a publisher for the human pose 2D message
# - [ ] Add a publisher for the human pose 3D message 

class MPROSWrapper(Node):
    def __init__(self):
        super().__init__('human_pose_node')
        self.bridge = CvBridge()

        self.pose_model = self.load_hpe_model()

        self.img_recv = False
        self.img_msg = None
        self.img_topic_name = '/dummy/image_raw'

        self._init_publishers()
        self._init_subscribers()

        test_img = True
        if test_img:
            self.image = cv2.imread('/root/ros2_ws/src/mp_ros_wrapper/michael_jordan.jpg')  # Replace with your image path
            self.publisher = self.create_publisher(Image, self.img_topic_name, 10)

        self.get_logger().info("Metrabs ROS 2 node initialized.")
        freq = 25 
        self.timer = self.create_timer(1/freq, self.timer_callback)

    def _init_publishers(self):
        self.image_pub = self.create_publisher(Image, 'human_pose_img', QUEUE_SIZE)
        self.loc_hpe3d_pub = self.create_publisher(MpHumanPose3D, 'loc/hpe3d', QUEUE_SIZE)

    def _init_subscribers(self):
        self.create_subscription(Image, self.img_topic_name, self.img_cb, QUEUE_SIZE)

    def load_hpe_model(self, model_link='https://bit.ly/metrabs_s_256'):
        # If loaded from the web link  
        model = tf_hub.load(model_link)
        return model 

    def img_cb(self, msg):
        self.img_msg = msg
        self.img_recv = True
        self.get_logger().debug("Image received at %s stamp" % msg.header.stamp)

    def publish_image(self):
        msg = self.bridge.cv2_to_imgmsg(self.image, encoding='bgr8')
        self.publisher.publish(msg)
        self.get_logger().info('Published dummy image.')

    def timer_callback(self):
        self.get_logger().info("Timer callback triggered.")
        self.publish_image()
        if self.img_recv:
            copied_img_msg = copy.deepcopy(self.img_msg)
            try:
                self.process_image(copied_img_msg)
            except Exception as e:
                self.get_logger().error(f"Error processing image: {e}")

    def plot_detections(self, img, detections): 
        for i in range(0, len(detections[:, :4]), 1):
            x, y, w, h = detections[i, :4]
            print("Detection %d: x=%d, y=%d, w=%d, h=%d" % (i, x, y, w, h))
            cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
        return img

    def plot_2d_pose(self, img, pose_2d):
        if PLOT_HPE_POSE:
            for i in range(len(pose_2d)):
                x, y = pose_2d[i]
                if x > 0 and y > 0:
                    cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
                    cv2.putText(img, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return img


    def process_image(self, img_msg):
        tf_img = self.ros_img_to_tf_tensor(img_msg, desired_encoding='rgb8')
        prediction = self.pose_model.detect_poses(tf_img, skeleton='smpl24')

        self.get_logger().info("Prediction 2d reciv: %s" % prediction['poses2d'].numpy())
        self.get_logger().info("Prediction 3d reciv: %s" % prediction['poses3d'].numpy()) 

        debug_proc_img = False
        if debug_proc_img: 
            self.get_logger().debug("RGB image type is %s" % type(rgb_image))
            self.get_logger().debug("RGB image shape is %s" % str(rgb_image.shape))
            self.get_logger().debug("RGB image encoding is %s" % img_msg.encoding)
            self.get_logger().debug("cv_image type is %s" % type(cv_img))
            self.get_logger().debug("cv_image shape is %s" % str(cv_img.shape))

        debug_plot = True
        if debug_plot: 
            cv2_img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
            cv2_img = self.plot_detections(cv2_img, prediction['boxes'].numpy())
            cv2_img = self.plot_2d_pose(cv2_img, prediction['poses2d'].numpy())
            self.image = cv2_img


        self.get_logger().info("Prediction type is %s" % prediction)
        
    def ros_img_to_tf_tensor(self, ros_img_msg: Image, desired_encoding='rgb8') -> tf.Tensor:
        # Convert ROS Image to OpenCV (NumPy) image
        cv_image = self.bridge.imgmsg_to_cv2(ros_img_msg, desired_encoding=desired_encoding)  # shape: (H, W, 3)
    
        # Convert NumPy array to TensorFlow tensor
        tf_image = tf.convert_to_tensor(cv_image, dtype=tf.uint8)  # shape: (H, W, 3)
    
        return tf_image



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
