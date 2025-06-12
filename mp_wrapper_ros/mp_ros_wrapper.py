#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import cv2
import mediapipe as mp
import copy

from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge

from hpe_ros_msgs.msg import MpGesture, MpHumanPose3D
from visualization_msgs.msg import Marker, MarkerArray

from mp_utils import packMPHPE3DMsg, getMarkerArray, createMarkerArrow

QUEUE_SIZE = 1
PLOT_LOC_MARKER = False
PLOT_GLOB_MARKER = True
PLOT_HPE_POSE = True
DETECT_HANDS = True
DETECT_GESTURES = False
GET_HAND_ORIENTATION = True


class MPROSWrapper(Node):
    def __init__(self):
        super().__init__('human_pose_node')
        self.bridge = CvBridge()

        self.pose = mp.solutions.pose.Pose()
        self.hand_tracking = mp.solutions.hands.Hands()
        self.drawing_utils = mp.solutions.drawing_utils

        base_options = mp.tasks.BaseOptions(model_asset_path='/root/ros2_ws/src/mp_wrapper_ros/models/gesture_recognizer.task')
        options = mp.tasks.vision.GestureRecognizerOptions(base_options=base_options)
        self.gest_recognizer = mp.tasks.vision.GestureRecognizer.create_from_options(options)

        self.img_recv = False
        self.img_msg = None

        self._init_publishers()
        self._init_subscribers()

        self.get_logger().info("Mediapipe ROS 2 node initialized.")

        self.timer = self.create_timer(0.01, self.timer_callback)

    def _init_publishers(self):
        self.image_pub = self.create_publisher(Image, 'human_pose_img', QUEUE_SIZE)
        self.loc_hpe3d_pub = self.create_publisher(MpHumanPose3D, 'loc/hpe3d', QUEUE_SIZE)
        self.glob_hpe3d_pub = self.create_publisher(MpHumanPose3D, 'glob/hpe3d', QUEUE_SIZE)
        self.r_gest_pub = self.create_publisher(MpGesture, 'right_gest', QUEUE_SIZE)
        self.l_gest_pub = self.create_publisher(MpGesture, 'left_gest', QUEUE_SIZE)
        self.loc_ma_pub = self.create_publisher(MarkerArray, 'loc_hpe_ma', 1)
        self.glob_ma_pub = self.create_publisher(MarkerArray, 'glob_hpe_ma', 1)
        self.nr_ma_pub = self.create_publisher(Marker, 'nr_ma', 1)
        self.nl_ma_pub = self.create_publisher(Marker, 'nl_ma', 1)

    def _init_subscribers(self):
        self.create_subscription(Image, '/camera/color/image_raw', self.img_cb, QUEUE_SIZE)

    def img_cb(self, msg):
        self.img_msg = msg
        self.img_recv = True

    def timer_callback(self):
        if self.img_recv:
            copied_img_msg = copy.deepcopy(self.img_msg)
            try:
                self.process_image(copied_img_msg)
            except Exception as e:
                self.get_logger().error(f"Error processing image: {e}")

    def process_image(self, img_msg):
        cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

        self.detect_pose(cv_image, rgb_image)

        if DETECT_HANDS:
            self.detect_hands(cv_image, rgb_image, gestures=DETECT_GESTURES)

        ros_image = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
        ros_image.header.stamp = self.get_clock().now().to_msg()
        self.image_pub.publish(ros_image)

    def detect_pose(self, cv_img, rgb_img):
        results_pose = self.pose.process(rgb_img)
        now = self.get_clock().now().to_msg()

        header = Header()
        header.stamp = now
        header.frame_id = "camera_color_link"

        loc_hpe3d_msg = packMPHPE3DMsg(header, results_pose.pose_landmarks)
        self.loc_hpe3d_pub.publish(loc_hpe3d_msg)

        glob_hpe3d_msg = packMPHPE3DMsg(header, results_pose.pose_world_landmarks)
        self.glob_hpe3d_pub.publish(glob_hpe3d_msg)

        if PLOT_HPE_POSE and results_pose.pose_landmarks:
            self.drawing_utils.draw_landmarks(
                cv_img, results_pose.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS
            )

        if GET_HAND_ORIENTATION:
            n_r = self.gen_hand_normal(glob_hpe3d_msg.r_thumb, glob_hpe3d_msg.r_wrist,
                                       glob_hpe3d_msg.r_index, glob_hpe3d_msg.r_pinky)
            n_l = self.gen_hand_normal(glob_hpe3d_msg.l_thumb, glob_hpe3d_msg.l_wrist,
                                       glob_hpe3d_msg.l_index, glob_hpe3d_msg.l_pinky)
            rw = np.array([glob_hpe3d_msg.r_wrist.x, glob_hpe3d_msg.r_wrist.y, glob_hpe3d_msg.r_wrist.z])
            lw = np.array([glob_hpe3d_msg.l_wrist.x, glob_hpe3d_msg.l_wrist.y, glob_hpe3d_msg.l_wrist.z])

            self.nr_ma_pub.publish(createMarkerArrow(now, rw, rw + n_r, 1, color=(255, 0, 0)))
            self.nl_ma_pub.publish(createMarkerArrow(now, lw, lw + n_l, 2, color=(0, 255, 0)))

        if PLOT_LOC_MARKER:
            mA = getMarkerArray(now, results_pose.pose_landmarks.landmark, color=(255, 0, 0))
            self.loc_ma_pub.publish(mA)

        if PLOT_GLOB_MARKER:
            mA = getMarkerArray(now, results_pose.pose_world_landmarks.landmark, color=(0, 255, 0))
            self.glob_ma_pub.publish(mA)

    def detect_hands(self, cv_img, rgb_img, gestures=False):
        results_hands = self.hand_tracking.process(rgb_img)

        if gestures and results_hands.multi_handedness:
            w, h = self.img_msg.width, self.img_msg.height

            for i, hand_info in enumerate(results_hands.multi_handedness):
                label = hand_info.classification[0].label
                crop = lambda img, min_x, max_x, min_y, max_y: img[int(min_y):int(max_y), int(min_x):int(max_x)]

                def crop_hand(img, landmarks):
                    x_ = [a.x * w for a in landmarks.landmark]
                    y_ = [a.y * h for a in landmarks.landmark]
                    return crop(img, max(min(x_) - 20, 0), min(max(x_) + 20, w),
                                     max(min(y_) - 20, 0), min(max(y_) + 20, h))

                hand_img = crop_hand(cv_img, results_hands.multi_hand_landmarks[i])
                rgb_crop = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
                gesture = self.detect_gesture(rgb_crop)

                if gesture.gestures:
                    gest_msg = self.create_gesture_msg(label.lower(), gesture)
                    if label == "Left":
                        self.r_gest_pub.publish(gest_msg)
                    else:
                        self.l_gest_pub.publish(gest_msg)

        if results_hands.multi_hand_landmarks:
            for landmarks in results_hands.multi_hand_landmarks:
                self.drawing_utils.draw_landmarks(cv_img, landmarks, mp.solutions.hands.HAND_CONNECTIONS)

    def detect_gesture(self, rgb_img):
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
        return self.gest_recognizer.recognize(mp_img)

    def gen_hand_normal(self, thumb, wrist, index, pinky):
        t, w, i, p = [np.array([j.x, j.y, j.z]) for j in [thumb, wrist, index, pinky]]
        vt, vi = (t - w), (i - w)
        return np.cross(vt / np.linalg.norm(vt), vi / np.linalg.norm(vi))

    def create_gesture_msg(self, hand, result):
        msg = MpGesture()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera_color_link"
        msg.hand.data = hand
        msg.gesture.data = result.gestures[0][0].category_name
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = MPROSWrapper()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
