#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import cv2
import copy

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge

from hpe_ros_msgs.msg import MpGesture, MpHumanPose3D
from visualization_msgs.msg import Marker, MarkerArray

from mp_wrapper_ros.mp_utils import packMPHPE3DMsg, getMarkerArray, createMarkerArrow

QUEUE_SIZE = 20
PLOT_HPE_POSE = True
DETECT_HANDS = False
DETECT_GESTURES = False
PLOT_MARKER = False
GET_HAND_ORIENTATION = False
OAK_CAMERA_TOPIC = "/oak/rgb/image_raw"
USB_CAMERA_TOPIC = "/camera1/image_raw"

# Mediapipe documentation/tutorials: 
# https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/index#models
# https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python 
# https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/pose_landmarker/python/%5BMediaPipe_Python_Tasks%5D_Pose_Landmarker.ipynb

# TODO: 
# - [x] Init pkg ROS 1 to ROS 2 migration
# - [x] Init HPE detection 
# - [x] Init hand detection
# - [x] Publish pose landmarks as mA 
# - [x] Packing ROS 2 messages
# - [ ] Hand rotation estimation
# - [ ] Init gesture detection
# - [x] Run on the GPU (if available)
# - [ ] Check duration of the processing

class MPROSWrapper(Node):
    def __init__(self):
        super().__init__('human_pose_node')
        self.bridge = CvBridge()

        self.pose = mp.solutions.pose.Pose()
        self.hand_tracking = mp.solutions.hands.Hands()
        self.drawing_utils = mp.solutions.drawing_utils

        hpe_path = '/root/uav_ws/src/mp_ros_wrapper/models/pose_landmarker_full.task'
        self.pose_model = self.load_hpe_model(hpe_path, GPU=False)

        # Load hand estimation model
        if DETECT_HANDS:
            hand_model_path = '/root/uav_ws/src/mp_ros_wrapper/models/hand_landmarker.task'
            self.hand_model = self.load_hand_model(hand_model_path, GPU=False)

        self.img_recv = False
        self.img_msg = None

        self._init_publishers()
        self._init_subscribers()

        self.get_logger().info("Mediapipe ROS 2 node initialized.")
        freq = 25 
        self.timer = self.create_timer(1/freq, self.timer_callback)

    def _init_publishers(self):
        self.image_pub = self.create_publisher(Image, 'human_pose_img', QUEUE_SIZE)
        self.hpe3d_pub = self.create_publisher(MpHumanPose3D, 'hpe3d', QUEUE_SIZE)
        self.glob_hpe3d_pub = self.create_publisher(MpHumanPose3D, 'glob/hpe3d', QUEUE_SIZE)
        self.r_gest_pub = self.create_publisher(MpGesture, 'right_gest', QUEUE_SIZE)
        self.l_gest_pub = self.create_publisher(MpGesture, 'left_gest', QUEUE_SIZE)
        self.ma_pub = self.create_publisher(MarkerArray, 'hpe_ma', 1)
        self.nr_ma_pub = self.create_publisher(Marker, 'nr_ma', 1)
        self.nl_ma_pub = self.create_publisher(Marker, 'nl_ma', 1)

    def _init_subscribers(self):
        self.create_subscription(Image, USB_CAMERA_TOPIC, self.img_cb, QUEUE_SIZE)

    def load_hpe_model(self, path, GPU=False): 
        # Load human pose estimation model
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        if GPU: 
            base_ = BaseOptions(model_asset_path=path, delegate=mp.tasks.BaseOptions.Delegate.GPU)
        else: 
            base_ = BaseOptions(model_asset_path=path)
        options = PoseLandmarkerOptions(base_options=base_,
                                        running_mode=VisionRunningMode.IMAGE)
        return PoseLandmarker.create_from_options(options) 

    # TODO: Add same GPU support for the hand model 
    def load_hand_model(self, path, GPU=False):
        # Load hand landmarker model
        BaseOptions = mp.tasks.BaseOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        if GPU: 
            base_ = BaseOptions(model_asset_path=path, delegate=mp.tasks.BaseOptions.Delegate.GPU)
        else: 
            base_ = BaseOptions(model_asset_path=path)
        options = HandLandmarkerOptions(base_options=base_, 
                                        num_hands=2,
                                        running_mode=VisionRunningMode.IMAGE)
        return HandLandmarker.create_from_options(options)

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
             
        
    def detect_pose(self, cv_img, mp_img, rgb_img):
        results_pose = self.pose_model.detect(mp_img)
        #self.get_logger().info("Pose landmarks detected: %s" % results_pose.pose_landmarks)

        now = self.get_clock().now().to_msg()

        header = Header()
        header.stamp = now
        # FRAME_ID based on the camera 
        header.frame_id = "camera_color_link"

        if PLOT_HPE_POSE and len(results_pose.pose_landmarks) != 0:
            cv_img = draw_landmarks_on_image(rgb_img, results_pose)

        if GET_HAND_ORIENTATION:
            n_r = self.gen_hand_normal(glob_hpe3d_msg.r_thumb, glob_hpe3d_msg.r_wrist,
                                       glob_hpe3d_msg.r_index, glob_hpe3d_msg.r_pinky)
            n_l = self.gen_hand_normal(glob_hpe3d_msg.l_thumb, glob_hpe3d_msg.l_wrist,
                                       glob_hpe3d_msg.l_index, glob_hpe3d_msg.l_pinky)
            rw = np.array([glob_hpe3d_msg.r_wrist.x, glob_hpe3d_msg.r_wrist.y, glob_hpe3d_msg.r_wrist.z])
            lw = np.array([glob_hpe3d_msg.l_wrist.x, glob_hpe3d_msg.l_wrist.y, glob_hpe3d_msg.l_wrist.z])

            self.nr_ma_pub.publish(createMarkerArrow(now, rw, rw + n_r, 1, color=(255, 0, 0)))
            self.nl_ma_pub.publish(createMarkerArrow(now, lw, lw + n_l, 2, color=(0, 255, 0)))

        if PLOT_MARKER:
            mA = self.getMarkerArray(now, results_pose.pose_landmarks[0], color=(255, 0, 0))
            self.get_logger().info("Publishing marker array with %d markers" % len(mA.markers))
            self.ma_pub.publish(mA)

        # ROS messages packing
        # ROS messages for the further processing of the pose landmarks if required
        # Maybe use only ROS messages for the further processing? 
        # TODO: World landmarks vs. landmarks in the camera frame 
        hpe3d_msg = packMPHPE3DMsg(header, results_pose.pose_landmarks[0])
        self.hpe3d_pub.publish(hpe3d_msg)

        #glob_hpe3d_msg = packMPHPE3DMsg(header, results_pose.pose_world_landmarks)
        #self.glob_hpe3d_pub.publish(glob_hpe3d_msg)

        return cv_img

    # Draw markers for the landmarks in the MarkerArray format
    def getMarkerArray(self, stamp, landmarks, color):
        hpe3d = [(landmark.x, landmark.y, landmark.z) for landmark in landmarks]
        mA = self.createMarkerArray(stamp, hpe3d, color)
        return mA

    def createMarkerArray(self, stamp, keypoints, color=(255, 0, 0)):
        mA = MarkerArray()
        i = 0
        for landmark in keypoints:
            x,y,z = landmark[0], landmark[1], landmark[2]
            m_ = self.createMarker(stamp, x, y, z, i, color)
            i+=1 
            mA.markers.append(m_)
        return mA

    def createMarker(self, stamp, x_, y_, z_, i, color=(255, 0, 0)):
        #self.get_logger().debug("Creating marker with id %d at position (%f, %f, %f)" % (i, x_, y_, z_))
        m_ = Marker()
        m_.header.frame_id = "oak_rgb_camera_frame"
        m_.header.stamp = stamp
        m_.type = m_.SPHERE
        m_.id = i
        m_.action = m_.ADD
        m_.scale.x = 0.01
        m_.scale.y = 0.01
        m_.scale.z = 0.01
        m_.color.r = color[0] / 255.0
        m_.color.g = color[1] / 255.0
        m_.color.b = color[2] / 255.0
        m_.color.a = 1.0
        m_.pose.position.x = 1.0 * float(x_)
        m_.pose.position.y = 1.0 * float(y_)
        m_.pose.position.z = 1.0 * float(z_)
        m_.pose.orientation.x = 0.0
        m_.pose.orientation.y = 0.0
        m_.pose.orientation.z = 0.0
        m_.pose.orientation.w = 1.0
        #self.get_logger().info("Marker is: %s" % str(m_))
        return m_

    def detect_hands(self, mp_img, rgb_img, gestures=False):
        # Detect hands and plot them 
        results_hands = self.hand_model.detect(mp_img)
        cv_img = draw_hand_landmarks_on_image(rgb_img, results_hands)
        # self.get_logger().info("Hand landmarks detected: %s" % results_hands)

        # Detect gestures and publish them if enabled
        if gestures:
            w, h = self.img_msg.width, self.img_msg.height
            # TODO: Modify with gestures (Check comment at the end of this file)
            # OLD CODE DEPRECATED DUE TO NEW MP API
        return cv_img

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

# TODO: Put this to utils  
def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    mp.solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      mp.solutions.pose.POSE_CONNECTIONS,
      mp.solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image

def draw_hand_landmarks_on_image(rgb_image, detection_result):
    MARGIN = 10  # pixels
    FONT_SIZE = 1
    FONT_THICKNESS = 1
    HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        mp.solutions.drawing_utils.draw_landmarks(
        annotated_image,
        hand_landmarks_proto,
        mp.solutions.hands.HAND_CONNECTIONS,
        mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
        mp.solutions.drawing_styles.get_default_hand_connections_style())

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        cv2.putText(annotated_image, f"{handedness[0].category_name}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image

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
