#!/usr/bin/python3

import rospy
import copy
import cv2
import mediapipe as mp

from sensor_msgs.msg import Image
from hpe_ros_msgs.msg import MpGesture, MpHumanPose3D
from std_msgs.msg import Header
from cv_bridge import CvBridge
from visualization_msgs.msg import MarkerArray

from mp_utils import packMPHPE3DMsg, getMarkerArray

# Google AI Edge API
# https://ai.google.dev/edge/api/mediapipe/python/mp/Image 

# TODO: 
# - [x] Create set of markers to visualize pose estimate
# - [ ] Test connecting with H2AMI 

QUEUE_SIZE=1
PLOT_LOC_MARKER=False
PLOT_GLOB_MARKER=True
PLOT_HPE_POSE=True
DETECT_HANDS=True
DETECT_GESTURES=False

class MPROSWrapper:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('human_pose_node', anonymous=True, log_level=rospy.DEBUG)

        # GPU is slower than CPU LOL
        self.pose = mp.solutions.pose.Pose()
        self.hand_tracking = mp.solutions.hands.Hands()
        self.drawing_utils = mp.solutions.drawing_utils

        # Gesture recognition
        base_options = mp.tasks.BaseOptions(model_asset_path='/root/catkin_ws/src/mp_wrapper_ros/models/gesture_recognizer.task')
        options = mp.tasks.vision.GestureRecognizerOptions(base_options=base_options)
        self.gest_recognizer = mp.tasks.vision.GestureRecognizer.create_from_options(options)

        self.bridge = CvBridge()
        self.img_recv = False
        self._init_subscribers()
        self._init_publishers()
        rospy.loginfo("Mediapipe node initialized.")

    def _init_publishers(self): 
        self.image_pub = rospy.Publisher('human_pose_img', Image, queue_size=QUEUE_SIZE)
        self.loc_hpe3d_pub = rospy.Publisher('loc/hpe3d', MpHumanPose3D, queue_size=QUEUE_SIZE)
        self.glob_hpe3d_pub = rospy.Publisher('glob/hpe3d', MpHumanPose3D, queue_size=QUEUE_SIZE)
        self.r_gest_pub = rospy.Publisher('right_gest', MpGesture, queue_size=QUEUE_SIZE)
        self.l_gest_pub = rospy.Publisher('left_gest', MpGesture, queue_size=QUEUE_SIZE)
        self.loc_ma_pub = rospy.Publisher('loc_hpe_ma', MarkerArray, queue_size=1)
        self.glob_ma_pub = rospy.Publisher('glob_hpe_ma', MarkerArray, queue_size=1) 
        #self.crop_right_hand = rospy.Publisher('/crop_right_hand', Image, queue_size=QUEUE_SIZE)
        #self.crop_left_hand = rospy.Publisher('/crop_left_hand', Image, queue_size=QUEUE_SIZE)
        
    def _init_subscribers(self): 
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.img_cb, queue_size=QUEUE_SIZE)
    
    def img_cb(self, msg):
        self.img_msg = msg
        self.img_recv = True
    
    def process_image(self, img_msg):

        cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)               
        self.detect_pose(cv_image, rgb_image)
        detect_hands = DETECT_HANDS
        if detect_hands:
            self.detect_hands(cv_image, rgb_image, gestures=DETECT_GESTURES)
        ros_image = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
        self.image_pub.publish(ros_image)

    def detect_pose(self, cv_img, rgb_img):
        results_pose = self.pose.process(rgb_img)
        
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "camera_color_link"
        # Local
        loc_hpe3d_msg = packMPHPE3DMsg(header, results_pose.pose_landmarks)
        self.loc_hpe3d_pub.publish(loc_hpe3d_msg)
        # Global   
        glob_hpe3d_msg = packMPHPE3DMsg(header, results_pose.pose_world_landmarks)
        self.glob_hpe3d_pub.publish(glob_hpe3d_msg)

        plot_pose = PLOT_HPE_POSE
        if plot_pose:
            if results_pose.pose_landmarks:
                self.drawing_utils.draw_landmarks(
                    cv_img, results_pose.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS
                )

        plot_loc_marker_array = PLOT_LOC_MARKER
        if plot_loc_marker_array: 
            landmarks = results_pose.pose_landmarks.landmark
            mA = getMarkerArray(header.stamp, landmarks, color=(255, 0, 0))
            self.loc_ma_pub.publish(mA)
        
        plot_glob_marker_array = PLOT_GLOB_MARKER
        if plot_glob_marker_array: 
            landmarks = results_pose.pose_world_landmarks.landmark
            mA = getMarkerArray(header.stamp, landmarks, color=(0, 255, 0))
            self.glob_ma_pub.publish(mA)

    def detect_hands(self, cv_img, rgb_img, gestures=False):

        results_hands = self.hand_tracking.process(rgb_img)
        if gestures: 
            # TODO: Move to utils and don't plot necessary
            def crop_img(cv_img, min_x, max_x, min_y, max_y):
                return cv_img[int(min_y):int(max_y), int(min_x):int(max_x)]
        
            def crop_hand(cv_img, hand_landmarks):
                x_ = [a.x * w for a in hand_landmarks.landmark]
                y_ = [a.y * h for a in hand_landmarks.landmark]
                min_x, max_x = min(x_), max(x_)
                if min_x < 0: min_x = 0
                else: 
                    min_x -= 20
                if max_x > w: max_x = w
                else: 
                    max_x += 20
                min_y, max_y = min(y_), max(y_)
                if min_y < 0: min_y = 0
                else:
                    min_y -= 20
                if max_y > h: max_y = h
                else: 
                    max_y += 20
                return crop_img(cv_img, min_x, max_x, min_y, max_y)

            w, h = self.img_msg.width, self.img_msg.height
            for i in range(len(results_hands.multi_handedness)):
                # Crop left hand [Left in the camera -> my right hand]
                if results_hands.multi_handedness[i].classification[0].label == 'Left':
                    r_hand_gesture = self.process_hand_gesture(cv_img, results_hands, crop_hand, i)
                    if r_hand_gesture.gestures:
                        rospy.logdebug(f"Right hand gesture detected {r_hand_gesture.gestures[0][0].category_name}")
                        rgest_msg = self.create_gesture_msg("right", r_hand_gesture)
                        self.r_gest_pub.publish(rgest_msg)

                # Crop right hand [Right in the camera -> my left hand]
                if results_hands.multi_handedness[i].classification[0].label == 'Right':
                    l_hand_gesture = self.process_hand_gesture(cv_img, results_hands, crop_hand, i)
                    if l_hand_gesture.gestures: 
                        rospy.logdebug(f"Left hand gesture detected {l_hand_gesture.gestures[0][0].category_name}")
                        lgest_msg = self.create_gesture_msg("left", l_hand_gesture)
                        self.l_gest_pub.publish(lgest_msg)
        if results_hands.multi_hand_landmarks:
            self.drawing_utils.draw_landmarks(
                cv_img, results_hands.multi_hand_landmarks[0], mp.solutions.hands.HAND_CONNECTIONS
            )
        if results_hands.multi_hand_landmarks:
            self.drawing_utils.draw_landmarks(
                cv_img, results_hands.multi_hand_landmarks[1], mp.solutions.hands.HAND_CONNECTIONS
            )

    def process_hand_gesture(self, cv_img, results_hands, crop_hand, i):
        cv_img = crop_hand(cv_img, results_hands.multi_hand_landmarks[i])
        #self.crop_left_hand.publish(self.bridge.cv2_to_imgmsg(left_cv_img, encoding='bgr8'))
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)   
        hand_gesture = self.detect_gesture(rgb_img)
        return hand_gesture

    def create_gesture_msg(self, hand, gesture_result):
        gesture_msg = MpGesture()
        gesture_msg.header.stamp = rospy.Time.now()
        gesture_msg.header.frame_id = "camera_color_link"
        gesture_msg.hand.data = hand
        gesture_msg.gesture.data = gesture_result.gestures[0][0].category_name
        return gesture_msg

    def detect_gesture(self, rgb_img):
        rgb_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
        results_gesture = self.gest_recognizer.recognize(rgb_frame)
        return results_gesture
    
    def run(self):
        while not rospy.is_shutdown():
            if self.img_recv: 
                copied_img_msg = copy.deepcopy(self.img_msg)
                try:
                    self.process_image(copied_img_msg)
                except Exception as e: 
                    rospy.logerr(f"Error processing image: {e}")
            rospy.sleep(0.01)
        

if __name__ == '__main__':
    try:
        node = MPROSWrapper()
        node.run()
    except rospy.ROSInterruptException:
        pass
