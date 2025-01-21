#!/usr/bin/python3

import rospy
import copy
import cv2
import mediapipe as mp

from sensor_msgs.msg import Image
from hpe_ros_msgs.msg import MpGesture
from std_msgs.msg import Header
from cv_bridge import CvBridge

from mp_utils import packMPHPE3DMsg

# Google AI Edge API
# https://ai.google.dev/edge/api/mediapipe/python/mp/Image 

QUEUE_SIZE=1
class HumanPoseNode:
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

        self.image_pub = rospy.Publisher('/human_pose/image', Image, queue_size=QUEUE_SIZE)
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.img_cb, queue_size=QUEUE_SIZE)
        self.gesture_pub = rospy.Publisher('/human_pose/gesture', MpGesture, queue_size=QUEUE_SIZE)
        self.crop_left_hand = rospy.Publisher('/crop_left_hand', Image, queue_size=QUEUE_SIZE)
        self.crop_right_hand = rospy.Publisher('/crop_right_hand', Image, queue_size=QUEUE_SIZE)

        self.img_recv = False
        rospy.loginfo("Mediapipe node initialized.")
    
    def img_cb(self, msg):
        self.img_msg = msg
        self.img_recv = True
    
    def process_image(self, img_msg):

        cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)               
        self.detect_pose(cv_image, rgb_image)
        self.detect_hands(cv_image, rgb_image)
        ros_image = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
        self.image_pub.publish(ros_image)

    def detect_pose(self, cv_img, rgb_img):
        results_pose = self.pose.process(rgb_img)
        
        # This is local pose I think
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "camera_color_link"
        hpe3d_msg = packMPHPE3DMsg(header, results_pose.pose_landmarks)

        plot = False
        if plot:
            if results_pose.pose_landmarks:
                self.drawing_utils.draw_landmarks(
                    cv_img, results_pose.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS
                )

    def detect_hands(self, cv_img, rgb_img):

        results_hands = self.hand_tracking.process(rgb_img)

        # TODO: Move to utils and don't plot necessary
        def crop_img(cv_img, min_x, max_x, min_y, max_y):
            return cv_img[int(min_y):int(max_y), int(min_x):int(max_x)]
    
        def crop_hand(cv_img, hand_landmarks):
            x_ = [a.x*w for a in hand_landmarks.landmark]
            y_ = [a.y*h for a in hand_landmarks.landmark]
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
            # Crop left hand
            if results_hands.multi_handedness[i].classification[0].label == 'Left':
                left_cv_img = crop_hand(cv_img, results_hands.multi_hand_landmarks[0])
                self.crop_left_hand.publish(self.bridge.cv2_to_imgmsg(left_cv_img, encoding='bgr8'))
                left_rgb_img = cv2.cvtColor(left_cv_img, cv2.COLOR_BGR2RGB)   
                left_hand_gesture = self.detect_gesture(left_rgb_img)
                if left_hand_gesture.gestures: 
                    rospy.loginfo(f"Left hand gesture detected {left_hand_gesture.gestures[0]}")
            # Crop right hand
            if results_hands.multi_handedness[i].classification[0].label == 'Right':
                right_cv_img = crop_hand(cv_img, results_hands.multi_hand_landmarks[1])
                self.crop_right_hand.publish(self.bridge.cv2_to_imgmsg(right_cv_img, encoding='bgr8'))
                right_rgb_img = cv2.cvtColor(right_cv_img, cv2.COLOR_BGR2RGB)
                right_hand_gesture = self.detect_gesture(right_rgb_img)
                if right_hand_gesture.gestures: 
                    rospy.loginfo(f"Right hand gesture detected {right_hand_gesture.gestures[0]}")
                else: 
                    rospy.loginfo("No right hand gesture detected")
    
        #gestures = self.detect_gesture(rgb_img)
        #rospy.loginfo(f"Gestures detected: {gestures.gestures}")

        if results_hands.multi_hand_landmarks:
            self.drawing_utils.draw_landmarks(
                cv_img, results_hands.multi_hand_landmarks[0], mp.solutions.hands.HAND_CONNECTIONS
            )
        if results_hands.multi_hand_landmarks:
            self.drawing_utils.draw_landmarks(
                cv_img, results_hands.multi_hand_landmarks[1], mp.solutions.hands.HAND_CONNECTIONS
            )

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
        node = HumanPoseNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
