#!/usr/bin/python3

import rospy
import copy
from sensor_msgs.msg import Image
from hpe_ros_msgs.msg import MpGesture
from cv_bridge import CvBridge
import cv2
import mediapipe as mp

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
        if results_pose.pose_landmarks:
            self.drawing_utils.draw_landmarks(
                cv_img, results_pose.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS
            )

    def detect_hands(self, cv_img, rgb_img):
        results_hands = self.hand_tracking.process(rgb_img)
        # TODO: Create parsers for the hand landmarks
        #print(len(results_hands.multi_handedness))
        #print(len(results_hands.multi_hand_landmarks))
        if results_hands.multi_hand_landmarks:
            self.drawing_utils.draw_landmarks(
                cv_img, results_hands.multi_hand_landmarks[0], mp.solutions.hands.HAND_CONNECTIONS
            )
        if results_hands.multi_hand_landmarks:
            self.drawing_utils.draw_landmarks(
                cv_img, results_hands.multi_hand_landmarks[1], mp.solutions.hands.HAND_CONNECTIONS
            )

    def detect_gesture(self, rgb_img):
        # rgb_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
        # results_gesture = self.gest_recognizer.recognize(rgb_frame)
        pass   
    
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
