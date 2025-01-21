import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import mediapipe as mp

class HumanPoseNode:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('human_pose_node', anonymous=True, log_level=rospy.DEBUG)

        # Mediapipe (POSE and HANDS) initialization
        # TODO: Move to the GPU if available
        self.pose = mp.solutions.pose.Pose()
        self.hand_tracking = mp.solutions.hands.Hands()
        self.drawing_utils = mp.solutions.drawing_utils

        # CV Bridge for ROS <-> OpenCV
        self.bridge = CvBridge()

        # Large queue size causes delays
        self.image_pub = rospy.Publisher('/human_pose/image', Image)
        self.enter_img_pub = rospy.Publisher('/human_pose/enter_image', Image)
        
        # ROS subscription and publication
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.img_cb, queue_size=1)

        self.init = True
        rospy.loginfo("Mediapipe node initialized.")

    
    def img_cb(self, msg):
        self.enter_img_pub.publish(msg)
        try:
            start_time_cb = rospy.Time.now()
            # Convert ROS Image message to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

            start_time = rospy.Time.now()
            # Process the image with Mediapipe
            results_pose = self.pose.process(rgb_image)
            results_hands = self.hand_tracking.process(rgb_image)
            duration = rospy.Time.now() - start_time
            rospy.logdebug(f"Processing time: {duration.to_sec()}s")

            plot_start_time = rospy.Time.now()
            # Draw pose landmarks on the image
            if results_pose.pose_landmarks:
                self.drawing_utils.draw_landmarks(
                    cv_image, results_pose.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS
                )
            # Right hand
            if results_hands.multi_hand_landmarks:
                self.drawing_utils.draw_landmarks(
                    cv_image, results_hands.multi_hand_landmarks[0], mp.solutions.hands.HAND_CONNECTIONS
                )
            # Left hand
            if results_hands.multi_hand_landmarks:
                self.drawing_utils.draw_landmarks(
                    cv_image, results_hands.multi_hand_landmarks[1], mp.solutions.hands.HAND_CONNECTIONS
                )
            plot_duration = rospy.Time.now() - plot_start_time
            rospy.logdebug(f"Plotting time: {plot_duration.to_sec()}s")

            # Convert OpenCV image back to ROS Image message
            ros_image = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')

            # Publish the annotated image
            self.image_pub.publish(ros_image)

            duration_cb = rospy.Time.now() - start_time_cb
            rospy.logdebug(f"Callback time: {duration_cb.to_sec()}s")
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")

    def run(self):
        while not rospy.is_shutdown():
            rospy.sleep(0.01)

if __name__ == '__main__':
    try:
        node = HumanPoseNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
