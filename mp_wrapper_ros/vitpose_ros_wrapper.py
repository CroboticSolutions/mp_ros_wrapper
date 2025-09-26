#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import copy
import torch
import supervision as sv
from PIL import Image
from transformers import AutoProcessor, RTDetrForObjectDetection, VitPoseForPoseEstimation, infer_device

from sensor_msgs.msg import Image as RosImage
from std_msgs.msg import Header

from hpe_ros_msgs.msg import MpHumanPose3D
from visualization_msgs.msg import Marker, MarkerArray

from mp_wrapper_ros.vit_utils import (
    ros_image_to_pil, 
    numpy_to_ros_image, 
    annotate_image_with_supervision,
    pack_vitpose_hpe3d_msg,
    create_vitpose_marker_array
)

QUEUE_SIZE = 1  # Reduced queue size for real-time processing
PLOT_HPE_POSE = True
PLOT_MARKER = True
OAK_CAMERA_TOPIC = "/oak/rgb/image_raw"
USB_CAMERA_TOPIC = "/camera1/image_raw"

class VitPoseROSWrapper(Node):
    def __init__(self):
        super().__init__('vitpose_ros_node')
        
        # Initialize device
        self.device = infer_device()
        self.get_logger().info(f"Using device: {self.device}")
        
        # Load person detection model
        self.get_logger().info("Loading person detection model...")
        self.person_image_processor = AutoProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
        self.person_model = RTDetrForObjectDetection.from_pretrained(
            "PekingU/rtdetr_r50vd_coco_o365", 
            device_map=self.device
        )
        
        # Load VitPose model
        self.get_logger().info("Loading VitPose model...")
        self.pose_image_processor = AutoProcessor.from_pretrained("usyd-community/vitpose-base-simple")
        self.pose_model = VitPoseForPoseEstimation.from_pretrained(
            "usyd-community/vitpose-base-simple", 
            device_map=self.device
        )
        
        self.img_recv = False
        self.img_msg = None
        
        self._init_publishers()
        self._init_subscribers()
        
        self.get_logger().info("🚀 VitPose ROS 2 node initialized and ready!")
        freq = 10  # Lower frequency due to heavier computation
        self.timer = self.create_timer(1/freq, self.timer_callback)

    def _init_publishers(self):
        self.image_pub = self.create_publisher(RosImage, 'vitpose/annotated_image', QUEUE_SIZE)
        self.hpe3d_pub = self.create_publisher(MpHumanPose3D, 'vitpose/hpe3d', QUEUE_SIZE)
        self.ma_pub = self.create_publisher(MarkerArray, 'vitpose/hpe_ma', 1)

    def _init_subscribers(self):
        self.create_subscription(RosImage, USB_CAMERA_TOPIC, self.img_cb, 1)

    def img_cb(self, msg):
        self.img_msg = msg
        self.img_recv = True
        self.get_logger().debug(f"Image received at {msg.header.stamp} stamp")

    def timer_callback(self):
        if self.img_recv:
            copied_img_msg = copy.deepcopy(self.img_msg)
            try:
                self.process_image(copied_img_msg)
            except Exception as e:
                self.get_logger().error(f"Error processing image: {e}")

    def process_image(self, img_msg):
        # Convert ROS image to PIL image (no cv_bridge needed!)
        pil_img = ros_image_to_pil(img_msg)
        
        self.get_logger().info("🔍 VitPose processing started!", throttle_duration_sec=2.0)
        
        start_time = self.get_clock().now()
        
        # Detect persons in the image
        person_boxes = self.detect_persons(pil_img)
        
        if len(person_boxes) == 0:
            self.get_logger().warn("No persons detected in image", throttle_duration_sec=5.0)
            return
        
        # Detect pose keypoints
        keypoints, scores, annotated_img = self.detect_pose(pil_img, person_boxes)
        
        duration = (self.get_clock().now() - start_time).nanoseconds / 1e6  # Convert to milliseconds
        self.get_logger().info(f"⚡ VitPose processing completed in {duration:.2f} ms", throttle_duration_sec=2.0)
        
        # Create ROS messages
        now = self.get_clock().now().to_msg()
        header = Header()
        header.stamp = now
        header.frame_id = "camera_color_link"
        
        # Publish HPE3D message for the first detected person
        if len(keypoints) > 0:
            # Debug: Check data types before packing
            kp = keypoints[0]
            sc = scores[0]
            self.get_logger().debug(f"Keypoints shape: {kp.shape}, type: {type(kp)}, dtype: {kp.dtype}")
            self.get_logger().debug(f"Scores shape: {sc.shape}, type: {type(sc)}, dtype: {sc.dtype}")
            self.get_logger().debug(f"Sample keypoint: {kp[0]} (type: {type(kp[0][0])})")
            self.get_logger().debug(f"Sample score: {sc[0]} (type: {type(sc[0])})")
            
            hpe3d_msg = pack_vitpose_hpe3d_msg(header, keypoints[0], scores[0])
            self.hpe3d_pub.publish(hpe3d_msg)
            
            # Publish marker array if enabled
            if PLOT_MARKER:
                ma_msg = create_vitpose_marker_array(header, keypoints[0], scores[0])
                self.ma_pub.publish(ma_msg)
        
        # Publish annotated image
        if PLOT_HPE_POSE:
            # Convert numpy array to ROS image (no cv_bridge needed!)
            ros_image = numpy_to_ros_image(
                annotated_img, 
                encoding='rgb8', 
                frame_id=header.frame_id, 
                stamp=now
            )
            self.image_pub.publish(ros_image)

    def detect_persons(self, pil_image):
        """Detect persons in the image using RTDetr"""
        inputs = self.person_image_processor(images=pil_image, return_tensors="pt").to(self.person_model.device)
        
        with torch.no_grad():
            outputs = self.person_model(**inputs)
        
        results = self.person_image_processor.post_process_object_detection(
            outputs, target_sizes=torch.tensor([(pil_image.height, pil_image.width)]), threshold=0.3
        )
        result = results[0]
        
        # Human label refers to index 0 in COCO dataset
        person_boxes = result["boxes"][result["labels"] == 0]
        person_boxes = person_boxes.cpu().numpy()
        
        # Convert boxes from VOC (x1, y1, x2, y2) to COCO (x1, y1, w, h) format
        if len(person_boxes) > 0:
            person_boxes[:, 2] = person_boxes[:, 2] - person_boxes[:, 0]  # width
            person_boxes[:, 3] = person_boxes[:, 3] - person_boxes[:, 1]  # height
        
        self.get_logger().debug(f"Detected {len(person_boxes)} persons")
        return person_boxes

    def detect_pose(self, pil_image, person_boxes):
        """Detect pose keypoints using VitPose"""
        if len(person_boxes) == 0:
            return [], [], np.array(pil_image)
            
        inputs = self.pose_image_processor(pil_image, boxes=[person_boxes], return_tensors="pt").to(self.pose_model.device)
        
        with torch.no_grad():
            outputs = self.pose_model(**inputs)
        
        pose_results = self.pose_image_processor.post_process_pose_estimation(outputs, boxes=[person_boxes])
        image_pose_result = pose_results[0]
        
        # Extract keypoints and scores
        keypoints_list = []
        scores_list = []
        
        for pose_result in image_pose_result:
            # Handle tensors that might already be on CPU
            kp_tensor = pose_result['keypoints']
            sc_tensor = pose_result['scores']
            
            keypoints = kp_tensor.cpu().numpy() if kp_tensor.is_cuda else kp_tensor.numpy()  # [17, 2]
            scores = sc_tensor.cpu().numpy() if sc_tensor.is_cuda else sc_tensor.numpy()       # [17]
            
            keypoints_list.append(keypoints)
            scores_list.append(scores)
        
        # Annotate image using Supervision
        annotated_img = annotate_image_with_supervision(pil_image, keypoints_list, scores_list)
        
        return keypoints_list, scores_list, annotated_img

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = VitPoseROSWrapper()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()