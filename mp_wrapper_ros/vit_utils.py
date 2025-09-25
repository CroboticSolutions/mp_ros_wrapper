#!/usr/bin/env python3
"""
Utility functions for VitPose ROS wrapper
"""

import numpy as np
import supervision as sv
from PIL import Image
from sensor_msgs.msg import Image as RosImage
from std_msgs.msg import Header
from hpe_ros_msgs.msg import MpHumanPose3D
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point


# VitPose COCO keypoint indices (17 keypoints)
VITPOSE_KEYPOINTS = {
    'nose': 0,
    'l_eye': 1,
    'r_eye': 2,
    'l_ear': 3,
    'r_ear': 4,
    'l_shoulder': 5,
    'r_shoulder': 6,
    'l_elbow': 7,
    'r_elbow': 8,
    'l_wrist': 9,
    'r_wrist': 10,
    'l_hip': 11,
    'r_hip': 12,
    'l_knee': 13,
    'r_knee': 14,
    'l_ankle': 15,
    'r_ankle': 16
}


def ros_image_to_pil(ros_image):
    """Convert ROS Image message to PIL Image (no cv_bridge needed!)"""
    # Assuming RGB8 or BGR8 encoding
    if ros_image.encoding == 'rgb8':
        # Reshape the data
        img_array = np.frombuffer(ros_image.data, dtype=np.uint8)
        img_array = img_array.reshape((ros_image.height, ros_image.width, 3))
        return Image.fromarray(img_array, mode='RGB')
    elif ros_image.encoding == 'bgr8':
        # Convert BGR to RGB
        img_array = np.frombuffer(ros_image.data, dtype=np.uint8)
        img_array = img_array.reshape((ros_image.height, ros_image.width, 3))
        # Swap BGR to RGB
        img_array = img_array[:, :, ::-1]
        return Image.fromarray(img_array, mode='RGB')
    else:
        raise ValueError(f"Unsupported encoding: {ros_image.encoding}")


def numpy_to_ros_image(numpy_array, encoding='rgb8', frame_id='', stamp=None):
    """Convert numpy array to ROS Image message (no cv_bridge needed!)"""
    ros_image = RosImage()
    ros_image.height, ros_image.width = numpy_array.shape[:2]
    ros_image.encoding = encoding
    ros_image.is_bigendian = False
    ros_image.step = ros_image.width * 3  # 3 bytes per pixel for RGB
    ros_image.data = numpy_array.tobytes()
    
    # Set header
    ros_image.header.frame_id = frame_id
    if stamp is not None:
        ros_image.header.stamp = stamp
    
    return ros_image


def annotate_image_with_supervision(pil_image, keypoints_list, scores_list, edge_color=sv.Color.GREEN, vertex_color=sv.Color.RED):
    """Draw keypoints on the image using Supervision (no OpenCV needed!)"""
    if len(keypoints_list) == 0:
        return np.array(pil_image)
    
    # Convert keypoints and scores to format expected by Supervision
    xy = np.stack(keypoints_list)  # Shape: [num_persons, 17, 2]
    confidence = np.stack(scores_list)  # Shape: [num_persons, 17]
    
    # Create Supervision KeyPoints object
    key_points = sv.KeyPoints(xy=xy, confidence=confidence)
    
    # Create annotators with nice colors
    edge_annotator = sv.EdgeAnnotator(
        color=edge_color,
        thickness=2
    )
    vertex_annotator = sv.VertexAnnotator(
        color=vertex_color,
        radius=4
    )
    
    # Start with PIL image copy
    scene = pil_image.copy()
    
    # Apply annotations
    annotated_scene = edge_annotator.annotate(
        scene=scene,
        key_points=key_points
    )
    annotated_scene = vertex_annotator.annotate(
        scene=annotated_scene,
        key_points=key_points
    )
    
    # Convert PIL to numpy array for ROS
    return np.array(annotated_scene)


def create_geometry_point(x, y, z):
    """Create a geometry_msgs/Point"""
    point = Point()
    try:
        # Ensure all values are Python float type (not numpy float)
        point.x = float(x) if x is not None else 0.0
        point.y = float(y) if y is not None else 0.0
        point.z = float(z) if z is not None else 0.0
        return point
    except (TypeError, ValueError) as e:
        print(f"ERROR in create_geometry_point: x={x} ({type(x)}), y={y} ({type(y)}), z={z} ({type(z)})")
        print(f"Error: {e}")
        # Fallback to zeros
        point.x = 0.0
        point.y = 0.0
        point.z = 0.0
        return point


def interpolate_neck_from_shoulders(l_shoulder, r_shoulder):
    """Interpolate neck position from shoulders (VitPose doesn't have neck)"""
    neck = create_geometry_point(
        (l_shoulder.x + r_shoulder.x) / 2.0,
        (l_shoulder.y + r_shoulder.y) / 2.0,
        (l_shoulder.z + r_shoulder.z) / 2.0
    )
    return neck


def pack_vitpose_hpe3d_msg(header, keypoints, scores, depth_scale=0.01):
    """
    Pack VitPose keypoints into MpHumanPose3D message
    
    Args:
        header: ROS message header
        keypoints: numpy array of keypoints [17, 2]  
        scores: numpy array of confidence scores [17]
        depth_scale: scaling factor for converting confidence to z-coordinate
    
    Returns:
        MpHumanPose3D message
    """
    
    def create_point_from_keypoint(idx):
        """Create a point from keypoint index with confidence-based depth"""
        if idx < len(keypoints) and len(scores) > idx and scores[idx] > 0.1:
            x, y = keypoints[idx]
            z = scores[idx] * depth_scale  # Use confidence as rough depth
            
            # Convert numpy types to Python float and handle NaN/inf
            x_float = float(x) if np.isfinite(x) else 0.0
            y_float = float(y) if np.isfinite(y) else 0.0
            z_float = float(z) if np.isfinite(z) else 0.0
            
            return create_geometry_point(x_float, y_float, z_float)
        else:
            return create_geometry_point(0.0, 0.0, 0.0)
    
    msg = MpHumanPose3D()
    msg.header = header
    
    # Map COCO keypoints to our message format
    msg.nose = create_point_from_keypoint(VITPOSE_KEYPOINTS['nose'])
    msg.l_eye = create_point_from_keypoint(VITPOSE_KEYPOINTS['l_eye'])
    msg.r_eye = create_point_from_keypoint(VITPOSE_KEYPOINTS['r_eye'])
    msg.l_ear = create_point_from_keypoint(VITPOSE_KEYPOINTS['l_ear'])
    msg.r_ear = create_point_from_keypoint(VITPOSE_KEYPOINTS['r_ear'])
    msg.l_shoulder = create_point_from_keypoint(VITPOSE_KEYPOINTS['l_shoulder'])
    msg.r_shoulder = create_point_from_keypoint(VITPOSE_KEYPOINTS['r_shoulder'])
    msg.l_elbow = create_point_from_keypoint(VITPOSE_KEYPOINTS['l_elbow'])
    msg.r_elbow = create_point_from_keypoint(VITPOSE_KEYPOINTS['r_elbow'])
    msg.l_wrist = create_point_from_keypoint(VITPOSE_KEYPOINTS['l_wrist'])
    msg.r_wrist = create_point_from_keypoint(VITPOSE_KEYPOINTS['r_wrist'])
    msg.l_hip = create_point_from_keypoint(VITPOSE_KEYPOINTS['l_hip'])
    msg.r_hip = create_point_from_keypoint(VITPOSE_KEYPOINTS['r_hip'])
    msg.l_knee = create_point_from_keypoint(VITPOSE_KEYPOINTS['l_knee'])
    msg.r_knee = create_point_from_keypoint(VITPOSE_KEYPOINTS['r_knee'])
    msg.l_ankle = create_point_from_keypoint(VITPOSE_KEYPOINTS['l_ankle'])
    msg.r_ankle = create_point_from_keypoint(VITPOSE_KEYPOINTS['r_ankle'])
    
    # VitPose doesn't directly provide these, so we'll estimate or set to zero
    # Note: MpHumanPose3D doesn't have a 'neck' field, so we skip it
    
    # Hand landmarks not available in VitPose COCO format
    msg.l_thumb = create_geometry_point(0.0, 0.0, 0.0)
    msg.r_thumb = create_geometry_point(0.0, 0.0, 0.0)
    msg.l_index = create_geometry_point(0.0, 0.0, 0.0)
    msg.r_index = create_geometry_point(0.0, 0.0, 0.0)
    msg.l_pinky = create_geometry_point(0.0, 0.0, 0.0)
    msg.r_pinky = create_geometry_point(0.0, 0.0, 0.0)
    
    # MediaPipe-specific fields that VitPose doesn't provide
    msg.l_eye_inner = create_geometry_point(0.0, 0.0, 0.0)
    msg.l_eye_outer = create_geometry_point(0.0, 0.0, 0.0)
    msg.r_eye_inner = create_geometry_point(0.0, 0.0, 0.0)
    msg.r_eye_outer = create_geometry_point(0.0, 0.0, 0.0)
    msg.mouth_l = create_geometry_point(0.0, 0.0, 0.0)
    msg.mouth_r = create_geometry_point(0.0, 0.0, 0.0)
    msg.l_heel = create_geometry_point(0.0, 0.0, 0.0)
    msg.r_heel = create_geometry_point(0.0, 0.0, 0.0)
    msg.l_foot_index = create_geometry_point(0.0, 0.0, 0.0)
    msg.r_foot_index = create_geometry_point(0.0, 0.0, 0.0)
    
    return msg


def create_vitpose_marker_array(header, keypoints, scores, confidence_threshold=0.3):
    """Create marker array for visualization of VitPose keypoints"""
    ma = MarkerArray()
    
    for i, (keypoint, score) in enumerate(zip(keypoints, scores)):
        if score > confidence_threshold:
            marker = Marker()
            marker.header = header
            marker.type = Marker.SPHERE
            marker.id = i
            marker.action = Marker.ADD
            marker.scale.x = 0.02
            marker.scale.y = 0.02
            marker.scale.z = 0.02
            
            # Color based on keypoint type
            if i in [VITPOSE_KEYPOINTS['nose']]:  # Head
                marker.color.r, marker.color.g, marker.color.b = 1.0, 1.0, 0.0  # Yellow
            elif i in [VITPOSE_KEYPOINTS['l_shoulder'], VITPOSE_KEYPOINTS['r_shoulder']]:  # Shoulders
                marker.color.r, marker.color.g, marker.color.b = 1.0, 0.0, 0.0  # Red
            elif i in [VITPOSE_KEYPOINTS['l_wrist'], VITPOSE_KEYPOINTS['r_wrist']]:  # Wrists
                marker.color.r, marker.color.g, marker.color.b = 0.0, 1.0, 0.0  # Green
            else:  # Other joints
                marker.color.r, marker.color.g, marker.color.b = 0.0, 0.0, 1.0  # Blue
            
            marker.color.a = 1.0
            
            x, y = keypoint
            marker.pose.position.x = float(x) * 0.001  # Convert to meters (rough scaling)
            marker.pose.position.y = float(y) * 0.001
            marker.pose.position.z = float(score) * 0.01  # Use confidence as depth (ensure float)
            
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            
            ma.markers.append(marker)
    
    return ma