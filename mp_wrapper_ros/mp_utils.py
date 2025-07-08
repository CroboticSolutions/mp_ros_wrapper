from hpe_ros_msgs.msg import MpHumanPose3D
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import cv2

def packMPHPE3DMsg(header, landmarks):
    """ Pack a mediapipe human pose estimation message into a ROS message """
    msg = MpHumanPose3D()
    msg.header = header
    msg.nose.x = landmarks[0].x; msg.nose.y = landmarks[0].y; msg.nose.z = landmarks[0].z
    msg.l_eye_inner.x = landmarks[1].x; msg.l_eye_inner.y = landmarks[1].y; msg.l_eye_inner.z = landmarks[1].z
    msg.l_eye.x = landmarks[2].x; msg.l_eye.y = landmarks[2].y; msg.l_eye.z = landmarks[2].z
    msg.l_eye_outer.x = landmarks[3].x; msg.l_eye_outer.y = landmarks[3].y; msg.l_eye_outer.z = landmarks[3].z
    msg.r_eye_inner.x = landmarks[4].x; msg.r_eye_inner.y = landmarks[4].y; msg.r_eye_inner.z = landmarks[4].z
    msg.r_eye.x = landmarks[5].x; msg.r_eye.y = landmarks[5].y; msg.r_eye.z = landmarks[5].z
    msg.r_eye_outer.x = landmarks[6].x; msg.r_eye_outer.y = landmarks[6].y; msg.r_eye_outer.z = landmarks[6].z
    msg.l_ear.x = landmarks[7].x; msg.l_ear.y = landmarks[7].y; msg.l_ear.z = landmarks[7].z
    msg.r_ear.x = landmarks[8].x; msg.r_ear.y = landmarks[8].y; msg.r_ear.z = landmarks[8].z
    msg.mouth_l.x = landmarks[9].x; msg.mouth_l.y = landmarks[9].y; msg.mouth_l.z = landmarks[9].z
    msg.mouth_r.x = landmarks[10].x; msg.mouth_r.y = landmarks[10].y; msg.mouth_r.z = landmarks[10].z
    msg.l_shoulder.x = landmarks[11].x; msg.l_shoulder.y = landmarks[11].y; msg.l_shoulder.z = landmarks[11].z
    msg.r_shoulder.x = landmarks[12].x; msg.r_shoulder.y = landmarks[12].y; msg.r_shoulder.z = landmarks[12].z
    msg.l_elbow.x = landmarks[13].x; msg.l_elbow.y = landmarks[13].y; msg.l_elbow.z = landmarks[13].z
    msg.r_elbow.x = landmarks[14].x; msg.r_elbow.y = landmarks[14].y; msg.r_elbow.z = landmarks[14].z
    msg.l_wrist.x = landmarks[15].x; msg.l_wrist.y = landmarks[15].y; msg.l_wrist.z = landmarks[15].z
    msg.r_wrist.x = landmarks[16].x; msg.r_wrist.y = landmarks[16].y; msg.r_wrist.z = landmarks[16].z
    msg.l_pinky.x = landmarks[17].x; msg.l_pinky.y = landmarks[17].y; msg.l_pinky.z = landmarks[17].z
    msg.r_pinky.x = landmarks[18].x; msg.r_pinky.y = landmarks[18].y; msg.r_pinky.z = landmarks[18].z
    msg.l_index.x = landmarks[19].x; msg.l_index.y = landmarks[19].y; msg.l_index.z = landmarks[19].z
    msg.r_index.x = landmarks[20].x; msg.r_index.y = landmarks[20].y; msg.r_index.z = landmarks[20].z
    msg.l_thumb.x = landmarks[21].x; msg.l_thumb.y = landmarks[21].y; msg.l_thumb.z = landmarks[21].z
    msg.r_thumb.x = landmarks[22].x; msg.r_thumb.y = landmarks[22].y; msg.r_thumb.z = landmarks[22].z
    msg.l_hip.x = landmarks[23].x; msg.l_hip.y = landmarks[23].y; msg.l_hip.z = landmarks[23].z
    msg.r_hip.x = landmarks[24].x; msg.r_hip.y = landmarks[24].y; msg.r_hip.z = landmarks[24].z
    msg.l_knee.x = landmarks[25].x; msg.l_knee.y = landmarks[25].y; msg.l_knee.z = landmarks[25].z
    msg.r_knee.x = landmarks[26].x; msg.r_knee.y = landmarks[26].y; msg.r_knee.z = landmarks[26].z
    msg.l_ankle.x = landmarks[27].x; msg.l_ankle.y = landmarks[27].y; msg.l_ankle.z = landmarks[27].z
    msg.r_ankle.x = landmarks[28].x; msg.r_ankle.y = landmarks[28].y; msg.r_ankle.z = landmarks[28].z
    msg.l_heel.x = landmarks[29].x; msg.l_heel.y = landmarks[29].y; msg.l_heel.z = landmarks[29].z
    msg.r_heel.x = landmarks[30].x; msg.r_heel.y = landmarks[30].y; msg.r_heel.z = landmarks[30].z
    msg.l_foot_index.x = landmarks[31].x; msg.l_foot_index.y = landmarks[31].y; msg.l_foot_index.z = landmarks[31].z
    msg.r_foot_index.x = landmarks[32].x; msg.r_foot_index.y = landmarks[32].y; msg.r_foot_index.z = landmarks[32].z
    print(msg)
    return msg


def getMarkerArray(stamp, landmarks, color):
    hpe3d = [(landmark.x, landmark.y, landmark.z) for landmark in landmarks]
    #mA = createMarkerArray(stamp, hpe3d, color)
    return mA

def createMarkerArray(stamp, keypoints, color=(255, 0, 0)):
    mA = MarkerArray()
    i = 0
    for landmark in keypoints:
        x,y,z = landmark[0], landmark[1], landmark[2]
        m_ = createMarker(stamp, x, y, z, i, color)
        i+=1 
        mA.markers.append(m_)
    return mA

def createMarker(stamp, x, y, z, i, color=(255, 0, 0)):
    m_ = Marker()
    m_.header.frame_id = "oak-d-base-frame"
    m_.header.stamp = stamp
    m_.type = m_.SPHERE
    m_.id = i
    m_.action = m_.ADD
    m_.scale.x = 0.1
    m_.scale.y = 0.1
    m_.scale.z = 0.1
    m_.color.a = 1.0
    m_.color.r = color[0] / 255.0
    m_.color.g = color[1] / 255.0
    m_.color.b = color[2] / 255.0
    m_.pose.position.x = float(x)
    m_.pose.position.y = float(y)
    m_.pose.position.z = float(z)
    m_.pose.orientation.x = 0
    m_.pose.orientation.y = 0
    m_.pose.orientation.z = 0
    m_.pose.orientation.w = 1
    return m_

def createMarkerArrow(stamp, start_point, end_point, i, color=(255, 0, 0)):
    m_ = Marker()
    m_.header.frame_id = "camera_color_frame"
    m_.header.stamp = stamp
    m_.type = m_.ARROW
    m_.id = i
    m_.action = m_.ADD
    m_.scale.x = 0.02  # shaft diameter
    m_.scale.y = 0.1  # head diameter
    m_.scale.z = 0.1  # head length
    m_.color.a = 1.0
    m_.color.r = color[0]
    m_.color.g = color[1]
    m_.color.b = color[2]
    pt1 = Point()
    pt2 = Point()
    pt1.x = start_point[0]
    pt1.y = start_point[1]
    pt1.z = start_point[2]
    pt2.x = end_point[0]
    pt2.y = end_point[1]
    pt2.z = end_point[2]
    m_.points.append(pt1)
    m_.points.append(pt2)
    return m_


# Drawing methods for the bounding box
def draw_bounding_box(image, bbox, color=(0, 255, 0), thickness=2, label=None):
    """
    Draws a bounding box on the image.

    Parameters:
        image (np.ndarray): The input image (BGR format).
        bbox (tuple): Bounding box in the format (x, y, w, h).
        color (tuple): Color of the bounding box in BGR (default is green).
        thickness (int): Line thickness of the bounding box (default is 2).
        label (str): Optional text label to display above the box.

    Returns:
        np.ndarray: Image with bounding box drawn.
    """
    x, y, w, h = bbox
    image_with_box = image.copy()
    cv2.rectangle(image_with_box, (x, y), (x + w, y + h), color, thickness)

    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(label, font, 0.5, 1)[0]
        text_x, text_y = x, y - 10 if y - 10 > 10 else y + 10
        cv2.rectangle(image_with_box, (x, y - text_size[1] - 4), (x + text_size[0] + 4, y), color, -1)
        cv2.putText(image_with_box, label, (x + 2, y - 2), font, 0.5, (255, 255, 255), 1)

    return image_with_box