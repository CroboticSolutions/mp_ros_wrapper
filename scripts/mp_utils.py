from hpe_ros_msgs.msg import MpHumanPose3D

def packMPHPE3DMsg(header, landmarks):
    """ Pack a mediapipe human pose estimation message into a ROS message """
    msg = MpHumanPose3D()
    msg.header = header
    msg.nose.x = landmarks.landmark[0].x; msg.nose.y = landmarks.landmark[0].y; msg.nose.z = landmarks.landmark[0].z
    msg.l_eye_inner.x = landmarks.landmark[1].x; msg.l_eye_inner.y = landmarks.landmark[1].y; msg.l_eye_inner.z = landmarks.landmark[1].z
    msg.l_eye.x = landmarks.landmark[2].x; msg.l_eye.y = landmarks.landmark[2].y; msg.l_eye.z = landmarks.landmark[2].z
    msg.l_eye_outer.x = landmarks.landmark[3].x; msg.l_eye_outer.y = landmarks.landmark[3].y; msg.l_eye_outer.z = landmarks.landmark[3].z
    msg.r_eye_inner.x = landmarks.landmark[4].x; msg.r_eye_inner.y = landmarks.landmark[4].y; msg.r_eye_inner.z = landmarks.landmark[4].z
    msg.r_eye.x = landmarks.landmark[5].x; msg.r_eye.y = landmarks.landmark[5].y; msg.r_eye.z = landmarks.landmark[5].z
    msg.r_eye_outer.x = landmarks.landmark[6].x; msg.r_eye_outer.y = landmarks.landmark[6].y; msg.r_eye_outer.z = landmarks.landmark[6].z
    msg.l_ear.x = landmarks.landmark[7].x; msg.l_ear.y = landmarks.landmark[7].y; msg.l_ear.z = landmarks.landmark[7].z
    msg.r_ear.x = landmarks.landmark[8].x; msg.r_ear.y = landmarks.landmark[8].y; msg.r_ear.z = landmarks.landmark[8].z
    msg.mouth_l.x = landmarks.landmark[9].x; msg.mouth_l.y = landmarks.landmark[9].y; msg.mouth_l.z = landmarks.landmark[9].z
    msg.mouth_r.x = landmarks.landmark[10].x; msg.mouth_r.y = landmarks.landmark[10].y; msg.mouth_r.z = landmarks.landmark[10].z
    msg.l_shoulder.x = landmarks.landmark[11].x; msg.l_shoulder.y = landmarks.landmark[11].y; msg.l_shoulder.z = landmarks.landmark[11].z
    msg.r_shoulder.x = landmarks.landmark[12].x; msg.r_shoulder.y = landmarks.landmark[12].y; msg.r_shoulder.z = landmarks.landmark[12].z
    msg.l_elbow.x = landmarks.landmark[13].x; msg.l_elbow.y = landmarks.landmark[13].y; msg.l_elbow.z = landmarks.landmark[13].z
    msg.r_elbow.x = landmarks.landmark[14].x; msg.r_elbow.y = landmarks.landmark[14].y; msg.r_elbow.z = landmarks.landmark[14].z
    msg.l_wrist.x = landmarks.landmark[15].x; msg.l_wrist.y = landmarks.landmark[15].y; msg.l_wrist.z = landmarks.landmark[15].z
    msg.r_wrist.x = landmarks.landmark[16].x; msg.r_wrist.y = landmarks.landmark[16].y; msg.r_wrist.z = landmarks.landmark[16].z
    msg.l_pinky.x = landmarks.landmark[17].x; msg.l_pinky.y = landmarks.landmark[17].y; msg.l_pinky.z = landmarks.landmark[17].z
    msg.r_pinky.x = landmarks.landmark[18].x; msg.r_pinky.y = landmarks.landmark[18].y; msg.r_pinky.z = landmarks.landmark[18].z
    msg.l_index.x = landmarks.landmark[19].x; msg.l_index.y = landmarks.landmark[19].y; msg.l_index.z = landmarks.landmark[19].z
    msg.r_index.x = landmarks.landmark[20].x; msg.r_index.y = landmarks.landmark[20].y; msg.r_index.z = landmarks.landmark[20].z
    msg.l_thumb.x = landmarks.landmark[21].x; msg.l_thumb.y = landmarks.landmark[21].y; msg.l_thumb.z = landmarks.landmark[21].z
    msg.r_thumb.x = landmarks.landmark[22].x; msg.r_thumb.y = landmarks.landmark[22].y; msg.r_thumb.z = landmarks.landmark[22].z
    msg.l_hip.x = landmarks.landmark[23].x; msg.l_hip.y = landmarks.landmark[23].y; msg.l_hip.z = landmarks.landmark[23].z
    msg.r_hip.x = landmarks.landmark[24].x; msg.r_hip.y = landmarks.landmark[24].y; msg.r_hip.z = landmarks.landmark[24].z
    msg.l_knee.x = landmarks.landmark[25].x; msg.l_knee.y = landmarks.landmark[25].y; msg.l_knee.z = landmarks.landmark[25].z
    msg.r_knee.x = landmarks.landmark[26].x; msg.r_knee.y = landmarks.landmark[26].y; msg.r_knee.z = landmarks.landmark[26].z
    msg.l_ankle.x = landmarks.landmark[27].x; msg.l_ankle.y = landmarks.landmark[27].y; msg.l_ankle.z = landmarks.landmark[27].z
    msg.r_ankle.x = landmarks.landmark[28].x; msg.r_ankle.y = landmarks.landmark[28].y; msg.r_ankle.z = landmarks.landmark[28].z
    msg.l_heel.x = landmarks.landmark[29].x; msg.l_heel.y = landmarks.landmark[29].y; msg.l_heel.z = landmarks.landmark[29].z
    msg.r_heel.x = landmarks.landmark[30].x; msg.r_heel.y = landmarks.landmark[30].y; msg.r_heel.z = landmarks.landmark[30].z
    msg.l_foot_index.x = landmarks.landmark[31].x; msg.l_foot_index.y = landmarks.landmark[31].y; msg.l_foot_index.z = landmarks.landmark[31].z
    msg.r_foot_index.x = landmarks.landmark[32].x; msg.r_foot_index.y = landmarks.landmark[32].y; msg.r_foot_index.z = landmarks.landmark[32].z
    return msg

