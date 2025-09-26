import rclpy
from rclpy.node import Node
import csv
import time

from hpe_ros_msgs.msg import MpHumanPose3D


class FingertipTracker(Node):

  def __init__(self):
    super().__init__('fingertip_tracker_node')

    self.subscription = self.create_subscription(MpHumanPose3D,'/mp_ros/hpe3d', self.listener_cb, 10) 

    self.filename = 'desni_kaziprst.csv'

    with open(self.filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['r_index_x', 'r_index_y', 'r_index_z'])
    

  def listener_cb(self, msg):

    rx = msg.r_index.x
    ry = msg.r_index.y
    rz = msg.r_index.z
    
    with open(self.filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([rx, ry, rz])
  

def main(args=None):
    rclpy.init(args=args)
    node = FingertipTracker()

    start_time = time.time()
    time_recording = 5.0

    while( time.time() - start_time) < time_recording:
      rclpy.spin_once(node, timeout_sec=0.1)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()




