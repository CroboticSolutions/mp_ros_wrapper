#!/usr/bin/env python3

import csv
import os
import time
from dataclasses import dataclass
from typing import List

import mediapipe as mp
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import Point, PointStamped
from hpe_ros_msgs.msg import MpHumanPose3D
from sensor_msgs.msg import Image

from mp_wrapper_ros.mp_utils import ros_image_to_numpy
from std_msgs.msg import Bool


@dataclass
class Sample:
    idx: int
    t_sec: float
    x: float
    y: float
    z: float


def _make_pose_qos(mode: str) -> QoSProfile:
    mode = (mode or 'default').strip().lower()

    if mode == 'default':
        return QoSProfile(depth=10)

    if mode == 'best_effort':
        return QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

    if mode == 'sensor_data':
        # Common choice for high-rate streams.
        return QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

    raise ValueError("pose_qos must be one of: default | best_effort | sensor_data")


class KeypointRecorder(Node):
    def __init__(self) -> None:
        super().__init__('keypoint_recorder')

        self.declare_parameter('trigger_topic', 'record_trigger')
        self.declare_parameter('pose_topic', 'hpe3d')
        self.declare_parameter('pose_qos', 'sensor_data')
        self.declare_parameter('keypoint_field', 'r_index')

        # If record_space == 'pixel', x/y will be stored as image pixel coordinates.
        # This assumes the pose keypoints coming in on MpHumanPose3D are MediaPipe-normalized coords (0..1).
        self.declare_parameter('record_space', 'pixel')  # 'pixel' | 'raw'
        self.declare_parameter('image_topic', '/image_raw')
        self.declare_parameter('invert_y', True)  # image coordinates usually have y pointing down


        # Data source:
        # - pose: subscribe to MpHumanPose3D and use keypoint_field
        # - hand: run MediaPipe HandLandmarker on the incoming image stream and track a hand landmark
        self.declare_parameter('source', 'pose')  # 'pose' | 'hand' | 'hand_topic'
        self.declare_parameter('handedness', 'right')  # 'right' | 'left'
        self.declare_parameter('hand_landmark_index', 8)  # 8 = INDEX_FINGER_TIP
        self.declare_parameter('hand_model_path', '')  # empty -> use package default models/hand_landmarker.task
        self.declare_parameter('hand_topic', '')  # PointStamped topic when source == 'hand_topic'
        self.declare_parameter('hand3d_topic', '')  # Optional PointStamped 3D topic for comparison/3D plots

        self.declare_parameter('duration_sec', 3.0)
        self.declare_parameter(
            'output_dir',
            os.path.join(os.path.expanduser('~'), '.ros', 'keypoint_records'),
        )
        self.declare_parameter('plot', True)
        # For "track right index finger in 2D" the most useful view is XY.
        self.declare_parameter('plot_mode', 'xy')  # '3d' | 'xy' | 'xz' | 'yz' (or comma-separated, e.g. 'xy,3d')

        self._trigger_topic = self.get_parameter('trigger_topic').get_parameter_value().string_value
        self._pose_topic = self.get_parameter('pose_topic').get_parameter_value().string_value
        self._pose_qos_mode = self.get_parameter('pose_qos').get_parameter_value().string_value
        self._keypoint_field = self.get_parameter('keypoint_field').get_parameter_value().string_value

        self._record_space = self.get_parameter('record_space').get_parameter_value().string_value.strip().lower()
        self._image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self._invert_y = self.get_parameter('invert_y').get_parameter_value().bool_value


        self._source = self.get_parameter('source').get_parameter_value().string_value.strip().lower()
        self._handedness = self.get_parameter('handedness').get_parameter_value().string_value.strip().lower()
        self._hand_landmark_index = self.get_parameter('hand_landmark_index').get_parameter_value().integer_value
        self._hand_model_path = self.get_parameter('hand_model_path').get_parameter_value().string_value.strip()
        self._hand_topic = self.get_parameter('hand_topic').get_parameter_value().string_value.strip()
        self._hand3d_topic = self.get_parameter('hand3d_topic').get_parameter_value().string_value.strip()

        if self._source == 'hand_topic' and not self._hand_topic:
            raise RuntimeError("Parameter 'hand_topic' must be set when source='hand_topic'")
        self._duration_sec = self.get_parameter('duration_sec').get_parameter_value().double_value
        self._output_dir = self.get_parameter('output_dir').get_parameter_value().string_value
        self._plot = self.get_parameter('plot').get_parameter_value().bool_value
        self._plot_mode = self.get_parameter('plot_mode').get_parameter_value().string_value

        self._recording: bool = False
        self._img_width = None
        self._img_height = None
        self._hand_landmarker = None
        self._samples: List[Sample] = []
        self._samples_hand3d: List[Sample] = []
        self._stop_timer = None

        self._pose_msg_count_total = 0

        if self._source in ('hand', 'hand_topic'):
            if not self._hand_model_path:
                share_dir = get_package_share_directory('mp_wrapper_ros')
                self._hand_model_path = os.path.join(share_dir, 'models', 'hand_landmarker.task')

            HandLandmarker = mp.tasks.vision.HandLandmarker
            HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
            VisionRunningMode = mp.tasks.vision.RunningMode
            BaseOptions = mp.tasks.BaseOptions

            options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=self._hand_model_path),
                running_mode=VisionRunningMode.IMAGE,
                num_hands=2,
            )
            self._hand_landmarker = HandLandmarker.create_from_options(options)

        self._pose_msg_count_during_record = 0
        self._last_pose_wall_time = None

        pose_qos = _make_pose_qos(self._pose_qos_mode)
        self._pose_sub = self.create_subscription(
            MpHumanPose3D,
            self._pose_topic,
            self._on_pose,
            pose_qos,
        )

        if self._source == 'hand_topic':
            self._hand_sub = self.create_subscription(
                PointStamped,
                self._hand_topic,
                self._on_hand_point,
                10,
            )
        else:
            self._hand_sub = None

        if self._source == 'hand_topic' and self._hand3d_topic:
            self._hand3d_sub = self.create_subscription(
                PointStamped,
                self._hand3d_topic,
                self._on_hand3d_point,
                10,
            )
        else:
            self._hand3d_sub = None

        # Image subscription is used for:
        # - hand source: run HandLandmarker on every frame while recording
        # - pose source with record_space==pixel: convert normalized keypoints to pixel coordinates
        if self._source in ('hand', 'hand_topic') or self._record_space == 'pixel':
            self._img_sub = self.create_subscription(
                Image,
                self._image_topic,
                self._on_image,
                2,
            )
        else:
            self._img_sub = None
        self._trigger_sub = self.create_subscription(
            Bool,
            self._trigger_topic,
            self._on_trigger,
            10,
        )

        self.get_logger().info(
            f"Ready. Waiting for Bool on '{self._trigger_topic}'. "
            f"Recording '{self._keypoint_field}' from '{self._pose_topic}' for {self._duration_sec:.2f}s. "
            f"record_space='{self._record_space}', source='{self._source}', image_topic='{self._image_topic}', invert_y={self._invert_y}. "
            f"pose_qos='{self._pose_qos_mode}'."
        )

    def _on_trigger(self, msg: Bool) -> None:
        if not msg.data:
            return
        if self._recording:
            self.get_logger().warn('Trigger received while already recording; ignoring.')
            return

        pub_count = 0
        try:
            pub_count = self._pose_sub.get_publisher_count()
        except Exception:
            pass

        self._recording = True
        self._samples = []
        self._samples_hand3d = []
        self._pose_msg_count_during_record = 0

        if self._stop_timer is not None:
            try:
                self._stop_timer.cancel()
            except Exception:
                pass
            self._stop_timer = None

        self.get_logger().info(
            f"Trigger=True received. Recording started for {self._duration_sec:.2f}s... "
            f"(pose publishers connected: {pub_count})"
        )
        if pub_count == 0:
            self.get_logger().warn(
                "No publishers connected on pose_topic right now. "
                "Run: ros2 topic list | grep hpe3d  (common topics: /hpe3d, /vitpose/hpe3d, /loc/hpe3d)."
            )

        self._stop_timer = self.create_timer(self._duration_sec, self._stop_recording_once)

    def _stop_recording_once(self) -> None:
        if not self._recording:
            return

        self._recording = False

        if self._stop_timer is not None:
            try:
                self._stop_timer.cancel()
            except Exception:
                pass
            self._stop_timer = None

        self.get_logger().info(
            f"Recording finished. Collected {len(self._samples)} samples "
            f"(pose msgs during record: {self._pose_msg_count_during_record}, total seen: {self._pose_msg_count_total})."
        )

        if self._source == 'hand_topic' and self._hand3d_topic:
            # Save 2D (pixel) and 3D (hand3d) streams separately so units don't get mixed.
            csv_path_2d = self._save_csv(self._samples, name_suffix='xy')
            if self._plot and len(self._samples) >= 2:
                try:
                    for plot_path in self._plot_trajectory(self._samples, csv_path_2d, plot_mode='xy'):
                        self.get_logger().info(f"Trajectory plot saved to: {plot_path}")
                except Exception as e:
                    self.get_logger().error(f"Failed to plot 2D trajectory: {e}")
            elif self._plot:
                self.get_logger().warn('Not plotting 2D because fewer than 2 samples were recorded.')

            if len(self._samples_hand3d) > 0:
                csv_path_3d = self._save_csv(self._samples_hand3d, name_suffix='3d')
                if self._plot and len(self._samples_hand3d) >= 2:
                    try:
                        for plot_path in self._plot_trajectory(self._samples_hand3d, csv_path_3d, plot_mode='3d'):
                            self.get_logger().info(f"Trajectory plot saved to: {plot_path}")
                    except Exception as e:
                        self.get_logger().error(f"Failed to plot 3D trajectory: {e}")
                elif self._plot:
                    self.get_logger().warn('Not plotting 3D because fewer than 2 samples were recorded.')
            else:
                self.get_logger().warn(
                    "No samples received from hand3d_topic during recording. "
                    "(Is the topic name correct, and is it publishing geometry_msgs/PointStamped?)"
                )
        else:
            csv_path = self._save_csv(self._samples)

            # Only plot when we have at least 2 points to draw a path.
            if self._plot and len(self._samples) >= 2:
                try:
                    for plot_path in self._plot_trajectory(self._samples, csv_path):
                        self.get_logger().info(f"Trajectory plot saved to: {plot_path}")
                except Exception as e:
                    self.get_logger().error(f"Failed to plot trajectory: {e}")
            elif self._plot:
                self.get_logger().warn('Not plotting because fewer than 2 samples were recorded.')



    def _on_image(self, msg: Image) -> None:
        # Keep last known image size for pixel conversion.
        self._img_width = int(msg.width)
        self._img_height = int(msg.height)

        if self._source != 'hand':
            return
        if not self._recording:
            return
        if self._hand_landmarker is None:
            return

        # Convert ROS Image -> numpy RGB -> MediaPipe Image
        try:
            rgb = ros_image_to_numpy(msg)
        except Exception as e:
            self.get_logger().error(f"Failed to decode image: {e}")
            return

        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._hand_landmarker.detect(mp_img)

        if not result.hand_landmarks:
            return

        want = 'right' if self._handedness != 'left' else 'left'
        chosen = None

        # Pick the requested hand by handedness.
        for i, h in enumerate(result.handedness):
            if not h:
                continue
            name = (h[0].category_name or '').strip().lower()
            if name == want:
                chosen = i
                break

        if chosen is None:
            # Fallback: just take the first detected hand
            chosen = 0

        hand = result.hand_landmarks[chosen]
        li = int(self._hand_landmark_index)
        if li < 0 or li >= len(hand):
            self.get_logger().error(f"hand_landmark_index out of range: {li}")
            return

        lm = hand[li]

        t_sec = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        idx = len(self._samples) + 1

        x = float(lm.x) * float(self._img_width)
        y = float(lm.y) * float(self._img_height)
        if self._invert_y:
            y = float(self._img_height) - y

        # Store 2D pixels (z=0)
        self._samples.append(Sample(idx=idx, t_sec=t_sec, x=x, y=y, z=0.0))

    def _on_hand_point(self, msg: PointStamped) -> None:
        if self._source != 'hand_topic':
            return
        if not self._recording:
            return

        t_sec = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        idx = len(self._samples) + 1
        x = float(msg.point.x)
        y = float(msg.point.y)
        if self._invert_y and self._img_height is not None:
            y = float(self._img_height) - y

        self._samples.append(Sample(idx=idx, t_sec=t_sec, x=x, y=y, z=0.0))
    def _on_hand3d_point(self, msg: PointStamped) -> None:
        # Optional 3D stream used for comparison when source == 'hand_topic'.
        if self._source != 'hand_topic' or not self._hand3d_topic:
            return
        if not self._recording:
            return

        t_sec = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        idx = len(self._samples_hand3d) + 1
        x = float(msg.point.x)
        y = float(msg.point.y)
        z = float(msg.point.z)

        self._samples_hand3d.append(Sample(idx=idx, t_sec=t_sec, x=x, y=y, z=z))

    def _on_pose(self, msg: MpHumanPose3D) -> None:
        self._pose_msg_count_total += 1
        self._last_pose_wall_time = self.get_clock().now()

        if self._source == 'hand':
            return

        if not self._recording:
            return

        self._pose_msg_count_during_record += 1

        try:
            pt = getattr(msg, self._keypoint_field)
        except AttributeError:
            self.get_logger().error(
                f"MpHumanPose3D has no field '{self._keypoint_field}'. "
                "Check hpe_ros_msgs/msg/MpHumanPose3D.msg for available names."
            )
            self._recording = False
            return

        if not isinstance(pt, Point):
            self.get_logger().error(
                f"Field '{self._keypoint_field}' is not geometry_msgs/Point (got {type(pt)})."
            )
            self._recording = False
            return

        t_sec = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        idx = len(self._samples) + 1

        x = float(pt.x)
        y = float(pt.y)
        z = float(pt.z)

        # MediaPipe pose landmarks are usually normalized image coords in [0,1].
        if self._record_space == 'pixel' and self._source != 'hand':
            if self._img_width is None or self._img_height is None:
                # Haven't seen an image yet; cannot convert to pixels.
                return

            # If values already look like pixels, keep them.
            if abs(x) <= 2.0 and abs(y) <= 2.0:
                x = x * float(self._img_width)
                y = y * float(self._img_height)

            if self._invert_y:
                y = float(self._img_height) - y

            z = 0.0

        self._samples.append(Sample(idx=idx, t_sec=t_sec, x=x, y=y, z=z))

    def _save_csv(self, samples: List[Sample], name_suffix: str = '') -> str:
        os.makedirs(self._output_dir, exist_ok=True)

        ts = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        suffix = f"_{name_suffix}" if name_suffix else ''
        base = f"{ts}_{self._keypoint_field}{suffix}_{self._duration_sec:.2f}s"
        csv_path = os.path.join(self._output_dir, base + '.csv')

        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['idx', 't_sec', 'x', 'y', 'z'])
            for s in samples:
                w.writerow([s.idx, f"{s.t_sec:.9f}", f"{s.x:.6f}", f"{s.y:.6f}", f"{s.z:.6f}"])

        self.get_logger().info(f"Saved recording to: {csv_path}")
        return csv_path

    def _plot_trajectory(self, samples: List[Sample], csv_path: str, plot_mode: str | None = None) -> List[str]:
        import matplotlib
        import re

        if os.environ.get('DISPLAY', '') == '':
            matplotlib.use('Agg')

        import matplotlib.pyplot as plt

        xs = [s.x for s in samples]
        ys = [s.y for s in samples]
        zs = [s.z for s in samples]

        raw = (plot_mode if plot_mode is not None else self._plot_mode).lower().strip()
        # Allow running multiple plot modes in one run, e.g. "xy,3d" or "xy+3d".
        parts = [p for p in re.split(r"[,+\s]+", raw) if p]

        modes: list[str] = []
        for p in parts:
            if p in ('both', '2d+3d', '3d+2d'):
                modes.extend(['xy', '3d'])
            else:
                modes.append(p)

        if not modes:
            modes = ['xy']

        # De-duplicate while preserving order.
        seen = set()
        modes = [m for m in modes if not (m in seen or seen.add(m))]

        allowed = {'3d', 'xy', 'xz', 'yz'}
        bad = [m for m in modes if m not in allowed]
        if bad:
            raise ValueError(
                f"Unsupported plot_mode value(s) {bad} from '{self._plot_mode}'. Use: 3d|xy|xz|yz (optionally comma-separated, e.g. 'xy,3d')"
            )

        out_paths: List[str] = []

        for mode in modes:
            fig = plt.figure(figsize=(7, 6))

            if mode == '3d':
                ax = fig.add_subplot(111, projection='3d')
                ax.plot(xs, ys, zs, linewidth=2)
                ax.scatter([xs[0]], [ys[0]], [zs[0]], label='start')
                ax.scatter([xs[-1]], [ys[-1]], [zs[-1]], label='end')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel('z')
                ax.legend(loc='best')
            else:
                ax = fig.add_subplot(111)

                # Pick the 2D projection to draw.
                if mode == 'xy':
                    x2, y2 = xs, ys
                    ax.set_xlabel('u [px]' if self._record_space == 'pixel' else 'x')
                    ax.set_ylabel('v [px]' if self._record_space == 'pixel' else 'y')
                elif mode == 'xz':
                    x2, y2 = xs, zs
                    ax.set_xlabel('u [px]' if self._record_space == 'pixel' else 'x')
                    ax.set_ylabel('z')
                else:  # 'yz'
                    x2, y2 = ys, zs
                    ax.set_xlabel('v [px]' if self._record_space == 'pixel' else 'y')
                    ax.set_ylabel('z')

                ax.plot(x2, y2, linewidth=2)
                ax.scatter(x2, y2, s=16)
                try:
                    ax.set_aspect('equal', adjustable='datalim')
                except Exception:
                    pass

                # Label every point with its ordinal index (1..N).
                for s in samples:
                    if mode == 'xy':
                        px, py = s.x, s.y
                    elif mode == 'xz':
                        px, py = s.x, s.z
                    else:  # 'yz'
                        px, py = s.y, s.z

                    ax.annotate(
                        str(s.idx),
                        (px, py),
                        textcoords='offset points',
                        xytext=(5, 5),
                        fontsize=8,
                    )

            fig.suptitle(f"Trajectory ({mode}): {self._keypoint_field} ({len(samples)} samples)")
            fig.tight_layout()

            out_path = os.path.splitext(csv_path)[0] + f"_{mode}.png"
            fig.savefig(out_path, dpi=160)
            plt.close(fig)
            out_paths.append(out_path)

        return out_paths

def main(args=None) -> None:
    rclpy.init(args=args)
    node = KeypointRecorder()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
