"""Read-only camera/ROI visualization of YOLO and the published driving path."""

import os
import time

import cv2
import message_filters
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from interfaces_pkg.msg import DetectionArray, LaneInfo, MotionCommand, PathPlanningResult

from .lane_tracking_render import points_array, render_views


class LaneTrackingVisualizer(Node):
    def __init__(self):
        super().__init__('lane_tracking_visualizer_node')
        self.show_image = self.declare_parameter('show_image', True).value
        if self.show_image and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
            self.get_logger().warning('No display available; publishing overlay images only')
            self.show_image = False
        self.bridge = CvBridge()
        self.frame_pair = None
        self.path, self.targets = [], []
        self.path_received_at = None
        self.motion = None
        self.camera_pub = self.create_publisher(Image, 'lane_tracking_camera', 1)
        self.roi_pub = self.create_publisher(Image, 'lane_tracking_roi', 1)
        camera_qos = QoSProfile(depth=3, reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST)
        self.image_sub = message_filters.Subscriber(self, Image, 'image_raw', qos_profile=camera_qos)
        self.detection_sub = message_filters.Subscriber(self, DetectionArray, 'detections', qos_profile=1)
        self.synchronizer = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.detection_sub], queue_size=10, slop=.1)
        self.synchronizer.registerCallback(self.paired_callback)
        self.create_subscription(PathPlanningResult, 'path_planning_result', self.path_callback, 1)
        self.create_subscription(LaneInfo, 'yolov8_lane_info', self.lane_callback, 1)
        self.create_subscription(MotionCommand, 'topic_control_signal', self.motion_callback, 1)
        self.create_timer(.1, self.render_callback)
        if self.show_image:
            for title, height in [('YOLO lanes + target', 548), ('ROI lanes + target', 248)]:
                cv2.namedWindow(title, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(title, 640, height)
        self.get_logger().info('Overlay topics: lane_tracking_camera, lane_tracking_roi')

    def paired_callback(self, image, detections):
        self.frame_pair = (image, detections, time.monotonic())

    def path_callback(self, msg):
        self.path = (points_array(list(zip(msg.x_points, msg.y_points))).tolist()
                     if len(msg.x_points) == len(msg.y_points) else [])
        self.path_received_at = time.monotonic()

    def lane_callback(self, msg):
        self.targets = points_array([(p.target_x, p.target_y) for p in msg.target_points]).tolist()

    def motion_callback(self, msg):
        self.motion = (msg.steering, msg.left_speed, msg.right_speed)

    def render_callback(self):
        if self.frame_pair is None:
            return
        image_msg, detection_msg, received_at = self.frame_pair
        try:
            frame = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        except CvBridgeError as exc:
            self.get_logger().error(f'Cannot convert camera encoding {image_msg.encoding}: {exc}')
            return
        records = []
        for d in detection_msg.detections:
            records.append({'name': d.class_name, 'score': d.score,
                            'height': d.mask.height, 'width': d.mask.width,
                            'polygon': [(p.x, p.y) for p in d.mask.data],
                            'bbox': (d.bbox.center.position.x-d.bbox.size.x/2,
                                     d.bbox.center.position.y-d.bbox.size.y/2,
                                     d.bbox.center.position.x+d.bbox.size.x/2,
                                     d.bbox.center.position.y+d.bbox.size.y/2)})
        age = time.monotonic() - received_at
        status = f'YOLO age {age:.1f}s'
        if self.path_received_at is not None:
            status += f' path age {time.monotonic()-self.path_received_at:.1f}s'
        if self.motion is not None:
            steer, left, right = self.motion
            status += f' steer {steer} PWM {left}/{right}'
        camera, roi = render_views(frame, records, self.path, self.targets, status)
        for publisher, output in ((self.camera_pub, camera), (self.roi_pub, roi)):
            msg = self.bridge.cv2_to_imgmsg(output, encoding='bgr8')
            msg.header = image_msg.header
            publisher.publish(msg)
        if self.show_image:
            cv2.imshow('YOLO lanes + target', camera)
            cv2.imshow('ROI lanes + target', roi)
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = LaneTrackingVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.show_image:
            cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.try_shutdown()
