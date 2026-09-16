import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from interfaces_pkg.msg import TargetPoint, LaneInfo, DetectionArray
from .lib import camera_perception_func_lib as CPFL

#---------------Variable Setting---------------
SUB_TOPIC_NAME = "detections"
PUB_TOPIC_NAME = "yolov8_lane_info"
ROI_IMAGE_TOPIC_NAME = "roi_image"
SHOW_IMAGE = True

LANE_CLASS_NAME = "lane2"
TARGET_Y_VALUES = (30, 70, 110)
TARGET_BAND_THICKNESS = 12
MIN_VALID_WIDTH = 60
SMOOTHING_ALPHA = 0.35
MAX_TARGET_STEP_PX = 20.0

# lane2가 잠깐 사라질 때 즉시 멈추지 않고 마지막 유효 경로를 잠시 유지한다.
# 20~25 Hz 기준 약 0.3~0.4초. 이보다 오래 사라지면 publish를 멈춰
# motion_planner의 path timeout이 차량을 정지시키도록 한다.
MAX_LANE_DROPOUT_FRAMES = 8
#----------------------------------------------


class Yolov8InfoExtractor(Node):
    def __init__(self):
        super().__init__('lane_info_extractor_node')

        self.sub_topic = self.declare_parameter('sub_detection_topic', SUB_TOPIC_NAME).value
        self.pub_topic = self.declare_parameter('pub_topic', PUB_TOPIC_NAME).value
        self.show_image = self.declare_parameter('show_image', SHOW_IMAGE).value
        self.lane_class_name = self.declare_parameter('lane_class_name', LANE_CLASS_NAME).value
        self.smoothing_alpha = float(
            self.declare_parameter('smoothing_alpha', SMOOTHING_ALPHA).value)
        self.max_target_step_px = float(
            self.declare_parameter('max_target_step_px', MAX_TARGET_STEP_PX).value)
        self.max_lane_dropout_frames = int(
            self.declare_parameter(
                'max_lane_dropout_frames', MAX_LANE_DROPOUT_FRAMES
            ).value
        )

        self.cv_bridge = CvBridge()
        self.prev_target_x = [None] * len(TARGET_Y_VALUES)
        self.last_valid_grad = 0.0
        self.lane_dropout_count = 0

        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )

        self.subscriber = self.create_subscription(
            DetectionArray,
            self.sub_topic,
            self.yolov8_detections_callback,
            self.qos_profile
        )
        self.publisher = self.create_publisher(
            LaneInfo, self.pub_topic, self.qos_profile)
        self.roi_image_publisher = self.create_publisher(
            Image, ROI_IMAGE_TOPIC_NAME, self.qos_profile)

    def build_lane_mask(self, detection_msg: DetectionArray):
        """가장 confidence가 높은 lane2 segmentation polygon을 채운 mask로 만든다."""
        lane_detections = [
            detection for detection in detection_msg.detections
            if detection.class_name == self.lane_class_name
            and detection.mask.height > 0
            and detection.mask.width > 0
            and len(detection.mask.data) >= 3
        ]

        if not lane_detections:
            return None

        best_detection = max(lane_detections, key=lambda detection: detection.score)
        mask_msg = best_detection.mask
        lane_mask = np.zeros((mask_msg.height, mask_msg.width), dtype=np.uint8)

        polygon = np.array(
            [[int(round(point.x)), int(round(point.y))] for point in mask_msg.data],
            dtype=np.int32
        )

        cv2.fillPoly(lane_mask, [polygon], 255)
        return lane_mask

    def estimate_target_x(self, roi_image, target_y, target_idx):
        """ROI의 해당 높이에서 가장 넓은 lane2 연속 구간의 중앙을 추정한다."""
        h, w = roi_image.shape[:2]
        half = TARGET_BAND_THICKNESS // 2
        upper = max(0, target_y - half)
        lower = min(h, target_y + half + 1)

        band = roi_image[upper:lower, :]
        occupancy = np.count_nonzero(band, axis=0)
        valid_columns = np.where(occupancy >= 2)[0]

        measured_x = None

        if valid_columns.size > 0:
            split_indices = np.where(np.diff(valid_columns) > 3)[0] + 1
            runs = np.split(valid_columns, split_indices)
            longest_run = max(runs, key=len)

            if len(longest_run) >= MIN_VALID_WIDTH:
                measured_x = float((longest_run[0] + longest_run[-1]) / 2.0)

        previous_x = self.prev_target_x[target_idx]

        if measured_x is None:
            target_x = previous_x if previous_x is not None else (w / 2.0)
        elif previous_x is None:
            target_x = measured_x
        else:
            # 한 프레임에서 target이 과도하게 튀는 것을 먼저 제한한다.
            delta = measured_x - previous_x
            delta = float(np.clip(delta, -self.max_target_step_px, self.max_target_step_px))
            limited_x = previous_x + delta

            # 이후 EMA smoothing으로 프레임 간 jitter를 줄인다.
            alpha = float(np.clip(self.smoothing_alpha, 0.0, 1.0))
            target_x = alpha * limited_x + (1.0 - alpha) * previous_x

        target_x = float(np.clip(target_x, 0, w - 1))
        self.prev_target_x[target_idx] = target_x
        return target_x

    def publish_lane(self, target_x_values, grad):
        lane = LaneInfo()
        lane.slope = float(grad)

        target_points = []
        for target_x, target_y in zip(target_x_values, TARGET_Y_VALUES):
            target_point = TargetPoint()
            target_point.target_x = int(round(target_x))
            target_point.target_y = int(target_y)
            target_points.append(target_point)

        lane.target_points = target_points
        self.publisher.publish(lane)
        return target_points

    def handle_lane_dropout(self):
        """짧은 lane2 dropout 동안 마지막 유효 target을 유지하고, 길어지면 fail-safe에 맡긴다."""
        self.lane_dropout_count += 1

        # OpenCV 창이 lane2 dropout 중에도 event loop를 처리하도록 한다.
        if self.show_image:
            cv2.waitKey(1)

        have_previous_target = all(x is not None for x in self.prev_target_x)

        if (
            have_previous_target
            and self.lane_dropout_count <= self.max_lane_dropout_frames
        ):
            self.publish_lane(self.prev_target_x, self.last_valid_grad)
            self.get_logger().warning(
                f"No valid '{self.lane_class_name}' mask: holding last path "
                f"({self.lane_dropout_count}/{self.max_lane_dropout_frames})"
            )
            return

        self.get_logger().warning(
            f"No valid '{self.lane_class_name}' mask: dropout "
            f"{self.lane_dropout_count} frames, stop publishing lane path"
        )

    def yolov8_detections_callback(self, detection_msg: DetectionArray):
        if len(detection_msg.detections) == 0:
            self.handle_lane_dropout()
            return

        lane2_mask_image = self.build_lane_mask(detection_msg)
        if lane2_mask_image is None or cv2.countNonZero(lane2_mask_image) == 0:
            self.handle_lane_dropout()
            return

        if self.lane_dropout_count > 0:
            self.get_logger().info(
                f"lane2 recovered after {self.lane_dropout_count} dropout frames"
            )
        self.lane_dropout_count = 0

        h, w = lane2_mask_image.shape[:2]
        dst_mat = [
            [round(w * 0.3), round(h * 0.0)],
            [round(w * 0.7), round(h * 0.0)],
            [round(w * 0.7), h],
            [round(w * 0.3), h]
        ]
        src_mat = [[238, 316], [402, 313], [501, 476], [155, 476]]

        lane2_bird_image = CPFL.bird_convert(
            lane2_mask_image, srcmat=src_mat, dstmat=dst_mat)
        roi_image = CPFL.roi_rectangle_below(
            lane2_bird_image, cutting_idx=300)
        roi_image = cv2.convertScaleAbs(roi_image)

        target_x_values = []

        for idx, target_point_y in enumerate(TARGET_Y_VALUES):
            target_point_x = self.estimate_target_x(
                roi_image, target_point_y, idx)
            target_x_values.append(target_point_x)

        # 메시지 slope는 디버깅용. 실제 steering은 path를 이용한다.
        dx = target_x_values[-1] - target_x_values[0]
        dy = TARGET_Y_VALUES[-1] - TARGET_Y_VALUES[0]
        grad = float(np.degrees(np.arctan(dx / dy))) if dy != 0 else 0.0
        self.last_valid_grad = grad

        target_points = self.publish_lane(target_x_values, grad)

        debug_roi = cv2.cvtColor(roi_image, cv2.COLOR_GRAY2BGR)
        for point in target_points:
            cv2.circle(
                debug_roi,
                (int(point.target_x), int(point.target_y)),
                5,
                (0, 0, 255),
                -1
            )
        cv2.line(
            debug_roi,
            (w // 2, 0),
            (w // 2, debug_roi.shape[0] - 1),
            (0, 255, 0),
            1
        )

        if self.show_image:
            cv2.imshow('lane2_mask_image', lane2_mask_image)
            cv2.imshow('lane2_bird_img', lane2_bird_image)
            cv2.imshow('roi_img', debug_roi)
            cv2.waitKey(1)

        try:
            roi_image_msg = self.cv_bridge.cv2_to_imgmsg(
                roi_image, encoding="mono8")
            self.roi_image_publisher.publish(roi_image_msg)
        except Exception as e:
            self.get_logger().error(
                f"Failed to convert and publish ROI image: {e}")

        self.get_logger().info(
            "target_x=" + ", ".join(f"{x:.1f}" for x in target_x_values) +
            f", lane_slope={grad:.2f}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = Yolov8InfoExtractor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n\nshutdown\n\n")
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
