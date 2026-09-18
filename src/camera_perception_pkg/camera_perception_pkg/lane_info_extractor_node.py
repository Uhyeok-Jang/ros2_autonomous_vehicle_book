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

SUB_TOPIC_NAME = "detections"
PUB_TOPIC_NAME = "yolov8_lane_info"
ROI_IMAGE_TOPIC_NAME = "roi_image"
SHOW_IMAGE = True
LANE_CLASS_NAME = "lane2"
STOP_ZONE_CLASS_NAME = "stop_zone"
BOUNDARY_CLASS_NAMES = ("dashed_line", "solid_line")

# 곡선의 중앙선을 3개 점으로만 근사하면 path가 chord처럼 코너를 잘라간다.
# ROI(약 180 px)에서 20 px 간격으로 중심점을 뽑아 실제 곡률을 더 잘 보존한다.
TARGET_Y_VALUES = (20, 40, 60, 80, 100, 120, 140)
TARGET_BAND_THICKNESS = 12
MIN_VALID_WIDTH = 40
MIN_CORRIDOR_WIDTH = 70
BOUNDARY_MARGIN_PX = 24
SMOOTHING_ALPHA = 0.35
MAX_TARGET_STEP_PX = 20.0
MAX_LANE_DROPOUT_FRAMES = 8


class Yolov8InfoExtractor(Node):
    def __init__(self):
        super().__init__('lane_info_extractor_node')
        self.sub_topic = self.declare_parameter('sub_detection_topic', SUB_TOPIC_NAME).value
        self.pub_topic = self.declare_parameter('pub_topic', PUB_TOPIC_NAME).value
        self.show_image = self.declare_parameter('show_image', SHOW_IMAGE).value
        self.lane_class_name = self.declare_parameter('lane_class_name', LANE_CLASS_NAME).value
        self.stop_zone_class_name = self.declare_parameter('stop_zone_class_name', STOP_ZONE_CLASS_NAME).value
        self.smoothing_alpha = float(self.declare_parameter('smoothing_alpha', SMOOTHING_ALPHA).value)
        self.max_target_step_px = float(self.declare_parameter('max_target_step_px', MAX_TARGET_STEP_PX).value)
        self.boundary_margin_px = float(self.declare_parameter('boundary_margin_px', BOUNDARY_MARGIN_PX).value)
        self.max_lane_dropout_frames = int(self.declare_parameter('max_lane_dropout_frames', MAX_LANE_DROPOUT_FRAMES).value)

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
        self.publisher = self.create_publisher(LaneInfo, self.pub_topic, self.qos_profile)
        self.roi_image_publisher = self.create_publisher(Image, ROI_IMAGE_TOPIC_NAME, self.qos_profile)

    @staticmethod
    def get_canvas_shape(detection_msg):
        for detection in detection_msg.detections:
            if detection.mask.height > 0 and detection.mask.width > 0:
                return int(detection.mask.height), int(detection.mask.width)
        return None

    def build_class_mask(self, detection_msg, class_names, shape):
        if isinstance(class_names, str):
            class_names = (class_names,)

        mask = np.zeros(shape, dtype=np.uint8)
        for detection in detection_msg.detections:
            if detection.class_name not in class_names:
                continue
            if detection.mask.height <= 0 or detection.mask.width <= 0 or len(detection.mask.data) < 3:
                continue

            polygon = np.array(
                [[int(round(point.x)), int(round(point.y))] for point in detection.mask.data],
                dtype=np.int32
            )
            cv2.fillPoly(mask, [polygon], 255)

        return mask

    @staticmethod
    def split_runs(columns, max_gap=3):
        if columns.size == 0:
            return []
        split_indices = np.where(np.diff(columns) > max_gap)[0] + 1
        return [run for run in np.split(columns, split_indices) if run.size > 0]

    def choose_drivable_run(self, roi_image, target_y, target_idx):
        h, w = roi_image.shape[:2]
        half = TARGET_BAND_THICKNESS // 2
        upper = max(0, target_y - half)
        lower = min(h, target_y + half + 1)

        band = roi_image[upper:lower, :]
        occupancy = np.count_nonzero(band, axis=0)
        valid_columns = np.where(occupancy >= 2)[0]
        runs = [run for run in self.split_runs(valid_columns) if len(run) >= MIN_VALID_WIDTH]

        if not runs:
            return None

        reference_x = self.prev_target_x[target_idx]
        if reference_x is None:
            reference_x = w / 2.0

        def run_distance(run):
            left, right = float(run[0]), float(run[-1])
            center = (left + right) / 2.0
            if left <= reference_x <= right:
                return 0.0
            return abs(center - reference_x)

        chosen = min(runs, key=run_distance)
        return float((chosen[0] + chosen[-1]) / 2.0)

    def get_boundary_centers(self, boundary_roi, target_y):
        h, _ = boundary_roi.shape[:2]
        half = TARGET_BAND_THICKNESS // 2
        upper = max(0, target_y - half)
        lower = min(h, target_y + half + 1)

        band = boundary_roi[upper:lower, :]
        occupancy = np.count_nonzero(band, axis=0)
        boundary_columns = np.where(occupancy >= 1)[0]
        runs = self.split_runs(boundary_columns, max_gap=4)

        return [
            float((run[0] + run[-1]) / 2.0)
            for run in runs
            if len(run) >= 2
        ]

    def estimate_target_x(self, drivable_roi, boundary_roi, target_y, target_idx):
        _, w = drivable_roi.shape[:2]
        base_x = self.choose_drivable_run(drivable_roi, target_y, target_idx)
        previous_x = self.prev_target_x[target_idx]
        reference_x = base_x if base_x is not None else (
            previous_x if previous_x is not None else w / 2.0
        )

        boundary_centers = self.get_boundary_centers(boundary_roi, target_y)
        left_boundaries = [x for x in boundary_centers if x < reference_x]
        right_boundaries = [x for x in boundary_centers if x > reference_x]
        left_boundary = max(left_boundaries) if left_boundaries else None
        right_boundary = min(right_boundaries) if right_boundaries else None

        measured_x = base_x
        mode = "drivable"

        if (
            left_boundary is not None
            and right_boundary is not None
            and (right_boundary - left_boundary) >= MIN_CORRIDOR_WIDTH
        ):
            safe_left = left_boundary + self.boundary_margin_px
            safe_right = right_boundary - self.boundary_margin_px
            measured_x = (
                (safe_left + safe_right) / 2.0
                if safe_left < safe_right
                else (left_boundary + right_boundary) / 2.0
            )
            mode = "boundary_pair"
        elif measured_x is not None and left_boundary is not None:
            measured_x = max(measured_x, left_boundary + self.boundary_margin_px)
            mode = "left_boundary"
        elif measured_x is not None and right_boundary is not None:
            measured_x = min(measured_x, right_boundary - self.boundary_margin_px)
            mode = "right_boundary"
        elif measured_x is None and previous_x is not None:
            measured_x = previous_x
            mode = "hold"
        elif measured_x is None:
            measured_x = w / 2.0
            mode = "fallback_center"

        if previous_x is None:
            target_x = measured_x
        else:
            delta = float(np.clip(
                measured_x - previous_x,
                -self.max_target_step_px,
                self.max_target_step_px
            ))
            limited_x = previous_x + delta
            alpha = float(np.clip(self.smoothing_alpha, 0.0, 1.0))
            target_x = alpha * limited_x + (1.0 - alpha) * previous_x

        target_x = float(np.clip(target_x, 0, w - 1))
        self.prev_target_x[target_idx] = target_x
        return target_x, left_boundary, right_boundary, mode

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
        self.lane_dropout_count += 1

        if self.show_image:
            cv2.waitKey(1)

        have_previous_target = all(x is not None for x in self.prev_target_x)

        if have_previous_target and self.lane_dropout_count <= self.max_lane_dropout_frames:
            self.publish_lane(self.prev_target_x, self.last_valid_grad)
            self.get_logger().warning(
                "No usable drivable/boundary geometry: holding last path "
                f"({self.lane_dropout_count}/{self.max_lane_dropout_frames})"
            )
            return

        self.get_logger().warning(
            "No usable drivable/boundary geometry: "
            f"dropout {self.lane_dropout_count} frames, stop publishing lane path"
        )

    def yolov8_detections_callback(self, detection_msg: DetectionArray):
        if len(detection_msg.detections) == 0:
            self.handle_lane_dropout()
            return

        shape = self.get_canvas_shape(detection_msg)
        if shape is None:
            self.handle_lane_dropout()
            return

        lane2_mask = self.build_class_mask(detection_msg, self.lane_class_name, shape)
        stop_zone_mask = self.build_class_mask(detection_msg, self.stop_zone_class_name, shape)
        boundary_mask = self.build_class_mask(detection_msg, BOUNDARY_CLASS_NAMES, shape)

        # stop_zone은 장애물이 아니라 주행 가능 영역 위의 semantic overlay다.
        drivable_mask = cv2.bitwise_or(lane2_mask, stop_zone_mask)

        h, w = shape
        dst_mat = [
            [round(w * 0.3), round(h * 0.0)],
            [round(w * 0.7), round(h * 0.0)],
            [round(w * 0.7), h],
            [round(w * 0.3), h]
        ]
        src_mat = [[238, 316], [402, 313], [501, 476], [155, 476]]

        drivable_bird = CPFL.bird_convert(drivable_mask, srcmat=src_mat, dstmat=dst_mat)
        boundary_bird = CPFL.bird_convert(boundary_mask, srcmat=src_mat, dstmat=dst_mat)

        drivable_roi = CPFL.roi_rectangle_below(drivable_bird, cutting_idx=300)
        boundary_roi = CPFL.roi_rectangle_below(boundary_bird, cutting_idx=300)

        drivable_roi = cv2.convertScaleAbs(drivable_roi)
        boundary_roi = cv2.convertScaleAbs(boundary_roi)
        _, drivable_roi = cv2.threshold(drivable_roi, 80, 255, cv2.THRESH_BINARY)
        _, boundary_roi = cv2.threshold(boundary_roi, 80, 255, cv2.THRESH_BINARY)

        has_geometry = (
            cv2.countNonZero(drivable_roi) > 0
            or cv2.countNonZero(boundary_roi) > 0
        )
        if not has_geometry:
            self.handle_lane_dropout()
            return

        target_x_values = []
        debug_boundary_pairs = []
        modes = []

        for idx, target_y in enumerate(TARGET_Y_VALUES):
            target_x, left_boundary, right_boundary, mode = self.estimate_target_x(
                drivable_roi,
                boundary_roi,
                target_y,
                idx
            )
            target_x_values.append(target_x)
            debug_boundary_pairs.append((left_boundary, right_boundary))
            modes.append(mode)

        fresh_geometry = any(
            mode not in ("hold", "fallback_center")
            for mode in modes
        )
        if not fresh_geometry:
            self.handle_lane_dropout()
            return

        if self.lane_dropout_count > 0:
            self.get_logger().info(
                f"lane geometry recovered after {self.lane_dropout_count} dropout frames"
            )
        self.lane_dropout_count = 0

        dx = target_x_values[-1] - target_x_values[0]
        dy = TARGET_Y_VALUES[-1] - TARGET_Y_VALUES[0]
        grad = float(np.degrees(np.arctan(dx / dy))) if dy != 0 else 0.0
        self.last_valid_grad = grad

        target_points = self.publish_lane(target_x_values, grad)

        # Debug:
        # - 회색/흰색: lane2 | stop_zone
        # - 노랑: dashed/solid boundary
        # - 빨강 점: target
        # - 초록 세로선: image center
        debug_roi = cv2.cvtColor(drivable_roi, cv2.COLOR_GRAY2BGR)
        debug_roi[boundary_roi > 0] = (0, 255, 255)

        for point, pair in zip(target_points, debug_boundary_pairs):
            left_boundary, right_boundary = pair
            y = int(point.target_y)

            if left_boundary is not None:
                cv2.circle(
                    debug_roi,
                    (int(round(left_boundary)), y),
                    4,
                    (255, 0, 0),
                    -1
                )

            if right_boundary is not None:
                cv2.circle(
                    debug_roi,
                    (int(round(right_boundary)), y),
                    4,
                    (255, 0, 255),
                    -1
                )

            cv2.circle(
                debug_roi,
                (int(point.target_x), y),
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
            cv2.imshow('drivable_mask_image', drivable_mask)
            cv2.imshow('drivable_bird_img', drivable_bird)
            cv2.imshow('roi_img', debug_roi)
            cv2.waitKey(1)

        try:
            roi_image_msg = self.cv_bridge.cv2_to_imgmsg(drivable_roi, encoding="mono8")
            self.roi_image_publisher.publish(roi_image_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to convert and publish ROI image: {e}")

        self.get_logger().info(
            "target_x="
            + ", ".join(f"{x:.1f}" for x in target_x_values)
            + f", modes={modes}, lane_slope={grad:.2f}"
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
