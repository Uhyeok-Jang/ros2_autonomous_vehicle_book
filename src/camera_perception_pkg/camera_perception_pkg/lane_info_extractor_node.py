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
DASHED_LINE_CLASS_NAME = "dashed_line"
SOLID_LINE_CLASS_NAME = "solid_line"
BOUNDARY_CLASS_NAMES = ("dashed_line", "solid_line")

# 곡선의 중앙선을 3개 점으로만 근사하면 path가 chord처럼 코너를 잘라간다.
# ROI(약 180 px)에서 20 px 간격으로 중심점을 뽑아 실제 곡률을 더 잘 보존한다.
TARGET_Y_VALUES = (20, 40, 60, 80, 100, 120, 140)
TARGET_BAND_THICKNESS = 12
MIN_VALID_WIDTH = 40
MIN_CORRIDOR_WIDTH = 70
BOUNDARY_MARGIN_PX = 24
OUTSIDE_BIAS_RATIO = 0.05
OUTSIDE_BIAS_MIN_SLOPE_DEG = 8.0
OUTER_BOUNDARY_MARGIN_PX = 45.0
SMOOTHING_ALPHA = 0.35
MAX_TARGET_STEP_PX = 20.0
MAX_LANE_DROPOUT_FRAMES = 8
STOP_ZONE_SMOOTHING_ALPHA = 0.18
STOP_ZONE_MAX_TARGET_STEP_PX = 6.0
STOP_ZONE_MIN_AREA_PX = 50
STOP_ZONE_GRACE_FRAMES = 45
# With the lane-following stop-zone logic, longitudinal lane boundaries are
# safety constraints and must remain visible. Wide transverse runs are
# rejected separately by BOUNDARY_MAX_RUN_WIDTH_PX.
STOP_ZONE_BOUNDARY_EXCLUSION_PX = 0
BOUNDARY_MAX_RUN_WIDTH_PX = 45
STOP_ZONE_DASHED_MARGIN_PX = 4
STOP_ZONE_DASHED_SEARCH_PX = 8
STOP_ZONE_DASHED_MIN_PIXELS = 20
STOP_ZONE_DASHED_MAX_SLOPE = 1.0
STOP_ZONE_BRIDGE_HISTORY_WEIGHT = 0.35
STOP_ZONE_BRIDGE_MAX_SLOPE = 0.35
STOP_ZONE_CORRIDOR_MAX_WIDTH_PX = 420.0
STOP_ZONE_CORRIDOR_ASSOCIATION_MARGIN_PX = 35.0
STOP_ZONE_CORRIDOR_MAX_SHIFT_PX = 60.0
STOP_ZONE_CORRIDOR_SMOOTHING_ALPHA = 0.45
STOP_ZONE_CORRIDOR_MAX_TARGET_STEP_PX = 12.0
LANE_WIDTH_SMOOTHING_ALPHA = 0.20
MAX_ADJACENT_TARGET_DELTA_PX = 20.0
MAX_TARGET_DELTA_CHANGE_PX = 8.0
LANE_MIN_SCORE = 0.30
STOP_ZONE_MIN_SCORE = 0.25
BOUNDARY_MIN_SCORE = 0.30


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
        self.outside_bias_ratio = float(self.declare_parameter('outside_bias_ratio', OUTSIDE_BIAS_RATIO).value)
        self.outside_bias_min_slope_deg = float(self.declare_parameter('outside_bias_min_slope_deg', OUTSIDE_BIAS_MIN_SLOPE_DEG).value)
        self.outer_boundary_margin_px = float(self.declare_parameter('outer_boundary_margin_px', OUTER_BOUNDARY_MARGIN_PX).value)
        self.max_lane_dropout_frames = int(self.declare_parameter('max_lane_dropout_frames', MAX_LANE_DROPOUT_FRAMES).value)
        self.stop_zone_smoothing_alpha = float(self.declare_parameter('stop_zone_smoothing_alpha', STOP_ZONE_SMOOTHING_ALPHA).value)
        self.stop_zone_max_target_step_px = float(self.declare_parameter('stop_zone_max_target_step_px', STOP_ZONE_MAX_TARGET_STEP_PX).value)
        self.stop_zone_grace_frames = int(self.declare_parameter('stop_zone_grace_frames', STOP_ZONE_GRACE_FRAMES).value)
        self.stop_zone_boundary_exclusion_px = int(
            self.declare_parameter(
                'stop_zone_boundary_exclusion_px',
                STOP_ZONE_BOUNDARY_EXCLUSION_PX,
            ).value
        )
        self.stop_zone_dashed_margin_px = int(
            self.declare_parameter(
                'stop_zone_dashed_margin_px',
                STOP_ZONE_DASHED_MARGIN_PX,
            ).value
        )
        self.stop_zone_bridge_history_weight = float(
            self.declare_parameter(
                'stop_zone_bridge_history_weight',
                STOP_ZONE_BRIDGE_HISTORY_WEIGHT,
            ).value
        )
        self.stop_zone_bridge_max_slope = float(
            self.declare_parameter(
                'stop_zone_bridge_max_slope',
                STOP_ZONE_BRIDGE_MAX_SLOPE,
            ).value
        )
        self.max_adjacent_target_delta_px = float(self.declare_parameter('max_adjacent_target_delta_px', MAX_ADJACENT_TARGET_DELTA_PX).value)
        self.max_target_delta_change_px = float(self.declare_parameter('max_target_delta_change_px', MAX_TARGET_DELTA_CHANGE_PX).value)

        self.cv_bridge = CvBridge()
        self.prev_target_x = [None] * len(TARGET_Y_VALUES)
        self.last_valid_grad = 0.0
        self.lane_dropout_count = 0
        self.stop_zone_visible = False
        self.stop_zone_raw_visible = False
        self.stop_zone_frames_remaining = 0
        self.learned_lane_width_px = None
        self.last_debug_roi = None
        self.debug_frame_count = 0

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

    def build_class_mask(self, detection_msg, class_names, shape, min_score=0.0):
        if isinstance(class_names, str):
            class_names = (class_names,)

        mask = np.zeros(shape, dtype=np.uint8)
        for detection in detection_msg.detections:
            if detection.class_name not in class_names:
                continue
            if float(detection.score) < min_score:
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
            if 2 <= len(run) <= BOUNDARY_MAX_RUN_WIDTH_PX
        ]

    @staticmethod
    def choose_boundary_corridor(
        dashed_centers,
        solid_centers,
        reference_x,
        learned_lane_width_px,
        min_width_px=MIN_CORRIDOR_WIDTH,
        max_width_px=STOP_ZONE_CORRIDOR_MAX_WIDTH_PX,
        association_margin_px=STOP_ZONE_CORRIDOR_ASSOCIATION_MARGIN_PX,
        max_shift_px=STOP_ZONE_CORRIDOR_MAX_SHIFT_PX,
    ):
        """Choose a lane center from longitudinal boundaries, not stop-zone area.

        A dashed/solid pair that contains the previous path has the highest
        confidence. If only one usable boundary remains, the previously
        learned bird-view lane width reconstructs the center. Candidates that
        require a large lateral jump are rejected so a single bad frame cannot
        swap lane sides.
        """
        reference_x = float(reference_x)
        valid_pairs = []
        for dashed_x in dashed_centers:
            for solid_x in solid_centers:
                left_x = min(float(dashed_x), float(solid_x))
                right_x = max(float(dashed_x), float(solid_x))
                width = right_x - left_x
                if not (min_width_px <= width <= max_width_px):
                    continue
                if not (
                    left_x - association_margin_px
                    <= reference_x
                    <= right_x + association_margin_px
                ):
                    continue

                center = (left_x + right_x) / 2.0
                width_error = (
                    0.0
                    if learned_lane_width_px is None
                    else abs(width - float(learned_lane_width_px))
                )
                score = abs(center - reference_x) + 0.25 * width_error
                valid_pairs.append((score, center, width))

        if valid_pairs:
            _, center, width = min(valid_pairs, key=lambda item: item[0])
            if abs(center - reference_x) <= max_shift_px:
                return float(center), float(width), "boundary_pair"

        if learned_lane_width_px is None:
            return None, None, "none"

        lane_width = float(learned_lane_width_px)
        single_candidates = []
        for boundary_x in list(dashed_centers) + list(solid_centers):
            boundary_x = float(boundary_x)
            if boundary_x < reference_x:
                center = boundary_x + lane_width / 2.0
            else:
                center = boundary_x - lane_width / 2.0
            shift = abs(center - reference_x)
            if shift <= max_shift_px:
                single_candidates.append((shift, center))

        if not single_candidates:
            return None, None, "none"

        _, center = min(single_candidates, key=lambda item: item[0])
        return float(center), lane_width, "single_boundary"

    def update_learned_lane_width(
        self,
        dashed_roi,
        solid_roi,
        reference_centers,
    ):
        """Learn a stable bird-view lane width only outside stop zones."""
        if self.stop_zone_visible:
            return

        measured_widths = []
        for target_y, reference_x in zip(TARGET_Y_VALUES, reference_centers):
            if reference_x is None:
                continue
            _, width, mode = self.choose_boundary_corridor(
                self.get_boundary_centers(dashed_roi, target_y),
                self.get_boundary_centers(solid_roi, target_y),
                reference_x,
                self.learned_lane_width_px,
            )
            if mode == "boundary_pair" and width is not None:
                measured_widths.append(width)

        if not measured_widths:
            return

        measured_width = float(np.median(measured_widths))
        if self.learned_lane_width_px is None:
            self.learned_lane_width_px = measured_width
            return

        alpha = LANE_WIDTH_SMOOTHING_ALPHA
        self.learned_lane_width_px = (
            alpha * measured_width
            + (1.0 - alpha) * self.learned_lane_width_px
        )

    def build_stop_zone_corridor_targets(
        self,
        dashed_roi,
        solid_roi,
        reference_centers,
    ):
        targets = []
        modes = []
        for target_y, reference_x in zip(TARGET_Y_VALUES, reference_centers):
            if reference_x is None:
                targets.append(None)
                modes.append("none")
                continue
            center, _, mode = self.choose_boundary_corridor(
                self.get_boundary_centers(dashed_roi, target_y),
                self.get_boundary_centers(solid_roi, target_y),
                reference_x,
                self.learned_lane_width_px,
            )
            targets.append(center)
            modes.append(mode)
        return targets, modes

    @staticmethod
    def build_lane2_from_stop_zone(
        stop_zone_roi,
        dashed_roi,
        reference_centers,
        target_y_values,
        dashed_margin_px,
    ):
        """Recover only the lane2 side of a stop zone split by dashed_line."""
        recovered = np.zeros_like(stop_zone_roi)
        if (
            cv2.countNonZero(stop_zone_roi) == 0
            or cv2.countNonZero(dashed_roi) == 0
        ):
            return recovered

        search_kernel_size = 2 * STOP_ZONE_DASHED_SEARCH_PX + 1
        stop_zone_search = cv2.dilate(
            stop_zone_roi,
            np.ones(
                (search_kernel_size, search_kernel_size),
                dtype=np.uint8,
            ),
        )
        dashed_inside_zone = cv2.bitwise_and(dashed_roi, stop_zone_search)
        dashed_y, dashed_x = np.where(dashed_inside_zone > 0)
        if dashed_x.size < STOP_ZONE_DASHED_MIN_PIXELS:
            return recovered

        sampled_y = np.unique(dashed_y)
        if sampled_y.size < 3:
            return recovered
        sampled_x = np.asarray([
            np.median(dashed_x[dashed_y == y]) for y in sampled_y
        ])

        slope = float(np.polyfit(sampled_y, sampled_x, 1)[0])
        slope = float(np.clip(
            slope,
            -STOP_ZONE_DASHED_MAX_SLOPE,
            STOP_ZONE_DASHED_MAX_SLOPE,
        ))
        intercept = float(np.median(sampled_x - slope * sampled_y))

        valid_reference = [
            (float(y), float(x))
            for y, x in zip(target_y_values, reference_centers)
            if x is not None
        ]
        if not valid_reference:
            return recovered

        reference_y = np.asarray(
            [item[0] for item in valid_reference],
            dtype=np.float64,
        )
        reference_x = np.asarray(
            [item[1] for item in valid_reference],
            dtype=np.float64,
        )
        interpolated_reference = np.interp(
            sampled_y.astype(np.float64),
            reference_y,
            reference_x,
        )
        side_offset = float(np.median(
            interpolated_reference - (slope * sampled_y + intercept)
        ))
        margin = max(0, int(dashed_margin_px))
        if abs(side_offset) <= margin:
            return recovered

        lane2_is_right = side_offset > 0.0
        stop_rows = np.where(np.any(stop_zone_roi > 0, axis=1))[0]
        for y in stop_rows:
            stop_columns = np.where(stop_zone_roi[y, :] > 0)[0]
            boundary_x = slope * float(y) + intercept
            if lane2_is_right:
                selected = stop_columns[
                    stop_columns >= int(np.ceil(boundary_x + margin))
                ]
            else:
                selected = stop_columns[
                    stop_columns <= int(np.floor(boundary_x - margin))
                ]
            recovered[y, selected] = 255

        return cv2.morphologyEx(
            recovered,
            cv2.MORPH_CLOSE,
            np.ones((3, 3), dtype=np.uint8),
        )

    @staticmethod
    def build_stop_zone_bridge(
        measured_centers,
        previous_targets,
        target_y_values,
        history_weight,
        max_slope,
    ):
        """Fill lane2 rows hidden by a stop-zone mask without using its shape.

        The visible lane2 centers determine the current travel direction. Only
        missing rows are extrapolated, then blended with the previous complete
        path so the stop-zone class cannot create a lateral avoidance command.
        """
        valid = [
            (float(y), float(x))
            for y, x in zip(target_y_values, measured_centers)
            if x is not None
        ]
        previous_is_complete = (
            len(previous_targets) == len(target_y_values)
            and all(value is not None for value in previous_targets)
        )
        if not valid:
            return (
                [float(value) for value in previous_targets]
                if previous_is_complete
                else [None] * len(target_y_values)
            )

        valid_y = np.asarray([item[0] for item in valid], dtype=np.float64)
        valid_x = np.asarray([item[1] for item in valid], dtype=np.float64)

        if len(valid) >= 2:
            slope = float(np.polyfit(valid_y, valid_x, 1)[0])
            slope = float(np.clip(slope, -abs(max_slope), abs(max_slope)))
            intercept = float(np.median(valid_x - slope * valid_y))
            predicted = [slope * float(y) + intercept for y in target_y_values]
        elif previous_is_complete:
            anchor_y, anchor_x = valid[0]
            anchor_index = min(
                range(len(target_y_values)),
                key=lambda idx: abs(float(target_y_values[idx]) - anchor_y),
            )
            offset = anchor_x - float(previous_targets[anchor_index])
            predicted = [float(value) + offset for value in previous_targets]
        else:
            predicted = [valid[0][1]] * len(target_y_values)

        history_weight = float(np.clip(history_weight, 0.0, 1.0))
        bridge = []
        for index, (measured_x, predicted_x) in enumerate(
            zip(measured_centers, predicted)
        ):
            if measured_x is not None:
                bridge.append(None)
                continue

            if previous_is_complete:
                predicted_x = (
                    (1.0 - history_weight) * predicted_x
                    + history_weight * float(previous_targets[index])
                )
            bridge.append(float(predicted_x))

        return bridge

    def exclude_stop_zone_boundaries(self, boundary_roi, stop_zone_roi):
        """Remove transverse stop-zone pixels from longitudinal boundaries."""
        radius = max(0, self.stop_zone_boundary_exclusion_px)
        if radius == 0 or cv2.countNonZero(stop_zone_roi) == 0:
            return boundary_roi

        kernel_size = 2 * radius + 1
        stop_zone_guard = cv2.dilate(
            stop_zone_roi,
            np.ones((kernel_size, kernel_size), dtype=np.uint8),
        )
        return cv2.bitwise_and(
            boundary_roi,
            cv2.bitwise_not(stop_zone_guard),
        )

    def estimate_target_x(
        self,
        drivable_roi,
        boundary_roi,
        target_y,
        target_idx,
        base_x=None,
        bridge_x=None,
        corridor_x=None,
    ):
        _, w = drivable_roi.shape[:2]
        previous_x = self.prev_target_x[target_idx]
        # 현재 프레임의 stop-zone/lane 마스크가 흔들려도 경계선의 좌우가
        # 뒤집히지 않도록 이전 경로점을 우선 기준으로 사용한다.
        reference_x = previous_x if previous_x is not None else (
            base_x if base_x is not None else (
                bridge_x if bridge_x is not None else w / 2.0
            )
        )

        boundary_centers = self.get_boundary_centers(boundary_roi, target_y)
        left_boundaries = [x for x in boundary_centers if x < reference_x]
        right_boundaries = [x for x in boundary_centers if x > reference_x]
        left_boundary = max(left_boundaries) if left_boundaries else None
        right_boundary = min(right_boundaries) if right_boundaries else None

        if self.stop_zone_visible and corridor_x is not None:
            measured_x = corridor_x
            mode = "stop_zone_corridor"
        else:
            measured_x = base_x if base_x is not None else bridge_x
            mode = "drivable" if base_x is not None else "stop_zone_bridge"

        if self.stop_zone_visible and measured_x is not None:
            # lane2 또는 그 연장선을 주 경로로 사용한다. 경계는 안전 범위로
            # 제한할 뿐 중심을 다시 계산하지 않으므로 stop zone을 피하지 않는다.
            if (
                left_boundary is not None
                and right_boundary is not None
                and (right_boundary - left_boundary) >= MIN_CORRIDOR_WIDTH
            ):
                safe_left = left_boundary + self.boundary_margin_px
                safe_right = right_boundary - self.boundary_margin_px
                if safe_left <= safe_right:
                    measured_x = float(np.clip(measured_x, safe_left, safe_right))
            elif left_boundary is not None:
                measured_x = max(
                    measured_x,
                    left_boundary + self.boundary_margin_px,
                )
            elif right_boundary is not None:
                measured_x = min(
                    measured_x,
                    right_boundary - self.boundary_margin_px,
                )
            if corridor_x is not None:
                mode = "stop_zone_corridor"
            else:
                mode = (
                    "stop_zone_lane_follow"
                    if base_x is not None
                    else "stop_zone_bridge"
                )
        elif (
            left_boundary is not None
            and right_boundary is not None
            and (right_boundary - left_boundary) >= MIN_CORRIDOR_WIDTH
        ):
            lane_width = right_boundary - left_boundary
            lane_center = (left_boundary + right_boundary) / 2.0

            # 곡선에서는 직전 프레임의 곡률 방향 반대편(아웃코스)으로
            # 목표점을 이동한다. 양쪽 경계가 모두 보일 때만 적용하고,
            # 어느 경계에서도 outer_boundary_margin_px 이상 떨어지도록 제한한다.
            if (
                not self.stop_zone_visible
                and self.outside_bias_ratio > 0.0
                and abs(self.last_valid_grad) >= self.outside_bias_min_slope_deg
            ):
                outside_direction = 1.0 if self.last_valid_grad > 0.0 else -1.0
                desired_x = (
                    lane_center
                    + outside_direction * lane_width * self.outside_bias_ratio
                )
                safe_left = left_boundary + self.outer_boundary_margin_px
                safe_right = right_boundary - self.outer_boundary_margin_px
                measured_x = (
                    float(np.clip(desired_x, safe_left, safe_right))
                    if safe_left <= safe_right
                    else lane_center
                )
                mode = "boundary_pair_outside"
            else:
                safe_left = left_boundary + self.boundary_margin_px
                safe_right = right_boundary - self.boundary_margin_px
                measured_x = (
                    (safe_left + safe_right) / 2.0
                    if safe_left < safe_right
                    else lane_center
                )
                mode = "boundary_pair"
        elif left_boundary is not None:
            if measured_x is None:
                measured_x = (
                    previous_x if previous_x is not None else w / 2.0
                )
            measured_x = max(measured_x, left_boundary + self.boundary_margin_px)
            mode = "left_boundary"
        elif right_boundary is not None:
            if measured_x is None:
                measured_x = (
                    previous_x if previous_x is not None else w / 2.0
                )
            measured_x = min(measured_x, right_boundary - self.boundary_margin_px)
            mode = "right_boundary"
        elif self.stop_zone_visible and previous_x is not None:
            measured_x = previous_x
            mode = "stop_zone_hold"
        elif measured_x is None and previous_x is not None:
            measured_x = previous_x
            mode = "hold"
        elif measured_x is None:
            measured_x = w / 2.0
            mode = "fallback_center"

        if previous_x is None:
            target_x = measured_x
        else:
            max_step = self.max_target_step_px
            alpha = self.smoothing_alpha
            if self.stop_zone_visible:
                if corridor_x is not None:
                    max_step = min(
                        max_step,
                        STOP_ZONE_CORRIDOR_MAX_TARGET_STEP_PX,
                    )
                    alpha = max(
                        alpha,
                        STOP_ZONE_CORRIDOR_SMOOTHING_ALPHA,
                    )
                else:
                    max_step = min(max_step, self.stop_zone_max_target_step_px)
                    alpha = min(alpha, self.stop_zone_smoothing_alpha)

            delta = float(np.clip(
                measured_x - previous_x,
                -max_step,
                max_step
            ))
            limited_x = previous_x + delta
            alpha = float(np.clip(alpha, 0.0, 1.0))
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

    def publish_roi_image(self, image):
        try:
            encoding = "bgr8" if image.ndim == 3 else "mono8"
            roi_image_msg = self.cv_bridge.cv2_to_imgmsg(
                image,
                encoding=encoding
            )
            self.roi_image_publisher.publish(roi_image_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to convert and publish ROI image: {e}")

    def handle_lane_dropout(self):
        self.lane_dropout_count += 1

        if self.show_image:
            cv2.waitKey(1)

        have_previous_target = all(x is not None for x in self.prev_target_x)

        if have_previous_target and self.lane_dropout_count <= self.max_lane_dropout_frames:
            self.publish_lane(self.prev_target_x, self.last_valid_grad)
            if self.last_debug_roi is not None:
                self.publish_roi_image(self.last_debug_roi)
            if self.lane_dropout_count == 1:
                self.get_logger().warning(
                    "No usable drivable/boundary geometry: holding last path "
                    f"(up to {self.max_lane_dropout_frames} frames)"
                )
            return

        if (
            self.lane_dropout_count == self.max_lane_dropout_frames + 1
            or self.lane_dropout_count % 30 == 0
        ):
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

        lane2_mask = self.build_class_mask(
            detection_msg, self.lane_class_name, shape, LANE_MIN_SCORE
        )
        stop_zone_mask = self.build_class_mask(
            detection_msg, self.stop_zone_class_name, shape, STOP_ZONE_MIN_SCORE
        )
        dashed_mask = self.build_class_mask(
            detection_msg,
            DASHED_LINE_CLASS_NAME,
            shape,
            BOUNDARY_MIN_SCORE,
        )
        solid_mask = self.build_class_mask(
            detection_msg,
            SOLID_LINE_CLASS_NAME,
            shape,
            BOUNDARY_MIN_SCORE,
        )
        boundary_mask = self.build_class_mask(
            detection_msg, BOUNDARY_CLASS_NAMES, shape, BOUNDARY_MIN_SCORE
        )

        # stop_zone은 차선의 일부가 아니라 횡방향 semantic overlay다. 이를 lane2와
        # 합치면 검출 여부에 따라 도로 폭이 매 프레임 바뀌므로 경로 계산에서는
        # lane2만 사용하고, stop_zone은 별도 상태/시각화 용도로 유지한다.
        drivable_mask = lane2_mask
        self.stop_zone_raw_visible = (
            cv2.countNonZero(stop_zone_mask) >= STOP_ZONE_MIN_AREA_PX
        )
        # 정지선 위를 통과하는 동안 segmentation은 차체/원근 때문에 몇 프레임씩
        # 사라질 수 있다. 즉시 일반 차선 모드로 돌아가면 한쪽 실선을 차선 중앙으로
        # 오인하므로 마지막 검출 뒤에도 잠시 보호 모드를 유지한다.
        if self.stop_zone_raw_visible:
            self.stop_zone_frames_remaining = self.stop_zone_grace_frames
        elif self.stop_zone_frames_remaining > 0:
            self.stop_zone_frames_remaining -= 1
        self.stop_zone_visible = (
            self.stop_zone_raw_visible
            or self.stop_zone_frames_remaining > 0
        )

        h, w = shape
        dst_mat = [
            [round(w * 0.3), round(h * 0.0)],
            [round(w * 0.7), round(h * 0.0)],
            [round(w * 0.7), h],
            [round(w * 0.3), h]
        ]
        src_mat = [[238, 316], [402, 313], [501, 476], [155, 476]]

        drivable_bird = CPFL.bird_convert(drivable_mask, srcmat=src_mat, dstmat=dst_mat)
        stop_zone_bird = CPFL.bird_convert(stop_zone_mask, srcmat=src_mat, dstmat=dst_mat)
        dashed_bird = CPFL.bird_convert(dashed_mask, srcmat=src_mat, dstmat=dst_mat)
        solid_bird = CPFL.bird_convert(solid_mask, srcmat=src_mat, dstmat=dst_mat)
        boundary_bird = CPFL.bird_convert(boundary_mask, srcmat=src_mat, dstmat=dst_mat)

        drivable_roi = CPFL.roi_rectangle_below(drivable_bird, cutting_idx=300)
        stop_zone_roi = CPFL.roi_rectangle_below(stop_zone_bird, cutting_idx=300)
        dashed_roi = CPFL.roi_rectangle_below(dashed_bird, cutting_idx=300)
        solid_roi = CPFL.roi_rectangle_below(solid_bird, cutting_idx=300)
        boundary_roi = CPFL.roi_rectangle_below(boundary_bird, cutting_idx=300)

        drivable_roi = cv2.convertScaleAbs(drivable_roi)
        stop_zone_roi = cv2.convertScaleAbs(stop_zone_roi)
        dashed_roi = cv2.convertScaleAbs(dashed_roi)
        solid_roi = cv2.convertScaleAbs(solid_roi)
        boundary_roi = cv2.convertScaleAbs(boundary_roi)
        _, drivable_roi = cv2.threshold(drivable_roi, 80, 255, cv2.THRESH_BINARY)
        _, stop_zone_roi = cv2.threshold(stop_zone_roi, 80, 255, cv2.THRESH_BINARY)
        _, dashed_roi = cv2.threshold(dashed_roi, 80, 255, cv2.THRESH_BINARY)
        _, solid_roi = cv2.threshold(solid_roi, 80, 255, cv2.THRESH_BINARY)
        _, boundary_roi = cv2.threshold(boundary_roi, 80, 255, cv2.THRESH_BINARY)

        drivable_roi = cv2.morphologyEx(
            drivable_roi,
            cv2.MORPH_CLOSE,
            np.ones((5, 5), dtype=np.uint8)
        )
        boundary_roi = cv2.morphologyEx(
            boundary_roi,
            cv2.MORPH_CLOSE,
            np.ones((3, 3), dtype=np.uint8)
        )
        dashed_roi = cv2.morphologyEx(
            dashed_roi,
            cv2.MORPH_CLOSE,
            np.ones((5, 3), dtype=np.uint8)
        )
        solid_roi = cv2.morphologyEx(
            solid_roi,
            cv2.MORPH_CLOSE,
            np.ones((5, 3), dtype=np.uint8)
        )

        previous_target_x = list(self.prev_target_x)
        observed_drivable_roi = drivable_roi.copy()
        observed_lane_centers = [
            self.choose_drivable_run(drivable_roi, target_y, idx)
            for idx, target_y in enumerate(TARGET_Y_VALUES)
        ]
        reference_centers = [
            measured_x if measured_x is not None else previous_target_x[idx]
            for idx, measured_x in enumerate(observed_lane_centers)
        ]
        # Learn the corridor width during ordinary driving. In a stop zone the
        # stop-zone polygon is never converted into lane2; longitudinal lane
        # boundaries define the path instead.
        self.update_learned_lane_width(
            dashed_roi,
            solid_roi,
            reference_centers,
        )
        corridor_targets = [None] * len(TARGET_Y_VALUES)
        corridor_modes = ["none"] * len(TARGET_Y_VALUES)
        if self.stop_zone_visible:
            corridor_targets, corridor_modes = (
                self.build_stop_zone_corridor_targets(
                    dashed_roi,
                    solid_roi,
                    reference_centers,
                )
            )

        if self.stop_zone_visible:
            boundary_roi = self.exclude_stop_zone_boundaries(
                boundary_roi,
                stop_zone_roi,
            )

        has_geometry = (
            cv2.countNonZero(drivable_roi) > 0
            or cv2.countNonZero(boundary_roi) > 0
        )
        if not has_geometry:
            if self.stop_zone_visible and all(
                x is not None for x in self.prev_target_x
            ):
                self.publish_lane(self.prev_target_x, self.last_valid_grad)
                if self.last_debug_roi is not None:
                    self.publish_roi_image(self.last_debug_roi)
                return
            self.handle_lane_dropout()
            return

        measured_lane_centers = [
            self.choose_drivable_run(drivable_roi, target_y, idx)
            for idx, target_y in enumerate(TARGET_Y_VALUES)
        ]
        bridge_targets = [None] * len(TARGET_Y_VALUES)
        if self.stop_zone_visible:
            bridge_targets = self.build_stop_zone_bridge(
                measured_lane_centers,
                previous_target_x,
                TARGET_Y_VALUES,
                self.stop_zone_bridge_history_weight,
                self.stop_zone_bridge_max_slope,
            )

        target_x_values = []
        debug_boundary_pairs = []
        modes = []

        for idx, target_y in enumerate(TARGET_Y_VALUES):
            target_x, left_boundary, right_boundary, mode = self.estimate_target_x(
                drivable_roi,
                boundary_roi,
                target_y,
                idx,
                base_x=measured_lane_centers[idx],
                bridge_x=bridge_targets[idx],
                corridor_x=corridor_targets[idx],
            )
            if corridor_targets[idx] is not None:
                mode = f"{mode}_{corridor_modes[idx]}"
            target_x_values.append(target_x)
            debug_boundary_pairs.append((left_boundary, right_boundary))
            modes.append(mode)

        # 서로 인접한 ROI 행이 반대편 경계를 선택하면 경로가 한 프레임 안에서
        # 실선을 가로질러 꺾인다. stop-zone에서는 이전의 연속적인 경로 전체를
        # 유지하고, 일반 구간에서도 행간 변화량을 물리적으로 가능한 범위로 제한한다.
        adjacent_deltas = np.abs(np.diff(target_x_values))
        discontinuous_path = (
            adjacent_deltas.size > 0
            and float(np.max(adjacent_deltas)) > self.max_adjacent_target_delta_px
        )
        have_previous_path = all(x is not None for x in previous_target_x)
        previous_path_is_continuous = (
            have_previous_path
            and float(np.max(np.abs(np.diff(previous_target_x))))
            <= self.max_adjacent_target_delta_px
        )

        if (
            self.stop_zone_visible
            and discontinuous_path
            and previous_path_is_continuous
            and cv2.countNonZero(drivable_roi) == 0
        ):
            target_x_values = [float(x) for x in previous_target_x]
            modes = ["stop_zone_path_hold"] * len(target_x_values)
        else:
            for idx in range(1, len(target_x_values)):
                previous_row_x = target_x_values[idx - 1]
                limited_x = float(np.clip(
                    target_x_values[idx],
                    previous_row_x - self.max_adjacent_target_delta_px,
                    previous_row_x + self.max_adjacent_target_delta_px
                ))
                if abs(limited_x - target_x_values[idx]) > 0.01:
                    modes[idx] = f"{modes[idx]}_row_limited"
                    target_x_values[idx] = limited_x

            # 한쪽 경계 검출에서 양쪽 경계 검출로 바뀌는 행에 꺾임이 생기지
            # 않도록 인접 구간의 기울기 변화(이산 2차 미분)도 제한한다.
            # 정상 곡선은 그대로 두고 계단형 목표 경로만 완만하게 만든다.
            if len(target_x_values) >= 3:
                previous_delta = target_x_values[1] - target_x_values[0]
                for idx in range(2, len(target_x_values)):
                    current_delta = (
                        target_x_values[idx] - target_x_values[idx - 1]
                    )
                    limited_delta = float(np.clip(
                        current_delta,
                        previous_delta - self.max_target_delta_change_px,
                        previous_delta + self.max_target_delta_change_px
                    ))
                    if abs(limited_delta - current_delta) > 0.01:
                        target_x_values[idx] = (
                            target_x_values[idx - 1] + limited_delta
                        )
                        modes[idx] = f"{modes[idx]}_curve_limited"
                    previous_delta = limited_delta

            spatially_smoothed_targets = list(target_x_values)

            # 공간 평활화가 검출된 실선/점선 바깥으로 목표점을 밀어내지 않도록
            # 마지막에 각 행의 안전 경계 안으로 다시 투영한다.
            for idx, (left_boundary, right_boundary) in enumerate(
                debug_boundary_pairs
            ):
                safe_x = target_x_values[idx]
                if left_boundary is not None and right_boundary is not None:
                    safe_left = left_boundary + self.boundary_margin_px
                    safe_right = right_boundary - self.boundary_margin_px
                    if safe_left <= safe_right:
                        safe_x = float(np.clip(
                            safe_x, safe_left, safe_right
                        ))
                    else:
                        safe_x = (left_boundary + right_boundary) / 2.0
                elif left_boundary is not None:
                    safe_x = max(
                        safe_x,
                        left_boundary + self.boundary_margin_px
                    )
                elif right_boundary is not None:
                    safe_x = min(
                        safe_x,
                        right_boundary - self.boundary_margin_px
                    )

                if abs(safe_x - target_x_values[idx]) > 0.01:
                    target_x_values[idx] = safe_x
                    modes[idx] = f"{modes[idx]}_boundary_clamped"

            # 한 프레임의 잘못된 경계 polygon이 안전 경로를 계단 모양으로
            # 만들면 해당 경계 투영만 취소한다. lane2에서 얻은 연속 경로가
            # 이 경우 더 신뢰할 수 있고, 불연속 경로를 다음 프레임 상태로
            # 저장하지 않는 것이 중요하다.
            boundary_projection_discontinuous = (
                len(target_x_values) >= 2
                and float(np.max(np.abs(np.diff(target_x_values))))
                > self.max_adjacent_target_delta_px
            )
            if boundary_projection_discontinuous:
                target_x_values = spatially_smoothed_targets
                modes = [
                    f"{mode}_boundary_rejected" for mode in modes
                ]

        # estimate_target_x가 행별 값을 먼저 갱신하므로 경로 전체 검증 결과로
        # 상태를 다시 덮어써 다음 프레임의 기준도 안전한 경로가 되게 한다.
        self.prev_target_x = list(target_x_values)

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
        # - 회색/흰색: 원래 검출된 lane2
        # - 주황: stop_zone
        # - stop_zone은 lane2로 합치지 않음; dashed/solid corridor를 사용
        # - 노랑: dashed/solid boundary
        # - 빨강 점: 검출된 lane2 기반 target
        # - 청록 점: stop-zone 때문에 누락된 lane2를 연장한 target
        # - 초록 세로선: image center
        debug_roi = cv2.cvtColor(observed_drivable_roi, cv2.COLOR_GRAY2BGR)
        debug_roi[stop_zone_roi > 0] = (0, 165, 255)
        debug_roi[boundary_roi > 0] = (0, 255, 255)

        for point, pair, mode in zip(
            target_points,
            debug_boundary_pairs,
            modes,
        ):
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
                (255, 255, 0) if "bridge" in mode else (0, 0, 255),
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

        self.last_debug_roi = debug_roi.copy()
        self.publish_roi_image(debug_roi)

        self.debug_frame_count += 1
        if self.debug_frame_count % 30 == 0:
            self.get_logger().info(
                "target_x="
                + ", ".join(f"{x:.1f}" for x in target_x_values)
                + f", modes={modes}, lane_slope={grad:.2f}, "
                + f"stop_zone={self.stop_zone_visible}"
                + f"(raw={self.stop_zone_raw_visible}, "
                + f"grace={self.stop_zone_frames_remaining}), "
                + "learned_lane_width_px="
                + (
                    "None"
                    if self.learned_lane_width_px is None
                    else f"{self.learned_lane_width_px:.1f}"
                )
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
