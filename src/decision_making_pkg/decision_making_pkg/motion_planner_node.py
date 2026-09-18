import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy

from std_msgs.msg import String, Bool
from interfaces_pkg.msg import PathPlanningResult, DetectionArray, MotionCommand

SUB_DETECTION_TOPIC_NAME = "detections"
SUB_PATH_TOPIC_NAME = "path_planning_result"
SUB_TRAFFIC_LIGHT_TOPIC_NAME = "yolov8_traffic_light_info"
SUB_LIDAR_OBSTACLE_TOPIC_NAME = "lidar_obstacle_info"
PUB_TOPIC_NAME = "topic_control_signal"

TIMER = 0.1
MAX_STEERING = 7.0
CAR_CENTER_X = 320.0

# ------------------------------------------------------------
# Curve state machine
#
# 핵심 수정:
#   "path heading이 기울어져 있다" != "도로가 곡선이다"
#
# 이전 버전은 far/near heading의 절대값이 크면 곡선으로 판단했기 때문에,
# 차량이 직선 차선에 비스듬히 놓인 경우에도 TURN_IN에 들어갈 수 있었다.
#
# 이제 곡선 여부는 far tangent와 near tangent의 차이,
# 즉 curvature proxy = far_heading - near_heading 으로 판단한다.
# ------------------------------------------------------------
CURVATURE_ENTER_DEG = 5.0
CURVATURE_EXIT_DEG = 2.0
CURVATURE_OPPOSITE_EXIT_DEG = 3.0
CURVE_CONFIRM_FRAMES = 3
CURVE_EXIT_CONFIRM_FRAMES = 3
HOLD_MIN_SEC = 0.40
TURN_IN_MAX_SEC = 1.50

# 100-point path 기준 near/far sampling.
NEAR_TRACK_FROM_END = 12
NEAR_TANGENT_SPAN = 10
FAR_TRACK_FROM_END = 45
FAR_TANGENT_SPAN = 12

# 내부 float steering ramp.
# TURN_IN은 의도적으로 천천히, EXIT은 그보다 조금 빠르게 푼다.
TURN_IN_RAMP_PER_TICK = 0.20
HOLD_ADJUST_PER_TICK = 0.15
EXIT_RAMP_PER_TICK = 0.30
STRAIGHT_RAMP_PER_TICK = 0.15

# 직선에서는 현재 heading/lateral error만 작은 범위에서 보정.
KP_STRAIGHT_HEADING = 0.08
KP_STRAIGHT_LATERAL = 0.015
STRAIGHT_STEERING_LIMIT = 2.0

# HOLD에서는 기본 조향을 유지하고 lateral error만 미세 보정.
KP_HOLD_LATERAL = 0.012
HOLD_CORRECTION_LIMIT = 0.50

# Speed
DRIVING_SPEED = 60
SHALLOW_CURVE_SPEED = 48
MEDIUM_CURVE_SPEED = 40
SHARP_CURVE_SPEED = 32
EXIT_SPEED = 40
APPROACH_SPEED = 35

PATH_TIMEOUT_SEC = 0.5
LIGHT_CONFIRM_FRAMES = 3
LIGHT_TIMEOUT_SEC = 0.7
DETECTION_TIMEOUT_SEC = 0.5
STOP_ZONE_MEMORY_SEC = 0.7
STOP_ZONE_STOP_Y = 360.0
GO_CLEAR_SEC = 1.5


class MotionPlanningNode(Node):
    def __init__(self):
        super().__init__('motion_planner_node')

        self.sub_detection_topic = self.declare_parameter(
            'sub_detection_topic', SUB_DETECTION_TOPIC_NAME).value
        self.sub_path_topic = self.declare_parameter(
            'sub_lane_topic', SUB_PATH_TOPIC_NAME).value
        self.sub_traffic_light_topic = self.declare_parameter(
            'sub_traffic_light_topic', SUB_TRAFFIC_LIGHT_TOPIC_NAME).value
        self.sub_lidar_obstacle_topic = self.declare_parameter(
            'sub_lidar_obstacle_topic', SUB_LIDAR_OBSTACLE_TOPIC_NAME).value
        self.pub_topic = self.declare_parameter(
            'pub_topic', PUB_TOPIC_NAME).value
        self.timer_period = self.declare_parameter('timer', TIMER).value

        self.driving_speed = int(
            self.declare_parameter('driving_speed', DRIVING_SPEED).value)
        self.approach_speed = int(
            self.declare_parameter('approach_speed', APPROACH_SPEED).value)
        self.path_timeout_sec = float(
            self.declare_parameter('path_timeout_sec', PATH_TIMEOUT_SEC).value)
        self.light_confirm_frames = int(
            self.declare_parameter(
                'light_confirm_frames', LIGHT_CONFIRM_FRAMES
            ).value
        )
        self.stop_zone_stop_y = float(
            self.declare_parameter('stop_zone_stop_y', STOP_ZONE_STOP_Y).value)
        self.go_clear_sec = float(
            self.declare_parameter('go_clear_sec', GO_CLEAR_SEC).value)

        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )

        self.detection_data = None
        self.detection_update_time_ns = None
        self.path_data = None
        self.path_update_time_ns = None
        self.traffic_light_data = None
        self.light_update_time_ns = None
        self.red_streak = 0
        self.green_streak = 0
        self.last_stop_zone_ymax = None
        self.last_stop_zone_time_ns = None
        self.lidar_data = None

        self.steering_command = 0
        self.left_speed_command = 0
        self.right_speed_command = 0
        self.filtered_steering = 0.0

        # Curve state
        self.curve_state = "STRAIGHT"
        self.curve_direction = 0
        self.curve_entry_count = 0
        self.curve_exit_count = 0
        self.curve_state_start_ns = self.now_ns()
        self.turn_target_steering = 0.0

        # Debug
        self.last_near_heading = 0.0
        self.last_far_heading = 0.0
        self.last_curvature = 0.0
        self.last_lateral_error = 0.0
        self.last_curve_speed = self.driving_speed
        self.last_turn_target = 0.0

        # Traffic state
        self.drive_state = "CRUISE"
        self.go_start_time_ns = None

        self.detection_sub = self.create_subscription(
            DetectionArray,
            self.sub_detection_topic,
            self.detection_callback,
            self.qos_profile
        )
        self.path_sub = self.create_subscription(
            PathPlanningResult,
            self.sub_path_topic,
            self.path_callback,
            self.qos_profile
        )
        self.traffic_light_sub = self.create_subscription(
            String,
            self.sub_traffic_light_topic,
            self.traffic_light_callback,
            self.qos_profile
        )
        self.lidar_sub = self.create_subscription(
            Bool,
            self.sub_lidar_obstacle_topic,
            self.lidar_callback,
            self.qos_profile
        )

        self.publisher = self.create_publisher(
            MotionCommand, self.pub_topic, self.qos_profile)
        self.timer = self.create_timer(
            self.timer_period, self.timer_callback)

    def now_ns(self):
        return self.get_clock().now().nanoseconds

    def elapsed_sec(self, start_ns):
        return (self.now_ns() - start_ns) / 1e9

    def detection_callback(self, msg: DetectionArray):
        self.detection_data = msg
        self.detection_update_time_ns = self.now_ns()

        zone_ymax = self.extract_stop_zone_ymax(msg)
        if zone_ymax is not None:
            self.last_stop_zone_ymax = zone_ymax
            self.last_stop_zone_time_ns = self.detection_update_time_ns

    def path_callback(self, msg: PathPlanningResult):
        self.path_data = list(zip(msg.x_points, msg.y_points))
        self.path_update_time_ns = self.now_ns()

    def traffic_light_callback(self, msg: String):
        self.traffic_light_data = msg
        self.light_update_time_ns = self.now_ns()

        color = msg.data
        if color == 'Red':
            self.red_streak += 1
            self.green_streak = 0
        elif color == 'Green':
            self.green_streak += 1
            self.red_streak = 0
        else:
            self.red_streak = 0
            self.green_streak = 0

    def lidar_callback(self, msg: Bool):
        self.lidar_data = msg

    def is_recent(self, timestamp_ns, timeout_sec):
        if timestamp_ns is None:
            return False

        age_sec = (self.now_ns() - timestamp_ns) / 1e9
        return age_sec <= timeout_sec

    def has_fresh_path(self):
        return (
            self.path_data is not None
            and self.is_recent(self.path_update_time_ns, self.path_timeout_sec)
        )

    def is_red_confirmed(self):
        return (
            self.is_recent(self.light_update_time_ns, LIGHT_TIMEOUT_SEC)
            and self.red_streak >= self.light_confirm_frames
        )

    def is_green_confirmed(self):
        return (
            self.is_recent(self.light_update_time_ns, LIGHT_TIMEOUT_SEC)
            and self.green_streak >= self.light_confirm_frames
        )

    @staticmethod
    def extract_stop_zone_ymax(detection_msg: DetectionArray):
        y_values = []

        for detection in detection_msg.detections:
            if detection.class_name != 'stop_zone':
                continue

            y_max = (
                detection.bbox.center.position.y
                + detection.bbox.size.y / 2.0
            )
            y_values.append(float(y_max))

        return max(y_values) if y_values else None

    def get_stop_zone_ymax(self):
        if (
            self.detection_data is not None
            and self.is_recent(
                self.detection_update_time_ns,
                DETECTION_TIMEOUT_SEC
            )
        ):
            current = self.extract_stop_zone_ymax(self.detection_data)
            if current is not None:
                return current

        if (
            self.last_stop_zone_ymax is not None
            and self.is_recent(
                self.last_stop_zone_time_ns,
                STOP_ZONE_MEMORY_SEC
            )
        ):
            return self.last_stop_zone_ymax

        return None

    @staticmethod
    def tangent_heading(point_a, point_b):
        denominator = point_a[1] - point_b[1]
        if abs(denominator) < 1e-6:
            return 0.0

        return math.degrees(
            math.atan(
                (point_b[0] - point_a[0]) / denominator
            )
        )

    def calculate_path_geometry(self):
        if not self.has_fresh_path() or len(self.path_data) < 12:
            return None

        n = len(self.path_data)

        near_idx = max(2, n - NEAR_TRACK_FROM_END)
        near_start = max(0, near_idx - NEAR_TANGENT_SPAN)

        far_idx = max(2, n - FAR_TRACK_FROM_END)
        far_start = max(0, far_idx - FAR_TANGENT_SPAN)

        near_point = self.path_data[near_idx]
        near_start_point = self.path_data[near_start]
        far_point = self.path_data[far_idx]
        far_start_point = self.path_data[far_start]

        near_heading = self.tangent_heading(
            near_start_point,
            near_point
        )
        far_heading = self.tangent_heading(
            far_start_point,
            far_point
        )

        # 직선이 카메라에 비스듬히 보이면 near/far heading 모두 클 수 있지만
        # 둘의 차이는 작다. 곡선에서는 앞/뒤 tangent가 달라진다.
        curvature = far_heading - near_heading

        lateral_error = float(near_point[0] - CAR_CENTER_X)

        return (
            near_heading,
            far_heading,
            curvature,
            lateral_error
        )

    @staticmethod
    def sign(value):
        if value > 0.0:
            return 1
        if value < 0.0:
            return -1
        return 0

    @staticmethod
    def ramp_toward(current, target, step):
        if current < target:
            return min(target, current + step)
        if current > target:
            return max(target, current - step)
        return current

    @staticmethod
    def steering_target_magnitude(curvature_deg):
        severity = abs(curvature_deg)

        if severity < 9.0:
            return 2.0
        if severity < 16.0:
            return 3.0
        if severity < 25.0:
            return 4.0
        return 5.0

    @staticmethod
    def curve_speed_from_target(target_magnitude):
        magnitude = abs(target_magnitude)

        if magnitude >= 4.0:
            return SHARP_CURVE_SPEED
        if magnitude >= 3.0:
            return MEDIUM_CURVE_SPEED
        return SHALLOW_CURVE_SPEED

    def set_curve_state(self, new_state):
        if new_state == self.curve_state:
            return

        self.get_logger().info(
            f"CURVE_STATE {self.curve_state} -> {new_state}"
        )

        self.curve_state = new_state
        self.curve_state_start_ns = self.now_ns()
        self.curve_entry_count = 0
        self.curve_exit_count = 0

        if new_state == "STRAIGHT":
            self.curve_direction = 0
            self.turn_target_steering = 0.0

    def reset_curve_controller(self):
        self.curve_state = "STRAIGHT"
        self.curve_direction = 0
        self.curve_entry_count = 0
        self.curve_exit_count = 0
        self.curve_state_start_ns = self.now_ns()
        self.turn_target_steering = 0.0
        self.filtered_steering = 0.0

    def update_curve_state(
        self,
        near_heading,
        far_heading,
        curvature
    ):
        curve_direction = self.sign(curvature)
        curve_strength = abs(curvature)

        if self.curve_state == "STRAIGHT":
            if (
                curve_direction != 0
                and curve_strength >= CURVATURE_ENTER_DEG
            ):
                self.curve_entry_count += 1
            else:
                self.curve_entry_count = 0

            if self.curve_entry_count >= CURVE_CONFIRM_FRAMES:
                self.curve_direction = curve_direction
                magnitude = self.steering_target_magnitude(
                    curve_strength
                )
                self.turn_target_steering = (
                    self.curve_direction * magnitude
                )
                self.set_curve_state("TURN_IN")

        elif self.curve_state == "TURN_IN":
            # 같은 방향으로 곡률이 더 커지면 target만 한 단계 높일 수 있다.
            if curve_direction == self.curve_direction:
                new_mag = self.steering_target_magnitude(
                    curve_strength
                )
                if new_mag > abs(self.turn_target_steering):
                    self.turn_target_steering = (
                        self.curve_direction * new_mag
                    )

            reached_target = (
                abs(
                    self.filtered_steering
                    - self.turn_target_steering
                ) <= 0.30
            )
            timed_out = (
                self.elapsed_sec(self.curve_state_start_ns)
                >= TURN_IN_MAX_SEC
            )

            if reached_target or timed_out:
                self.set_curve_state("HOLD")

        elif self.curve_state == "HOLD":
            if self.elapsed_sec(self.curve_state_start_ns) < HOLD_MIN_SEC:
                self.curve_exit_count = 0
                return

            # 현재 코너의 곡률이 사라졌거나,
            # S-curve처럼 반대 방향 곡률이 나타나면 EXIT.
            curve_is_flat = (
                curve_strength <= CURVATURE_EXIT_DEG
            )
            curve_is_opposite = (
                curve_direction == -self.curve_direction
                and curve_strength >= CURVATURE_OPPOSITE_EXIT_DEG
            )

            if curve_is_flat or curve_is_opposite:
                self.curve_exit_count += 1
            else:
                self.curve_exit_count = 0

            if self.curve_exit_count >= CURVE_EXIT_CONFIRM_FRAMES:
                self.set_curve_state("EXIT")

        elif self.curve_state == "EXIT":
            if abs(self.filtered_steering) <= 0.25:
                self.filtered_steering = 0.0

                # S-curve이면 현재 curvature의 방향으로 다음 TURN_IN.
                if (
                    curve_direction != 0
                    and curve_strength >= CURVATURE_ENTER_DEG
                ):
                    self.curve_direction = curve_direction
                    magnitude = self.steering_target_magnitude(
                        curve_strength
                    )
                    self.turn_target_steering = (
                        self.curve_direction * magnitude
                    )
                    self.set_curve_state("TURN_IN")
                else:
                    self.set_curve_state("STRAIGHT")

    def calculate_curve_command(self):
        geometry = self.calculate_path_geometry()
        if geometry is None:
            return None

        (
            near_heading,
            far_heading,
            curvature,
            lateral_error
        ) = geometry

        self.update_curve_state(
            near_heading,
            far_heading,
            curvature
        )

        if self.curve_state == "STRAIGHT":
            # 직선에서 heading이 약간 기울어진 것은 곡선으로 보지 않는다.
            # 작은 P-control로 차선 중앙/방향만 맞춘다.
            raw_target = (
                KP_STRAIGHT_HEADING * near_heading
                + KP_STRAIGHT_LATERAL * lateral_error
            )
            raw_target = max(
                -STRAIGHT_STEERING_LIMIT,
                min(STRAIGHT_STEERING_LIMIT, raw_target)
            )

            self.filtered_steering = self.ramp_toward(
                self.filtered_steering,
                raw_target,
                STRAIGHT_RAMP_PER_TICK
            )
            speed = self.driving_speed

        elif self.curve_state == "TURN_IN":
            # 목표 조향까지 매우 천천히 만든다.
            correction = max(
                -0.35,
                min(
                    0.35,
                    KP_HOLD_LATERAL * lateral_error
                )
            )
            target = (
                self.turn_target_steering + correction
            )

            self.filtered_steering = self.ramp_toward(
                self.filtered_steering,
                target,
                TURN_IN_RAMP_PER_TICK
            )
            speed = self.curve_speed_from_target(
                self.turn_target_steering
            )

        elif self.curve_state == "HOLD":
            # 코너 중간에서는 steering을 거의 고정한다.
            correction = max(
                -HOLD_CORRECTION_LIMIT,
                min(
                    HOLD_CORRECTION_LIMIT,
                    KP_HOLD_LATERAL * lateral_error
                )
            )
            target = (
                self.turn_target_steering + correction
            )

            self.filtered_steering = self.ramp_toward(
                self.filtered_steering,
                target,
                HOLD_ADJUST_PER_TICK
            )
            speed = self.curve_speed_from_target(
                self.turn_target_steering
            )

        else:  # EXIT
            self.filtered_steering = self.ramp_toward(
                self.filtered_steering,
                0.0,
                EXIT_RAMP_PER_TICK
            )
            speed = EXIT_SPEED

        self.filtered_steering = max(
            -MAX_STEERING,
            min(MAX_STEERING, self.filtered_steering)
        )

        command = int(round(self.filtered_steering))

        self.last_near_heading = near_heading
        self.last_far_heading = far_heading
        self.last_curvature = curvature
        self.last_lateral_error = lateral_error
        self.last_curve_speed = speed
        self.last_turn_target = self.turn_target_steering

        return (
            command,
            near_heading,
            curvature,
            lateral_error,
            speed
        )

    def stop_vehicle(self):
        self.steering_command = 0
        self.left_speed_command = 0
        self.right_speed_command = 0
        self.reset_curve_controller()

    def set_lane_following_command(self, speed_override=None):
        result = self.calculate_curve_command()

        if result is None:
            self.stop_vehicle()
            return 0.0, 0.0

        (
            command,
            near_heading,
            curvature,
            lateral_error,
            curve_speed
        ) = result

        if speed_override is None:
            speed = curve_speed
        else:
            speed = min(
                int(speed_override),
                int(curve_speed)
            )

        self.steering_command = command
        self.left_speed_command = speed
        self.right_speed_command = speed

        return near_heading, self.filtered_steering

    def set_state(self, new_state):
        if new_state == self.drive_state:
            return

        self.get_logger().info(
            f"STATE {self.drive_state} -> {new_state}"
        )
        self.drive_state = new_state

        if new_state == "GO":
            self.go_start_time_ns = self.now_ns()

    def go_clear_elapsed(self):
        if self.go_start_time_ns is None:
            return True

        return (
            (self.now_ns() - self.go_start_time_ns) / 1e9
            >= self.go_clear_sec
        )

    def run_traffic_state_machine(self):
        zone_ymax = self.get_stop_zone_ymax()
        red_confirmed = self.is_red_confirmed()
        green_confirmed = self.is_green_confirmed()

        near_heading = 0.0
        steering_float = 0.0

        if self.drive_state == "CRUISE":
            if red_confirmed and zone_ymax is not None:
                if zone_ymax >= self.stop_zone_stop_y:
                    self.set_state("STOPPED_RED")
                    self.stop_vehicle()
                else:
                    self.set_state("APPROACH_RED")
                    near_heading, steering_float = (
                        self.set_lane_following_command(
                            self.approach_speed
                        )
                    )
            else:
                near_heading, steering_float = (
                    self.set_lane_following_command()
                )

        elif self.drive_state == "APPROACH_RED":
            if green_confirmed:
                self.set_state("GO")
                near_heading, steering_float = (
                    self.set_lane_following_command()
                )
            elif zone_ymax is None:
                self.set_state("CRUISE")
                near_heading, steering_float = (
                    self.set_lane_following_command()
                )
            elif zone_ymax >= self.stop_zone_stop_y:
                self.set_state("STOPPED_RED")
                self.stop_vehicle()
            else:
                near_heading, steering_float = (
                    self.set_lane_following_command(
                        self.approach_speed
                    )
                )

        elif self.drive_state == "STOPPED_RED":
            if green_confirmed:
                self.set_state("GO")
                near_heading, steering_float = (
                    self.set_lane_following_command()
                )
            else:
                self.stop_vehicle()

        elif self.drive_state == "GO":
            near_heading, steering_float = (
                self.set_lane_following_command()
            )
            if self.go_clear_elapsed():
                self.set_state("CRUISE")

        else:
            self.set_state("CRUISE")
            near_heading, steering_float = (
                self.set_lane_following_command()
            )

        return (
            near_heading,
            steering_float,
            zone_ymax,
            red_confirmed,
            green_confirmed
        )

    def timer_callback(self):
        near_heading = 0.0
        steering_float = 0.0
        zone_ymax = self.get_stop_zone_ymax()
        red_confirmed = self.is_red_confirmed()
        green_confirmed = self.is_green_confirmed()

        if (
            self.lidar_data is not None
            and self.lidar_data.data is True
        ):
            self.stop_vehicle()
        else:
            (
                near_heading,
                steering_float,
                zone_ymax,
                red_confirmed,
                green_confirmed
            ) = self.run_traffic_state_machine()

        zone_text = (
            "None"
            if zone_ymax is None
            else f"{zone_ymax:.1f}"
        )

        self.get_logger().info(
            f"traffic={self.drive_state}, "
            f"curve={self.curve_state}, "
            f"zone_ymax={zone_text}, "
            f"red={red_confirmed}, green={green_confirmed}, "
            f"near={self.last_near_heading:.2f}, "
            f"far={self.last_far_heading:.2f}, "
            f"curvature={self.last_curvature:.2f}, "
            f"lateral={self.last_lateral_error:.1f}px, "
            f"turn_target={self.last_turn_target:.2f}, "
            f"steering_float={self.filtered_steering:.2f}, "
            f"steering={self.steering_command}, "
            f"speed={self.left_speed_command}"
        )

        motion_command_msg = MotionCommand()
        motion_command_msg.steering = int(self.steering_command)
        motion_command_msg.left_speed = int(self.left_speed_command)
        motion_command_msg.right_speed = int(self.right_speed_command)
        self.publisher.publish(motion_command_msg)


def main(args=None):
    rclpy.init(args=args)
    node = MotionPlanningNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n\nshutdown\n\n")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
