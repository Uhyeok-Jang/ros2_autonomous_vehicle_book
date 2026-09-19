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
MAX_STEERING = 5.0
CAR_CENTER_X = 320.0

# Curve tracking controller
# 기존에는 차량 중심 -> 먼 lookahead 점의 기울기만 사용해서 코너를 안쪽으로 자르는
# 경향이 있었다. 이제 가까운 구간의 path tangent + lateral error를 함께 사용한다.
KP_HEADING = 0.055
KP_LATERAL = 0.030
STEERING_DEADBAND_DEG = 1.0
STEERING_ALPHA = 0.20
STEERING_STEP_TURN_IN = 0.50
STEERING_STEP_RECOVER = 0.70
STEERING_STEP_REVERSE = 0.45
# 값이 작을수록 차량에 가까운 경로만 사용하므로 코너 선행 조향이 줄어든다.
TRACKING_FROM_END = 6
TANGENT_SPAN = 3
STEERING_DELAY_SEC = 0.20

# Curve-aware speed
DRIVING_SPEED = 200
MEDIUM_CURVE_SPEED = 130
SHARP_CURVE_SPEED = 95
VERY_SHARP_CURVE_SPEED = 75
APPROACH_SPEED = 70
STOP_ZONE_SPEED = 120
STOP_ZONE_SPEED_HOLD_SEC = 1.2
SPEED_RECOVERY_STEP = 10
MEDIUM_CURVE_DEG = 8.0
SHARP_CURVE_DEG = 18.0
VERY_SHARP_CURVE_DEG = 40.0
LATERAL_MEDIUM_PX = 28.0
LATERAL_SHARP_PX = 48.0
LATERAL_VERY_SHARP_PX = 80.0

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

        self.kp_heading = float(
            self.declare_parameter('kp_heading', KP_HEADING).value)
        self.kp_lateral = float(
            self.declare_parameter('kp_lateral', KP_LATERAL).value)
        self.steering_alpha = float(
            self.declare_parameter('steering_alpha', STEERING_ALPHA).value)
        self.steering_step_turn_in = float(
            self.declare_parameter(
                'steering_step_turn_in', STEERING_STEP_TURN_IN
            ).value
        )
        self.steering_step_recover = float(
            self.declare_parameter(
                'steering_step_recover', STEERING_STEP_RECOVER
            ).value
        )
        self.steering_step_reverse = float(
            self.declare_parameter(
                'steering_step_reverse', STEERING_STEP_REVERSE
            ).value
        )
        self.tracking_from_end = int(
            self.declare_parameter(
                'tracking_from_end', TRACKING_FROM_END
            ).value
        )
        self.tangent_span = int(
            self.declare_parameter('tangent_span', TANGENT_SPAN).value)
        self.steering_delay_sec = float(
            self.declare_parameter(
                'steering_delay_sec', STEERING_DELAY_SEC
            ).value
        )

        self.driving_speed = int(
            self.declare_parameter('driving_speed', DRIVING_SPEED).value)
        self.medium_curve_speed = int(
            self.declare_parameter(
                'medium_curve_speed', MEDIUM_CURVE_SPEED
            ).value
        )
        self.sharp_curve_speed = int(
            self.declare_parameter(
                'sharp_curve_speed', SHARP_CURVE_SPEED
            ).value
        )
        self.very_sharp_curve_speed = int(
            self.declare_parameter(
                'very_sharp_curve_speed', VERY_SHARP_CURVE_SPEED
            ).value
        )
        self.approach_speed = int(
            self.declare_parameter('approach_speed', APPROACH_SPEED).value)
        self.stop_zone_speed = int(
            self.declare_parameter('stop_zone_speed', STOP_ZONE_SPEED).value)
        self.stop_zone_speed_hold_sec = float(
            self.declare_parameter(
                'stop_zone_speed_hold_sec',
                STOP_ZONE_SPEED_HOLD_SEC
            ).value
        )
        self.speed_recovery_step = int(
            self.declare_parameter(
                'speed_recovery_step', SPEED_RECOVERY_STEP
            ).value
        )
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

        self.last_heading_error = 0.0
        self.last_lateral_error = 0.0
        self.last_curve_speed = self.driving_speed
        self.last_step_limit = self.steering_step_turn_in
        self.curve_entry_time_ns = None
        self.last_steering_delay_remaining = 0.0
        self.stop_zone_speed_recovery_active = False
        self.log_counter = 0

        # CRUISE -> APPROACH_RED -> STOPPED_RED -> GO -> CRUISE
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

    def is_stop_zone_speed_active(self):
        """Keep the speed cap briefly after the stop zone leaves the image."""
        return (
            self.last_stop_zone_time_ns is not None
            and self.is_recent(
                self.last_stop_zone_time_ns,
                self.stop_zone_speed_hold_sec
            )
        )

    @staticmethod
    def limit_speed_recovery(target_speed, current_speed, recovery_step):
        """Apply deceleration immediately and make only acceleration gradual."""
        target_speed = max(0, int(target_speed))
        current_speed = max(0, int(current_speed))
        recovery_step = max(0, int(recovery_step))

        if (
            recovery_step > 0
            and current_speed > 0
            and target_speed > current_speed
        ):
            return min(target_speed, current_speed + recovery_step)
        return target_speed

    def calculate_path_errors(self):
        """가까운 path의 tangent와 lateral error를 함께 계산한다.

        예전 방식은 차량 중심에서 먼 lookahead 한 점까지의 chord 각도를 썼기 때문에
        급커브에서 코너 안쪽을 잘라가는 경향이 있었다. 여기서는 차량에 더 가까운
        path 구간의 local tangent를 사용하고, target path가 차량 중심에서 얼마나
        좌/우로 벗어나 있는지도 별도 오차로 사용한다.
        """
        if not self.has_fresh_path() or len(self.path_data) < 4:
            return None

        n = len(self.path_data)
        track_idx = max(1, n - max(4, self.tracking_from_end))
        track_idx = min(track_idx, n - 2)
        far_idx = max(0, track_idx - max(2, self.tangent_span))

        far_point = self.path_data[far_idx]
        track_point = self.path_data[track_idx]

        denominator = far_point[1] - track_point[1]
        if abs(denominator) < 1e-6:
            heading_error = 0.0
        else:
            # 기존 조향 부호 체계 유지:
            # path가 왼쪽으로 꺾이면 negative, 오른쪽이면 positive.
            heading_error = math.degrees(
                math.atan(
                    (track_point[0] - far_point[0]) / denominator
                )
            )

        lateral_error = float(track_point[0] - CAR_CENTER_X)
        return heading_error, lateral_error

    def select_curve_speed(self, heading_error, lateral_error):
        abs_heading = abs(heading_error)
        abs_lateral = abs(lateral_error)

        if (
            abs_heading >= VERY_SHARP_CURVE_DEG
            or abs_lateral >= LATERAL_VERY_SHARP_PX
        ):
            return self.very_sharp_curve_speed

        if (
            abs_heading >= SHARP_CURVE_DEG
            or abs_lateral >= LATERAL_SHARP_PX
        ):
            return self.sharp_curve_speed

        if (
            abs_heading >= MEDIUM_CURVE_DEG
            or abs_lateral >= LATERAL_MEDIUM_PX
        ):
            return self.medium_curve_speed

        return self.driving_speed

    def steering_delay_active(self, heading_error, lateral_error):
        """코너를 처음 감지한 뒤 설정된 시간 동안 조향 진입을 늦춘다."""
        abs_heading = abs(heading_error)
        abs_lateral = abs(lateral_error)
        curve_detected = (
            abs_heading >= MEDIUM_CURVE_DEG
            or abs_lateral >= LATERAL_MEDIUM_PX
        )
        curve_cleared = (
            abs_heading < MEDIUM_CURVE_DEG * 0.5
            and abs_lateral < LATERAL_MEDIUM_PX * 0.5
        )

        now_ns = self.now_ns()

        if self.curve_entry_time_ns is None:
            if not curve_detected:
                self.last_steering_delay_remaining = 0.0
                return False
            self.curve_entry_time_ns = now_ns
        elif curve_cleared:
            self.curve_entry_time_ns = None
            self.last_steering_delay_remaining = 0.0
            return False

        elapsed_sec = (now_ns - self.curve_entry_time_ns) / 1e9
        self.last_steering_delay_remaining = max(
            0.0,
            self.steering_delay_sec - elapsed_sec
        )
        return self.last_steering_delay_remaining > 0.0

    def calculate_stable_steering(self):
        errors = self.calculate_path_errors()
        if errors is None:
            return None, 0.0, 0.0, 0.0, self.driving_speed

        heading_error, lateral_error = errors

        curve_speed = self.select_curve_speed(
            heading_error,
            lateral_error
        )

        if self.steering_delay_active(heading_error, lateral_error):
            # 대기 중 조향 필터가 내부적으로 누적되어 2초 뒤 갑자기 튀지 않게 한다.
            self.filtered_steering = 0.0
            self.last_heading_error = heading_error
            self.last_lateral_error = lateral_error
            self.last_curve_speed = curve_speed
            self.last_step_limit = self.steering_step_turn_in
            return 0, heading_error, lateral_error, 0.0, curve_speed

        heading_component = (
            0.0
            if abs(heading_error) < STEERING_DEADBAND_DEG
            else self.kp_heading * heading_error
        )
        lateral_component = self.kp_lateral * lateral_error
        raw_steering = heading_component + lateral_component
        raw_steering = max(
            -MAX_STEERING,
            min(MAX_STEERING, raw_steering)
        )

        alpha = max(0.0, min(1.0, self.steering_alpha))
        smoothed = (
            alpha * raw_steering
            + (1.0 - alpha) * self.filtered_steering
        )

        # 코너 감지 시점은 그대로 유지하되 조향 명령의 변화만 부드럽게 만든다.
        # 특히 segmentation 경계가 바뀌는 한두 프레임 때문에 반대 방향으로
        # 급전환하지 않도록 방향 반전에는 가장 작은 변화량을 사용한다.
        same_direction = self.filtered_steering * raw_steering >= 0.0
        increasing_magnitude = abs(raw_steering) > abs(self.filtered_steering)

        reversing_direction = (
            self.filtered_steering * raw_steering < 0.0
            and abs(self.filtered_steering) >= 0.5
            and abs(raw_steering) >= 0.5
        )

        if reversing_direction:
            step_limit = self.steering_step_reverse
        elif same_direction and increasing_magnitude:
            step_limit = self.steering_step_turn_in
        else:
            step_limit = self.steering_step_recover

        delta = max(
            -step_limit,
            min(step_limit, smoothed - self.filtered_steering)
        )
        self.filtered_steering += delta
        self.filtered_steering = max(
            -MAX_STEERING,
            min(MAX_STEERING, self.filtered_steering)
        )

        self.last_heading_error = heading_error
        self.last_lateral_error = lateral_error
        self.last_curve_speed = curve_speed
        self.last_step_limit = step_limit

        return (
            int(round(self.filtered_steering)),
            heading_error,
            lateral_error,
            raw_steering,
            curve_speed
        )

    def stop_vehicle(self):
        self.steering_command = 0
        self.filtered_steering = 0.0
        self.left_speed_command = 0
        self.right_speed_command = 0
        self.curve_entry_time_ns = None
        self.last_steering_delay_remaining = 0.0

    def set_lane_following_command(self, speed_override=None):
        (
            command,
            heading_error,
            lateral_error,
            raw_steering,
            curve_speed
        ) = self.calculate_stable_steering()

        if command is None:
            self.stop_vehicle()
            return 0.0, 0.0

        if speed_override is None:
            speed = curve_speed
        else:
            # 신호 접근 속도와 curve-aware 속도 중 더 느린 값을 사용한다.
            speed = min(int(speed_override), int(curve_speed))

        if (
            self.stop_zone_speed_recovery_active
            and speed_override is None
        ):
            current_speed = min(
                int(self.left_speed_command),
                int(self.right_speed_command)
            )
            speed = self.limit_speed_recovery(
                speed,
                current_speed,
                self.speed_recovery_step
            )
            if speed >= self.driving_speed:
                self.stop_zone_speed_recovery_active = False

        self.steering_command = command
        self.left_speed_command = speed
        self.right_speed_command = speed

        return heading_error, raw_steering

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
        # stop_zone 자체는 통과 가능하다.
        # 안정적으로 확인된 Red + stop_zone일 때만 정지 절차를 시작한다.
        zone_ymax = self.get_stop_zone_ymax()
        red_confirmed = self.is_red_confirmed()
        green_confirmed = self.is_green_confirmed()
        stop_zone_speed_active = self.is_stop_zone_speed_active()
        stop_zone_speed_limit = (
            self.stop_zone_speed if stop_zone_speed_active else None
        )
        if stop_zone_speed_active:
            self.stop_zone_speed_recovery_active = True

        target_slope = 0.0
        raw_steering = 0.0

        if self.drive_state == "CRUISE":
            if red_confirmed and zone_ymax is not None:
                if zone_ymax >= self.stop_zone_stop_y:
                    self.set_state("STOPPED_RED")
                    self.stop_vehicle()
                else:
                    self.set_state("APPROACH_RED")
                    target_slope, raw_steering = (
                        self.set_lane_following_command(
                            self.approach_speed
                        )
                    )
            else:
                target_slope, raw_steering = (
                    self.set_lane_following_command(stop_zone_speed_limit)
                )

        elif self.drive_state == "APPROACH_RED":
            if green_confirmed:
                self.set_state("GO")
                target_slope, raw_steering = (
                    self.set_lane_following_command(stop_zone_speed_limit)
                )
            elif zone_ymax is None:
                self.set_state("CRUISE")
                target_slope, raw_steering = (
                    self.set_lane_following_command(stop_zone_speed_limit)
                )
            elif zone_ymax >= self.stop_zone_stop_y:
                self.set_state("STOPPED_RED")
                self.stop_vehicle()
            else:
                target_slope, raw_steering = (
                    self.set_lane_following_command(
                        self.approach_speed
                    )
                )

        elif self.drive_state == "STOPPED_RED":
            if green_confirmed:
                self.set_state("GO")
                target_slope, raw_steering = (
                    self.set_lane_following_command(stop_zone_speed_limit)
                )
            else:
                self.stop_vehicle()

        elif self.drive_state == "GO":
            target_slope, raw_steering = (
                self.set_lane_following_command(stop_zone_speed_limit)
            )
            if self.go_clear_elapsed():
                self.set_state("CRUISE")

        else:
            self.set_state("CRUISE")
            target_slope, raw_steering = (
                self.set_lane_following_command(stop_zone_speed_limit)
            )

        return (
            target_slope,
            raw_steering,
            zone_ymax,
            red_confirmed,
            green_confirmed
        )

    def timer_callback(self):
        target_slope = 0.0
        raw_steering = 0.0
        zone_ymax = self.get_stop_zone_ymax()
        red_confirmed = self.is_red_confirmed()
        green_confirmed = self.is_green_confirmed()
        stop_zone_speed_active = self.is_stop_zone_speed_active()

        if (
            self.lidar_data is not None
            and self.lidar_data.data is True
        ):
            self.stop_vehicle()
        else:
            (
                target_slope,
                raw_steering,
                zone_ymax,
                red_confirmed,
                green_confirmed
            ) = self.run_traffic_state_machine()

        zone_text = (
            "None"
            if zone_ymax is None
            else f"{zone_ymax:.1f}"
        )

        self.log_counter += 1
        if self.log_counter % 10 == 0:
            self.get_logger().info(
                f"state={self.drive_state}, "
                f"zone_ymax={zone_text}, "
                f"zone_speed_cap={stop_zone_speed_active}, "
                f"red={red_confirmed}, green={green_confirmed}, "
                f"heading={self.last_heading_error:.2f}, "
                f"lateral={self.last_lateral_error:.1f}px, "
                f"raw_steer={raw_steering:.2f}, "
                f"steering={self.steering_command}, "
                f"steering_delay={self.last_steering_delay_remaining:.2f}s, "
                f"step_limit={self.last_step_limit:.2f}, "
                f"curve_speed={self.last_curve_speed}, "
                f"left_speed={self.left_speed_command}, "
                f"right_speed={self.right_speed_command}"
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
