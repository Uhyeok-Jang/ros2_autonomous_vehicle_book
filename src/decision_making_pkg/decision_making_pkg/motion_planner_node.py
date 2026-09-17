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
KP_HEADING = 0.18
STEERING_DEADBAND_DEG = 1.5
STEERING_ALPHA = 0.40
MAX_STEERING_STEP = 1.5
DRIVING_SPEED = 60
APPROACH_SPEED = 35
LOOKAHEAD_FROM_END = 35
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
        self.steering_alpha = float(
            self.declare_parameter('steering_alpha', STEERING_ALPHA).value)
        self.max_steering_step = float(
            self.declare_parameter('max_steering_step', MAX_STEERING_STEP).value)
        self.driving_speed = int(
            self.declare_parameter('driving_speed', DRIVING_SPEED).value)
        self.approach_speed = int(
            self.declare_parameter('approach_speed', APPROACH_SPEED).value)
        self.lookahead_from_end = int(
            self.declare_parameter('lookahead_from_end', LOOKAHEAD_FROM_END).value)
        self.path_timeout_sec = float(
            self.declare_parameter('path_timeout_sec', PATH_TIMEOUT_SEC).value)
        self.light_confirm_frames = int(
            self.declare_parameter('light_confirm_frames', LIGHT_CONFIRM_FRAMES).value)
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
            and self.is_recent(self.detection_update_time_ns, DETECTION_TIMEOUT_SEC)
        ):
            current = self.extract_stop_zone_ymax(self.detection_data)
            if current is not None:
                return current

        if (
            self.last_stop_zone_ymax is not None
            and self.is_recent(self.last_stop_zone_time_ns, STOP_ZONE_MEMORY_SEC)
        ):
            return self.last_stop_zone_ymax

        return None

    def calculate_target_heading(self):
        if not self.has_fresh_path() or len(self.path_data) < 2:
            return None

        lookahead_index = max(
            0,
            len(self.path_data) - max(2, self.lookahead_from_end)
        )
        p1 = self.path_data[lookahead_index]
        p2 = self.path_data[-1]

        denominator = p1[1] - p2[1]
        if abs(denominator) < 1e-6:
            return 0.0

        return math.degrees(
            math.atan((p2[0] - p1[0]) / denominator)
        )

    def calculate_stable_steering(self):
        target_slope = self.calculate_target_heading()
        if target_slope is None:
            return None, 0.0, 0.0

        raw_steering = (
            0.0
            if abs(target_slope) < STEERING_DEADBAND_DEG
            else self.kp_heading * target_slope
        )
        raw_steering = max(-MAX_STEERING, min(MAX_STEERING, raw_steering))

        alpha = max(0.0, min(1.0, self.steering_alpha))
        smoothed = (
            alpha * raw_steering
            + (1.0 - alpha) * self.filtered_steering
        )

        delta = max(
            -self.max_steering_step,
            min(self.max_steering_step, smoothed - self.filtered_steering)
        )
        self.filtered_steering += delta
        self.filtered_steering = max(
            -MAX_STEERING,
            min(MAX_STEERING, self.filtered_steering)
        )

        return int(round(self.filtered_steering)), target_slope, raw_steering

    def stop_vehicle(self):
        self.steering_command = 0
        self.filtered_steering = 0.0
        self.left_speed_command = 0
        self.right_speed_command = 0

    def set_lane_following_command(self, speed_override=None):
        command, target_slope, raw_steering = self.calculate_stable_steering()

        if command is None:
            self.stop_vehicle()
            return 0.0, 0.0

        speed = self.driving_speed if speed_override is None else int(speed_override)
        self.steering_command = command
        self.left_speed_command = speed
        self.right_speed_command = speed
        return target_slope, raw_steering

    def set_state(self, new_state):
        if new_state == self.drive_state:
            return

        self.get_logger().info(f"STATE {self.drive_state} -> {new_state}")
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

        target_slope = 0.0
        raw_steering = 0.0

        if self.drive_state == "CRUISE":
            if red_confirmed and zone_ymax is not None:
                if zone_ymax >= self.stop_zone_stop_y:
                    self.set_state("STOPPED_RED")
                    self.stop_vehicle()
                else:
                    self.set_state("APPROACH_RED")
                    target_slope, raw_steering = self.set_lane_following_command(
                        self.approach_speed
                    )
            else:
                target_slope, raw_steering = self.set_lane_following_command()

        elif self.drive_state == "APPROACH_RED":
            if green_confirmed:
                self.set_state("GO")
                target_slope, raw_steering = self.set_lane_following_command()
            elif zone_ymax is None:
                self.set_state("CRUISE")
                target_slope, raw_steering = self.set_lane_following_command()
            elif zone_ymax >= self.stop_zone_stop_y:
                self.set_state("STOPPED_RED")
                self.stop_vehicle()
            else:
                target_slope, raw_steering = self.set_lane_following_command(
                    self.approach_speed
                )

        elif self.drive_state == "STOPPED_RED":
            if green_confirmed:
                self.set_state("GO")
                target_slope, raw_steering = self.set_lane_following_command()
            else:
                self.stop_vehicle()

        elif self.drive_state == "GO":
            target_slope, raw_steering = self.set_lane_following_command()
            if self.go_clear_elapsed():
                self.set_state("CRUISE")

        else:
            self.set_state("CRUISE")
            target_slope, raw_steering = self.set_lane_following_command()

        return target_slope, raw_steering, zone_ymax, red_confirmed, green_confirmed

    def timer_callback(self):
        target_slope = 0.0
        raw_steering = 0.0
        zone_ymax = self.get_stop_zone_ymax()
        red_confirmed = self.is_red_confirmed()
        green_confirmed = self.is_green_confirmed()

        if self.lidar_data is not None and self.lidar_data.data is True:
            self.stop_vehicle()
        else:
            (
                target_slope,
                raw_steering,
                zone_ymax,
                red_confirmed,
                green_confirmed
            ) = self.run_traffic_state_machine()

        zone_text = "None" if zone_ymax is None else f"{zone_ymax:.1f}"
        self.get_logger().info(
            f"state={self.drive_state}, "
            f"zone_ymax={zone_text}, "
            f"red={red_confirmed}, green={green_confirmed}, "
            f"slope={target_slope:.2f}, raw_steer={raw_steering:.2f}, "
            f"steering={self.steering_command}, "
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
