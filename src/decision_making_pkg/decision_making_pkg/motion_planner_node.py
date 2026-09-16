import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy

from std_msgs.msg import String, Bool
from interfaces_pkg.msg import PathPlanningResult, DetectionArray, MotionCommand

#---------------Variable Setting---------------
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
LOOKAHEAD_FROM_END = 35
PATH_TIMEOUT_SEC = 0.5
#----------------------------------------------


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
        self.lookahead_from_end = int(
            self.declare_parameter('lookahead_from_end', LOOKAHEAD_FROM_END).value)
        self.path_timeout_sec = float(
            self.declare_parameter('path_timeout_sec', PATH_TIMEOUT_SEC).value)

        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )

        self.detection_data = None
        self.path_data = None
        self.path_update_time_ns = None
        self.traffic_light_data = None
        self.lidar_data = None

        self.steering_command = 0
        self.left_speed_command = 0
        self.right_speed_command = 0
        self.filtered_steering = 0.0

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

    def detection_callback(self, msg: DetectionArray):
        self.detection_data = msg

    def path_callback(self, msg: PathPlanningResult):
        self.path_data = list(zip(msg.x_points, msg.y_points))
        self.path_update_time_ns = self.get_clock().now().nanoseconds

    def traffic_light_callback(self, msg: String):
        self.traffic_light_data = msg

    def lidar_callback(self, msg: Bool):
        self.lidar_data = msg

    def has_fresh_path(self):
        if self.path_data is None or self.path_update_time_ns is None:
            return False

        age_sec = (
            self.get_clock().now().nanoseconds - self.path_update_time_ns
        ) / 1e9
        return age_sec <= self.path_timeout_sec

    def calculate_target_heading(self):
        """차량 중심보다 충분히 앞쪽의 path point를 사용해 목표 heading을 계산한다."""
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

        # 기존 제공 함수와 같은 부호 체계를 유지한다.
        slope = math.degrees(
            math.atan((p2[0] - p1[0]) / denominator)
        )
        return slope

    def calculate_stable_steering(self):
        target_slope = self.calculate_target_heading()
        if target_slope is None:
            return None, 0.0, 0.0

        if abs(target_slope) < STEERING_DEADBAND_DEG:
            raw_steering = 0.0
        else:
            raw_steering = self.kp_heading * target_slope

        raw_steering = max(
            -MAX_STEERING,
            min(MAX_STEERING, raw_steering)
        )

        alpha = max(0.0, min(1.0, self.steering_alpha))
        smoothed = (
            alpha * raw_steering
            + (1.0 - alpha) * self.filtered_steering
        )

        delta = smoothed - self.filtered_steering
        delta = max(
            -self.max_steering_step,
            min(self.max_steering_step, delta)
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

    def set_lane_following_command(self):
        command, target_slope, raw_steering = self.calculate_stable_steering()

        if command is None:
            # path가 없는데 직진하는 기존 동작을 없애고 안전하게 정지한다.
            self.stop_vehicle()
            return 0.0, 0.0

        self.steering_command = command
        self.left_speed_command = self.driving_speed
        self.right_speed_command = self.driving_speed
        return target_slope, raw_steering

    def timer_callback(self):
        target_slope = 0.0
        raw_steering = 0.0

        if self.lidar_data is not None and self.lidar_data.data is True:
            self.stop_vehicle()

        elif (
            self.traffic_light_data is not None
            and self.traffic_light_data.data == 'Red'
        ):
            should_stop = False

            if self.detection_data is not None:
                for detection in self.detection_data.detections:
                    if detection.class_name == 'traffic_light':
                        y_max = int(
                            detection.bbox.center.position.y
                            + detection.bbox.size.y / 2
                        )

                        if y_max < 150:
                            should_stop = True
                            break

            if should_stop:
                self.stop_vehicle()
            else:
                target_slope, raw_steering = self.set_lane_following_command()

        else:
            target_slope, raw_steering = self.set_lane_following_command()

        self.get_logger().info(
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
