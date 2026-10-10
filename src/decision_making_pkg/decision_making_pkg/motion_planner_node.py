import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy

from std_msgs.msg import String, Bool
from interfaces_pkg.msg import PathPlanningResult, DetectionArray, MotionCommand
from .lib import decision_making_func_lib as DMFL

#---------------Variable Setting---------------
SUB_DETECTION_TOPIC_NAME = "detections"
SUB_PATH_TOPIC_NAME = "path_planning_result"
SUB_TRAFFIC_LIGHT_TOPIC_NAME = "yolov8_traffic_light_info"
SUB_LIDAR_OBSTACLE_TOPIC_NAME = "lidar_obstacle_info"
PUB_TOPIC_NAME = "topic_control_signal"

# 모션 플랜 발행 주기 (초)
TIMER = 0.1

# lane following controller
MAX_STEERING = 7
KP_HEADING = 0.08        # deg -> steering command
STEERING_DEADBAND_DEG = 1.0
STEERING_ALPHA = 0.3    # EMA: 작을수록 조향 변화가 부드러움
MAX_STEERING_STEP = 2.0  # 한 제어 주기당 최대 변화량
DRIVING_SPEED = 210        # 첫 안정화 테스트에서는 기존 100보다 낮게 시작
LANE_LOSS_SPEED = 140      # lane2 누락 중 마지막 경로로 주행할 때의 PWM
LANE_DETECTION_TIMEOUT = 0.5  # 새 인식/경로가 없는 경우 감속까지 기다리는 초
#----------------------------------------------


class MotionPlanningNode(Node):
    def __init__(self):
        super().__init__('motion_planner_node')

        # 토픽 이름 설정
        self.sub_detection_topic = self.declare_parameter(
            'sub_detection_topic', SUB_DETECTION_TOPIC_NAME).value
        self.sub_path_topic = self.declare_parameter(
            'sub_lane_topic', SUB_PATH_TOPIC_NAME).value
        self.sub_traffic_light_topic = self.declare_parameter(
            'sub_traffic_light_topic', SUB_TRAFFIC_LIGHT_TOPIC_NAME).value
        self.sub_lidar_obstacle_topic = self.declare_parameter(
            'sub_lidar_obstacle_topic', SUB_LIDAR_OBSTACLE_TOPIC_NAME).value
        self.pub_topic = self.declare_parameter('pub_topic', PUB_TOPIC_NAME).value
        self.timer_period = self.declare_parameter('timer', TIMER).value

        self.kp_heading = float(self.declare_parameter('kp_heading', KP_HEADING).value)
        self.steering_alpha = float(self.declare_parameter('steering_alpha', STEERING_ALPHA).value)
        self.max_steering_step = float(self.declare_parameter('max_steering_step', MAX_STEERING_STEP).value)
        self.driving_speed = int(self.declare_parameter('driving_speed', DRIVING_SPEED).value)
        self.lane_loss_speed = int(self.declare_parameter('lane_loss_speed', LANE_LOSS_SPEED).value)
        self.lane_detection_timeout = max(0.0, float(self.declare_parameter(
            'lane_detection_timeout', LANE_DETECTION_TIMEOUT).value))

        # QoS 설정
        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )

        # 변수 초기화
        self.detection_data = None
        self.path_data = None
        self.lane_detected = None
        self.last_detection_time = None
        self.last_valid_path_time = None
        # A path proves that camera -> YOLO -> lane extraction is ready.
        # Latch once; brief lane losses during driving must not restart this gate.
        self.startup_ready = False
        self.traffic_light_data = None
        self.lidar_data = None

        self.steering_command = 0
        self.left_speed_command = 0
        self.right_speed_command = 0
        self.filtered_steering = 0.0

        # 서브스크라이버 설정
        self.detection_sub = self.create_subscription(
            DetectionArray, self.sub_detection_topic,
            self.detection_callback, self.qos_profile)
        self.path_sub = self.create_subscription(
            PathPlanningResult, self.sub_path_topic,
            self.path_callback, self.qos_profile)
        self.traffic_light_sub = self.create_subscription(
            String, self.sub_traffic_light_topic,
            self.traffic_light_callback, self.qos_profile)
        self.lidar_sub = self.create_subscription(
            Bool, self.sub_lidar_obstacle_topic,
            self.lidar_callback, self.qos_profile)

        # 퍼블리셔 설정
        self.publisher = self.create_publisher(
            MotionCommand, self.pub_topic, self.qos_profile)

        # 타이머 설정
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        self.get_logger().info('Waiting for the first valid camera/YOLO lane path; motors stopped')

    def detection_callback(self, msg: DetectionArray):
        self.detection_data = msg
        self.last_detection_time = time.monotonic()
        self.lane_detected = any(
            detection.class_name == 'lane2'
            and detection.mask.width > 0 and detection.mask.height > 0
            and len(detection.mask.data) >= 3
            for detection in msg.detections
        )

    def path_callback(self, msg: PathPlanningResult):
        if len(msg.x_points) != len(msg.y_points) or len(msg.x_points) < 10:
            return
        path = list(zip(msg.x_points, msg.y_points))
        if not all(math.isfinite(x) and math.isfinite(y) for x, y in path):
            return
        if path[-10][1] == path[-1][1]:
            return

        self.path_data = path
        self.last_valid_path_time = time.monotonic()
        if not self.startup_ready:
            self.startup_ready = True
            self.get_logger().info('Camera/YOLO lane path ready; steering and speed control enabled')

    def traffic_light_callback(self, msg: String):
        self.traffic_light_data = msg

    def lidar_callback(self, msg: Bool):
        self.lidar_data = msg

    def calculate_stable_steering(self):
        """Path의 차량 근처 heading을 비례 제어하고 급격한 좌우 반전을 제한한다."""
        if self.path_data is None or len(self.path_data) < 10:
            return 0, 0.0, 0.0

        target_slope = DMFL.calculate_slope_between_points(
            self.path_data[-10], self.path_data[-1])

        if not isinstance(target_slope, (int, float)):
            return 0, 0.0, 0.0

        # 작은 오차에서는 직진하여 +7/-7 반복 스위칭을 막는다.
        if abs(target_slope) < STEERING_DEADBAND_DEG:
            raw_steering = 0.0
        else:
            raw_steering = self.kp_heading * target_slope

        raw_steering = max(-MAX_STEERING, min(MAX_STEERING, raw_steering))

        # EMA smoothing
        alpha = max(0.0, min(1.0, self.steering_alpha))
        smoothed = alpha * raw_steering + (1.0 - alpha) * self.filtered_steering

        # 한 주기에서 조향이 너무 크게 바뀌지 않도록 rate limit
        delta = smoothed - self.filtered_steering
        delta = max(-self.max_steering_step, min(self.max_steering_step, delta))
        self.filtered_steering += delta
        self.filtered_steering = max(-MAX_STEERING, min(MAX_STEERING, self.filtered_steering))

        command = int(round(self.filtered_steering))
        return command, float(target_slope), float(raw_steering)

    def stop_vehicle(self):
        self.steering_command = 0
        self.filtered_steering = 0.0
        self.left_speed_command = 0
        self.right_speed_command = 0

    def tracking_speed(self):
        """Missing lane2 or stale perception reduces speed while keeping the last path."""
        now = time.monotonic()
        lane_lost = (
            self.lane_detected is False
            or self.last_valid_path_time is None
            or now - self.last_valid_path_time > self.lane_detection_timeout
            or (self.last_detection_time is not None
                and now - self.last_detection_time > self.lane_detection_timeout)
        )
        normal_speed = max(0, min(255, self.driving_speed))
        speed = min(normal_speed, max(0, min(255, self.lane_loss_speed))) if lane_lost else normal_speed
        return speed, lane_lost

    def timer_callback(self):
        target_slope = 0.0
        raw_steering = 0.0
        speed, lane_lost = self.tracking_speed()

        if not self.startup_ready:
            self.stop_vehicle()

        elif self.lidar_data is not None and self.lidar_data.data is True:
            self.stop_vehicle()

        elif self.traffic_light_data is not None and self.traffic_light_data.data == 'Red':
            should_stop = False

            if self.detection_data is not None:
                for detection in self.detection_data.detections:
                    if detection.class_name == 'traffic_light':
                        y_max = int(
                            detection.bbox.center.position.y +
                            detection.bbox.size.y / 2
                        )

                        if y_max < 150:
                            should_stop = True
                            break

            if should_stop:
                self.stop_vehicle()
            else:
                self.steering_command, target_slope, raw_steering = self.calculate_stable_steering()
                self.left_speed_command = speed
                self.right_speed_command = speed

        else:
            self.steering_command, target_slope, raw_steering = self.calculate_stable_steering()
            self.left_speed_command = speed
            self.right_speed_command = speed

        self.get_logger().info(
            f"slope={target_slope:.2f}, raw_steer={raw_steering:.2f}, "
            f"steering={self.steering_command}, "
            f"lane_lost={lane_lost}, "
            f"left_speed={self.left_speed_command}, right_speed={self.right_speed_command}"
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
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
