import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, QoSReliabilityPolicy
from interfaces_pkg.msg import LaneInfo, PathPlanningResult
import numpy as np

#---------------Variable Setting---------------
SUB_LANE_TOPIC_NAME = "yolov8_lane_info"
PUB_TOPIC_NAME = "path_planning_result"
CAR_CENTER_POINT = (320, 179)
#----------------------------------------------


class PathPlannerNode(Node):
    def __init__(self):
        super().__init__('path_planner_node')

        self.sub_lane_topic = self.declare_parameter(
            'sub_lane_topic', SUB_LANE_TOPIC_NAME).value
        self.pub_topic = self.declare_parameter(
            'pub_topic', PUB_TOPIC_NAME).value
        self.car_center_point = self.declare_parameter(
            'car_center_point', CAR_CENTER_POINT).value

        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )

        self.target_points = []

        self.lane_sub = self.create_subscription(
            LaneInfo,
            self.sub_lane_topic,
            self.lane_callback,
            self.qos_profile
        )
        self.publisher = self.create_publisher(
            PathPlanningResult,
            self.pub_topic,
            self.qos_profile
        )

    def lane_callback(self, msg: LaneInfo):
        self.target_points = msg.target_points
        if len(self.target_points) >= 3:
            self.plan_path()

    def plan_path(self):
        if not self.target_points:
            self.get_logger().warning("No target points available")
            return

        x_points, y_points = zip(*[
            (tp.target_x, tp.target_y) for tp in self.target_points
        ])

        # 차량 앞 범퍼 중심을 마지막 anchor로 추가한다.
        y_points_list = list(y_points)
        x_points_list = list(x_points)
        y_points_list.append(self.car_center_point[1])
        x_points_list.append(self.car_center_point[0])

        sorted_points = sorted(
            zip(y_points_list, x_points_list),
            key=lambda point: point[0]
        )

        y_points, x_points = zip(*sorted_points)
        y_points = np.asarray(y_points, dtype=float)
        x_points = np.asarray(x_points, dtype=float)

        # CubicSpline은 소수의 noisy anchor에서 overshoot할 수 있다.
        # lane-following에서는 shape-preserving한 선형 보간을 사용해
        # target point 사이에서 경로가 튀는 현상을 막는다.
        y_new = np.linspace(float(y_points.min()), float(y_points.max()), 100)
        x_new = np.interp(y_new, y_points, x_points)

        path_msg = PathPlanningResult()
        path_msg.x_points = x_new.tolist()
        path_msg.y_points = y_new.tolist()
        self.publisher.publish(path_msg)

        self.get_logger().info(
            "anchors=" + ", ".join(
                f"({x:.1f},{y:.1f})" for x, y in zip(x_points, y_points)
            )
        )

        self.target_points.clear()


def main(args=None):
    rclpy.init(args=args)
    node = PathPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n\nshutdown\n\n")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
