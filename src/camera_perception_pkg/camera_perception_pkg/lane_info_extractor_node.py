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
# Subscribe할 토픽 이름
SUB_TOPIC_NAME = "detections"

# Publish할 토픽 이름
PUB_TOPIC_NAME = "yolov8_lane_info"
ROI_IMAGE_TOPIC_NAME = "roi_image"

# 화면에 이미지를 처리하는 과정을 띄울것인지 여부
SHOW_IMAGE = True

# 차선 중앙점 추출 설정
LANE_CLASS_NAME = "lane2"
TARGET_Y_VALUES = (5, 55, 105)
TARGET_BAND_THICKNESS = 10
MIN_LANE_PIXELS = 20
SMOOTHING_ALPHA = 0.4
#----------------------------------------------


class Yolov8InfoExtractor(Node):
    def __init__(self):
        super().__init__('lane_info_extractor_node')

        self.sub_topic = self.declare_parameter('sub_detection_topic', SUB_TOPIC_NAME).value
        self.pub_topic = self.declare_parameter('pub_topic', PUB_TOPIC_NAME).value
        self.show_image = self.declare_parameter('show_image', SHOW_IMAGE).value
        self.lane_class_name = self.declare_parameter('lane_class_name', LANE_CLASS_NAME).value
        self.smoothing_alpha = float(self.declare_parameter('smoothing_alpha', SMOOTHING_ALPHA).value)

        self.cv_bridge = CvBridge()
        self.prev_target_x = [None] * len(TARGET_Y_VALUES)

        # QoS settings
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

    def build_lane_mask(self, detection_msg: DetectionArray):
        """YOLO segmentation의 lane2 polygon을 채운 binary mask로 변환한다."""
        lane_mask = None

        for detection in detection_msg.detections:
            if detection.class_name != self.lane_class_name:
                continue

            mask_msg = detection.mask
            if mask_msg.height <= 0 or mask_msg.width <= 0 or len(mask_msg.data) < 3:
                continue

            if lane_mask is None:
                lane_mask = np.zeros((mask_msg.height, mask_msg.width), dtype=np.uint8)

            polygon = np.array(
                [[int(round(point.x)), int(round(point.y))] for point in mask_msg.data],
                dtype=np.int32
            )

            if len(polygon) >= 3:
                cv2.fillPoly(lane_mask, [polygon], 255)

        return lane_mask

    def estimate_target_x(self, roi_image, target_y, target_idx):
        """해당 높이의 lane2 영역 중앙을 구하고 이전 프레임과 EMA smoothing한다."""
        h, w = roi_image.shape[:2]
        half = TARGET_BAND_THICKNESS // 2
        upper = max(0, target_y - half)
        lower = min(h, target_y + half + 1)

        detected_x = np.where(roi_image[upper:lower, :] > 0)[1]

        if detected_x.size >= MIN_LANE_PIXELS:
            # 극단 outlier에 덜 민감하도록 5~95 percentile 경계를 사용한다.
            left_x, right_x = np.percentile(detected_x, [5, 95])
            measured_x = float((left_x + right_x) / 2.0)

            previous_x = self.prev_target_x[target_idx]
            if previous_x is None:
                smoothed_x = measured_x
            else:
                alpha = np.clip(self.smoothing_alpha, 0.0, 1.0)
                smoothed_x = alpha * measured_x + (1.0 - alpha) * previous_x
        else:
            # lane 일부가 순간적으로 사라지면 x=150 같은 임의값으로 튀지 않고
            # 직전 유효 target 또는 화면 중앙을 유지한다.
            previous_x = self.prev_target_x[target_idx]
            smoothed_x = previous_x if previous_x is not None else (w / 2.0)

        smoothed_x = float(np.clip(smoothed_x, 0, w - 1))
        self.prev_target_x[target_idx] = smoothed_x
        return smoothed_x

    def yolov8_detections_callback(self, detection_msg: DetectionArray):
        if len(detection_msg.detections) == 0:
            return

        lane2_mask_image = self.build_lane_mask(detection_msg)
        if lane2_mask_image is None or cv2.countNonZero(lane2_mask_image) == 0:
            self.get_logger().warning(f"No valid '{self.lane_class_name}' mask detected")
            return

        h, w = lane2_mask_image.shape[:2]
        dst_mat = [
            [round(w * 0.3), round(h * 0.0)],
            [round(w * 0.7), round(h * 0.0)],
            [round(w * 0.7), h],
            [round(w * 0.3), h]
        ]
        src_mat = [[238, 316], [402, 313], [501, 476], [155, 476]]

        lane2_bird_image = CPFL.bird_convert(
            lane2_mask_image,
            srcmat=src_mat,
            dstmat=dst_mat
        )
        roi_image = CPFL.roi_rectangle_below(lane2_bird_image, cutting_idx=300)
        roi_image = cv2.convertScaleAbs(roi_image)

        target_points = []
        target_x_values = []

        for idx, target_point_y in enumerate(TARGET_Y_VALUES):
            target_point_x = self.estimate_target_x(roi_image, target_point_y, idx)
            target_x_values.append(target_point_x)

            target_point = TargetPoint()
            target_point.target_x = int(round(target_point_x))
            target_point.target_y = int(target_point_y)
            target_points.append(target_point)

        # slope는 현재 path planner에서 직접 사용하지 않지만, 디버깅을 위해
        # far-to-near target의 방향을 계산해 메시지에 유지한다.
        if len(target_points) >= 2:
            dx = target_x_values[-1] - target_x_values[0]
            dy = TARGET_Y_VALUES[-1] - TARGET_Y_VALUES[0]
            grad = float(np.degrees(np.arctan2(dx, dy))) if dy != 0 else 0.0
        else:
            grad = 0.0

        if self.show_image:
            debug_roi = cv2.cvtColor(roi_image, cv2.COLOR_GRAY2BGR)
            for point in target_points:
                cv2.circle(
                    debug_roi,
                    (int(point.target_x), int(point.target_y)),
                    5,
                    (0, 0, 255),
                    -1
                )
            cv2.line(debug_roi, (w // 2, 0), (w // 2, debug_roi.shape[0] - 1), (0, 255, 0), 1)

            cv2.imshow('lane2_mask_image', lane2_mask_image)
            cv2.imshow('lane2_bird_img', lane2_bird_image)
            cv2.imshow('roi_img', debug_roi)
            cv2.waitKey(1)

        try:
            roi_image_msg = self.cv_bridge.cv2_to_imgmsg(roi_image, encoding="mono8")
            self.roi_image_publisher.publish(roi_image_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to convert and publish ROI image: {e}")

        lane = LaneInfo()
        lane.slope = grad
        lane.target_points = target_points
        self.publisher.publish(lane)

        self.get_logger().info(
            "target_x=" + ", ".join(f"{x:.1f}" for x in target_x_values) +
            f", slope={grad:.2f}"
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
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
