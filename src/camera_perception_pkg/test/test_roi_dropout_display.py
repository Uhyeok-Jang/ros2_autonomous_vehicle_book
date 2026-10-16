"""A missed lane refreshes diagnostics without creating a driving target."""
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from cv_bridge import CvBridge
from interfaces_pkg.msg import DetectionArray, Detection, Point2D
from camera_perception_pkg.lane_info_extractor_node import Yolov8InfoExtractor


def extractor():
    node = object.__new__(Yolov8InfoExtractor)
    node.cv_bridge = CvBridge()
    node.lane_class_name = 'lane2'
    node.smoothing_alpha = .4
    node.lane_target_ratio = .5
    node.prev_target_x = [310., 315., 320.]
    node.last_mask_shape = (480, 640)
    node.show_image = False
    images, lanes = [], []
    node.roi_image_publisher = SimpleNamespace(publish=images.append)
    node.publisher = SimpleNamespace(publish=lanes.append)
    node.get_logger = lambda: Mock()
    return node, images, lanes


def test_empty_detection_refreshes_blank_roi_without_lane_or_smoothing_changes():
    node, images, lanes = extractor()
    msg = DetectionArray()
    msg.header.stamp.sec = 12
    node.yolov8_detections_callback(msg)
    assert len(images) == 1 and not lanes
    assert images[0].header.stamp.sec == 12
    roi = node.cv_bridge.imgmsg_to_cv2(images[0])
    assert roi.shape == (180, 640) and not roi.any()
    assert node.prev_target_x == [310., 315., 320.]


def test_valid_lane_then_dropout_pumps_window_events_and_keeps_last_targets():
    node, images, lanes = extractor()
    detection = Detection()
    detection.class_name = 'lane2'
    detection.mask.height, detection.mask.width = 480, 640
    detection.mask.data = [Point2D(x=float(x), y=float(y)) for x,y in
                           [(270, 300), (370, 300), (420, 479), (230, 479)]]
    msg = DetectionArray(detections=[detection])
    node.yolov8_detections_callback(msg)
    assert len(lanes) == 1 and len(lanes[0].target_points) == 3
    assert np.any(node.cv_bridge.imgmsg_to_cv2(images[-1]))
    previous = list(node.prev_target_x)
    node.show_image = True
    with patch('cv2.imshow') as show, patch('cv2.waitKey') as events:
        node.yolov8_detections_callback(DetectionArray())
        assert show.call_count == 3
        events.assert_called_once_with(1)
    assert len(lanes) == 1 and node.prev_target_x == previous
    assert not np.any(node.cv_bridge.imgmsg_to_cv2(images[-1]))
