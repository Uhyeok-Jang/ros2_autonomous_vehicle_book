"""Camera frames must retain timestamps used to synchronize YOLO overlays."""
from types import SimpleNamespace

import numpy as np
from builtin_interfaces.msg import Time
from cv_bridge import CvBridge
from camera_perception_pkg.image_publisher_node import ImagePublisherNode


def test_camera_publishes_the_stamped_frame():
    node = object.__new__(ImagePublisherNode)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    node.data_source, node.logger = 'camera', False
    node.cap = SimpleNamespace(read=lambda: (True, frame))
    node.br = CvBridge()
    node.get_clock = lambda: SimpleNamespace(
        now=lambda: SimpleNamespace(to_msg=lambda: Time(sec=123, nanosec=456)))
    outputs = []
    node.publisher = SimpleNamespace(publish=outputs.append)
    node.timer_callback()
    assert len(outputs) == 1
    assert outputs[0].header.stamp == Time(sec=123, nanosec=456)
    assert outputs[0].header.frame_id == 'image_frame'
    assert outputs[0].height == 480 and outputs[0].width == 640
    assert outputs[0].encoding == 'bgr8'
    restored = node.br.imgmsg_to_cv2(outputs[0], desired_encoding='bgr8')
    np.testing.assert_array_equal(restored, frame)
