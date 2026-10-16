"""Start both motors only after a usable camera/YOLO path has arrived."""
from types import SimpleNamespace

import numpy as np
import pytest
from interfaces_pkg.msg import Detection, DetectionArray, PathPlanningResult
from std_msgs.msg import Bool, String
from decision_making_pkg.motion_planner_node import MotionPlanningNode


@pytest.fixture
def controller(monkeypatch):
    monkeypatch.setattr('decision_making_pkg.motion_planner_node.time.monotonic', lambda: 100.)
    node = object.__new__(MotionPlanningNode)
    node.startup_ready, node.path_data = False, None
    node.lane_detected = None
    node.last_detection_time = node.last_valid_path_time = None
    node.detection_data = node.traffic_light_data = node.lidar_data = None
    node.stop_reason = node.stop_started_time = None
    node.steering_command = node.left_speed_command = node.right_speed_command = 0
    node.filtered_steering = 0.
    node.kp_heading, node.steering_alpha = .09, .30
    node.max_steering_step, node.driving_speed = 2., 230
    node.lane_loss_speed, node.lane_detection_timeout = 140, .5
    outputs = []
    node.publisher = SimpleNamespace(publish=outputs.append)
    node.get_logger = lambda: SimpleNamespace(info=lambda _: None, warning=lambda _: None)
    return node, outputs


def valid_path():
    return PathPlanningResult(x_points=np.linspace(120., 320., 100).tolist(),
                              y_points=np.linspace(5., 179., 100).tolist())


def test_startup_waits_indefinitely_instead_of_using_a_fixed_delay(controller):
    node, outputs = controller
    node.detection_callback(DetectionArray())
    for _ in range(50):
        node.timer_callback()
    assert all((m.steering, m.left_speed, m.right_speed) == (0, 0, 0) for m in outputs)
    assert not node.startup_ready


@pytest.mark.parametrize('path', [
    PathPlanningResult(),
    PathPlanningResult(x_points=[320.]*9, y_points=list(map(float, range(9)))),
    PathPlanningResult(x_points=[320.]*10, y_points=[5.]*9),
    PathPlanningResult(x_points=[float('nan')]*10, y_points=list(map(float, range(10)))),
    PathPlanningResult(x_points=[320.]*10, y_points=[float('inf')]*10),
    PathPlanningResult(x_points=[320.]*10, y_points=[5.]*10),
])
def test_invalid_first_path_does_not_release_motors(controller, path):
    node, outputs = controller
    node.path_callback(path)
    node.timer_callback()
    assert not node.startup_ready
    assert outputs[-1].left_speed == outputs[-1].right_speed == 0


def test_first_driving_message_already_contains_path_based_steering(controller):
    node, outputs = controller
    node.timer_callback()
    node.path_callback(valid_path())
    node.timer_callback()
    assert node.startup_ready
    assert outputs[-1].left_speed == outputs[-1].right_speed == 230
    assert outputs[-1].steering < 0


def test_brief_dropout_preserves_the_ready_latch_and_last_path(controller):
    node, outputs = controller
    node.path_callback(valid_path())
    node.timer_callback()
    previous_path = list(node.path_data)
    node.detection_callback(DetectionArray())
    node.path_callback(PathPlanningResult())
    for _ in range(10):
        node.timer_callback()
    assert node.startup_ready and node.path_data == previous_path
    assert outputs[0].left_speed == 230
    assert all(m.left_speed == m.right_speed == 140 and m.steering < 0 for m in outputs[1:])


def test_obstacle_still_prevents_driving_after_camera_is_ready(controller):
    node, outputs = controller
    node.lidar_callback(Bool(data=True))
    node.path_callback(valid_path())
    node.timer_callback()
    assert node.startup_ready and outputs[-1].left_speed == 0
    node.lidar_callback(Bool(data=False))
    node.timer_callback()
    assert outputs[-1].left_speed == 230


def test_disabled_red_light_stop_keeps_driving_after_camera_is_ready(controller):
    node, outputs = controller
    signal = Detection(class_name='traffic_light')
    signal.bbox.center.position.y, signal.bbox.size.y = 100., 20.
    node.detection_callback(DetectionArray(detections=[signal]))
    node.traffic_light_callback(String(data='Red'))
    node.path_callback(valid_path())
    node.timer_callback()
    # The validated real configuration leaves should_stop=True commented out.
    assert outputs[-1].left_speed == outputs[-1].right_speed == node.lane_loss_speed
