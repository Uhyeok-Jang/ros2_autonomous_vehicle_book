"""Behavior checks for the selected lane-following-stability branch."""

from types import SimpleNamespace

import numpy as np
import pytest

from camera_perception_pkg.lane_info_extractor_node import Yolov8InfoExtractor
from decision_making_pkg.motion_planner_node import MotionPlanningNode
from decision_making_pkg.path_planner_node import PathPlannerNode
from serial_communication_pkg.lib.protocol_convert_func_lib import convert_serial_message
from interfaces_pkg.msg import TargetPoint


def test_lane_center_smoothing_and_missing_band_retention():
    node = object.__new__(Yolov8InfoExtractor)
    node.prev_target_x = [None] * 3
    node.smoothing_alpha = .4
    node.lane_target_ratio = .5
    roi = np.zeros((180, 640), dtype=np.uint8)
    roi[:, 200:301] = 255
    assert node.estimate_target_x(roi, 55, 1) == pytest.approx(250)
    roi[:] = 0
    roi[:, 300:401] = 255
    assert node.estimate_target_x(roi, 55, 1) == pytest.approx(290)
    roi[:] = 0
    assert node.estimate_target_x(roi, 55, 1) == pytest.approx(290)


def test_linear_path_preserves_anchors_without_overshoot():
    node = object.__new__(PathPlannerNode)
    node.target_points = [TargetPoint(target_x=x, target_y=y)
                          for x, y in [(260, 5), (380, 55), (260, 105)]]
    node.car_center_point = (320, 179)
    outputs = []
    node.publisher = SimpleNamespace(publish=outputs.append)
    node.get_logger = lambda: SimpleNamespace(info=lambda _: None)
    node.plan_path()
    path = outputs[0]
    assert len(path.x_points) == len(path.y_points) == 100
    assert min(path.x_points) >= 260 and max(path.x_points) <= 380
    assert path.x_points[-1] == 320 and path.y_points[-1] == 179
    assert node.target_points == []


def test_steering_reversal_is_limited_and_converges():
    node = object.__new__(MotionPlanningNode)
    node.kp_heading, node.steering_alpha = .20, .35
    node.max_steering_step, node.filtered_steering = 2., 0.
    node.path_data = [(260., float(y)) for y in range(90)] + [(320., 179.)]
    states = []
    for _ in range(10):
        command, _, _ = node.calculate_stable_steering()
        states.append(node.filtered_steering)
        assert -7 <= command <= 7
    assert states[-1] < 0
    node.path_data = [(380., float(y)) for y in range(90)] + [(320., 179.)]
    for _ in range(10):
        node.calculate_stable_steering()
        states.append(node.filtered_steering)
    assert states[-1] > 0
    assert np.max(np.abs(np.diff([0.] + states))) <= 2.


def test_motion_command_keeps_the_real_arduino_protocol():
    assert convert_serial_message(-7, 210, 210) == "s-7l210r210\n"
    assert convert_serial_message(0, 0, 0) == "s0l0r0\n"
