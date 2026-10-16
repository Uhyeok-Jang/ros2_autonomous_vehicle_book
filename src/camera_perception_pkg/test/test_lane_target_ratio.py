"""The lane target can move right without changing masks or inventing targets."""
from types import SimpleNamespace

import numpy as np
import pytest
from interfaces_pkg.msg import TargetPoint

from camera_perception_pkg.lane_info_extractor_node import Yolov8InfoExtractor
from decision_making_pkg.motion_planner_node import MotionPlanningNode
from decision_making_pkg.path_planner_node import PathPlannerNode


def extractor(ratio):
    node = object.__new__(Yolov8InfoExtractor)
    node.lane_target_ratio = ratio
    node.smoothing_alpha = .4
    node.prev_target_x = [None] * 3
    return node


@pytest.mark.parametrize('ratio,expected', [(.4, 282.), (.5, 300.), (.6, 318.)])
def test_ratio_selects_position_within_lane_boundaries(ratio, expected):
    node = extractor(ratio)
    roi = np.zeros((180, 640), dtype=np.uint8)
    roi[:, 200:401] = 255
    original = roi.copy()
    assert node.estimate_target_x(roi, 55, 1) == pytest.approx(expected)
    assert np.array_equal(roi, original)


def test_right_shift_scales_with_detected_lane_width():
    narrow = np.zeros((180, 640), dtype=np.uint8)
    wide = narrow.copy()
    narrow[:, 200:401] = 255
    wide[:, 100:501] = 255
    narrow_target = extractor(.6).estimate_target_x(narrow, 55, 1)
    wide_target = extractor(.6).estimate_target_x(wide, 55, 1)
    assert narrow_target - 300 == pytest.approx(18)
    assert wide_target - 300 == pytest.approx(36)


def test_missing_band_keeps_last_biased_target_without_accumulating_shift():
    node = extractor(.6)
    roi = np.zeros((180, 640), dtype=np.uint8)
    roi[:, 200:401] = 255
    target = node.estimate_target_x(roi, 55, 1)
    roi[:] = 0
    for _ in range(30):
        assert node.estimate_target_x(roi, 55, 1) == target


def test_no_initial_lane_band_keeps_existing_center_fallback():
    roi = np.zeros((180, 640), dtype=np.uint8)
    assert extractor(.6).estimate_target_x(roi, 55, 1) == 320.


def test_right_target_produces_right_steering_and_preserves_vehicle_anchor():
    lane = extractor(.6)
    roi = np.zeros((180, 640), dtype=np.uint8)
    roi[:, 200:441] = 255
    planner = object.__new__(PathPlannerNode)
    planner.target_points = [
        TargetPoint(target_x=int(round(lane.estimate_target_x(roi, y, idx))), target_y=y)
        for idx, y in enumerate((5, 55, 105))
    ]
    planner.car_center_point = (320, 179)
    outputs = []
    planner.publisher = SimpleNamespace(publish=outputs.append)
    planner.get_logger = lambda: SimpleNamespace(info=lambda _: None)
    planner.plan_path()
    path = outputs[0]
    assert path.x_points[0] > 320
    assert (path.x_points[-1], path.y_points[-1]) == (320, 179)

    controller = object.__new__(MotionPlanningNode)
    controller.path_data = list(zip(path.x_points, path.y_points))
    controller.kp_heading, controller.steering_alpha = .07, .3
    controller.max_steering_step, controller.filtered_steering = 4., 0.
    for _ in range(15):
        command, slope, raw = controller.calculate_stable_steering()
    assert command > 0 and slope > 0 and raw > 0
