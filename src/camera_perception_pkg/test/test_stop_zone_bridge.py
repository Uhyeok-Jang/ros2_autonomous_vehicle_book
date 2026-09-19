# Copyright 2026 Uhyeok-Jang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cv2
import numpy as np

from camera_perception_pkg.lane_info_extractor_node import Yolov8InfoExtractor


def make_split_stop_zone():
    stop_zone = np.zeros((180, 640), dtype=np.uint8)
    stop_zone[20:81, 100:501] = 255
    dashed = np.zeros_like(stop_zone)
    dashed[15:86, 249:252] = 255
    return stop_zone, dashed


def test_dashed_line_recovers_only_right_lane2_part_of_stop_zone():
    stop_zone, dashed = make_split_stop_zone()

    recovered = Yolov8InfoExtractor.build_lane2_from_stop_zone(
        stop_zone,
        dashed,
        reference_centers=(375.0,) * 7,
        target_y_values=(20, 40, 60, 80, 100, 120, 140),
        dashed_margin_px=4,
    )

    assert recovered[40, 400] == 255
    assert recovered[40, 150] == 0
    assert recovered[40, 250] == 0


def test_dashed_line_recovers_only_left_lane2_part_of_stop_zone():
    stop_zone, dashed = make_split_stop_zone()

    recovered = Yolov8InfoExtractor.build_lane2_from_stop_zone(
        stop_zone,
        dashed,
        reference_centers=(175.0,) * 7,
        target_y_values=(20, 40, 60, 80, 100, 120, 140),
        dashed_margin_px=4,
    )

    assert recovered[40, 150] == 255
    assert recovered[40, 400] == 0
    assert recovered[40, 250] == 0


def test_stop_zone_bridge_extends_only_missing_lane_rows():
    target_y = (20, 40, 60, 80, 100, 120, 140)
    measured = (None, None, None, 310.0, 312.0, 314.0, 316.0)
    previous = (304.0, 306.0, 308.0, 310.0, 312.0, 314.0, 316.0)

    bridge = Yolov8InfoExtractor.build_stop_zone_bridge(
        measured,
        previous,
        target_y,
        history_weight=0.0,
        max_slope=0.35,
    )

    np.testing.assert_allclose(bridge[:3], [304.0, 306.0, 308.0])
    assert bridge[3:] == [None, None, None, None]


def test_stop_zone_bridge_blends_prediction_with_previous_path():
    bridge = Yolov8InfoExtractor.build_stop_zone_bridge(
        measured_centers=(None, 120.0, 124.0),
        previous_targets=(100.0, 110.0, 120.0),
        target_y_values=(0, 10, 20),
        history_weight=0.5,
        max_slope=1.0,
    )

    assert np.isclose(bridge[0], 108.0)
    assert bridge[1:] == [None, None]


def test_stop_zone_bridge_holds_previous_path_when_lane_is_fully_hidden():
    previous = (300.0, 301.0, 302.0)

    bridge = Yolov8InfoExtractor.build_stop_zone_bridge(
        measured_centers=(None, None, None),
        previous_targets=previous,
        target_y_values=(20, 40, 60),
        history_weight=0.35,
        max_slope=0.35,
    )

    assert bridge == list(previous)


def test_stop_zone_pixels_are_removed_from_boundary_mask():
    node = object.__new__(Yolov8InfoExtractor)
    node.stop_zone_boundary_exclusion_px = 1

    boundary = np.full((7, 7), 255, dtype=np.uint8)
    stop_zone = np.zeros((7, 7), dtype=np.uint8)
    stop_zone[3, 3] = 255

    filtered = node.exclude_stop_zone_boundaries(boundary, stop_zone)

    assert cv2.countNonZero(filtered[2:5, 2:5]) == 0
    assert filtered[0, 0] == 255


def test_stop_zone_uses_lane_center_instead_of_avoiding_zone():
    node = object.__new__(Yolov8InfoExtractor)
    node.stop_zone_visible = True
    node.prev_target_x = [280.0] * 7
    node.boundary_margin_px = 24.0
    node.max_target_step_px = 20.0
    node.smoothing_alpha = 0.35
    node.stop_zone_max_target_step_px = 6.0
    node.stop_zone_smoothing_alpha = 0.18

    drivable = np.zeros((180, 640), dtype=np.uint8)
    boundaries = np.zeros_like(drivable)
    boundaries[:, 99:102] = 255
    boundaries[:, 499:502] = 255

    target_x, _, _, mode = node.estimate_target_x(
        drivable,
        boundaries,
        target_y=20,
        target_idx=0,
        base_x=280.0,
    )

    assert target_x == 280.0
    assert mode == "stop_zone_lane_follow"
