"""Validate overlay coordinates, empty-path clearing and unchanged inputs."""

from types import SimpleNamespace
from cv_bridge import CvBridge

import numpy as np

from debug_pkg.lane_tracking_render import (
    SOURCE, DESTINATION, PATH_COLOR, camera_points, render_views)
from debug_pkg.lane_tracking_visualizer_node import LaneTrackingVisualizer


def records():
    return [{'name': 'lane2', 'score': .82, 'width': 640, 'height': 480,
             'polygon': SOURCE.tolist(), 'bbox': (155, 313, 501, 476)}]


def test_roi_projection_uses_the_driving_crop_and_scales_with_camera_size():
    roi = DESTINATION.copy()
    roi[:, 1] -= 300
    before = roi.copy()
    np.testing.assert_allclose(camera_points(roi), SOURCE, atol=.001)
    np.testing.assert_allclose(camera_points(roi, 1280, 960), SOURCE * 2, atol=.002)
    np.testing.assert_array_equal(roi, before)


def test_both_views_show_lane_masks_and_the_actual_published_path():
    frame = np.full((480, 640, 3), 80, dtype=np.uint8)
    path = [[320, y] for y in range(20, 180)]
    targets = [[320, y] for y in range(20, 141, 20)]
    camera, roi = render_views(frame, records(), path, targets)
    assert camera.shape == (548, 640, 3) and roi.shape == (248, 640, 3)
    assert np.count_nonzero(np.all(roi[68:] == PATH_COLOR, axis=2)) > 100
    assert np.count_nonzero(np.all(camera[68:] == PATH_COLOR, axis=2)) > 20
    # Interior of the lane polygon becomes green, outside it stays gray.
    assert roi[68+90, 300, 1] > roi[68+90, 300, 2]
    np.testing.assert_array_equal(frame, 80)
    assert path[0] == [320, 20] and records()[0]['polygon'] == SOURCE.tolist()


def test_empty_path_clears_old_lines_on_the_next_render():
    frame = np.full((480, 640, 3), 80, dtype=np.uint8)
    camera, roi = render_views(frame, [], [[320, 20], [320, 140]])
    assert np.any(np.all(roi[68:] == PATH_COLOR, axis=2))
    camera, roi = render_views(frame, [], [], [])
    assert not np.any(np.all(roi[68:] == PATH_COLOR, axis=2))
    assert not np.any(np.all(camera[68:] == PATH_COLOR, axis=2))


def test_empty_or_nonfinite_paths_are_cleared_by_the_subscriber():
    node = object.__new__(LaneTrackingVisualizer)
    node.path_callback(SimpleNamespace(x_points=[320., 320.], y_points=[20., 140.]))
    assert node.path == [[320., 20.], [320., 140.]]
    for msg in [SimpleNamespace(x_points=[], y_points=[]),
                SimpleNamespace(x_points=[float('nan')], y_points=[20.]),
                SimpleNamespace(x_points=[320.], y_points=[])]:
        node.path_callback(msg)
        assert node.path == []


def test_no_extra_detection_threshold_or_history_is_applied():
    frame = np.full((480, 640, 3), 80, dtype=np.uint8)
    low_confidence = records()
    low_confidence[0]['score'] = .25
    _, roi = render_views(frame, low_confidence)
    assert roi[68+90, 300, 1] > roi[68+90, 300, 2]
    _, blank = render_views(frame, [])
    assert not np.array_equal(roi[68:], blank[68:])


def test_invalid_camera_encoding_does_not_terminate_the_visualizer():
    node = object.__new__(LaneTrackingVisualizer)
    node.bridge = CvBridge()
    image = node.bridge.cv2_to_imgmsg(np.zeros((480, 640, 3), dtype=np.uint8))
    node.frame_pair = (image, None, 0.)
    errors = []
    node.get_logger = lambda: SimpleNamespace(error=errors.append)
    node.render_callback()
    assert errors and '8UC3' in errors[0]
