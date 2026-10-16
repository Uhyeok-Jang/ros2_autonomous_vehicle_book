import pytest
"""Reduced PWM on lane loss must retain steering and obey stop conditions."""
from interfaces_pkg.msg import Detection, DetectionArray, Point2D
from std_msgs.msg import Bool, String

from test_startup_sync import controller, valid_path


def lane_detection():
    lane = Detection(class_name='lane2')
    lane.mask.width, lane.mask.height = 640, 480
    lane.mask.data = [Point2D(x=200., y=300.), Point2D(x=400., y=300.),
                      Point2D(x=400., y=479.)]
    return DetectionArray(detections=[lane])


@pytest.mark.parametrize('missing', [
    DetectionArray(),
    DetectionArray(detections=[Detection(class_name='solid_line')]),
    DetectionArray(detections=[Detection(class_name='lane2')]),
])
def test_loss_reduces_both_motors_and_recovery_restores_speed(controller, missing):
    node, outputs = controller
    node.detection_callback(lane_detection())
    node.path_callback(valid_path())
    node.timer_callback()
    assert outputs[-1].left_speed == outputs[-1].right_speed == 230
    last_path = list(node.path_data)
    node.detection_callback(missing)
    node.timer_callback()
    assert outputs[-1].left_speed == outputs[-1].right_speed == 140
    assert outputs[-1].steering < 0 and node.path_data == last_path
    assert node.startup_ready
    node.detection_callback(lane_detection())
    node.path_callback(valid_path())
    node.timer_callback()
    assert outputs[-1].left_speed == outputs[-1].right_speed == 230


def test_silent_yolo_or_stalled_path_slows_after_configured_timeout(controller, monkeypatch):
    node, outputs = controller
    node.lane_detection_timeout = .75
    node.detection_callback(lane_detection())
    node.path_callback(valid_path())
    monkeypatch.setattr('decision_making_pkg.motion_planner_node.time.monotonic', lambda: 100.74)
    node.timer_callback()
    assert outputs[-1].left_speed == 230
    monkeypatch.setattr('decision_making_pkg.motion_planner_node.time.monotonic', lambda: 100.76)
    node.timer_callback()
    assert outputs[-1].left_speed == 140 and outputs[-1].steering < 0
    # Detections alone cannot restore normal speed when valid paths have stalled.
    node.detection_callback(lane_detection())
    node.timer_callback()
    assert outputs[-1].left_speed == 140
    node.path_callback(valid_path())
    node.timer_callback()
    assert outputs[-1].left_speed == 230


@pytest.mark.parametrize('speed,expected', [(85, 85), (0, 0), (-1, 0), (255, 230), (300, 230)])
def test_reduced_speed_is_configurable_and_never_exceeds_normal_pwm(controller, speed, expected):
    node, outputs = controller
    node.lane_loss_speed = speed
    node.path_callback(valid_path())
    node.detection_callback(DetectionArray())
    node.timer_callback()
    assert outputs[-1].left_speed == outputs[-1].right_speed == expected


def test_lidar_stops_but_disabled_red_light_stop_keeps_lane_loss_speed(controller):
    node, outputs = controller
    node.path_callback(valid_path())
    node.detection_callback(DetectionArray())
    node.lidar_callback(Bool(data=True))
    node.timer_callback()
    assert outputs[-1].left_speed == outputs[-1].right_speed == 0
    node.lidar_callback(Bool(data=False))
    signal = Detection(class_name='traffic_light')
    signal.bbox.center.position.y, signal.bbox.size.y = 100., 20.
    node.detection_callback(DetectionArray(detections=[signal]))
    node.traffic_light_callback(String(data='Red'))
    node.timer_callback()
    # Red-light stopping is disabled in this real-vehicle snapshot.
    assert outputs[-1].left_speed == outputs[-1].right_speed == 140
    node.traffic_light_callback(String(data='Green'))
    node.timer_callback()
    assert outputs[-1].left_speed == outputs[-1].right_speed == 140
