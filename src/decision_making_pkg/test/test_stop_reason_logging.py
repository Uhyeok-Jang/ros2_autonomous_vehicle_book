"""Stop diagnostics must describe emitted commands without changing control."""
from types import SimpleNamespace

import pytest
from std_msgs.msg import Bool, String

from test_startup_sync import controller, valid_path


@pytest.fixture
def logged_controller(controller):
    node, outputs = controller
    records = []
    node.get_logger = lambda: SimpleNamespace(
        info=lambda text: records.append(('INFO', text)),
        warning=lambda text: records.append(('WARN', text)),
    )
    return node, outputs, records


def stop_records(records):
    return [(level, text) for level, text in records if text.startswith('STOP ')]


def test_startup_logged_once_and_resume_includes_duration(logged_controller, monkeypatch):
    node, outputs, records = logged_controller
    for _ in range(5):
        node.timer_callback()
    assert len(stop_records(records)) == 1
    assert stop_records(records)[0][0] == 'INFO'
    assert 'reason=startup_wait' in stop_records(records)[0][1]
    assert all(msg.left_speed == msg.right_speed == 0 for msg in outputs)

    monkeypatch.setattr('decision_making_pkg.motion_planner_node.time.monotonic', lambda: 100.35)
    node.path_callback(valid_path())
    node.timer_callback()
    endings = [text for _, text in records if text.startswith('STOP_END ')]
    assert len(endings) == 1
    assert 'duration=0.350s, next_reason=none' in endings[0]
    assert node.stop_reason is None
    assert outputs[-1].left_speed == outputs[-1].right_speed == 230
    node.timer_callback()
    assert len([text for _, text in records if text.startswith('STOP_END ')]) == 1


def test_lidar_stop_is_logged_once_with_input_snapshot(logged_controller):
    node, outputs, records = logged_controller
    node.path_callback(valid_path())
    node.traffic_light_callback(String(data='Red'))
    node.lidar_callback(Bool(data=True))
    for _ in range(5):
        node.timer_callback()
    assert len(stop_records(records)) == 1
    level, message = stop_records(records)[0]
    assert level == 'WARN'
    assert 'reason=lidar_obstacle' in message
    assert 'traffic_light=Red, lidar_obstacle=True' in message
    assert all(msg.left_speed == msg.right_speed == 0 for msg in outputs)
    node.lidar_callback(Bool(data=False))
    node.timer_callback()
    assert node.stop_reason is None
    assert outputs[-1].left_speed == outputs[-1].right_speed == 230


@pytest.mark.parametrize('normal,lost,lane_detected,reason', [
    (0, 140, True, 'driving_speed_zero'),
    (-1, 140, False, 'driving_speed_zero'),
    (230, 0, False, 'lane_loss_speed_zero'),
    (230, -1, False, 'lane_loss_speed_zero'),
])
def test_zero_speed_parameters_have_distinct_reasons(
        logged_controller, normal, lost, lane_detected, reason):
    node, outputs, records = logged_controller
    node.path_callback(valid_path())
    node.driving_speed, node.lane_loss_speed = normal, lost
    node.lane_detected = lane_detected
    node.timer_callback()
    assert f'reason={reason}' in stop_records(records)[0][1]
    assert outputs[-1].left_speed == outputs[-1].right_speed == 0


def test_changing_stop_reason_does_not_report_a_driving_resume(logged_controller):
    node, outputs, records = logged_controller
    node.timer_callback()
    node.path_callback(valid_path())
    node.lidar_callback(Bool(data=True))
    node.timer_callback()
    assert node.stop_reason == 'lidar_obstacle'
    assert len(stop_records(records)) == 2
    endings = [text for _, text in records if text.startswith('STOP_END ')]
    assert len(endings) == 1
    assert 'next_reason=lidar_obstacle' in endings[0]
    assert all(msg.left_speed == msg.right_speed == 0 for msg in outputs)
