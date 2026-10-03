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

from decision_making_pkg.motion_planner_node import MotionPlanningNode


def make_node():
    node = object.__new__(MotionPlanningNode)
    node.steering_delay_sec = 0.2
    node.steering_delay_rearm_sec = 0.6
    node.steering_delay_bypass_heading_deg = 18.0
    node.steering_delay_bypass_lateral_px = 48.0
    node.curve_entry_time_ns = None
    node.curve_clear_start_time_ns = None
    node.last_steering_delay_remaining = 0.0
    node.test_now_ns = 0
    node.now_ns = lambda: node.test_now_ns
    return node


def test_short_straight_inside_compound_curve_does_not_rearm_delay():
    node = make_node()

    assert node.steering_delay_active(10.0, 0.0)

    node.test_now_ns = 300_000_000
    assert not node.steering_delay_active(10.0, 0.0)

    node.test_now_ns = 400_000_000
    assert not node.steering_delay_active(0.0, 0.0)

    node.test_now_ns = 550_000_000
    assert not node.steering_delay_active(-10.0, 0.0)
    assert node.last_steering_delay_remaining == 0.0


def test_sustained_straight_rearms_delay_for_next_curve():
    node = make_node()

    assert node.steering_delay_active(10.0, 0.0)
    node.test_now_ns = 300_000_000
    assert not node.steering_delay_active(0.0, 0.0)
    node.test_now_ns = 950_000_000
    assert not node.steering_delay_active(0.0, 0.0)
    assert node.curve_entry_time_ns is None

    node.test_now_ns = 1_000_000_000
    assert node.steering_delay_active(10.0, 0.0)


def test_sharp_curve_bypasses_entry_delay():
    node = make_node()

    assert not node.steering_delay_active(18.0, 0.0)
    assert node.last_steering_delay_remaining == 0.0


def test_stop_zone_bypasses_entry_delay():
    node = make_node()

    assert not node.steering_delay_active(
        10.0,
        0.0,
        bypass_delay=True,
    )
    assert node.last_steering_delay_remaining == 0.0
