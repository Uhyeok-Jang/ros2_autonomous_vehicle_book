#!/bin/bash
# Select real-workspace packages even after simulation was sourced in this shell.
set -eo pipefail

task_workspace=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
unset AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH PYTHONPATH LD_LIBRARY_PATH
unset COLCON_CURRENT_PREFIX AMENT_CURRENT_PREFIX
source /opt/ros/humble/setup.bash
# Use only this workspace, without any underlay recorded by a previous build.
source "$task_workspace/install/local_setup.bash"

exec ros2 launch camera_perception_pkg driving_real.launch.py "$@"
