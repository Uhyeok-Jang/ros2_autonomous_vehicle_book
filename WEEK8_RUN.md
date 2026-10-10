# Week 8 lane-following snapshot

This branch records the current lane2 controller and perception/target-path overlays. The controller waits for its first valid lane path, then preserves that path and reduces PWM during lane2 loss or stale perception. Red-light and obstacle stop conditions remain in the controller.

Current launch values: `driving_speed=245`, `lane_loss_speed=180`, `lane_detection_timeout=0.2`, `kp_heading=0.09`, `steering_alpha=1.0`, `max_steering_step=4.0`, `timer=0.1`. Edit the motion-planner parameters in the launch file for tuning.

Build from the workspace root with ROS2 Humble and the required Python/ROS dependencies installed:

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install
./run_driving_real.sh
```

The runner selects this workspace without the other workspace's overlay.

The launch is `src/camera_perception_pkg/launch/driving_real.launch.py`. It uses USB camera 2 by default and the serial sender uses `/dev/ttyACM0` at 9600 baud. The Arduino source snapshot is `src/control/driving/driving.ino`; committing this file does not upload it to the board.

The model defaults to `~/yolo/models/lane_segmentation/best.pt`, which is outside this repository and must be supplied separately. Override the path without editing the controller:

```bash
./run_driving_real.sh yolo_model:=/absolute/path/to/best.pt
```

The equivalent model is committed as `src/simulation_pkg/simulation_pkg/data/sim.pt` in the simulation repository's same branch. Its SHA256 is `8444c0351c789cc137d8f410f81c8383a4834b801a64d1746d8de55ff2b09643`.

Validation: 34 behavior/image tests. A software build or test does not verify the firmware installed on a physical vehicle.
