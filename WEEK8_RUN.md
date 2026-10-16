# Week 8 lane-following snapshot

The real driving stack started from the simulation lane-following logic and now includes real-vehicle tuning. For the validated `fix/midterm` settings and run instructions, see [MIDTERM_RUN.md](MIDTERM_RUN.md). RGB/BGR camera input is normalized to BGR; YOLO inference explicitly uses FP32. Missing lane2 updates the diagnostic ROI while preserving the last path.

Launch values: `lane_target_ratio=0.6`, `driving_speed=255`, `lane_loss_speed=220`, `lane_detection_timeout=0.2`, `kp_heading=0.09`, `steering_alpha=0.4`, `max_steering_step=4.0`, `timer=0.05`. Detection thresholds: common=0.3, lane2=0.15. The controller waits for its first valid path, then immediately reduces PWM on a missing lane2 detection, retaining the last path. The timeout applies to stale detection/path updates. Obstacle stopping is enabled; the red-light should_stop assignment is commented out.

Default model: `model/v3.2_finetuned_corner4_with_v3.pt`, SHA256 `b34cb3d8e6633d873f982a73c12d46090c5260e145b780bc9383ad2e09067f86`. Both repositories contain all three named models. The compatibility best.pt/sim.pt copies also match v3.2. Original v3/v3.1 files are preserved. See `MODEL_COMPARE_2026-10-16.md` for experiment commands.

```bash
./run_model_compare.sh v3
./run_model_compare.sh v3.1
./run_model_compare.sh v3.2
```

Run one driving session at a time. The wrapper selects only model weights and forwards other launch arguments to `run_driving_real.sh`. The runner isolates the selected workspace's overlay. Existing installed source links reflect these updates without rebuilding.

Real camera defaults to USB2. Serial sender remains /dev/ttyACM0,9600baud. Camera acquisition, serial sender and Arduino firmware sources were preserved. No physical driving or flashing was performed during synchronization.
