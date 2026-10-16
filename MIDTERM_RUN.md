# 실차 중간평가 주행 설정 — 2026-10-16

`fix/midterm`은 실차 테스트로 확인한 로직·모델·파라미터를 보존한다.
파라미터는 `src/camera_perception_pkg/launch/driving_real.launch.py`에서 조정한다.
저장 후 실행을 종료하고 재시작해야 적용된다.

| 파라미터 | 값 | 의미 |
|---|---|---|
| lane_target_ratio | 0.6 | 인식된 좌우 경계 사이의 목표 위치. 0.5는 중앙, 클수록 오른쪽 |
| driving_speed | 255 | 정상 주행 PWM |
| lane_loss_speed | 220 | lane2 미검출 중 PWM |
| lane_detection_timeout | 0.2 | 인식 또는 유효 경로 갱신이 중단된 경우 감속 기준 시간(초) |
| kp_heading | 0.09 | 경로 방향 오차에 대한 조향 게인 |
| steering_alpha | 0.4 | 조향 EMA 계수 |
| max_steering_step | 4.0 | 제어 주기당 조향 변화량 제한 |
| timer | 0.05 | 제어 주기(초), 20 Hz |

YOLO는 `cuda:0`, FP32 추론을 사용한다. 일반 검출 임계값은 0.3, lane2 임계값은 0.15다.
카메라 기본 번호는 2이며, 시리얼 기본 연결은 `/dev/ttyACM0`, 9600 baud다.

기본 모델은 `model/v3.2_finetuned_corner4_with_v3.pt`다.
SHA256: `b34cb3d8e6633d873f982a73c12d46090c5260e145b780bc9383ad2e09067f86`.
`model/best.pt`도 동일한 모델이다.
비교용 v3, v3.1 모델과 실행 방법은 `MODEL_COMPARE_2026-10-16.md`에 기록했다.

첫 유효 경로가 들어오기 전에는 정지한다. 경로를 한 번 받으면 출발 허용 상태를 유지한다.
주행 중 lane2 미검출 메시지가 오면 즉시 220으로 감속하고 마지막 경로를 유지한다.
0.2초는 미검출 메시지의 유예 시간이 아니라 인식/경로 갱신 중단을 판단하는 시간이다.
LiDAR 장애물 입력에 의한 정지는 활성 상태다. 빨간 신호등의 `should_stop = True` 줄은
주석 처리되어 있어 신호등 정지는 비활성 상태다. 정지 사유와 해제 시간은 `STOP`, `STOP_END` 로그로 기록한다.

최초 설치 시 ROS 2 Humble 환경에서 빌드한다.

```bash
cd /home/autolab/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
```

카메라와 모든 주행 노드가 이 PC에서 실행되는 실차 구성의 실행 명령:

```bash
ROS_LOCALHOST_ONLY=1 /home/autolab/ros2_ws/run_driving_real.sh
```

`ROS_LOCALHOST_ONLY=1`은 같은 네트워크의 다른 ROS 실행에서 카메라·경로·모터 명령이
섞이는 것을 방지한다. 이 값은 위 명령에만 지정하며 runner가 강제하지는 않는다.
runner는 실차 workspace의 설치 환경을 선택한다. 모델만 바꾸어 비교하려면:

```bash
ROS_LOCALHOST_ONLY=1 /home/autolab/ros2_ws/run_model_compare.sh v3.2
```

현재 symlink 설치에서는 launch/Python 소스 수정 후 재빌드 없이 재시작으로 반영된다.
새로 clone한 환경에서는 먼저 빌드해야 한다. Arduino 소스의 포텐셔미터 보정값은
왼쪽 158, 오른쪽 76이며, 커밋 작업에서는 펌웨어 업로드나 실차 실행을 하지 않았다.
