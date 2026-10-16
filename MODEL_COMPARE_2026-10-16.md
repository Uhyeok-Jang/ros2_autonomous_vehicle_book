# 모델 비교 실험 — 2026-10-16

기본 모델은 v3.2다. 아래 비교 실행기는 모델 경로만 변경하며, 카메라/주행/인식 임계값 등은 같은 launch 설정을 사용한다.

|선택값|모델 파일|설명|
|---|---|---|
|v3|v3_good_corner4_issue.pt|기존 모델, epoch93|
|v3.1|v3.1_not_good_corner4_good.pt|COCO에서 새 학습한 모델, epoch74|
|v3.2|v3.2_finetuned_corner4_with_v3.pt|v3에서 전체 데이터로 튜닝, epoch29|

실행 중인 주행을 종료한 뒤 모델별로 하나씩 실행한다.

```bash
/home/autolab/ros2_ws/run_model_compare.sh v3
```

```bash
/home/autolab/ros2_ws/run_model_compare.sh v3.1
```

```bash
/home/autolab/ros2_ws/run_model_compare.sh v3.2
```

기존 실행 명령(`/home/autolab/ros2_ws/run_driving_real.sh`)은 기본 v3.2를 사용한다. 필요하면 비교 실행기 뒤에도 기존 launch 인자를 전달할 수 있다(예: `cam_num:=2`, `yolo_device:=cuda:0`).

현재 실차 설정: lane_target_ratio=.6, driving_speed=255, lane_loss_speed=220, lane_detection_timeout=.2초, kp_heading=.09, steering_alpha=.4, max_steering_step=4, timer=.05초. YOLO threshold=.3, lane_threshold=.15, FP32 추론. 상세 설정과 실행 방법은 `MIDTERM_RUN.md`를 참고한다. 모델별 비교 중에는 이 값을 고정하고 조명/카메라 설치/출발 위치도 가능한 한 같게 유지한다.

비교 기록은 4번 코너뿐 아니라 1·2번 코너 탈출, 3·5번 코너, 직진을 구분해 남긴다. lane2의 연속 미검출 시간과 차로 이탈/정지 여부를 확인한다. v3.2는 동일 test에서 4번 코너 lane2 AP가 개선됐지만 다른 구간 AP는 낮아졌으므로 실차에서도 같은 경향인지 확인한다.

실차 카메라/serial I/O와 시뮬레이터 전송 어댑터는 각 환경의 기존 코드를 유지한다. 실차 목표 위치와 파라미터는 이후 실차 테스트로 튜닝되어 시뮬레이션 설정과 다를 수 있다. 실차 구동/Arduino 업로드는 동기화 작업에서 실행하지 않았다. 인식 경로는 BGR 입력 정규화와 미검출 ROI 갱신을 사용한다. 설치 launch/Python 경로는 소스를 참조하므로 현재 설치에서는 재빌드가 필요 없다.
