"""Low-confidence lane2 may pass without admitting other weak detections."""
from types import SimpleNamespace

import numpy as np
import torch
from cv_bridge import CvBridge
from ultralytics.engine.results import Results
from camera_perception_pkg.yolov8_node import Yolov8Node


def run_callback(scores):
    names = {0: 'lane2', 1: 'traffic_light', 2: 'dashed_line'}
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    boxes = np.array([[10+i*15, 10, 20+i*15, 50, score, cls]
                      for i, (score, cls) in enumerate(scores)], dtype=np.float32)
    masks = np.zeros((len(scores), 80, 100), dtype=np.float32)
    for i in range(len(scores)):
        masks[i, 10:50, 10+i*15:20+i*15] = 1
    results = Results(frame, 'test.png', names,
                      boxes=torch.from_numpy(boxes), masks=torch.from_numpy(masks))
    node = object.__new__(Yolov8Node)
    node.enable, node.device = True, 'cpu'
    node.threshold, node.lane_threshold = .3, .15
    calls, outputs = [], []
    node.yolo = SimpleNamespace(names=names, predict=lambda **kwargs: (calls.append(kwargs) or [results]))
    node.cv_bridge = CvBridge()
    node._pub = SimpleNamespace(publish=outputs.append)
    image = node.cv_bridge.cv2_to_imgmsg(frame, encoding='bgr8')
    image.header.frame_id = 'camera_frame'
    node.image_cb(image)
    return outputs[0], calls[0]


def test_lane_threshold_preserves_box_mask_alignment_and_other_object_threshold():
    msg, inference = run_callback([(.18, 0), (.19, 1), (.91, 2), (.14, 0)])
    assert inference['conf'] == .15
    assert [d.class_name for d in msg.detections] == ['lane2', 'dashed_line']
    assert msg.header.frame_id == 'camera_frame'
    first_x = min(p.x for p in msg.detections[0].mask.data)
    second_x = min(p.x for p in msg.detections[1].mask.data)
    assert first_x == 10 and second_x == 40


def test_all_filtered_detections_publish_an_empty_message():
    msg, _ = run_callback([(.14, 0), (.19, 1)])
    assert msg.detections == []
