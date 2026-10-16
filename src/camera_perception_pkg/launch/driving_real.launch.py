#!/usr/bin/env python3
"""Run the validated simulation lane-following logic with real camera/serial I/O."""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_model = os.path.expanduser('~/ros2_ws/model/v3.2_finetuned_corner4_with_v3.pt')
    return LaunchDescription([
        DeclareLaunchArgument('cam_num', default_value='2',
                              description='USB camera device number'),
        DeclareLaunchArgument('yolo_model', default_value=default_model,
                              description='YOLO segmentation model path'),
        DeclareLaunchArgument('yolo_device', default_value='cuda:0',
                              description='YOLO inference device: cpu or cuda:0'),
        Node(
            package='camera_perception_pkg', executable='image_publisher_node',
            output='screen',
            parameters=[{
                'data_source': 'camera', 'cam_num': LaunchConfiguration('cam_num'),
                'pub_topic': 'image_raw', 'logger': True,
            }]
        ),
        Node(
            package='camera_perception_pkg', executable='yolov8_node',
            output='screen',
            parameters=[{
                'model': LaunchConfiguration('yolo_model'),
                'device': LaunchConfiguration('yolo_device'), 'threshold': 0.3,
                # Real track glare lowers lane2 confidence; other classes stay at 0.3.
                'lane_threshold': 0.15,
            }],
            remappings=[('camera/image_raw', 'image_raw')]
        ),
        Node(
            package='camera_perception_pkg', executable='traffic_light_detector_node',
            output='screen', parameters=[{'sub_image_topic': 'image_raw'}]
        ),
        Node(
            package='camera_perception_pkg', executable='lane_info_extractor_node',
            output='screen',
            parameters=[{
                'lane_target_ratio': 0.6,  # 0.5=차선 중앙, 클수록 오른쪽 목표 (0~1)
            }]
        ),
        Node(
            package='decision_making_pkg', executable='path_planner_node',
            output='screen'
        ),
        Node(
            package='decision_making_pkg', executable='motion_planner_node',
            output='screen',
            # Real vehicle speed and steering settings.
            # Speed is PWM and steering is an integer stage (-7..7).
            parameters=[{
                'driving_speed': 255,
                'lane_loss_speed': 220,          # lane2 누락 시 감속 PWM (0~255)
                'lane_detection_timeout': 0.2,  # 인식/경로 갱신이 없으면 감속 (초)
                'kp_heading': 0.09,
                'steering_alpha': 0.4,
                'max_steering_step': 4.0,
                'timer': 0.05,
            }]
        ),
        Node(
            package='debug_pkg', executable='yolov8_visualizer_node',
            output='screen', remappings=[('camera/image_raw', 'image_raw')]
        ),
        Node(
            package='debug_pkg', executable='path_visualizer_node', output='screen'
        ),
        Node(
            package='debug_pkg', executable='lane_tracking_visualizer_node',
            output='screen'
        ),
        Node(
            package='serial_communication_pkg', executable='serial_sender_node',
            output='screen'
        ),
    ])
