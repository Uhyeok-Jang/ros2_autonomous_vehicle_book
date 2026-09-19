#!/usr/bin/env python3

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    default_model = os.path.expanduser(
        '~/yolo/models/lane_segmentation/best.pt'
    )

    return LaunchDescription([

        DeclareLaunchArgument(
            'cam_num',
            default_value='2',
            description='USB camera device number'
        ),

        DeclareLaunchArgument(
            'yolo_model',
            default_value=default_model,
            description='YOLO segmentation model path'
        ),

        DeclareLaunchArgument(
            'yolo_device',
            default_value='cuda:0',
            description='YOLO inference device: auto, cpu, or cuda:0'
        ),

        # USB Camera
        Node(
            package='camera_perception_pkg',
            executable='image_publisher_node',
            output='screen',
            parameters=[{
                'data_source': 'camera',
                'cam_num': LaunchConfiguration('cam_num'),
                'pub_topic': 'image_raw',
                'logger': True,
            }]
        ),

        # YOLOv8 segmentation - GPU
        Node(
            package='camera_perception_pkg',
            executable='yolov8_node',
            output='screen',
            parameters=[{
                'model': LaunchConfiguration('yolo_model'),
                'device': LaunchConfiguration('yolo_device'),
                'threshold': 0.3,
            }]
        ),

        # Traffic-light classification
        Node(
            package='camera_perception_pkg',
            executable='traffic_light_detector_node',
            output='screen',
            parameters=[{
                'sub_image_topic': 'image_raw',
            }]
        ),

        # Lane extraction
        Node(
            package='camera_perception_pkg',
            executable='lane_info_extractor_node',
            output='screen',
            parameters=[{
                'stop_zone_boundary_exclusion_px': 10,
                'stop_zone_dashed_margin_px': 4,
                'stop_zone_bridge_history_weight': 0.35,
                'stop_zone_bridge_max_slope': 0.35,
                'stop_zone_smoothing_alpha': 0.18,
                'stop_zone_max_target_step_px': 6.0,
            }]
        ),

        # Path planning
        Node(
            package='decision_making_pkg',
            executable='path_planner_node',
            output='screen'
        ),

        # Motion planning
        Node(
            package='decision_making_pkg',
            executable='motion_planner_node',
            output='screen',
            parameters=[{
                'stop_zone_speed': 120,
                'stop_zone_speed_hold_sec': 1.2,
                'speed_recovery_step': 10,
            }]
        ),

        # Arduino serial output
        Node(
            package='serial_communication_pkg',
            executable='serial_sender_node',
            output='screen'
        ),

    ])
