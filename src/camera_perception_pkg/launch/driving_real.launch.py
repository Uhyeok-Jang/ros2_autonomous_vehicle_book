#!/usr/bin/env python3

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    default_model = os.path.expanduser(
        '~/yolo/runs/segment/lane_v2_hard_aug/weights/best.pt'
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
                'device': 'cuda:0',
                'threshold': 0.3,
            }]
        ),

        # Lane extraction
        Node(
            package='camera_perception_pkg',
            executable='lane_info_extractor_node',
            output='screen'
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
            output='screen'
        ),

        # Arduino serial output
        Node(
            package='serial_communication_pkg',
            executable='serial_sender_node',
            output='screen'
        ),

    ])
