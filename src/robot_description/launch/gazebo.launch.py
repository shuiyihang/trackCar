
import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess,LogInfo
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition, UnlessCondition

def generate_launch_description():
    robot_name_in_model = 'fishbot'
    package_name = 'robot_description'
    urdf_name = "robot_base.urdf"

    ld = LaunchDescription()
    pkg_share = FindPackageShare(package=package_name).find(package_name) 
    urdf_model_path = os.path.join(pkg_share, f'urdf/{urdf_name}')

    gazebo_world_path = os.path.join(pkg_share, 'world/track_env_v2.world')

    use_gui_arg = DeclareLaunchArgument(
        'use_gui', default_value='true',
        description='Whether to start Gazebo with GUI (true) or headless (false)'
    )
    ld.add_action(use_gui_arg)
    use_gui = LaunchConfiguration('use_gui')
    
    ld.add_action(LogInfo(msg=['use_gui value: ', use_gui]))

    # Start Gazebo server
    start_gazebo_cmd =  ExecuteProcess(
        cmd=['gazebo', '--verbose','-s', 'libgazebo_ros_init.so', '-s', 'libgazebo_ros_factory.so',gazebo_world_path],
        output='screen',
        condition=IfCondition(use_gui)

        # 训练使用gzserver，需要GUI查看用gazebo
        # cmd=['gzserver', '--verbose','-s', 'libgazebo_ros_init.so', '-s', 'libgazebo_ros_factory.so', '--headless-rendering',gazebo_world_path],
        # output='screen'
        )
    
    start_gzserver_cmd =  ExecuteProcess(
        # 训练使用gzserver，需要GUI查看用gazebo
        cmd=['gzserver', '--verbose','-s', 'libgazebo_ros_init.so', '-s', 'libgazebo_ros_factory.so', '--headless-rendering',gazebo_world_path],
        output='screen',
        condition=UnlessCondition(use_gui)
        )
        
    # Launch the robot
    spawn_entity_cmd = Node(
        package='gazebo_ros', 
        executable='spawn_entity.py',
        arguments=['-entity', robot_name_in_model,  '-file', urdf_model_path ], output='screen')
    
    # Start Robot State publisher
    start_robot_state_publisher_cmd = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        arguments=[urdf_model_path]
    )

    # Launch RViz
    start_rviz_cmd = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        # arguments=['-d', default_rviz_config_path]
        )

    ld.add_action(start_gazebo_cmd)
    ld.add_action(start_gzserver_cmd)
    ld.add_action(spawn_entity_cmd)
    ld.add_action(start_robot_state_publisher_cmd)
    # ld.add_action(start_rviz_cmd)


    return ld
