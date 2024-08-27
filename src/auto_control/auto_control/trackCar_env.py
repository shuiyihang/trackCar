
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry

from cv_bridge import CvBridge          # ROS与OpenCV图像转换类
import cv2                              # Opencv图像处理库

from rclpy.executors import SingleThreadedExecutor
from geometry_msgs.msg import Twist

from std_srvs.srv import Empty

from threading import Thread

import numpy as np

from time import sleep

import math

import logging

import time

TIME_DELTA = 0.1



class TrackCarEnv():
    def __init__(self) -> None:
        self.interact_node = rclpy.create_node("track_env_v1")

        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.interact_node)

        self.pub_cmd_vel = self.interact_node.create_publisher(Twist,"cmd_vel",1)

        self.sub_camera = self.interact_node.create_subscription(Image,'camera_node/image_raw',self.camera_data_callback,10)

        self.sub_car_speed = self.interact_node.create_subscription(Odometry,'odom',self.speed_data_callback,10)

        # 开关引擎
        self.pause_proxy = self.interact_node.create_client(Empty,"pause_physics")
        self.unpause_proxy = self.interact_node.create_client(Empty,"unpause_physics")

        self.reset_world = self.interact_node.create_client(Empty,"reset_world")
        self.reset_simulation = self.interact_node.create_client(Empty,"reset_simulation")

        # camera 转 模拟5个光电传感器数据
        self.sensor_data = np.zeros(5,dtype=int)
        # vel_x,ang_z
        self.vel_x = 0
        self.ang_z = 0

        self.done = False

        self.middle = -1

        # 轮距
        self.Wheel_L = 0.126

        self.v_left = 0
        self.v_right = 0

        self.bridge = CvBridge()

        self.step_nums = 0

        # 配置日志
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            filename='app.log',
            filemode='w'
        )

        self.logger = logging.getLogger('my_logger')

    
        self.t = Thread(target=self.executor.spin)
        self.t.start()
    
    def cleanup(self):
        self.t.join()

    def camera_data_callback(self,msg):

        cv_img = self.bridge.imgmsg_to_cv2(msg,'mono8')

        flat_img = cv_img.flatten()

        max_val = int(flat_img.max())
        min_val = int(flat_img.min())

        # 向下取整
        threshold = 100 # math.floor((max_val + min_val)/2)


        pixel_nums = flat_img.size
        left_edge = -1
        right_edge = -1

        for i in range(pixel_nums-3):
            if flat_img[i] > threshold and flat_img[i+1] > threshold and flat_img[i+2] <= threshold and flat_img[i+3] <= threshold:
                left_edge = i+1
                break
        
        for i in reversed(range(pixel_nums-3)):
            if flat_img[i] <= threshold and flat_img[i+1] <= threshold and flat_img[i+2] > threshold and flat_img[i+3] > threshold:
                right_edge = i+1
                break
        
        middle = -1
        
        if left_edge == -1 and right_edge == -1:
            # 完全丢线
            pass
        elif left_edge == -1:
            # 左丢线
            middle = right_edge//2
        elif right_edge == -1:
            middle = (left_edge+pixel_nums)//2
        else:
            middle = (left_edge+right_edge)//2

        self.middle = middle


        # zones = [8, 32, 56, 80, 104]
        # result = np.zeros(5,dtype=int)

        # for i in range(len(zones)):
        #     black_cnt = 0
        #     for j in range(16):
        #         if flat_img[zones[i] + j] < threshold:
        #             black_cnt += 1

        #     result[i] = black_cnt

        # self.sensor_data = result





        # self.logger.info("flat_img:\n")
        # self.logger.info("{}\n".format(flat_img))
        # self.logger.info("middle:\n")
        # self.logger.info("{}\n".format(self.middle))
        # self.logger.info("sensor data:\n")
        # self.logger.info("{}\n\n".format(self.sensor_data))



    def speed_data_callback(self,msg):
        
        self.vel_x = msg.twist.twist.linear.x
        self.ang_z = msg.twist.twist.angular.z

        # 计算左右轮速度
        self.v_left = self.vel_x - (self.Wheel_L/2)*self.ang_z
        self.v_right = self.vel_x + (self.Wheel_L/2)*self.ang_z

        # self.logger.info("vel_x: {} ang_z: {}".format(self.vel_x,self.ang_z))

    def stop_action(self):
        vel_cmd = Twist()
        vel_cmd.linear.x = 0.0
        vel_cmd.angular.z = 0.0
        self.pub_cmd_vel.publish(vel_cmd)


    # 与中线偏离不多，奖励0
    # 偏离较多，奖励-10
    # 完全丢线，奖励-100，结束
    def get_reward(self):
        done = False

        offset = abs(self.middle-64)
        
        reward = 1 - offset/32
        if self.middle == -1:
            done = True
            reward = -1
    
        return reward,done

    def step(self,action):
        vel_cmd = Twist()
        vel_cmd.linear.x = 0.1
        vel_cmd.angular.z = action[0]
        self.pub_cmd_vel.publish(vel_cmd)

        while not self.unpause_proxy.wait_for_service(timeout_sec=1.0):
            print("unpause physics service is not available...")
        self.unpause_proxy.call_async(Empty.Request())

        sleep(TIME_DELTA)

        while not self.pause_proxy.wait_for_service(timeout_sec=1.0):
            print("pause physics service is not available...")
        self.pause_proxy.call_async(Empty.Request())

        # 返回 state,reward,done
        # state = np.append(self.sensor_data,[self.vel_x,self.ang_z])

        # state = np.array(self.sensor_data)

        # self.logger.info("tar_z:{} vel_x:{} ang_z:{} | middle:{} v_left:{} v_right:{}".format(action[0],self.vel_x,self.ang_z,self.middle,self.v_left,self.v_right))
        self.logger.info("tar_z:{} | ang_z:{} | middle:{}".format(action[0],self.ang_z,self.middle))

        state = np.array([self.middle,self.v_left,self.v_right])

        reward,self.done = self.get_reward()

        # (-1,1)->(0,1)
        reward = (reward + 1)/2

        self.step_nums += 1

        # 达到1000时间步，终止
        if self.step_nums > 1000:
            self.done = True
            self.step_nums = 0


        return state,reward,self.done
        

    def reset(self):
        self.step_nums = 0
        self.stop_action()
        while not self.reset_world.wait_for_service(timeout_sec=1.0):
            print("reset world service is not available...")
        
        # 复位到原来位置
        self.reset_world.call_async(Empty.Request())

        while not self.reset_simulation.wait_for_service(timeout_sec=1.0):
            print("reset simulation service is not available...")
        
        # 复位仿真
        self.reset_simulation.call_async(Empty.Request())

        while not self.unpause_proxy.wait_for_service(timeout_sec=1.0):
            print("unpause physics service is not available...")
        self.unpause_proxy.call_async(Empty.Request())

        sleep(TIME_DELTA)

        while not self.pause_proxy.wait_for_service(timeout_sec=1.0):
            print("pause physics service is not available...")
        self.pause_proxy.call_async(Empty.Request())



        # 返回状态  [[camera raw_data] [x_vel,z_ang]]
        # state = np.append(self.sensor_data,[self.vel_x,self.ang_z])
        # state = np.array(self.sensor_data)
        state = np.array([self.middle,self.v_left,self.v_right])
        return state



# ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{'linear': {'x': 0.1, 'y': 0.0, 'z': 0.0}, 'angular': {'x': 0.0, 'y': 0.0, 'z': 0.0}}"
# ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{'linear': {'x': 0, 'y': 0.0, 'z': 0.0}, 'angular': {'x': 0.0, 'y': 0.0, 'z': 0.0}}"
# ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{'linear': {'x': 0.0, 'y': 0.0, 'z': 0.0}, 'angular': {'x': 0.0, 'y': 0.0, 'z': 0.9}}"
# ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{'linear': {'x': 0.1, 'y': 0.0, 'z': 0.0}, 'angular': {'x': 0.0, 'y': 0.0, 'z': 0.1}}"