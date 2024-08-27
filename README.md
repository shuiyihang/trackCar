
## 使用强化学习在ROS2中实现巡线小车


### 编译
工作空间目录下，使用`colcon build`编译


在工作空间目录下`source install/setup.sh`

### 模型训练
1. 启动环境
2. 启动训练
```
ros2 launch robot_description gazebo.launch.py use_gui:=false

ros2 run auto_control track_car_train
```

### 模型测试
```
ros2 run auto_control track_car_test
```

### 结果

![环境](pic/env.png)


![奖励曲线](pic/sac_track_car.png)