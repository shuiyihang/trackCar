
## 1. 使用强化学习在ROS2中实现巡线小车

### 目录说明

**track_plane**
使用到的轨迹环境track_env_v2.world中用到的track_plane,是自定义贴图环境
将track_plane放入到/home/jetson/.gazebo/models中,gazebo才能正常加载环境。

**checkpoint**
训练好的模型文件，可以用来测试

**src**
代码文件

**pic**
运行结果图片


### 编译
工作空间目录下，使用`colcon build`编译


在工作空间目录下`source install/setup.sh`

### 模型训练
use_gui:=true将会打开gazebo GUI界面，训练为了资源，使用无界面形式

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

为了防止程序一直运行，设置环境，机器人步数>1000，终止本轮。

![环境](pic/env.png)


![奖励曲线](pic/sac_track_car.png)


## 一些调试记录

cmd_vel中设定目标角速度为正，向左转
比如：
-0.5向左转
+0.5向右转
但是订阅odom查看到的值是相反的值


### TODO
1. 测试修改状态空间为原来的5个区域 和 1个黑线中值(状态空间中的速度是否真的有用??)
2. 使用当前的状态空间在AC和DDPG中做测试
3. 似乎存在的问题，有些时候，训练的结果并不如上图理想，待解决。


## 2. 在jetsonNano中配置环境，进行测试
上面的训练和运行是在无显卡的虚拟机中进行的，不推荐使用jetsonNano训练，容易卡死

下面尝试将代码放入jetsonNano。方便进行后续实际巡线的测试
具体看我的这篇博客:[jetsonNano烧录Ubuntu20.04镜像使用ROS2](https://blog.csdn.net/shuiyihang0981/article/details/141717463)
