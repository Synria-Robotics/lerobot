# 如何向LeRobot框架添加自定义机器人

本指南详细介绍如何将自定义机器人集成到LeRobot框架中，以便使用框架的数据收集、训练和部署功能。我们将使用XuanyaArm作为示例，展示完整的集成过程。

## 目录

1. [框架概述](#框架概述)
2. [添加新机器人的步骤](#添加新机器人的步骤)
   - [步骤1：创建机器人配置类](#步骤1创建机器人配置类)
   - [步骤2：实现机器人类](#步骤2实现机器人类)
   - [步骤3：注册机器人到框架](#步骤3注册机器人到框架)
   - [步骤4：实现校准逻辑（可选）](#步骤4实现校准逻辑可选)
3. [使用自定义机器人](#使用自定义机器人)
   - [数据收集](#数据收集)
   - [训练模型](#训练模型)
   - [部署策略](#部署策略)
4. [常见集成案例](#常见集成案例)
   - [ROS机器人集成](#ros机器人集成)
   - [自定义硬件集成](#自定义硬件集成)
   - [远程控制机器人](#远程控制机器人)
5. [故障排除](#故障排除)

## 框架概述

LeRobot框架采用模块化设计，支持多种机器人硬件和控制接口。主要组件包括：

- **机器人设备管理**：定义在`lerobot/common/robot_devices/`下
- **数据收集工具**：主要通过`lerobot/scripts/control_robot.py`实现
- **策略训练**：使用`lerobot/scripts/train.py`进行
- **模型部署**：通过`control_robot.py`和自定义机器人接口实现

## 添加新机器人的步骤

### 步骤1：创建机器人配置类

在`lerobot/common/robot_devices/robots/configs.py`中定义机器人配置类：

```python
@RobotConfig.register_subclass("xuanya_arm")  # 注册类型名称
@dataclass
class XuanyaArmRobotConfig(ManipulatorRobotConfig):
    robot_type: str = "xuanya_arm"
    calibration_dir: str = ".cache/calibration/xuanya_arm"
    # 可配置相对目标限制以确保安全
    max_relative_target: int | None = None
    
    # ROS相关配置
    ros_node_name: str = "lerobot_xuanya_control"
    read_lead_arm_topic: str = "/arm_joint_state"
    read_follow_arm_topic: str = "/arm_joint_state"
    control_arm_topic: str = "/arm_joint_command"
    
    # 摄像头配置
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "overview": OpenCVCameraConfig(
                camera_index=0,
                fps=30,
                width=640,
                height=480,
            ),
            "hand": OpenCVCameraConfig(
                camera_index=2,
                fps=30,
                width=640,
                height=480,
            ),
            "front": OpenCVCameraConfig(
                camera_index=4,
                fps=30,
                width=640,
                height=480,
            ),
        }
    )
    
    # 由于使用ROS接口，不需要直接配置motors
    # 但为了与框架兼容，我们保留空的leader_arms和follower_arms字段
    leader_arms: dict[str, MotorsBusConfig] = field(default_factory=dict)
    follower_arms: dict[str, MotorsBusConfig] = field(default_factory=dict)
    
    mock: bool = False
```

配置类继承关系：
- `RobotConfig`：基础抽象类 (使用`draccus.ChoiceRegistry`实现类型注册)
- `ManipulatorRobotConfig`：机械臂机器人通用配置
- 或直接继承`RobotConfig`：对于非机械臂类型机器人（如移动机器人）

**重要字段说明**：
- `robot_type`：字符串标识符，与注册名保持一致
- `calibration_dir`：校准文件存储路径
- `cameras`：相机配置，键为相机名称，值为`CameraConfig`派生类实例
- `leader_arms`/`follower_arms`：对于带有物理电机接口的机器人，定义电机总线配置
- 自定义配置：添加机器人特定参数（例如ROS话题）

### 步骤2：实现机器人类

创建新文件`lerobot/common/robot_devices/robots/xuanya_arm.py`：

```python
class XuanyaArmRobot(ManipulatorRobot):
    """
    Xuanya机械臂实现，通过ROS话题与机器人通信
    """
    @property
    def available_arms(self):
        """定义XuanyaArm可用的机械臂ID"""
        return [
            get_arm_id("main", "leader"),
            get_arm_id("main", "follower")
        ]
    
    @property
    def motor_features(self) -> dict:
        # 定义正确的7维向量形状，需与ROS消息匹配
        joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]
        return {
            "action": {
                "dtype": "float32",
                "shape": (7,),
                "names": joint_names,
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (7,),
                "names": joint_names,
            },
        }

    @property
    def features(self):
        # 合并电机和相机特征
        return {**self.motor_features, **self.camera_features}
        
    def __init__(self, config: XuanyaArmRobotConfig):
        """初始化机器人"""
        self.config = config
        self.robot_type = config.robot_type
        self.calibration_dir = Path(config.calibration_dir)
        
        # 初始化摄像头
        self.cameras = make_cameras_from_configs(config.cameras)
        
        # 初始化空的arms字典以保持与基类兼容
        self.leader_arms = {}
        self.follower_arms = {}
        
        self.is_connected = False
        self.ros_initialized = False
        
        # ROS相关变量
        self.node_name = config.ros_node_name
        self.lead_arm_topic = config.read_lead_arm_topic
        self.follow_arm_topic = config.read_follow_arm_topic
        self.control_topic = config.control_arm_topic
        
        # 状态存储
        self.leader_state = None
        self.follower_state = None
        self.logs = {}
```

**必须实现的方法**：

```python
def connect(self):
    """连接到ROS和相机"""
    # 实现连接逻辑

def disconnect(self):
    """断开连接"""
    # 实现断开连接逻辑

def teleop_step(self, record_data=False) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """进行一步遥操作，并可选地返回观察和动作数据"""
    # 实现遥操作逻辑

def capture_observation(self):
    """捕获观察，没有批次维度"""
    # 实现观察捕获逻辑

def send_action(self, action: torch.Tensor) -> torch.Tensor:
    """发送动作到机器人"""
    # 实现动作发送逻辑
```

**重要接口规范**：

1. `teleop_step`返回格式：
   - `None`：当`record_data=False`时
   - 或`(observation, action)`：两个字典，包含传感器数据和动作
   
2. `observation`字典必须包含：
   - `"observation.state"`：机器人状态张量
   - `"observation.images.<camera_name>"`：每个相机的图像张量

3. `action`字典必须包含：
   - `"action"`：动作向量张量

所有返回的张量应为`torch.Tensor`类型，且不带批次维度。

### 步骤3：注册机器人到框架

在`lerobot/common/robot_devices/robots/utils.py`中注册机器人：

1. 导入配置类：
```python
from lerobot.common.robot_devices.robots.configs import (
    # ...其他配置类...
    XuanyaArmRobotConfig,
)
```

2. 在`make_robot_from_config`函数中添加支持：
```python
def make_robot_from_config(config: RobotConfig):
    if isinstance(config, XuanyaArmRobotConfig):
        from lerobot.common.robot_devices.robots.xuanya_arm import XuanyaArmRobot
        return XuanyaArmRobot(config)
    # ...其他机器人类型...
```

3. (可选) 如果机器人不属于现有类别，在`make_robot_config`中添加：
```python
def make_robot_config(robot_type: str, **kwargs) -> RobotConfig:
    # ...现有代码...
    elif robot_type == "xuanya_arm":
        return XuanyaArmRobotConfig(**kwargs)
    # ...其他类型...
```

### 步骤4：实现校准逻辑（可选）

如果机器人需要校准，可实现校准逻辑：

1. 为自定义机器人创建校准文件，例如：
```python
# lerobot/common/robot_devices/robots/xuanya_calibration.py
def run_arm_calibration(arm, robot_type, name, arm_type):
    # 实现校准逻辑
    # 返回校准数据
```

2. 或者在机器人类中重写`activate_calibration`方法：
```python
def activate_calibration(self):
    """执行自定义校准逻辑"""
    # 实现校准过程
```

## 使用自定义机器人

### 数据收集

使用`control_robot.py`脚本收集数据：

```bash
python lerobot/scripts/control_robot.py \
    --robot.type=xuanya_arm \
    --control.type=record \
    --control.fps=30 \
    --control.repo_id=user/xuanya_dataset \
    --control.num_episodes=10
```

### 训练模型

使用收集的数据训练模型：

```bash
python lerobot/scripts/train.py \
    --dataset.repo_id=user/xuanya_dataset \
    --policy.type=diffusion \
    --output_dir=outputs/diffusion_xuanya
```

### 部署策略

将训练好的模型部署到机器人上：

```bash
python lerobot/scripts/control_robot.py \
    --robot.type=xuanya_arm \
    --control.type=record \
    --control.fps=30 \
    --control.repo_id=user/xuanya_eval \
    --control.num_episodes=5 \
    --control.policy.path=outputs/diffusion_xuanya/checkpoints/last/pretrained_model
```

## 常见集成案例

### ROS机器人集成

XuanyaArm是ROS机器人集成的典型示例：

1. 添加ROS依赖：
```python
import rospy
from your_ros_package.msg import YourMessage
```

2. 在`connect`方法中初始化ROS节点和话题：
```python
def connect(self):
    rospy.init_node(self.node_name, anonymous=True)
    self.subscriber = rospy.Subscriber(self.topic, YourMessage, self._callback)
    self.publisher = rospy.Publisher(self.control_topic, YourMessage, queue_size=1)
```

3. 实现回调和控制方法：
```python
def _callback(self, msg):
    # 处理传入的ROS消息
    
def send_action(self, action):
    # 创建ROS消息并发布
    msg = YourMessage()
    # 设置消息字段
    self.publisher.publish(msg)
```

### 自定义硬件集成

对于直接通过串口、USB等接口控制的硬件：

1. 使用相应的通信库：
```python
import serial  # 或其他设备通信库
```

2. 在`connect`中初始化连接：
```python
def connect(self):
    self.device = serial.Serial(self.config.port, self.config.baudrate)
```

3. 实现命令发送和状态读取：
```python
def send_action(self, action):
    command = self._format_command(action)
    self.device.write(command)
    
def capture_observation(self):
    data = self.device.read(self.buffer_size)
    return self._parse_observation(data)
```

### 远程控制机器人

对于通过网络控制的机器人：

1. 使用网络通信：
```python
import socket
import requests
```

2. 实现连接和通信：
```python
def connect(self):
    self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.socket.connect((self.config.ip, self.config.port))
    
def send_action(self, action):
    data = json.dumps({"action": action.tolist()})
    self.socket.sendall(data.encode())
```

## 故障排除

常见问题及解决方案：

1. **机器人连接失败**：
   - 检查硬件连接
   - 确认配置中的端口、地址等参数正确
   - 验证必要服务（如ROS节点）正在运行

2. **数据形状不匹配**：
   - 确保`motor_features`方法返回正确的形状
   - 验证`teleop_step`和`capture_observation`返回符合接口规范的数据

3. **校准文件问题**：
   - 检查校准目录权限
   - 手动删除校准文件并重新运行校准过程

4. **相机问题**：
   - 验证相机索引/标识符
   - 检查相机分辨率设置是否受支持
   - 确认USB带宽足够多个相机同时使用

为获取更多帮助，请参考具体机器人类的实现，如`xuanya_arm.py`、`manipulator.py`等。 