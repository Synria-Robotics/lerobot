# Alicia Duo机械臂集成总结

我们已成功将Alicia Duo机械臂集成到LeRobot框架中。下面是集成的主要组件和使用方法。

## 已完成的工作

1. **配置类 - AliciaDuoRobotConfig**
   - 添加到`lerobot/common/robot_devices/robots/configs.py`
   - 使用`@RobotConfig.register_subclass("alicia_duo")`注册
   - 包含机械臂的基本参数：端口、波特率、调试模式等
   - 支持摄像头配置和安全控制参数

2. **控制类 - AliciaDuoRobot**
   - 创建`lerobot/common/robot_devices/robots/alicia_duo.py`文件
   - 封装了Alicia Duo SDK的`ArmController`
   - 实现了LeRobot框架需要的Robot协议接口
   - 提供了连接、数据读取和控制功能

3. **工厂函数支持**
   - 更新`utils.py`中的`make_robot_config`和`make_robot_from_config`函数
   - 使系统能识别并创建Alicia Duo机械臂实例

4. **使用示例**
   - 直接使用Python API的示例：`examples/use_alicia_duo.py`
   - 通过命令行控制的说明：`examples/control_alicia_duo.md`

## 主要特性

1. **基本功能**
   - 连接/断开机械臂
   - 读取机械臂状态
   - 发送控制命令
   - 支持夹爪控制

2. **安全控制**
   - 通过`max_relative_target`限制每次移动的幅度
   - 防止机械臂运动过快导致危险

3. **摄像头支持**
   - 支持集成多个摄像头
   - 可在观测数据中包含图像

4. **遥操作和数据记录**
   - 支持通过`control_robot.py`脚本进行遥操作
   - 支持数据记录，可用于后续训练

## 使用方法

### 1. 直接使用Python API

```python
from lerobot.common.robot_devices.robots.utils import make_robot_config, make_robot_from_config

# 创建配置
config = make_robot_config(
    robot_type="alicia_duo",
    port="/dev/ttyUSB0",  # 或留空自动搜索
    baudrate=921600
)

# 创建机器人实例
robot = make_robot_from_config(config)

# 连接
robot.connect()

# 读取状态
observation = robot.capture_observation()
current_state = observation["observation.state"]
print(f"当前关节角度: {current_state[:-1].tolist()}")
print(f"当前夹爪角度: {current_state[-1].item()}")

# 发送动作
import torch
action = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.5])  # 6个关节 + 1个夹爪
robot.send_action(action)

# 断开连接
robot.disconnect()
```

### 2. 通过命令行使用

遥操作：

```bash
python lerobot/scripts/control_robot.py \
    --robot.type=alicia_duo \
    --control.type=teleoperate \
    --control.fps=30 \
    --control.display_data=true
```

记录数据：

```bash
python lerobot/scripts/control_robot.py \
    --robot.type=alicia_duo \
    --control.type=record \
    --control.fps=30 \
    --control.repo_id=YOUR_USERNAME/alicia_duo_dataset \
    --control.single_task="使用Alicia Duo机械臂抓取物体" \
    --control.num_episodes=5
```

## 注意事项

1. **串口连接**
   - 默认自动搜索可用串口
   - 可以通过`port`参数指定串口

2. **安全设置**
   - 建议设置`max_relative_target`参数
   - 初始设置为0.1弧度（约5.7度）

3. **调试模式**
   - 设置`debug_mode=True`获取更多日志信息

4. **模拟模式**
   - 当硬件不可用时，可以设置`mock=True`进行测试 