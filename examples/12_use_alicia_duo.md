# 使用Alicia Duo机械臂

本指南将帮助您设置和使用Alicia Duo机械臂与LeRobot框架。

## 环境准备

1. 确保已经安装LeRobot框架和Alicia Duo SDK：

```bash
# 安装LeRobot
pip install -e .

# 安装Alicia Duo SDK
cd /path/to/Alicia_duo_sdk
pip install -e .
```

2. 确保您的机械臂已正确连接到计算机的USB端口。

## 机械臂校准

首次使用或需要重新校准机械臂时，您可以运行校准命令：

```bash
python lerobot/scripts/control_robot.py \
    --robot.type=alicia_duo \
    --control.type=calibrate
```

这将设置当前位置为零点，便于后续遥操作和数据记录。

## 遥操作机械臂

使用预定义的配置进行遥操作：

```bash
python lerobot/scripts/control_robot.py \
    --config-path configs \
    --config-name alicia_duo_teleop.yaml
```

或者使用命令行参数：

```bash
python lerobot/scripts/control_robot.py \
    --robot.type=alicia_duo \
    --control.type=teleoperate \
    --control.fps=30 \
    --control.display_data=true
```

### 安全注意事项

遥操作时，建议设置`max_relative_target`参数以限制每次移动的最大幅度，防止机械臂运动过快：

```bash
# 限制每次移动最大幅度约为5.7度(0.1弧度)
python lerobot/scripts/control_robot.py \
    --robot.type=alicia_duo \
    --control.type=teleoperate \
    --robot.max_relative_target=0.1
```

## 记录数据集

要记录机械臂的操作数据，可以使用预定义的配置：

```bash
python lerobot/scripts/control_robot.py \
    --config-path configs \
    --config-name alicia_duo_record.yaml
```

或者使用命令行参数：

```bash
python lerobot/scripts/control_robot.py \
    --robot.type=alicia_duo \
    --control.type=record \
    --control.fps=30 \
    --control.repo_id=YOUR_USERNAME/alicia_duo_dataset \
    --control.single_task="使用Alicia Duo机械臂抓取物体并放置到指定位置。" \
    --control.num_episodes=5 \
    --control.warmup_time_s=5 \
    --control.episode_time_s=30 \
    --control.reset_time_s=10
```

记录过程中的键盘控制：
- 按`→`（右方向键）提前结束当前回合并进入环境重置阶段
- 按`←`（左方向键）取消当前回合并重新记录
- 按`ESC`（退出键）停止整个记录过程

## 添加摄像头

如果需要在遥操作或数据记录中使用摄像头，可以在配置文件中添加摄像头配置，或使用命令行参数：

```bash
python lerobot/scripts/control_robot.py \
    --robot.type=alicia_duo \
    --control.type=record \
    --robot.cameras.webcam._target_=lerobot.common.robot_devices.cameras.configs.OpenCVCameraConfig \
    --robot.cameras.webcam.camera_index=0 \
    --robot.cameras.webcam.fps=30 \
    --robot.cameras.webcam.width=640 \
    --robot.cameras.webcam.height=480 \
    --control.display_data=true
```

## 高级配置

您可以编辑`configs/alicia_duo_teleop.yaml`和`configs/alicia_duo_record.yaml`文件，根据需要调整参数。

### 串口设置

默认情况下，SDK会自动搜索可用的串口。如果需要指定串口：

```yaml
robot:
  port: "/dev/ttyUSB0"  # 在Linux上
  # 或
  port: "COM3"  # 在Windows上
```

### 调试模式

如果需要更详细的日志信息，可以启用调试模式：

```yaml
robot:
  debug_mode: true
```

## 常见问题

1. **无法连接到机械臂**
   - 检查USB连接是否正常
   - 确认串口路径是否正确
   - 确认用户有串口访问权限（Linux上可能需要`sudo`或将用户添加到`dialout`组）

2. **移动不稳定或抖动**
   - 减小`max_relative_target`值
   - 检查夹具是否安装牢固
   - 确保电源供应充足

3. **找不到或无法导入AliciaDuo模块**
   - 确认已正确安装Alicia Duo SDK
   - 检查Python路径是否正确

## 自定义机械臂配置

如果需要更深入地自定义Alicia Duo机械臂的行为，可以修改以下文件：

- `lerobot/common/robot_devices/robots/configs.py`中的`AliciaDuoRobotConfig`类
- `lerobot/common/robot_devices/robots/alicia_duo.py`中的`AliciaDuoRobot`类

## 进一步学习

参考其他示例文件了解更多关于LeRobot框架的功能：

- [7_get_started_with_real_robot.md](7_get_started_with_real_robot.md) - 如何开始使用真实机器人
- [3_train_policy.py](3_train_policy.py) - 如何使用收集的数据训练策略
- [4_train_policy_with_script.md](4_train_policy_with_script.md) - 使用脚本训练策略 