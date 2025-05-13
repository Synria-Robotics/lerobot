# 通过命令行控制Alicia Duo机械臂

本文档介绍如何使用`control_robot.py`脚本来控制Alicia Duo机械臂。

## 环境准备

1. 确保已安装LeRobot框架和Alicia Duo SDK
2. 确保机械臂已正确连接到计算机

## 遥操作机械臂

使用以下命令进行基本遥操作：

```bash
python lerobot/scripts/control_robot.py \
    --robot.type=alicia_duo \
    --control.type=teleoperate \
    --control.fps=30 \
    --control.display_data=true
```

### 设置端口和波特率

如果需要指定串口和波特率：

```bash
python lerobot/scripts/control_robot.py \
    --robot.type=alicia_duo \
    --robot.port=/dev/ttyUSB0 \
    --robot.baudrate=921600 \
    --control.type=teleoperate
```

### 安全设置

设置动作限制，防止机械臂运动过快：

```bash
python lerobot/scripts/control_robot.py \
    --robot.type=alicia_duo \
    --robot.max_relative_target=0.05 \
    --control.type=teleoperate
```

## 添加摄像头

连接摄像头并显示图像：

```bash
python lerobot/scripts/control_robot.py \
    --robot.type=alicia_duo \
    --robot.cameras.webcam._target_=lerobot.common.robot_devices.cameras.configs.OpenCVCameraConfig \
    --robot.cameras.webcam.camera_index=0 \
    --robot.cameras.webcam.width=640 \
    --robot.cameras.webcam.height=480 \
    --control.type=teleoperate \
    --control.display_data=true
```

## 数据记录

记录机械臂操作数据：

```bash
python lerobot/scripts/control_robot.py \
    --robot.type=alicia_duo \
    --control.type=record \
    --control.fps=30 \
    --control.repo_id=YOUR_USERNAME/alicia_duo_dataset \
    --control.single_task="使用Alicia Duo机械臂抓取物体" \
    --control.num_episodes=5 \
    --control.warmup_time_s=2 \
    --control.episode_time_s=30 \
    --control.reset_time_s=10
```

记录过程中的键盘控制：
- 按`→`（右方向键）提前结束当前回合并进入环境重置阶段
- 按`←`（左方向键）取消当前回合并重新记录
- 按`ESC`（退出键）停止整个记录过程

## 调试模式

启用调试模式获取更多日志信息：

```bash
python lerobot/scripts/control_robot.py \
    --robot.type=alicia_duo \
    --robot.debug_mode=true \
    --control.type=teleoperate
```

## 模拟模式

当硬件不可用时，可以使用模拟模式进行测试：

```bash
python lerobot/scripts/control_robot.py \
    --robot.type=alicia_duo \
    --robot.mock=true \
    --control.type=teleoperate
``` 