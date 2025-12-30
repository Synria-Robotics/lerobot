# Alicia-D 机械臂与 LeRobot 集成指南

本指南提供了使用 Alicia-D 机械臂与 LeRobot 框架进行数据集录制和策略训练的完整说明。

## 目录

- [安装](#安装)
- [硬件设置](#硬件设置)
- [数据集录制](#数据集录制)
- [策略训练](#策略训练)
- [键盘快捷键](#键盘快捷键)
- [故障排除](#故障排除)

---

## 安装

### 前置要求

- Python 3.10
- Conda（推荐）或虚拟环境
- Ubuntu Linux（推荐）或兼容的 Linux 发行版

### 步骤 1：创建 Conda 环境

```bash
conda create -n lerobot python=3.10
conda activate lerobot
```

### 步骤 2：安装 Alicia-D SDK

```bash
# 创建工作目录
mkdir -p alicia_lerobot
cd alicia_lerobot

# 克隆并安装 Alicia-D SDK
git clone https://github.com/Synria-Robotics/Alicia-D-SDK.git -b v6.1.0
cd Alicia-D-SDK
pip install -e .
cd ..
```

### 步骤 3：安装 LeRobot

```bash
# 克隆并安装 LeRobot
git clone https://github.com/Synria-Robotics/lerobot.git -b v6.1.0-beta1
cd lerobot
pip install -e .
```

### 步骤 4：验证安装

```bash
# 测试命令是否可用
lerobot-record --help
lerobot-train --help
```

---

## 硬件设置

### 连接要求

1. **从动臂（Follower Arm）**：使用 Type-C USB 线将从动臂连接到计算机
2. **主动臂（Leader Arm）**：主动臂通过硬件控制线直接连接到从动臂（无需连接计算机）
3. **摄像头**：将 USB 摄像头连接到计算机

### 端口检测

使用以下命令识别可用的串口：

```bash
lerobot-find-port
```

常见端口位置：
- Linux: `/dev/ttyACM0`, `/dev/ttyACM1`, `/dev/ttyUSB0`
- 检查摄像头设备: `ls /dev/video*`

---

## 数据集录制

### 概述

Alicia-D 主动臂通过硬件控制线直接控制从动臂，绕过计算机。录制过程中，系统：
- 从从动臂读取关节位置（反映主动臂的命令）
- 捕获摄像头图像
- 基于从动臂观测记录动作（因为主动臂直接控制从动臂）

### 单臂配置

**命令：**

```bash
lerobot-record \
    --robot.type=alicia_d_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.cameras="{laptop: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30}}" \
    --robot.id=black \
    --teleop.type=alicia_d_leader \
    --teleop.id=leader_arm \
    --dataset.repo_id=ubuntu/grab-cube-dataset \
    --dataset.root=/home/ubuntu/Data/LerobotData \
    --dataset.num_episodes=10 \
    --dataset.single_task="Grab the cube" \
    --dataset.episode_time_s=60 \
    --dataset.reset_time_s=30 \
    --display_data=true \
    --dataset.push_to_hub=false
```

**参数说明：**
- `--robot.port`: 从动臂的串口（使用 `lerobot-find-port` 检测）
- `--robot.cameras`: 摄像头配置字典
- `--dataset.repo_id`: 数据集仓库标识符（格式：`用户名/数据集名称`）
- `--dataset.root`: 保存数据集的本地目录（可选，默认为缓存目录）
- `--dataset.num_episodes`: 要录制的回合数
- `--dataset.episode_time_s`: 每个回合的持续时间（秒）
- `--dataset.reset_time_s`: 回合之间的环境重置时间

### 双臂（双手）配置

**命令：**

```bash
lerobot-record \
    --robot.type=bi_alicia_d_follower \
    --robot.left_arm_port=/dev/ttyACM0 \
    --robot.right_arm_port=/dev/ttyACM1 \
    --robot.cameras='{
        camera1: {type: opencv, index_or_path: /dev/video10, width: 640, height: 480, fps: 30},
        camera2: {type: opencv, index_or_path: /dev/video18, width: 640, height: 480, fps: 30},
        camera3: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30}
    }' \
    --robot.id=bimanual_follower \
    --teleop.type=bi_alicia_d_leader \
    --teleop.id=bimanual_leader \
    --dataset.repo_id=ubuntu/bimanual-grab-cube-dataset \
    --dataset.root=/home/ubuntu/Data/LerobotData/test2 \
    --dataset.num_episodes=20 \
    --dataset.single_task="Grab the cloth with both arms" \
    --dataset.episode_time_s=60 \
    --dataset.reset_time_s=30 \
    --dataset.chunks_size=10 \
    --dataset.data_files_size_in_mb=50 \
    --dataset.video_files_size_in_mb=100 \
    --display_data=true \
    --dataset.push_to_hub=false
```

**额外参数：**
- `--robot.left_arm_port`: 左从动臂的串口
- `--robot.right_arm_port`: 右从动臂的串口
- `--dataset.chunks_size`: 每个分块目录的最大文件数（默认：1000）
- `--dataset.data_files_size_in_mb`: 数据 parquet 文件的最大大小（MB，默认：100）
- `--dataset.video_files_size_in_mb`: 视频文件的最大大小（MB，默认：200）

### 恢复录制

要在现有数据集上继续录制：

```bash
lerobot-record \
    ... \
    --resume=true
```

这将：
- 加载现有数据集
- 从最后一个回合继续
- 保持数据集兼容性

**注意：** 确保您的机器人配置与原始录制设置匹配。

---

## 策略训练

### 概述

在录制的数据集上训练模仿学习策略（ACT、Diffusion Policy 等）。

### ACT 策略训练

**基本命令：**

```bash
lerobot-train \
    --dataset.repo_id=ubuntu/bimanual-grab-cube-dataset \
    --dataset.root=/home/ubuntu/Data/LerobotData/test2 \
    --dataset.video_backend=pyav \
    --policy.type=act \
    --policy.push_to_hub=false \
    --output_dir=outputs/train/act_bimanual_grab_cube \
    --job_name=act_bimanual_grab_cube \
    --policy.device=cuda \
    --wandb.enable=true \
    --wandb.project=alicia-d-bimanual \
    --steps=50000 \
    --batch_size=32 \
    --save_freq=5000 \
    --log_freq=100 \
    --eval_freq=5000
```

### Diffusion Policy 策略训练

**命令：**

```bash
lerobot-train \
    --dataset.repo_id=ubuntu/bimanual-grab-cube-dataset \
    --dataset.root=/home/ubuntu/Data/LerobotData/test2 \
    --dataset.video_backend=pyav \
    --policy.type=diffusion \
    --policy.push_to_hub=false \
    --output_dir=outputs/train/dp_bimanual_grab_cube \
    --job_name=dp_bimanual_grab_cube \
    --policy.device=cuda \
    --wandb.enable=true \
    --wandb.project=alicia-d-bimanual \
    --steps=50000 \
    --batch_size=32 \
    --save_freq=5000 \
    --log_freq=100 \
    --eval_freq=5000
```

### 训练参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--dataset.repo_id` | 数据集仓库 ID | 必需 |
| `--dataset.root` | 本地数据集路径（加载更快） | 缓存目录 |
| `--dataset.video_backend` | 视频解码器：`pyav` 或 `torchcodec` | 自动检测 |
| `--policy.type` | 策略类型：`act`、`diffusion` 等 | 必需 |
| `--policy.device` | 设备：`cuda` 或 `cpu` | `cpu` |
| `--policy.push_to_hub` | 训练完成后将模型推送到 Hugging Face Hub | `true` |
| `--steps` | 训练步数 | 50000 |
| `--batch_size` | 批次大小 | 32 |
| `--save_freq` | 检查点保存频率 | 5000 |
| `--log_freq` | 日志记录频率 | 100 |
| `--eval_freq` | 评估频率（0 表示禁用） | 5000 |

### 视频后端选择

`--dataset.video_backend` 参数选择视频解码器：

- **`pyav`**（推荐）：稳定，适用于系统 FFmpeg，兼容性更好
- **`torchcodec`**：速度更快，但需要特定版本的 FFmpeg 库

如果遇到 FFmpeg 库错误，请使用 `--dataset.video_backend=pyav`。

### 禁用模型上传到 Hub

默认情况下，LeRobot 会在训练完成后尝试将训练好的模型推送到 Hugging Face Hub。如果您不想上传模型（例如，仅进行本地训练），请设置：

```bash
--policy.push_to_hub=false
```

**注意：** 如果 `push_to_hub=true`（默认值），您必须：
- 配置 Hugging Face 身份验证（`huggingface-cli login`）
- 或者设置 `--policy.push_to_hub=false` 以避免身份验证错误

无论此设置如何，模型都会保存在 `output_dir` 目录中。

---

## 键盘快捷键

在数据集录制期间，可使用以下键盘快捷键：

| 快捷键 | 操作 | 说明 |
|--------|------|------|
| **← (左箭头)** | 重新录制回合 | 清除当前回合缓冲区并重新开始录制同一回合编号 |
| **→ (右箭头)** | 提前退出 | 退出当前循环（录制或重置阶段） |
| **ESC** | 停止录制 | 停止整个数据录制会话 |

### 重新录制回合

如果某个回合出现问题：

1. 在录制期间或之后按 **← (左箭头)**
2. 回合缓冲区被清除
3. 重新开始录制同一回合编号
4. 回合计数器不递增

这允许您丢弃不良回合并重新录制，而不影响总回合数。

---

## 故障排除

### 常见问题

#### 1. FFmpeg/TorchCodec 库错误

**错误：** `RuntimeError: Could not load libtorchcodec`

**解决方案：** 使用 `pyav` 后端：
```bash
--dataset.video_backend=pyav
```

#### 2. 端口未找到

**错误：** `ConnectionError: Failed to connect to Alicia-D robot`

**解决方案：**
- 使用以下命令检查端口：`lerobot-find-port`
- 验证 USB 线连接
- 检查权限：`sudo usermod -a -G dialout $USER`（需要注销/登录）

#### 3. 摄像头未检测到

**错误：** 摄像头初始化失败

**解决方案：**
- 列出摄像头：`ls /dev/video*`
- 检查摄像头权限
- 验证摄像头未被其他进程使用

#### 4. 数据集兼容性错误

**错误：** `ValueError: Dataset metadata compatibility check failed`

**解决方案：**
- 确保机器人配置与原始录制匹配
- 检查 FPS、特征和机器人类型是否匹配

#### 5. Hugging Face Hub 身份验证错误

**错误：** `401 Client Error: Unauthorized for url: https://huggingface.co/api/repos/create`

**解决方案：** 禁用模型上传到 Hub：
```bash
--policy.push_to_hub=false
```

或者，使用 Hugging Face 进行身份验证：
```bash
huggingface-cli login
```

### 获取帮助

- **官方文档：** [LeRobot 文档](https://huggingface.co/docs/lerobot/il_robots#record-a-dataset)
- **GitHub 问题：** [LeRobot Issues](https://github.com/huggingface/lerobot/issues)
- **Discord：** [LeRobot Discord](https://discord.gg/3gxM6Avj)

---

## 其他资源

- [Alicia-D 产品手册](https://docs.sparklingrobo.com/)
- [LeRobot 策略文档](https://huggingface.co/docs/lerobot/bring_your_own_policies)
- [LeRobot 硬件集成指南](https://huggingface.co/docs/lerobot/integrate_hardware)

