# Alicia-D 机械臂 - LeRobot 框架快速上手指南

欢迎使用 Alicia-D 机械臂与 LeRobot 框架！本指南将帮助您快速设置环境、安装必要的软件，并开始使用 Alicia-D 机械臂收集数据。即使您是机器人或编程新手，也能轻松上手。

## 目录

1.  [系统要求](#1-系统要求)
2.  [安装 Alicia-D SDK](#2-安装-alicia-duo-sdk)
3.  [安装 LeRobot 框架](#3-安装-lerobot-框架)
4.  [连接 Alicia-D 机械臂](#4-连接-alicia-duo-机械臂)
5.  [配置数据收集参数](#5-配置数据收集参数)
6.  [开始数据收集](#6-开始数据收集)
7.  [常见问题与故障排除](#7-常见问题与故障排除)
8.  [可视化已收集的数据集](#8-可视化已收集的数据集)

---

## 1. 系统要求

*   **操作系统**: 推荐使用 Linux (例如 Ubuntu 20.04 或更高版本)。本指南主要基于 Linux 环境。
*   **Python**: 版本 3.10 或更高。
*   **硬件**:
    *   Alicia-D 机械臂。
    *   一台用于连接和控制机械臂的计算机。
    *   USB 数据线，用于连接计算机和 Alicia-D 机械臂。
    *   (可选) 如果您希望收集视觉数据，需要至少一个兼容的USB摄像头 (例如普通的网络摄像头)。

---


## 2. 安装 LeRobot 框架

LeRobot 是一个用于机器人学习的开源框架，我们将用它来控制 Alicia-D 并收集数据。

1.  **获取 LeRobot**:
    使用 Git 从 GitHub 克隆 LeRobot 仓库。如果您没有安装 Git，请先安装它 (搜索 "如何安装 Git on [您的操作系统]")。

    ```bash
    # 导航到您希望存放 LeRobot 项目的文件夹
    cd /path/to/your/projects_directory

    # For Alicia-D-SDK not installed:
    git clone --recursive git@github.com:Synria-Robotics/lerobot.git -b v6.0.0
    # For ALicia-D-SDK installed:
    git clone https://github.com/Synria-Robotics/lerobot.git -b v6.0.0


    # 进入 LeRobot 文件夹
    cd lerobot
    ```

2.  **创建虚拟环境 (推荐)**:
    创建`conda`环境：

    ```bash
    conda create -y -n lerobot python=3.10
    conda activate lerobot
    ```
    安装环境依赖：
    ```bash
    conda install ffmpeg -c conda-forge
    ```


3. **安装 Alicia-D SDK**

    Alicia-D SDK (Software Development Kit) 是控制 Alicia-D 机械臂和读取其数据的核心软件库。
    ```
    # cd /path/to/Alicia-D-SDK
    cd Alicia-D-SDK
    # 使用 pip 安装 SDK
    pip install -e .
    ```

---

4.  **安装 LeRobot 及其依赖**:

    ```bash
    # 确保您在 lerobot 文件夹的根目录下
    # 安装 LeRobot 及其核心依赖
    cd .. # Enter the path for lerobot setup
    pip install -e .
    ```
    这将安装 LeRobot 框架本身以及运行它所必需的库。且请注意torch torchvision cuda ffmpeg 的版本匹配问题

---

## 4. 连接 Alicia-D 机械臂

1.  **物理连接**:
    *   使用 USB 数据线将 Alicia-D 遥操套件的示教臂连接到您的计算机。
    *   确保机械臂已通电 (如果需要外部电源)。
    *   (可选) 如果您要使用摄像头，也将摄像头连接到计算机的USB端口。

2.  **检查连接 (Linux)**:
    在 Linux 系统上，连接机械臂后，它通常会显示为一个串口设备，例如 `/dev/ttyUSB0` 或 `/dev/ttyACM0`。您可以通过以下命令查看新出现的设备：
    ```bash
    ls /dev/ttyUSB*
    ```
    LeRobot 框架中的 Alicia-D 驱动默认会自动搜索可用的串口。如果自动搜索失败，您可能需要手动指定端口号。

    ```
    # Add serial port permission
    sudo chmod 666 /dev/ttyUSB*  # temporally
    sudo usermod -a -G dialout $USER  # permanently
    ```

3.  **检查连接 (Windows)**:
    在 Windows 系统上，连接机械臂后，它会显示为一个 `COM` 串口（例如 `COM3`、`COM5`）。您可以通过以下方式查看端口：

    - 打开“设备管理器” → 展开“端口 (COM 和 LPT)” → 查找类似 “USB-SERIAL CH340 (COM3)” 或 “Silicon Labs CP210x USB to UART Bridge (COM5)” 的设备名称。
    - 或使用 `mode` 命令（CMD/PowerShell 均可）：
      ```cmd
      mode
      ```

    如自动搜索失败，可在命令行通过 `--robot.port=COM3` 显式指定端口（将 `COM3` 替换为您的实际端口）。

    注意事项：
    - 若首次连接后未识别端口，可能需要安装对应的 USB 转串口驱动（常见为 CH340 或 CP210x），可从芯片官方或硬件厂商处下载并安装。
    - 确保没有其他程序占用该串口（如串口调试助手等）。
    - 保持默认波特率 `1000000`；如需修改，请与设备端设置一致。
---

## 5. 配置数据收集参数

LeRobot 使用命令行参数来配置数据收集任务。以下是一些关键参数：

- `--robot.type=alicia_d` :设置操作臂类型
- `--robot.port=/dev/ttyACM0`：设置操作臂端口
- `--robot.baudrate=1000000`：设置操作臂波特率
- `--robot.execute_motion=false`：设置不复现下发（默认为false即可）
- `--teleop.type=leader`：控制方式为示教臂
- `--teleop.port=/dev/ttyACM0`：示教臂端口
- `--teleop.baudrate=1000000`：示教臂波特率
- `--dataset.repo_id=yourname/alicia_leader_dataset`：数据集ID
- `--dataset.root=/home/ubuntu/vla/vla_datasets`：数据集本地路径
- `--dataset.num_episodes=5`：采集轮次
- `--dataset.episode_time_s=60`：采集单轮时间
- `--dataset.reset_time_s=30`：采集环境重置时间
- `--dataset.fps=30`：采集数据频率
- `--dataset.single_task="pick and place"`：动作命令（language），注意数据集的命令要具有多样性，这样训练出来的模型泛化性要强一些

**添加摄像头 (可选):**

如果您想同时记录来自一个或多个摄像头的视觉数据，您需要直接在 LeRobot 框架的配置文件中进行设置。

1.  **打开配置文件**:
    找到并打开 `lerobot/src/lerobot/robots/alicia_d/config_alicia_d.py` 文件。

2.  **修改 `AliciaDRobotConfig`**:
    在该文件中，找到 `AliciaDRobotConfig` 类。您可以修改其 `cameras` 属性来定义您的摄像头。

    下面是一个示例，展示了如何配置一个名为 "front" 的前置USB摄像头和一个名为 "wrist" 的腕部USB摄像头:
    ```python
  @RobotConfig.register_subclass("alicia_d")
  @dataclass
  class AliciaDConfig(RobotConfig):
    @staticmethod
    def default_cameras_config() -> dict[str, CameraConfig]:
        return {
            "wrist": OpenCVCameraConfig(
                index_or_path="/dev/video0", fps=30, width=480, height=640, rotation=Cv2Rotation.ROTATE_90
            ),
            "front": OpenCVCameraConfig(
                index_or_path="/dev/video6", fps=30, width=480, height=640, rotation=Cv2Rotation.ROTATE_90
            ),
            "top": OpenCVCameraConfig(
                index_or_path="/dev/video7", fps=30, width=480, height=640, rotation=Cv2Rotation.ROTATE_90
            ),
        }
    # 串口/波特率
    port: str | None = None  # None 表示让 SDK 自行扫描
    baudrate: int = 1_000_000

    # 断开连接时的安全选项（与其它机器人保持一致语义）
    disable_torque_on_disconnect: bool = True

    # 安全限制：每个关节相对目标的最大允许变化（弧度）。
    # 可设为 float（统一值）或按关节名的 dict[str, float]
    max_relative_target: float | dict[str, float] | None = None

    # 摄像头
    cameras: dict[str, CameraConfig] = field(default_factory=lambda: AliciaDConfig.default_cameras_config())

    # 是否实际执行动作（False=只记录，不下发到硬件）
    execute_motion: bool = False
    ```

    *   **`cameras` 字典**: 这是一个字典，键是您为摄像头指定的名称 (例如 `"front"`, `"wrist_cam"`), 值是 `OpenCVCameraConfig` (或其他摄像头类型的配置对象)。
    *   **`OpenCVCameraConfig` 参数**:
        *   `camera_index`: 对于USB摄像头，这通常是一个数字索引 (0, 1, ...)，或者是设备文件的路径 (例如 `"/dev/video0"`)。
        *   `fps`: 摄像头的帧率。
        *   `width`, `height`: 图像的分辨率。
        *   `rotation`: 如果您的摄像头安装方向导致图像是旋转的，可以使用此参数进行校正 (例如 `90`, `180`, `-90`)。
    *   您可以根据您的实际摄像头数量和参数修改此部分。如果不需要摄像头，可以将 `cameras` 字典设置为空 `field(default_factory=dict)`。

修改完 `configs.py` 文件并保存后，当您运行数据收集脚本时，LeRobot 将自动使用这些配置来连接和记录摄像头数据。命令行中不再需要添加 `--robot.cameras...` 参数。

---

## 6. 开始数据收集

一切准备就绪后，打开您的终端，确保您处于已激活 LeRobot 虚拟环境的 `lerobot` 文件夹根目录下。然后运行 `control_robot.py` 脚本并附带上您配置好的参数。

**示例命令 (假设摄像头已在 `configs.py` 中配置):**

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

lerobot-record \
  --robot.type=alicia_d \
  --robot.port=/dev/ttyUSB1 \
  --robot.baudrate=1000000 \
  --robot.execute_motion=false \
  --teleop.type=leader \
  --teleop.port=/dev/ttyUSB0 \
  --teleop.baudrate=1000000 \
  --dataset.repo_id=local/alicia_dataset \
  --dataset.root=/home/ubuntu/vla/datasets \
  --dataset.num_episodes=5 \
  --dataset.episode_time_s=60 \
  --dataset.reset_time_s=30 \
  --dataset.fps=30 \
  --dataset.single_task="pick and place" \
  --dataset.video=true \
  --dataset.push_to_hub=false \
  --display_data=true \
  --play_sounds=false
```
**请务必将 `/home/ubuntu/lerobot_datasets` 和 `ubuntu/alicia_demo_dataset` 替换为您自己的路径和数据集标识符（用户名/数据集名称格式）。**


**数据收集中:**

*   脚本运行后，会首先连接机械臂和摄像头。
*   **记录阶段**: LeRobot 会提示开始记录。在此期间，您操作 Alicia-D 机械臂执行任务，LeRobot 会记录下机械臂的关节状态、夹爪状态以及摄像头图像 (如果配置了)。此阶段持续 `episode_time_s` 秒。
*   **重置阶段**: 一个回合记录完成后，您有 `reset_time_s` 秒的时间将场景和机械臂复位，为下一个回合做准备。
*   这个过程会重复 `num_episodes` 次。

数据收集完成后，您可以在您指定的 `--dataset.root` 路径下找到生成的数据集文件夹。

---

## 7. 数据集训练

LeRobot 支持两种数据集训练方式：使用本地数据集和使用 HuggingFace Hub 上的数据集。两种方式都使用相同的 `repo_id` 格式，主要区别在于是否需要 `root` 参数：

- **本地数据集**: `repo_id` 使用 `username/dataset_name` 格式，需要配合 `root` 参数指定数据集的父目录
- **HuggingFace Hub 数据集**: `repo_id` 使用 `username/dataset_name` 格式，无需 `root` 参数（自动从 Hub 下载）

### 本地数据集训练
对于本地数据集，使用与数据收集时相同的 `repo_id` 格式（`username/dataset_name`），`root` 参数应该指向包含数据集的父目录：

**重要说明**: 
- 本地数据集目录结构：`root_directory/username/dataset_name/`
- 数据收集时创建的文件夹结构会是：`/your/root/path/username/dataset_name/`

**配置方法**:
- `repo_id`: 与数据收集时使用的相同格式（例如 `my_user/alicia_visual_demo_dataset`）
- `root`: 包含数据集文件夹的父目录路径（例如 `/home/ubuntu/lerobot_datasets`）

```bash
python lerobot/scripts/train.py \
    --policy.type=diffusion \
    --dataset.repo_id=username/dataset_name \
    --dataset.root=/path/to/parent/directory \
    --output_dir=/path/to/training_result
```

**示例**:
```bash
lerobot-train \
  --policy.type=act \
  --dataset.root=/home/ubuntu/lerobot/datasets \
  --dataset.repo_id=local/alicia_leader_dataset \
  --policy.repo_id=local/alicia_pi_ft \
  --output_dir=/home/ubuntu/vla/outputs \
  --job_name=smolvla_ft_alicia \
  --policy.device=cuda \
  --wandb.enable=false
```

## 8. 模型验证
进入训练结果
```/path_to_training_result/checkpoints/last/pretrained_model/config.json```

参考`inference.py`修改对应参数验证训练结果

## 9. 常见问题与故障排除

*   **"未找到 Alicia-D SDK" 或 "ArmController 未初始化"**:
    *   请确保您已正确安装 Alicia-D SDK (参见步骤2)。
    *   确认您在运行 LeRobot 命令时，Alicia-D SDK 所在的 Python 环境是激活的 (或者它已安装到全局 Python 环境中，并且 LeRobot 使用的是同一个 Python 解释器)。

*   **"无法连接到 Alicia-D 机械臂"**:
    *   检查 USB 连接是否牢固，机械臂是否已通电。
    *   确认机械臂的串口是否被其他程序占用。
    *   尝试手动指定 `--robot.port` 参数，例如 `--robot.port=/dev/ttyUSB0`。您可能需要尝试不同的数字 (ttyUSB0, ttyUSB1 等)。
    *   在 Linux 上，您可能需要串口的读写权限。尝试将您的用户添加到 `dialout` 组：`sudo usermod -a -G dialout $USER`，然后**重启计算机**或重新登录。

*   **"AttributeError: 'AliciaDuoRobot' object has no attribute 'some_feature'"**:
    *   这通常表示 Alicia-D 的 LeRobot 驱动实现 (`alicia_d.py`) 可能缺少了框架期望的某些属性或方法。请确保您使用的是最新或兼容版本的 LeRobot 和 Alicia-D 驱动。如果问题是最近集成的，可能需要开发者进一步调试。

*   **摄像头无法工作或报错**:
    *   确保摄像头已正确连接到 USB 端口。
    *   使用 `--robot.cameras.YOUR_CAM_NAME.camera_index` 指定的摄像头索引是否正确。您可以使用 `lerobot/common/robot_devices/cameras/opencv.py --images-dir outputs/cam_test` 来测试和识别摄像头索引。
    *   尝试降低摄像头的 `--fps` 或分辨率 (`--width`, `--height`)，某些 USB 总线或摄像头可能不支持高参数配置。

*   **数据记录频率不理想**:
    *   如果 `--control.fps` 设置得很高，但实际感觉卡顿或日志显示帧率较低，可能是计算机性能瓶颈，或者摄像头/机械臂通信延迟。
    *   确保您的 `--robot.max_relative_target` (在 `lerobot/common/robot_devices/robots/configs.py` 中 `AliciaDRobotConfig` 定义或通过命令行覆盖) 设置合理，以允许平滑运动。

如果您遇到其他问题，建议查看终端输出的详细错误信息，并可以查阅 LeRobot 的 GitHub Issues 或向 Synria Robotics 技术支持寻求帮助。

---
