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
    git clone --recursive git@github.com:Synria-Robotics/lerobot.git -b v5.5.0
    # For ALicia-D-SDK installed:
    git clone https://github.com/Synria-Robotics/lerobot.git -b v5.5.0


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
    pip install -r requirement.txt
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
    这将安装 LeRobot 框架本身以及运行它所必需的库。

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

*   `--robot.type=alicia_d`: 指定我们要使用的机器人类型是 Alicia-D。
*   `--control.type=record`: 指定我们要执行的任务是数据记录。
*   `--control.fps=30`: 设置数据记录的帧率 (每秒捕获多少帧数据)。常用的值是 15、30。
*   `--control.single_task="在这里描述您的任务"`: 对您正在演示或记录的任务进行简短描述，例如 `"机械臂抓取红色的积木并放入盒子中"`。
*   `--control.root=/path/to/your/datasets/my_alicia_dataset`: 指定收集的数据集存储在本地计算机的哪个文件夹。请确保此路径存在，或者 LeRobot 有权限创建它。
*   `--control.repo_id=username/dataset_name`: (**必需**) 指定数据集标识符。必须使用 `用户名/数据集名称` 的格式（例如：`my_user/alicia_demo`）。即使您不上传到 Hugging Face Hub (`--control.push_to_hub=false`)，这个格式也是必需的。
*   `--control.num_episodes=10`: 您希望记录多少个 "回合" 或 "演示" 的数据。
*   `--control.warmup_time_s=5`: 每个回合开始前，等待多少秒。这给您时间准备。
*   `--control.episode_time_s=60`: 每个回合计划记录多长时间 (秒)。
*   `--control.reset_time_s=30`: 每个回合结束后，给您多少时间来重置场景或机械臂到初始状态。
*   `--control.push_to_hub=false`: 是否在数据收集完成后自动将数据集上传到 Hugging Face Hub。初次使用建议设为 `false`。
*   `--robot.port=""`: (可选) 指定机械臂连接的串口。留空则自动搜索。如果自动搜索失败，请手动设置：Linux 示例 `"/dev/ttyUSB0"`，Windows 示例 `COM3`（将其替换为设备管理器中显示的实际端口）。
*   `--robot.baudrate=1000000`: (可选) 串口通信的波特率。Alicia-D 通常使用 921600，这是默认值。

**添加摄像头 (可选):**

如果您想同时记录来自一个或多个摄像头的视觉数据，您需要直接在 LeRobot 框架的配置文件中进行设置。

1.  **打开配置文件**:
    找到并打开 `lerobot/common/robot_devices/robots/configs.py` 文件。

2.  **修改 `AliciaDRobotConfig`**:
    在该文件中，找到 `AliciaDRobotConfig` 类。您可以修改其 `cameras` 属性来定义您的摄像头。

    下面是一个示例，展示了如何配置一个名为 "front" 的前置USB摄像头和一个名为 "wrist" 的腕部USB摄像头:
    ```python
    @RobotConfig.register_subclass("alicia_d")
    @dataclass
    class AliciaDRobotConfig(RobotConfig):
        """Alicia-D机械臂的配置类"""
        
        # 串口设置
        port: str = ""  # 留空则自动搜索
        baudrate: int = 1000000
        debug_mode: bool = False
        
        # 摄像头配置
        cameras: dict[str, CameraConfig] = field(default_factory=lambda: {
                "front": OpenCVCameraConfig(
                    # 摄像头名称，您可以自定义，例如 "front_cam", "webcam"
                    camera_index=0,       # OpenCV 摄像头索引 (通常从0开始)，或设备路径如 "/dev/video0" (Windows无路径，直接给摄像头的整数索引)
                    fps=30,               # 期望的帧率
                    width=640,            # 图像宽度
                    height=480,           # 图像高度
                    rotation=0            # 旋转角度 (可以是 0, 90, 180, -90)
                ),
                "wrist": OpenCVCameraConfig(
                    camera_index="/dev/video2", # 另一个摄像头的设备路径或索引 (Windows无路径，直接给摄像头的整数索引)
                    fps=30,
                    width=640,
                    height=480,
                    rotation=180
                ),
                # 您可以根据需要添加更多摄像头，或者删除不需要的摄像头
            })
        
        # 安全控制参数
        max_relative_target: list[float] | float | None = None
        
        # 模拟模式
        mock: bool = False
        
        def __post_init__(self):
            if self.mock:
                for cam in self.cameras.values():
                    if not cam.mock:
                        cam.mock = True
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
python lerobot/scripts/control_robot.py \
    --robot.type=alicia_d \
    --control.type=record  \
    --control.fps=30  \
    --control.single_task="演示如何用Alicia-D机械臂移动一个方块" \
    --control.root=/home/ubuntu/lerobot_datasets \
    --control.repo_id=ubuntu/alicia_demo_dataset \
    --control.num_episodes=5  \
    --control.warmup_time_s=5  \
    --control.episode_time_s=60  \
    --control.reset_time_s=20  \
    --control.push_to_hub=false
```
**请务必将 `/home/ubuntu/lerobot_datasets` 和 `ubuntu/alicia_demo_dataset` 替换为您自己的路径和数据集标识符（用户名/数据集名称格式）。**

**示例命令 (带一个前置摄像头):**
如果您已在 `lerobot/common/robot_devices/robots/configs.py` 中的 `AliciaDRobotConfig` 配置了摄像头，则运行数据收集脚本时，无需在命令行中再次指定摄像头参数。脚本会自动加载 `configs.py` 中的设置。

**一套遥操作套件**（一个操作臂一个示教臂）

```bash
export XDG_RUNTIME_DIR=/tmp
python lerobot/scripts/control_robot.py \
  --robot.type=alicia_d \
  --control.type=record \
  --control.fps=30 \
  --control.play_sounds=false \
  --control.root=/home/ubuntu/lerobot_datasets \
  --control.repo_id=ubuntu/alicia_visual_demo_v2 \
  --control.num_episodes=5 \
  --control.warmup_time_s=2 \
  --control.episode_time_s=5 \
  --control.reset_time_s=5 \
  --control.push_to_hub=false \
  --control.single_task="pick and place demo" \
  --control.display_data=true
```

**两套遥操作套件**（两个操作臂两个示教臂）
```bash
python lerobot/scripts/control_robot.py \
    --robot.type=alicia_d_multi \
    --control.type=record \
    --control.fps=30 \
    --control.play_sounds=false \
    --control.single_task="演示如何用Alicia-D机械臂移动一个方块（带视觉）" \
    --control.root=/home/ubuntu/lerobot_datasets_multi \
    --control.repo_id=ubuntu/alicia_visual_demo_multi \
    --control.num_episodes=10 \
    --control.warmup_time_s=10 \
    --control.episode_time_s=18 \
    --control.reset_time_s=8 \
    --control.push_to_hub=false \
    --control.video=false
```

**数据收集中:**

*   脚本运行后，会首先连接机械臂和摄像头。
*   **预热阶段**: 您有 `warmup_time_s` 秒的时间将机械臂移动到起始姿态。
*   **记录阶段**: LeRobot 会提示开始记录。在此期间，您操作 Alicia-D 机械臂执行任务，LeRobot 会记录下机械臂的关节状态、夹爪状态以及摄像头图像 (如果配置了)。此阶段持续 `episode_time_s` 秒。
*   **重置阶段**: 一个回合记录完成后，您有 `reset_time_s` 秒的时间将场景和机械臂复位，为下一个回合做准备。
*   这个过程会重复 `num_episodes` 次。

**键盘控制 (在记录过程中):**
*   按 `→` (右箭头键): 提前结束当前回合的记录 (或重置阶段) 并进入下一阶段。
*   按 `←` (左箭头键): 取消当前回合的记录，并重新开始记录当前回合。
*   按 `ESC` (退出键): 停止整个数据收集过程。

数据收集完成后，您可以在您指定的 `--control.root` 路径下找到生成的数据集文件夹。

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
python lerobot/scripts/train.py \
    --policy.type=diffusion \
    --dataset.repo_id=ubuntu/alicia_visual_demo_dataset \
    --dataset.root=/home/ubuntu/lerobot_datasets \
    --output_dir=/home/ubuntu/lerobot_trained_result
```

### HuggingFace Hub 数据集训练
对于 HuggingFace Hub 上的数据集，只需要指定 `repo_id`：
```bash
python lerobot/scripts/train.py \
    --policy.type=diffusion \
    --dataset.repo_id=username/dataset_name \
    --output_dir=/path/to/training_result
```

## 8. 模型验证
进入训练结果
```/path_to_training_result/checkpoints/last/pretrained_model/config.json```
确保首行已添加训练类型
```
    "type": "difussion",
```
参考`examples/5_inference_dp.py`修改对应参数验证训练结果

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
