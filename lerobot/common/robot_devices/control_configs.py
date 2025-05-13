# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from pathlib import Path

import draccus

from lerobot.common.robot_devices.robots.configs import RobotConfig
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig


@dataclass
class ControlConfig(draccus.ChoiceRegistry):
    pass


@ControlConfig.register_subclass("calibrate")
@dataclass
class CalibrateControlConfig(ControlConfig):
    # List of arms to calibrate (e.g. `--arms='["left_follower","right_follower"]' left_leader`)
    arms: list[str] | None = None


@ControlConfig.register_subclass("teleoperate")
@dataclass
class TeleoperateControlConfig(ControlConfig):
    # Limit the maximum frames per second. By default, no limit.
    fps: int | None = None
    teleop_time_s: float | None = None
    # Display all cameras on screen
    display_data: bool = False


@ControlConfig.register_subclass("record")
@dataclass
class RecordControlConfig(ControlConfig):
    # 数据集标识符。按照惯例，应与'{hf_username}/{dataset_name}'匹配（例如：`lerobot/test`）。
    repo_id: str
    # 对录制过程中执行任务的简短但准确的描述（例如："拿起乐高积木并将其放入右侧的盒子中。"）
    single_task: str
    # 数据集将被存储的根目录（例如：'dataset/path'）。
    root: str | Path | None = None
    policy: PreTrainedConfig | None = None
    # 限制每秒帧数。默认情况下，使用策略的fps。
    fps: int | None = None
    # 开始数据收集前的预热秒数。它允许机器人设备预热和同步。
    warmup_time_s: int | float = 10
    # 每个回合的数据记录秒数。
    episode_time_s: int | float = 60
    # 每个回合后重置环境的秒数。
    reset_time_s: int | float = 60
    # 要记录的回合数。
    num_episodes: int = 50
    # 将数据集中的帧编码为视频
    video: bool = True
    # 上传数据集到Hugging Face hub。
    push_to_hub: bool = True
    # 上传到Hugging Face hub上的私有仓库。
    private: bool = False
    # 在hub上为您的数据集添加标签。
    tags: list[str] | None = None
    # 处理将帧保存为PNG的子进程数量。设置为0仅使用线程；
    # 设置为≥1使用子进程，每个子进程使用线程写入图像。最佳的进程
    # 和线程数量取决于您的系统。我们推荐每个摄像头使用4个线程，0个进程。
    # 如果fps不稳定，调整线程数量。如果仍然不稳定，尝试使用1个或更多子进程。
    num_image_writer_processes: int = 0
    # 每个摄像头在磁盘上将帧写为png图像的线程数量。
    # 过多的线程可能会导致远程操作fps不稳定，因为主线程被阻塞。
    # 线程不足可能会导致摄像头fps较低。
    num_image_writer_threads_per_camera: int = 4
    # 在屏幕上显示所有摄像头
    display_data: bool = False
    # 使用语音合成读取事件。
    play_sounds: bool = True
    # 在现有数据集上继续记录。
    resume: bool = False

    def __post_init__(self):
        # HACK：在这里我们再次解析cli参数以获取预训练路径（如果存在的话）。
        policy_path = parser.get_path_arg("control.policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("control.policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path


@ControlConfig.register_subclass("replay")
@dataclass
class ReplayControlConfig(ControlConfig):
    # Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).
    repo_id: str
    # Index of the episode to replay.
    episode: int
    # Root directory where the dataset will be stored (e.g. 'dataset/path').
    root: str | Path | None = None
    # Limit the frames per second. By default, uses the dataset fps.
    fps: int | None = None
    # Use vocal synthesis to read events.
    play_sounds: bool = True


@ControlConfig.register_subclass("remote_robot")
@dataclass
class RemoteRobotConfig(ControlConfig):
    log_interval: int = 100
    # Display all cameras on screen
    display_data: bool = False
    # Rerun configuration for remote robot (https://ref.rerun.io/docs/python/0.22.1/common/initialization_functions/#rerun.connect_tcp)
    viewer_ip: str | None = None
    viewer_port: str | None = None


@dataclass
class ControlPipelineConfig:
    robot: RobotConfig
    control: ControlConfig

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["control.policy"]
