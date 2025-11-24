#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass, field

from lerobot.cameras.configs import CameraConfig, Cv2Rotation
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

from ..config import RobotConfig


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


