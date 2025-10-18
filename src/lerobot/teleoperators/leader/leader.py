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

import logging
from time import perf_counter
from typing import Any

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.robots.alicia_d.sdk_registry import get_controller, release_controller

from ..teleoperator import Teleoperator
from .config_leader import LeaderConfig

logger = logging.getLogger(__name__)


class Leader(Teleoperator):
    """Alicia-D 示例教臂（leader arm）遥操作器。

    读取 leader 臂的关节与夹爪，输出动作键：
      joint1.pos ... joint6.pos, gripper.pos
    """

    config_class = LeaderConfig
    name = "leader"

    def __init__(self, config: LeaderConfig):
        super().__init__(config)
        self.config = config
        self._controller = None

        self._joint_names = [f"joint{i}" for i in range(1, 7)]
        self._gripper_name = "gripper"

    @property
    def action_features(self) -> dict[str, type]:
        ft = {f"{n}.pos": float for n in self._joint_names}
        ft[f"{self._gripper_name}.pos"] = float
        return ft

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._controller is not None and (
            getattr(self._controller, "is_connected", None) is None or self._controller.is_connected()
        )

    @property
    def is_calibrated(self) -> bool:
        # 示教臂通过 SDK 直接读取，无需在此层做标定
        return True

    def calibrate(self) -> None:
        # 不适用：标定流程在硬件/SDK 层完成
        return None

    def configure(self) -> None:
        # 无需额外配置
        return None

    def connect(self, calibrate: bool = True) -> None:  # noqa: ARG002 (unused)
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        self._controller = get_controller(
            port=self.config.port,
            baudrate=self.config.baudrate,
            robot_version=self.config.robot_version,
            gripper_type=self.config.gripper_type,
            debug_mode=self.config.debug_mode,
        )
        logger.info(f"{self} connected.")

    def get_action(self) -> dict[str, float]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        t0 = perf_counter()
        joints = self._controller.get_joints() or [0.0] * 6
        gripper = self._controller.get_gripper() or 0.0  # 0..100
        act = {f"{n}.pos": float(v) for n, v in zip(self._joint_names, joints)}
        act[f"{self._gripper_name}.pos"] = float(gripper)
        dt_ms = (perf_counter() - t0) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return act

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # 暂不实现力反馈
        return None

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        try:
            release_controller(
                port=self.config.port,
                baudrate=self.config.baudrate,
                robot_version=self.config.robot_version,
                gripper_type=self.config.gripper_type,
                debug_mode=self.config.debug_mode,
            )
        finally:
            self._controller = None
        logger.info(f"{self} disconnected.")


