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

from __future__ import annotations

import logging
import time
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_alicia_d import AliciaDConfig


logger = logging.getLogger(__name__)


class AliciaD(Robot):
    """
    Alicia-D 六轴机械臂（带夹爪）对接公司 SDK 的新版 LeRobot 机器人实现。

    观测/动作采用新版字典风格：
      - 关节位置键: joint1.pos ... joint6.pos, gripper.pos
      - 相机键: 与配置中的相机键一致（值为 HxWxC 的 numpy 数组）
    """

    config_class = AliciaDConfig
    name = "alicia_d"

    def __init__(self, config: AliciaDConfig):
        super().__init__(config)
        self.config = config

        # 延迟导入 SDK，允许在无 SDK 环境下导入模块但不可连接
        self._sdk_available = False
        self._controller = None
        self._session = None

        try:
            # 兼容旧版命名
            from alicia_d_sdk.controller import get_default_session, ControlApi  # type: ignore

            self._get_default_session = get_default_session
            self._ControlApi = ControlApi
            self._sdk_available = True
        except Exception:
            logger.warning("未找到 Alicia-D SDK。请安装 `alicia_d_sdk` 包以连接真实硬件。")

        self.cameras = make_cameras_from_configs(config.cameras)

        # 关节命名（与旧版一致 6 关节 + 夹爪）
        self._joint_names = [f"joint{i}" for i in range(1, 7)]
        self._gripper_name = "gripper"

    # ===== Features =====
    @property
    def _motors_ft(self) -> dict[str, type]:
        ft = {f"{name}.pos": float for name in self._joint_names}
        ft[f"{self._gripper_name}.pos"] = float
        return ft

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    # ===== Connection =====
    @property
    def is_connected(self) -> bool:
        hw_connected = self._controller is not None
        cams_connected = all(cam.is_connected for cam in self.cameras.values())
        return hw_connected and cams_connected

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # 建立硬件连接
        if not self._sdk_available:
            raise DeviceNotConnectedError("Alicia-D SDK 不可用，无法连接硬件。请安装 `alicia_d_sdk`。")

        self._session = self._get_default_session(port=self.config.port, baudrate=self.config.baudrate)
        self._controller = self._ControlApi(session=self._session)

        # 连接相机
        for cam in self.cameras.values():
            cam.connect()

        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        # 该机器人不需要在 LeRobot 层面做电机标定，返回 True
        return True

    def calibrate(self) -> None:
        # 不适用：标定流程在 SDK/设备侧完成
        return None

    def configure(self) -> None:
        # 目前无额外配置；如需启用 SDK 在线平滑，可在此启动
        return None

    # ===== IO =====
    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()

        joint_rad = self._controller.get_joints()
        gripper_rad = self._controller.get_gripper()

        obs_dict: dict[str, Any] = {}

        for name, val in zip(self._joint_names, joint_rad):
            obs_dict[f"{name}.pos"] = float(val)
        obs_dict[f"{self._gripper_name}.pos"] = float(gripper_rad)

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # 解析关节/夹爪目标
        goal_pos = {key.removesuffix(".pos"): float(val) for key, val in action.items() if key.endswith(".pos")}

        # 安全限制（相对幅度裁剪）
        if self.config.max_relative_target is not None:
            present = self.get_observation()  # 读取一次当前位姿（仅关节相关键）
            present_pos = {k: float(v) for k, v in present.items() if k.endswith(".pos")}
            goal_present = {f"{name}.pos": (goal_pos[name], present_pos[f"{name}.pos"]) for name in goal_pos}
            safe_goal = ensure_safe_goal_position(goal_present, self.config.max_relative_target)
            goal_pos = {k.removesuffix(".pos"): v for k, v in safe_goal.items()}

        # 下发到 SDK
        joint_targets = [goal_pos.get(name, None) for name in self._joint_names]
        if any(v is None for v in joint_targets):
            # 填补未提供的关节为当前值，避免 SDK 接口拒绝
            present = self.get_observation()
            for i, name in enumerate(self._joint_names):
                if joint_targets[i] is None:
                    joint_targets[i] = float(present[f"{name}.pos"])  # type: ignore

        gripper_target = goal_pos.get(self._gripper_name, None)
        if gripper_target is None:
            present = self.get_observation()
            gripper_target = float(present[f"{self._gripper_name}.pos"])  # type: ignore

        # 调用控制接口（默认使用关节空间一次性下发）
        try:
            # 若需要在线平滑，可替换为 setJointTargetOnline / setGripperTargetOnline
            self._controller.joint_controller.set_joint_angles(joint_targets)  # type: ignore[attr-defined]
            self._controller.joint_controller.set_gripper(gripper_target)  # type: ignore[attr-defined]
        except AttributeError:
            # 兼容部分 SDK 仅提供一体化接口的情况
            if hasattr(self._controller, "setJointTargetOnline") and hasattr(self._controller, "setGripperTargetOnline"):
                self._controller.setJointTargetOnline(joint_targets)
                self._controller.setGripperTargetOnline(gripper_target)
            else:
                raise

        return {**{f"{n}.pos": float(v) for n, v in zip(self._joint_names, joint_targets)}, f"{self._gripper_name}.pos": float(gripper_target)}

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # 相机先断开
        for cam in self.cameras.values():
            cam.disconnect()

        # 断开硬件会话
        try:
            if hasattr(self._session, "joint_controller"):
                self._session.joint_controller.disconnect()  # type: ignore[attr-defined]
        finally:
            self._controller = None
            self._session = None

        logger.info(f"{self} disconnected.")


