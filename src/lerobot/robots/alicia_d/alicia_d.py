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
import numpy as np
import time
from functools import cached_property
from typing import Any, List, Dict

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_alicia_d import AliciaDConfig
from .sdk_registry import get_controller, release_controller


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

        # 延迟导入 SDK（v6.0.0），支持从本仓库子模块回退导入
        import sys
        from pathlib import Path

        self._sdk_available = False
        self._controller = None
        self._sdk_mod = None

        try:
            import alicia_d_sdk  # type: ignore

            self._sdk_mod = alicia_d_sdk
            self._sdk_available = True
        except Exception:
            # fallback: 尝试从本仓库子模块路径导入
            try:
                repo_root = Path(__file__).resolve().parents[5]
                sdk_dir = repo_root / "lerobot" / "Alicia-D-SDK"
                if str(sdk_dir) not in sys.path:
                    sys.path.insert(0, str(sdk_dir))
                import alicia_d_sdk  # type: ignore

                self._sdk_mod = alicia_d_sdk
                self._sdk_available = True
            except Exception:
                logger.warning("未找到 Alicia-D SDK。请安装 `alicia_d_sdk` 或初始化子模块 `Alicia-D-SDK`。")

        # 单臂相机
        self.cameras = make_cameras_from_configs(config.cameras)

        # 关节命名（与旧版一致 6 关节 + 夹爪）
        self._joint_names = [f"joint{i}" for i in range(1, 7)]
        self._gripper_name = "gripper"

    # ===== Features =====
    @property
    def _motors_ft(self) -> dict[str, type]:
        # 为了兼容 hw_to_dataset_features，将关节位置定义为单独的 float 类型
        # 但实际观测中我们使用 joint_positions 列表
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
        hw_connected = bool(self._controller) and (
            getattr(self._controller, "is_connected", None) is None or self._controller.is_connected()
        )
        cams_connected = all(cam.is_connected for cam in self.cameras.values())
        return hw_connected and cams_connected

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # 建立硬件连接
        if not self._sdk_available:
            raise DeviceNotConnectedError("Alicia-D SDK 不可用，无法连接硬件。请安装 `alicia_d_sdk` 或初始化子模块。")

        robot_version = getattr(self.config, "robot_version", "v5_6")
        gripper_type = getattr(self.config, "gripper_type", "50mm")
        debug_mode = getattr(self.config, "debug_mode", False)

        # 使用共享控制器，便于与示教臂复用同一端口
        self._controller = get_controller(
            port=self.config.port,
            baudrate=self.config.baudrate,
            robot_version=robot_version,
            gripper_type=gripper_type,
            debug_mode=debug_mode,
        )

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
        pose_info = self._controller.get_pose()
        gripper_rad = self._controller.get_gripper()
        obs_dict: dict[str, Any] = {}

        # 末端位姿数组： [x, y, z, qx, qy, qz, qw]
        pos = pose_info.get("position")
        quat = pose_info.get("quaternion_xyzw")
        if pos is None or quat is None:
            ee_pose = []
        else:
            pos_arr = np.asarray(pos).flatten()
            quat_arr = np.asarray(quat).flatten()
            if pos_arr.size == 0 or quat_arr.size == 0:
                ee_pose = []
            else:
                ee_pose = [float(x) for x in np.concatenate([pos_arr, quat_arr]).tolist()]
        obs_dict["ee_pose"] = ee_pose

        # 关节与夹爪
        for name, val in zip(self._joint_names, joint_rad):
            obs_dict[f"{name}.pos"] = float(val)
        # 兼容旧逻辑：提供 joint_positions 列表
        obs_dict["joint_positions"] = [float(v) for v in joint_rad]
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
        if "joint_positions" in action:
            # 新的结构：joint_positions 是一个包含6个关节位置的列表
            joint_targets = [float(val) for val in action["joint_positions"]]
        else:
            # 旧的结构：分别的关节键
            goal_pos = {key.removesuffix(".pos"): float(val) for key, val in action.items() if key.endswith(".pos")}
            joint_targets = [goal_pos.get(name) for name in self._joint_names]
            if any(v is None for v in joint_targets):
                present = self.get_observation()
                for i, name in enumerate(self._joint_names):
                    if joint_targets[i] is None:
                        joint_targets[i] = float(present["joint_positions"][i])  # type: ignore

        gripper_target = action.get(f"{self._gripper_name}.pos")

        # 仅当 execute_motion=True 时才下发到硬件
        if getattr(self.config, "execute_motion", False):
            self._controller.set_joint_target(joint_targets, joint_format="rad", speed_factor=1.0)
            if gripper_target is not None:
                gt = max(0.0, min(100.0, float(gripper_target)))
                self._controller.set_gripper_target(value=gt)

        sent = {f"{n}.pos": float(v) for n, v in zip(self._joint_names, joint_targets)}
        if gripper_target is not None:
            sent[f"{self._gripper_name}.pos"] = float(gripper_target)
        return sent

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # 相机先断开
        for cam in self.cameras.values():
            cam.disconnect()

        # 断开硬件
        try:
            if self._controller is not None:
                robot_version = getattr(self.config, "robot_version", "v5_6")
                gripper_type = getattr(self.config, "gripper_type", "50mm")
                debug_mode = getattr(self.config, "debug_mode", False)
                release_controller(
                    port=self.config.port,
                    baudrate=self.config.baudrate,
                    robot_version=robot_version,
                    gripper_type=gripper_type,
                    debug_mode=debug_mode,
                )
        finally:
            self._controller = None

        logger.info(f"{self} disconnected.")

    # ===== High-level IK API (proxy to SDK) =====
    def set_pose_target(
        self,
        *,
        target_pose: List[float],
        backend: str = "torch",
        method: str = "dls",
        display: bool = False,
        tolerance: float = 1e-4,
        max_iters: int = 100,
        multi_start: int = 0,
        use_random_init: bool = False,
        speed_factor: float = 1.0,
        execute: bool = True,
    ) -> Dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if not isinstance(target_pose, (list, tuple)) or len(target_pose) < 7:
            raise ValueError("target_pose 需为长度>=7的列表：[x,y,z,qx,qy,qz,qw]")

        # 归一化四元数，提升 IK 稳定性
        pos = [float(x) for x in target_pose[:3]]
        quat = [float(x) for x in target_pose[3:7]]
        q = np.asarray(quat, dtype=np.float64)
        n = np.linalg.norm(q)
        if not np.isfinite(n) or n < 1e-8:
            qn = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        else:
            qn = (q / n).astype(np.float64)
        target_pose_norm = pos + qn.tolist()

        return self._controller.set_pose_target(
            target_pose=target_pose_norm,
            backend=backend,
            method=method,
            display=display,
            tolerance=tolerance,
            max_iters=max_iters,
            multi_start=multi_start,
            use_random_init=use_random_init,
            speed_factor=speed_factor,
            execute=execute,
        )



