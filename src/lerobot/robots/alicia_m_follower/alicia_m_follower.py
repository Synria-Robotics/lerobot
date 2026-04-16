#!/usr/bin/env python3
# Copyright (c) 2025 Synria Robotics Co., Ltd.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# Author: Synria Robotics Team
# Website: https://synriarobotics.ai

import logging
import math
import time
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_alicia_m_follower import AliciaMFollowerConfig

logger = logging.getLogger(__name__)


def _import_sdk():
    """Lazy import of Alicia-M SDK."""
    try:
        import alicia_m_sdk

        return alicia_m_sdk, True
    except ImportError as e:
        logger.warning(
            "Alicia-M SDK not available. Please install alicia_m_sdk package and dependencies. "
            f"Error: {e}"
        )
        return None, False
    except Exception as e:
        logger.warning(f"Failed to import Alicia-M SDK. Error: {e}")
        return None, False


class AliciaMFollower(Robot):
    """Alicia-M follower arm implementation for LeRobot."""

    config_class = AliciaMFollowerConfig
    name = "alicia_m_follower"

    def __init__(self, config: AliciaMFollowerConfig):
        super().__init__(config)
        self.config = config
        self._arm_connected = False
        self._arm_connection_required = True

        alicia_m_sdk, sdk_available = _import_sdk()
        if not sdk_available:
            raise ImportError("Alicia-M SDK is not available. Please install alicia_m_sdk package.")

        # Keep auto_connect=False so LeRobot controls lifecycle in connect()/disconnect().
        self.robot_api: Any = alicia_m_sdk.create_robot(
            port=self.config.port,
            version=self.config.version,
            variant=self.config.variant,
            auto_connect=False,
            base_link=self.config.base_link,
            end_link=self.config.end_link,
            baudrate=self.config.baudrate,
            control_aim=self.config.control_aim,
            control_mode=self.config.control_mode,
            skip_mit_init=self.config.skip_mit_init,
            debug_mode=self.config.debug_mode,
        )

        self._joint_names = [f"joint{i}" for i in range(1, 7)]
        self._gripper_name = "gripper"
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        ft = {f"{name}.pos": float for name in self._joint_names}
        ft[f"{self._gripper_name}.pos"] = float
        return ft

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam_key: (self.config.cameras[cam_key].height, self.config.cameras[cam_key].width, 3)
            for cam_key in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        if self.robot_api is None:
            return False
        arm_ok = True
        if self._arm_connection_required:
            arm_ok = self._arm_connected and self.robot_api.is_connected()
        return arm_ok and all(cam.is_connected for cam in self.cameras.values())

    @property
    def uses_teleop_state_for_observation(self) -> bool:
        return self.config.use_teleop_state_for_observation

    def _should_connect_arm(self) -> bool:
        if self.config.connect_arm is None:
            return bool(self.config.port)
        return self.config.connect_arm

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self._arm_connection_required = self._should_connect_arm()
        if self._arm_connection_required:
            if not self.robot_api.connect():
                raise ConnectionError("Failed to connect to Alicia-M robot")
            self._arm_connected = True

            if not self.is_calibrated and calibrate:
                logger.info("No calibration action is required for Alicia-M by default.")
                self.calibrate()
        else:
            logger.info(f"{self} skipping arm connection; using cameras only.")

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        logger.info("Alicia-M robots are typically pre-calibrated. Use zero_calibration() if needed.")

    def configure(self) -> None:
        control_mode = (self.config.control_mode or "").lower()
        if control_mode == "mit":
            logger.info(
                "%s using MIT %s mode.",
                self,
                "interpolation" if self.config.use_interpolation else "direct PD",
            )
        elif not self.config.use_interpolation:
            logger.warning(
                "%s use_interpolation is ignored when control_mode=%s.",
                self,
                self.config.control_mode,
            )
        logger.debug(f"{self} configured.")

    def _get_joint_state_with_retry(self, retries: int = 3, sleep_s: float = 0.02):
        """Read joint_gripper state with retry for transient SDK read failures."""
        last_error: Exception | None = None
        for _ in range(retries):
            try:
                state = self.robot_api.get_robot_state("joint_gripper")
                if state is not None:
                    return state
            except Exception as e:  # noqa: BLE001
                last_error = e
            time.sleep(sleep_s)
        if last_error is not None:
            raise last_error
        return None

    def _get_joint_angles_deg(self, retries: int = 3, sleep_s: float = 0.001) -> list[float] | None:
        """Read current joint angles (deg) with retry."""
        last_error: Exception | None = None
        for _ in range(retries):
            try:
                joints_rad = self.robot_api.get_robot_state("joint")
                if joints_rad is not None and len(joints_rad) >= 6:
                    return [float(angle) * 180.0 / math.pi for angle in joints_rad[:6]]
            except Exception as e:  # noqa: BLE001
                last_error = e
            time.sleep(sleep_s)
        if last_error is not None:
            raise last_error
        return None

    def _extract_joint_gripper_state(self, joint_state: Any) -> tuple[list[float], float]:
        """Support both dict-style and object-style SDK state payloads."""
        if isinstance(joint_state, dict):
            joint_angles_rad = joint_state.get("angles")
            gripper_raw = joint_state.get("gripper")
        else:
            joint_angles_rad = getattr(joint_state, "angles", None)
            gripper_raw = getattr(joint_state, "gripper", None)

        if joint_angles_rad is None:
            raise ValueError(f"{self} joint_gripper state is missing 'angles': {joint_state!r}")
        if len(joint_angles_rad) != 6:
            raise ValueError(f"Expected 6 joint angles, got {len(joint_angles_rad)}")

        joint_angles_rad = [float(angle) for angle in joint_angles_rad]
        gripper_value = 0.0 if gripper_raw is None else float(gripper_raw)
        return joint_angles_rad, gripper_value

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self._arm_connection_required:
            start = time.perf_counter()
            joint_state = self._get_joint_state_with_retry()
            if joint_state is None:
                raise DeviceNotConnectedError(f"Failed to read robot state from {self}")

            joint_angles_rad, gripper_value = self._extract_joint_gripper_state(joint_state)
            joint_angles_deg = [angle * 180.0 / math.pi for angle in joint_angles_rad]
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read state: {dt_ms:.1f}ms")
        else:
            joint_angles_deg = [0.0 for _ in self._joint_names]
            gripper_value = 0.0

        obs_dict: dict[str, Any] = {}

        # print("======================================")
        # print("joint_angles_deg: ", joint_angles_deg)
        # print("======================================")
        for i, joint_name in enumerate(self._joint_names):
            obs_dict[f"{joint_name}.pos"] = float(joint_angles_deg[i])
        obs_dict[f"{self._gripper_name}.pos"] = float(gripper_value)

        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        if not self._arm_connection_required:
            logger.warning(f"{self} arm connection disabled; skipping send_action.")
            return action

        goal_pos_deg = {
            key.removesuffix(".pos"): float(val)
            for key, val in action.items()
            if key.endswith(".pos") and not key.startswith("gripper")
        }
        gripper_key = f"{self._gripper_name}.pos"
        if gripper_key in action:
            gripper_value = int(float(action[gripper_key]))
            gripper_value = max(0, min(1000, gripper_value))
        else:
            gripper_value = None

        # Leader action format is degrees. Keep current joint position when a joint key is missing.
        current_joints_deg = self._get_joint_angles_deg()
        if current_joints_deg is None:
            current_joints_deg = [0.0] * len(self._joint_names)

        if self.config.max_relative_target is not None:
            goal_present_pos = {}
            for i, joint_name in enumerate(self._joint_names):
                if joint_name in goal_pos_deg:
                    goal_present_pos[joint_name] = (goal_pos_deg[joint_name], current_joints_deg[i])
            goal_present_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)
            for joint_name, g_pos_deg in goal_present_pos.items():
                goal_pos_deg[joint_name] = g_pos_deg

        goal_joints_deg: list[float] = []
        for i, joint_name in enumerate(self._joint_names):
            goal_joints_deg.append(float(goal_pos_deg.get(joint_name, current_joints_deg[i])))

        # print("======================================")
        # print("goal_joints_deg: ", goal_joints_deg)
        # print("======================================")
        # success = True
        success = self.robot_api.set_robot_state(
            target_joints=goal_joints_deg,
            gripper_value=gripper_value,
            joint_format="deg",
            speed=self.config.speed,
            wait_for_completion=False,
            use_interpolation=self.config.use_interpolation,
        )
        if not success:
            logger.warning(f"Failed to send action to {self}")

        sent_action = {f"{joint_name}.pos": float(goal_joints_deg[i]) for i, joint_name in enumerate(self._joint_names)}
        if gripper_value is not None:
            sent_action[f"{self._gripper_name}.pos"] = float(gripper_value)
        return sent_action

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self._arm_connection_required and self.robot_api is not None and self._arm_connected:
            if self.config.disable_torque_on_disconnect:
                try:
                    self.robot_api.torque_control("off")
                except Exception as e:
                    logger.warning(f"Failed to disable torque on disconnect: {e}")
            self.robot_api.disconnect()
            self._arm_connected = False

        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
