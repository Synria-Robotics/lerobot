#!/usr/bin/env python
"""Bimanual Alicia-D Follower Arms - LeRobot Integration

This module provides LeRobot-compatible interface for bimanual Alicia-D robot arms,
using the SynriaRobotAPI from Alicia-D SDK.

Copyright (c) 2025 Synria Robotics Co., Ltd.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

Author: Synria Robotics Team
Website: https://synriarobotics.ai
"""

import logging
import time
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots.alicia_d_follower import AliciaDFollower
from lerobot.robots.alicia_d_follower.config_alicia_d_follower import AliciaDFollowerConfig

from ..robot import Robot
from .config_bi_alicia_d_follower import BiAliciaDFollowerConfig

logger = logging.getLogger(__name__)


class BiAliciaDFollower(Robot):
    """
    Bimanual Alicia-D Follower Arms - LeRobot integration using SynriaRobotAPI from Alicia-D SDK.
    
    This bimanual robot uses two Alicia-D follower arms (left and right) and provides
    a unified interface for controlling both arms simultaneously.
    """

    config_class = BiAliciaDFollowerConfig
    name = "bi_alicia_d_follower"

    def __init__(self, config: BiAliciaDFollowerConfig):
        super().__init__(config)
        self.config = config

        # Create left arm configuration
        left_arm_config = AliciaDFollowerConfig(
            id=f"{config.id}_left" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.left_arm_port,
            connect_arm=config.left_arm_connect,
            disable_torque_on_disconnect=config.left_arm_disable_torque_on_disconnect,
            max_relative_target=config.left_arm_max_relative_target,
            gripper_type=config.left_arm_gripper_type,
            debug_mode=config.left_arm_debug_mode,
            speed_deg_s=config.left_arm_speed_deg_s,
            cameras={},
        )

        # Create right arm configuration
        right_arm_config = AliciaDFollowerConfig(
            id=f"{config.id}_right" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.right_arm_port,
            connect_arm=config.right_arm_connect,
            disable_torque_on_disconnect=config.right_arm_disable_torque_on_disconnect,
            max_relative_target=config.right_arm_max_relative_target,
            gripper_type=config.right_arm_gripper_type,
            debug_mode=config.right_arm_debug_mode,
            speed_deg_s=config.right_arm_speed_deg_s,
            cameras={},
        )

        # Create left and right arm instances
        self.left_arm = AliciaDFollower(left_arm_config)
        self.right_arm = AliciaDFollower(right_arm_config)
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        """Motor/joint features dictionary with left_ and right_ prefixes."""
        # Remove ".pos" suffix from keys, then add prefix
        left_ft = {
            f"left_{key.removesuffix('.pos')}.pos": float 
            for key in self.left_arm._motors_ft.keys()
        }
        right_ft = {
            f"right_{key.removesuffix('.pos')}.pos": float 
            for key in self.right_arm._motors_ft.keys()
        }
        return {**left_ft, **right_ft}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """Camera features dictionary."""
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """Observation features combining motors and cameras."""
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        """Action features (same as motor features)."""
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        """Check if both arms and cameras are connected."""
        return (
            self.left_arm.is_connected
            and self.right_arm.is_connected
            and all(cam.is_connected for cam in self.cameras.values())
        )

    @property
    def uses_teleop_state_for_observation(self) -> bool:
        """Whether the recorder should replace joint observations with teleop state."""
        return self.config.use_teleop_state_for_observation

    def connect(self, calibrate: bool = True) -> None:
        """Connect both arms and cameras."""
        self.left_arm.connect(calibrate)
        self.right_arm.connect(calibrate)

        for cam in self.cameras.values():
            cam.connect()

    @property
    def is_calibrated(self) -> bool:
        """Check if both arms are calibrated."""
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated

    def calibrate(self) -> None:
        """Calibrate both arms."""
        self.left_arm.calibrate()
        self.right_arm.calibrate()

    def configure(self) -> None:
        """Configure both arms."""
        self.left_arm.configure()
        self.right_arm.configure()

    def get_observation(self) -> dict[str, Any]:
        """Get observation from both arms and cameras."""
        obs_dict = {}

        # Add "left_" prefix to left arm observations
        left_obs = self.left_arm.get_observation()
        obs_dict.update({f"left_{key}": value for key, value in left_obs.items()})

        # Add "right_" prefix to right arm observations
        right_obs = self.right_arm.get_observation()
        obs_dict.update({f"right_{key}": value for key, value in right_obs.items()})

        # Add camera observations
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Send action to both arms.
        
        Args:
            action: Dictionary with "left_" and "right_" prefixed joint positions and gripper values.
            
        Returns:
            The action actually sent to both arms, potentially clipped.
        """
        # Remove "left_" prefix for left arm
        left_action = {
            key.removeprefix("left_"): value for key, value in action.items() if key.startswith("left_")
        }
        
        # Remove "right_" prefix for right arm
        right_action = {
            key.removeprefix("right_"): value for key, value in action.items() if key.startswith("right_")
        }

        # Send actions to respective arms
        send_action_left = self.left_arm.send_action(left_action)
        send_action_right = self.right_arm.send_action(right_action)

        # Add prefixes back to returned actions
        prefixed_send_action_left = {f"left_{key}": value for key, value in send_action_left.items()}
        prefixed_send_action_right = {f"right_{key}": value for key, value in send_action_right.items()}

        return {**prefixed_send_action_left, **prefixed_send_action_right}

    def disconnect(self) -> None:
        """Disconnect both arms and cameras."""
        self.left_arm.disconnect()
        self.right_arm.disconnect()

        for cam in self.cameras.values():
            cam.disconnect()
