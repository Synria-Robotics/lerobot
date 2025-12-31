#!/usr/bin/env python
"""Alicia-D Follower Arm - LeRobot Integration

This module provides LeRobot-compatible interface for the Alicia-D robot arm,
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
import math
import time
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_alicia_d_follower import AliciaDFollowerConfig

logger = logging.getLogger(__name__)

# Lazy import function for Alicia-D SDK
def _import_sdk():
    """Lazy import of Alicia-D SDK to avoid warnings when module is imported but not used."""
    try:
        import alicia_d_sdk
        return alicia_d_sdk, True
    except ImportError as e:
        logger.warning(
            f"Alicia-D SDK not available. Please install alicia_d_sdk package and its dependencies. "
            f"Error: {e}"
        )
        return None, False
    except Exception as e:
        logger.warning(
            f"Failed to import Alicia-D SDK. This may be due to missing dependencies. "
            f"Error: {e}"
        )
        return None, False


class AliciaDFollower(Robot):
    """
    Alicia-D Follower Arm - LeRobot integration using Alicia-D SDK.
    
    This implementation provides a LeRobot-compatible interface for the Alicia-D robot arm,
    using alicia_d_sdk.create_robot() for robot control, following the same pattern as
    the SDK examples.
    """

    config_class = AliciaDFollowerConfig
    name = "alicia_d_follower"

    def __init__(self, config: AliciaDFollowerConfig):
        super().__init__(config)
        self.config = config
        
        # Lazy import SDK only when robot is instantiated
        alicia_d_sdk, sdk_available = _import_sdk()
        if not sdk_available:
            raise ImportError("Alicia-D SDK is not available. Please install alicia_d_sdk package.")
        
        # Create robot instance using SDK (same pattern as demos)
        self.robot_api: Any = alicia_d_sdk.create_robot(
            port=self.config.port,
            gripper_type=self.config.gripper_type,
            debug_mode=self.config.debug_mode,
            auto_connect=False,  # Manual connection in connect() method
        )
        
        # Joint names: 6 joints (joint1-joint6) + 1 separate gripper
        # Note: Gripper is NOT a joint - it's a separate actuator
        self._joint_names = [f"joint{i}" for i in range(1, 7)]  # 6 joints only
        self._gripper_name = "gripper"  # Separate from joints
        
        # Cameras
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        """Motor/joint features dictionary.
        
        Returns features for 6 joints (joint1.pos through joint6.pos) 
        """
        ft = {f"{name}.pos": float for name in self._joint_names}  # 6 joints
        ft[f"{self._gripper_name}.pos"] = float  # 1 separate gripper
        return ft

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """Camera features dictionary."""
        return {
            cam_key: (self.config.cameras[cam_key].height, self.config.cameras[cam_key].width, 3)
            for cam_key in self.cameras
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
        """Check if robot is connected."""
        if self.robot_api is None:
            return False
        return self.robot_api.is_connected() and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        """
        Connect to the robot.
        
        We assume that at connection time, arm is in a rest position,
        and torque can be safely disabled to run calibration.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # Manually connect the robot API
        if not self.robot_api.connect():
            raise ConnectionError("Failed to connect to Alicia-D robot")

        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()

        # Connect cameras
        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        """
        Check if robot is calibrated.
        
        Note: Alicia-D robots typically don't require calibration in the same way
        as Feetech motors. This always returns True.
        """
        return True

    def calibrate(self) -> None:
        """
        Calibrate the robot.
        
        Note: Alicia-D robots use zero_calibration() from the SDK if needed.
        This is a no-op by default as calibration is typically done at the factory.
        """
        logger.info("Alicia-D robots are typically pre-calibrated. Use zero_calibration() if needed.")
        # If calibration is needed, user can call robot_api.zero_calibration() directly

    def configure(self) -> None:
        """Apply configuration to the robot."""
        # Alicia-D SDK handles most configuration automatically
        # Additional configuration can be added here if needed
        logger.debug(f"{self} configured.")

    def get_observation(self) -> dict[str, Any]:
        """
        Get current observation from the robot.
        
        Returns:
            Dictionary with joint positions and camera images.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Read joint positions in degrees (record in degrees)
        start = time.perf_counter()
        joint_state = self.robot_api.get_robot_state("joint_gripper")
        
        if joint_state is None:
            raise DeviceNotConnectedError(f"Failed to read robot state from {self}")
        
        # Extract joints and gripper separately (gripper is NOT a joint)
        # Joints: 6 angles in radians
        joint_angles_rad = joint_state.angles
        if len(joint_angles_rad) != 6:
            raise ValueError(f"Expected 6 joint angles, got {len(joint_angles_rad)}")
        joint_angles_deg = [angle * 180.0 / math.pi for angle in joint_angles_rad]
        
        # Gripper: separate actuator, not a joint
        gripper_value = joint_state.gripper if joint_state.gripper is not None else 0.0
        
        # Convert to observation dictionary format (in degrees for recording)
        obs_dict = {}
        for i, joint_name in enumerate(self._joint_names):
            obs_dict[f"{joint_name}.pos"] = float(joint_angles_deg[i])
        obs_dict[f"{self._gripper_name}.pos"] = float(gripper_value)
        
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Send action command to the robot.
        
        Args:
            action: Dictionary with joint positions and gripper value.
            
        Returns:
            The action actually sent to the robot, potentially clipped.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Extract joint positions from action (actions are in degrees)
        goal_pos_deg = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos") and not key.startswith("gripper")}
        
        # Extract gripper value
        gripper_key = f"{self._gripper_name}.pos"
        if gripper_key in action:
            gripper_value = int(action[gripper_key])
            gripper_value = max(0, min(1000, gripper_value))  # Clip to valid range
        else:
            gripper_value = None  # Keep current gripper position

        # Apply safety limits if configured
        # /!\ Slower fps expected due to reading from the follower.
        if self.config.max_relative_target is not None:
            # Read joints in degrees for safety comparison (both goal and present in degrees)
            present_joints_rad = self.robot_api.get_robot_state("joint")
            if present_joints_rad is not None:
                # Convert from radians to degrees
                present_joints_deg = [angle * 180.0 / math.pi for angle in present_joints_rad]
                # Convert goal_pos dict to goal_present_pos format for safety clipping (in degrees)
                goal_present_pos = {}
                for i, joint_name in enumerate(self._joint_names):
                    if joint_name in goal_pos_deg:
                        goal_present_pos[joint_name] = (goal_pos_deg[joint_name], present_joints_deg[i])
                
                # Apply safety clipping (in degrees)
                goal_present_pos = ensure_safe_goal_position(
                    goal_present_pos, 
                    self.config.max_relative_target
                )
                
                # Update goal_pos_deg with clipped values
                for joint_name, g_pos_deg in goal_present_pos.items():
                    goal_pos_deg[joint_name] = g_pos_deg

        # Convert goal_pos from dict to list format for API (in degrees)
        goal_joints_deg = []
        for joint_name in self._joint_names:
            if joint_name in goal_pos_deg:
                goal_joints_deg.append(goal_pos_deg[joint_name])
            else:
                logger.warning(f"Action missing joint: {joint_name}. Using 0.0 for missing joint.")
                goal_joints_deg.append(0.0)

        # Send command to robot (using degrees format)
        success = self.robot_api.set_robot_state(
            target_joints=goal_joints_deg,
            gripper_value=gripper_value,
            joint_format='deg',
            speed_deg_s=self.config.speed_deg_s,
            wait_for_completion=False  # Don't wait, return immediately
        )
        
        if not success:
            logger.warning(f"Failed to send action to {self}")

        # Return the action that was sent (in degrees for recording)
        sent_action = {f"{motor}.pos": val for motor, val in goal_pos_deg.items()}
        if gripper_value is not None:
            sent_action[f"{self._gripper_name}.pos"] = float(gripper_value)
        
        return sent_action

    def disconnect(self) -> None:
        """Disconnect from the robot and perform cleanup."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Disconnect robot
        if self.robot_api is not None:
            if self.config.disable_torque_on_disconnect:
                try:
                    self.robot_api.torque_control("off")
                except Exception as e:
                    logger.warning(f"Failed to disable torque on disconnect: {e}")
            self.robot_api.disconnect()

        # Disconnect cameras
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")

