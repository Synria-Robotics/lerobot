#!/usr/bin/env python
"""Alicia-D Leader Arm - LeRobot Teleoperator Integration

This module provides LeRobot-compatible teleoperator interface for the Alicia-D leader arm,
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
from typing import Any

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from .config_alicia_d_leader import AliciaDLeaderConfig

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


class AliciaDLeader(Teleoperator):
    """
    Alicia-D Leader Arm - LeRobot teleoperator integration using Alicia-D SDK.
    
    This teleoperator reads joint positions and button status from the leader arm
    and provides them as actions for follower arms. Leader arms use the same API
    as follower arms but don't have pose data.
    """

    config_class = AliciaDLeaderConfig
    name = "alicia_d_leader"

    def __init__(self, config: AliciaDLeaderConfig):
        super().__init__(config)
        self.config = config
        
        # Lazy import SDK only when teleoperator is instantiated
        alicia_d_sdk, sdk_available = _import_sdk()
        if not sdk_available:
            raise ImportError("Alicia-D SDK is not available. Please install alicia_d_sdk package.")
        
        # Create robot instance using SDK (leader arms use the same API)
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

    @property
    def action_features(self) -> dict[str, type]:
        """Action features dictionary.
        
        Returns features for 6 joints (joint1.pos through joint6.pos) 
        plus 1 separate gripper (gripper.pos).
        """
        ft = {f"{name}.pos": float for name in self._joint_names}  # 6 joints
        ft[f"{self._gripper_name}.pos"] = float  # 1 separate gripper
        return ft

    @property
    def feedback_features(self) -> dict[str, type]:
        """Feedback features dictionary (not used for leader arms)."""
        return {}

    @property
    def directly_controls_robot(self) -> bool:
        """
        Whether the leader arm directly controls the follower arm via hardware wire.
        
        Returns the value from configuration. If True, actions don't need to be sent
        through the computer. If False, actions will be sent through the computer.
        """
        return self.config.directly_controls_robot

    @property
    def uses_action_as_observation(self) -> bool:
        """Whether to use leader actions as robot observations during recording."""
        return self.config.use_action_as_observation

    @property
    def action_observation_delay_frames(self) -> int:
        """Frame delay between observation and action when using leader state."""
        return max(0, int(self.config.action_observation_delay_frames))

    @property
    def is_connected(self) -> bool:
        """Check if leader arm is connected."""
        if self.robot_api is None:
            return False
        return self.robot_api.is_connected()

    def connect(self, calibrate: bool = True) -> None:
        """Connect to the leader arm."""
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # Manually connect the robot API
        if not self.robot_api.connect():
            raise ConnectionError("Failed to connect to Alicia-D leader arm")

        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        """
        Check if leader arm is calibrated.
        
        Note: Alicia-D robots typically don't require calibration in the same way
        as Feetech motors. This always returns True.
        """
        return True

    def calibrate(self) -> None:
        """
        Calibrate the leader arm.
        
        Note: Alicia-D robots use zero_calibration() from the SDK if needed.
        This is a no-op by default as calibration is typically done at the factory.
        """
        logger.info("Alicia-D leader arms are typically pre-calibrated. Use zero_calibration() if needed.")

    def configure(self) -> None:
        """Apply configuration to the leader arm."""
        # Alicia-D SDK handles most configuration automatically
        logger.debug(f"{self} configured.")

    def get_action(self) -> dict[str, float]:
        """
        Get current action from the leader arm.
        
        Reads joint positions and gripper value from the leader arm.
        Leader arms don't have pose data, only joint positions.
        
        Returns:
            Dictionary with joint positions and gripper value in the format:
            {'joint1.pos': 0.5, 'joint2.pos': 0.3, ..., 'gripper.pos': 0.5}
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()
        
        # Get robot state once to avoid duplicate API calls
        state = self.robot_api.get_robot_state("joint_gripper")
        
        if state is None:
            raise DeviceNotConnectedError(f"Failed to read robot state from {self}")
        
        # Extract joints and gripper separately (gripper is NOT a joint)
        # Joints: 6 angles in radians
        joint_angles_rad = state.angles
        if len(joint_angles_rad) != 6:
            raise ValueError(f"Expected 6 joint angles, got {len(joint_angles_rad)}")
        joint_angles_deg = [angle * 180.0 / math.pi for angle in joint_angles_rad]
        
        # Gripper: separate actuator, not a joint
        gripper_value = state.gripper if state.gripper is not None else 0.0
        
        # Button status: leader arms have button status
        button_status = state.run_status_text if state else "idle"
        
        # Format as action dictionary (in degrees for recording)
        action = {}
        for i, joint_name in enumerate(self._joint_names):
            action[f"{joint_name}.pos"] = float(joint_angles_deg[i])
        action[f"{self._gripper_name}.pos"] = float(gripper_value)
        
        # Log button status for debugging (leader arms have buttons - any status is acceptable)
        logger.debug(f"{self} button status: {button_status}")
        
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        """
        Send feedback to the leader arm.
        
        Note: Force feedback is not currently implemented for Alicia-D leader arms.
        """
        # TODO: Implement force feedback if supported by hardware
        logger.debug(f"Feedback not implemented for {self}")

    def disconnect(self) -> None:
        """Disconnect from the leader arm and perform cleanup."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Disconnect robot
        if self.robot_api is not None:
            self.robot_api.disconnect()

        logger.info(f"{self} disconnected.")
