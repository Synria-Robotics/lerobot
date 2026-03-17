#!/usr/bin/env python
"""Bimanual Alicia-D Leader Arms - LeRobot Teleoperator Integration

This module provides LeRobot-compatible teleoperator interface for bimanual Alicia-D leader arms,
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
from functools import cached_property

from lerobot.teleoperators.alicia_d_leader import AliciaDLeader
from lerobot.teleoperators.alicia_d_leader.config_alicia_d_leader import AliciaDLeaderConfig

from ..teleoperator import Teleoperator
from .config_bi_alicia_d_leader import BiAliciaDLeaderConfig

logger = logging.getLogger(__name__)


class BiAliciaDLeader(Teleoperator):
    """
    Bimanual Alicia-D Leader Arms - LeRobot teleoperator integration using Alicia-D SDK.
    
    This bimanual teleoperator uses two Alicia-D leader arms (left and right) and provides
    a unified interface for reading joint positions and button status from both arms.
    Leader arms use the same API as follower arms but don't have pose data.
    """

    config_class = BiAliciaDLeaderConfig
    name = "bi_alicia_d_leader"

    def __init__(self, config: BiAliciaDLeaderConfig):
        super().__init__(config)
        self.config = config

        # Create left arm configuration
        left_arm_config = AliciaDLeaderConfig(
            id=f"{config.id}_left" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.left_arm_port,
            gripper_type=config.left_arm_gripper_type,
            debug_mode=config.left_arm_debug_mode,
            use_action_as_observation=config.use_action_as_observation,
            action_observation_delay_frames=config.action_observation_delay_frames,
        )

        # Create right arm configuration
        right_arm_config = AliciaDLeaderConfig(
            id=f"{config.id}_right" if config.id else None,
            calibration_dir=config.calibration_dir,
            port=config.right_arm_port,
            gripper_type=config.right_arm_gripper_type,
            debug_mode=config.right_arm_debug_mode,
            use_action_as_observation=config.use_action_as_observation,
            action_observation_delay_frames=config.action_observation_delay_frames,
        )

        # Create left and right arm instances
        self.left_arm = AliciaDLeader(left_arm_config)
        self.right_arm = AliciaDLeader(right_arm_config)

    @cached_property
    def action_features(self) -> dict[str, type]:
        """Action features dictionary with left_ and right_ prefixes."""
        # Remove ".pos" suffix from keys, then add prefix
        left_ft = {
            f"left_{key.removesuffix('.pos')}.pos": float 
            for key in self.left_arm.action_features.keys()
        }
        right_ft = {
            f"right_{key.removesuffix('.pos')}.pos": float 
            for key in self.right_arm.action_features.keys()
        }
        return {**left_ft, **right_ft}

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        """Feedback features dictionary (not used for leader arms)."""
        return {}

    @property
    def directly_controls_robot(self) -> bool:
        """
        Whether the leader arms directly control the follower arms via hardware wire.
        
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
        """Check if both leader arms are connected."""
        return self.left_arm.is_connected and self.right_arm.is_connected

    def connect(self, calibrate: bool = True) -> None:
        """Connect both leader arms."""
        self.left_arm.connect(calibrate)
        self.right_arm.connect(calibrate)

    @property
    def is_calibrated(self) -> bool:
        """Check if both leader arms are calibrated."""
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated

    def calibrate(self) -> None:
        """Calibrate both leader arms."""
        self.left_arm.calibrate()
        self.right_arm.calibrate()

    def configure(self) -> None:
        """Configure both leader arms."""
        self.left_arm.configure()
        self.right_arm.configure()

    def get_action(self) -> dict[str, float]:
        """
        Get current action from both leader arms.
        
        Reads joint positions and gripper values from both leader arms.
        Leader arms don't have pose data, only joint positions.
        
        Returns:
            Dictionary with left_ and right_ prefixed joint positions and gripper values.
        """
        action_dict = {}

        # Add "left_" prefix to left arm actions
        left_action = self.left_arm.get_action()
        action_dict.update({f"left_{key}": value for key, value in left_action.items()})

        # Add "right_" prefix to right arm actions
        right_action = self.right_arm.get_action()
        action_dict.update({f"right_{key}": value for key, value in right_action.items()})

        return action_dict

    def send_feedback(self, feedback: dict[str, float]) -> None:
        """
        Send feedback to both leader arms.
        
        Note: Force feedback is not currently implemented for Alicia-D leader arms.
        """
        # Remove "left_" prefix
        left_feedback = {
            key.removeprefix("left_"): value for key, value in feedback.items() if key.startswith("left_")
        }
        # Remove "right_" prefix
        right_feedback = {
            key.removeprefix("right_"): value for key, value in feedback.items() if key.startswith("right_")
        }

        if left_feedback:
            self.left_arm.send_feedback(left_feedback)
        if right_feedback:
            self.right_arm.send_feedback(right_feedback)

    def disconnect(self) -> None:
        """Disconnect both leader arms."""
        self.left_arm.disconnect()
        self.right_arm.disconnect()
