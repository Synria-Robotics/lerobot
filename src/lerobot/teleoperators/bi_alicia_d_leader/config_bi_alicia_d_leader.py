#!/usr/bin/env python
"""Bimanual Alicia-D Leader Configuration

Configuration class for Bimanual Alicia-D Leader teleoperator integration with LeRobot.

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

from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("bi_alicia_d_leader")
@dataclass
class BiAliciaDLeaderConfig(TeleoperatorConfig):
    # Ports to connect to the leader arms
    left_arm_port: str = ""
    right_arm_port: str = ""
    
    # Gripper types for Alicia-D (e.g., "50mm" or "100mm")
    left_arm_gripper_type: str | None = None
    right_arm_gripper_type: str | None = None
    
    # Debug mode for SDK
    left_arm_debug_mode: bool = False
    right_arm_debug_mode: bool = False
    
    # Whether the leader arms directly control the follower arms via hardware wire.
    # If True, actions don't need to be sent through the computer (default: True).
    # If False, actions will be sent through the computer from teleoperator to robot.
    directly_controls_robot: bool = True

