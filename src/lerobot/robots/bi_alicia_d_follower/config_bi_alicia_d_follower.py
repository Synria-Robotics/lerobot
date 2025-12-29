#!/usr/bin/env python
"""Bimanual Alicia-D Follower Configuration

Configuration class for Bimanual Alicia-D Follower robot integration with LeRobot.

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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("bi_alicia_d_follower")
@dataclass
class BiAliciaDFollowerConfig(RobotConfig):
    left_arm_port: str = ""
    right_arm_port: str = ""

    # Optional left arm settings
    left_arm_disable_torque_on_disconnect: bool = False
    left_arm_max_relative_target: float | dict[str, float] | None = None
    left_arm_gripper_type: str | None = None
    left_arm_debug_mode: bool = False
    left_arm_speed_deg_s: int = 20

    # Optional right arm settings
    right_arm_disable_torque_on_disconnect: bool = False
    right_arm_max_relative_target: float | dict[str, float] | None = None
    right_arm_gripper_type: str | None = None
    right_arm_debug_mode: bool = False
    right_arm_speed_deg_s: int = 20

    # cameras (shared between both arms)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

