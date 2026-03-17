#!/usr/bin/env python
"""Alicia-D Follower Configuration

Configuration class for Alicia-D Follower robot integration with LeRobot.

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


@RobotConfig.register_subclass("alicia_d_follower")
@dataclass
class AliciaDFollowerConfig(RobotConfig):
    # Port to connect to the arm
    port: str = ""

    disable_torque_on_disconnect: bool = False

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a dictionary that maps motor
    # names to the max_relative_target value for that motor.
    max_relative_target: float | dict[str, float] | None = None

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Gripper type for Alicia-D (e.g., "50mm" or "100mm")
    gripper_type: str | None = None

    # Debug mode for SDK
    debug_mode: bool = False

    # Speed in degrees per second for motion commands
    speed_deg_s: int = 100

    # Allow teleop state to overwrite joint observations when the leader is
    # directly wired to the follower.
    use_teleop_state_for_observation: bool = True

    # Whether to connect to the follower arm over serial.
    # If None, connect only when `port` is set.
    connect_arm: bool | None = None
