#!/usr/bin/env python
"""Alicia-D Leader Configuration

Configuration class for Alicia-D Leader teleoperator integration with LeRobot.

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


@TeleoperatorConfig.register_subclass("alicia_d_leader")
@dataclass
class AliciaDLeaderConfig(TeleoperatorConfig):
    # Port to connect to the leader arm
    port: str = ""
    
    # Gripper type for Alicia-D (e.g., "50mm" or "100mm")
    gripper_type: str | None = None
    
    # Debug mode for SDK
    debug_mode: bool = False
    
    # Whether the leader arm directly controls the follower arm via hardware wire.
    # If True, actions don't need to be sent through the computer (default: True).
    # If False, actions will be sent through the computer from teleoperator to robot.
    directly_controls_robot: bool = True

    # When directly controlling the follower via hardware, choose whether the
    # leader state should overwrite robot joint observations during recording.
    # True: teleop-only joint observations (case 1).
    # False: robot observations + teleop actions (case 2).
    use_action_as_observation: bool = True

    # Delay (in frames) between observation (leader state) and action (leader command).
    # A value of 1 means observation is the previous frame's leader state.
    action_observation_delay_frames: int = 1
