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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("alicia_m_follower")
@dataclass
class AliciaMFollowerConfig(RobotConfig):
    # Serial port used by the follower arm.
    port: str = ""

    # Alicia-M model/configuration selection.
    version: str = "v1_1"
    variant: str | None = None
    base_link: str = "base_link"
    end_link: str = "tool0"

    # Serial/control behavior.
    baudrate: int = 1_000_000
    control_aim: str | None = "operation"  # "teach" | "operation" | None(auto)
    control_mode: str | None = "mit"  # "pv" | "mit" | None(SDK default)
    skip_mit_init: bool = False
    # Effective only in MIT mode. False sends direct MIT PD commands without firmware interpolation.
    use_interpolation: bool = False
    debug_mode: bool = False

    disable_torque_on_disconnect: bool = False

    # Maximum relative per-step target for safety clipping.
    max_relative_target: float | dict[str, float] | None = None

    # Robot motion speed used by M-SDK set_robot_state(speed=...).
    # Valid M-SDK range is usually [0, 400].
    speed: int = 100

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Allow teleop state to overwrite joint observations when wired leader->follower.
    use_teleop_state_for_observation: bool = True

    # Whether to connect to the follower arm over serial.
    # If None, connect only when `port` is set.
    connect_arm: bool | None = None
