#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("leader")
@dataclass
class LeaderConfig(TeleoperatorConfig):
    # Alicia-D leader arm (teaching arm) connection
    port: str
    baudrate: int = 1_000_000

    # Alicia-D SDK v6 parameters
    robot_version: str = "v5_6"
    gripper_type: str = "50mm"
    debug_mode: bool = False


