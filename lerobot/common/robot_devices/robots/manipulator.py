# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""包含实例化机器人、读取其电机和摄像头信息以及向电机发送指令的逻辑。
"""
# TODO(rcadene, aliberts): 重组代码库为每个机器人一个文件，包含相关的
# 校准程序，以便人们轻松添加自己的机器人。

import json
import logging
import time
import warnings
from pathlib import Path

import numpy as np
import torch

from lerobot.common.robot_devices.cameras.utils import make_cameras_from_configs
from lerobot.common.robot_devices.motors.utils import MotorsBus, make_motors_buses_from_configs
from lerobot.common.robot_devices.robots.configs import ManipulatorRobotConfig
from lerobot.common.robot_devices.robots.utils import get_arm_id
from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError, RobotDeviceNotConnectedError


def ensure_safe_goal_position(
    goal_pos: torch.Tensor, present_pos: torch.Tensor, max_relative_target: float | list[float]
):
    # 为安全起见，限制相对动作目标幅度
    diff = goal_pos - present_pos
    max_relative_target = torch.tensor(max_relative_target)
    safe_diff = torch.minimum(diff, max_relative_target)
    safe_diff = torch.maximum(safe_diff, -max_relative_target)
    safe_goal_pos = present_pos + safe_diff

    if not torch.allclose(goal_pos, safe_goal_pos):
        logging.warning(
            "相对目标位置幅度必须被限制以确保安全。\n"
            f"  请求的相对目标位置: {diff}\n"
            f"    被限制的相对目标位置: {safe_diff}"
        )

    return safe_goal_pos


class ManipulatorRobot:
    # TODO(rcadene): 实现力反馈
    """这个类允许控制各种具有不同电机数量的机械臂机器人。

    非详尽的机器人列表:
    - [Koch v1.0](https://github.com/AlexanderKoch-Koch/low_cost_robot)，带或不带腕到肘的扩展，由
    [Tau Robotics](https://tau-robotics.com) 的 Alexander Koch 开发
    - [Koch v1.1](https://github.com/jess-moss/koch-v1-1) 由 Jess Moss 开发
    - [Aloha](https://www.trossenrobotics.com/aloha-kits) 由 Trossen Robotics 开发

    实例化示例，需要预定义的机器人配置:
    ```python
    robot = ManipulatorRobot(KochRobotConfig())
    ```

    在实例化过程中覆盖电机的示例:
    ```python
    # 定义如何与主导臂和跟随臂的电机通信
    leader_arms = {
        "main": DynamixelMotorsBusConfig(
            port="/dev/tty.usbmodem575E0031751",
            motors={
                # 名称: (索引, 型号)
                "shoulder_pan": (1, "xl330-m077"),
                "shoulder_lift": (2, "xl330-m077"),
                "elbow_flex": (3, "xl330-m077"),
                "wrist_flex": (4, "xl330-m077"),
                "wrist_roll": (5, "xl330-m077"),
                "gripper": (6, "xl330-m077"),
            },
        ),
    }
    follower_arms = {
        "main": DynamixelMotorsBusConfig(
            port="/dev/tty.usbmodem575E0032081",
            motors={
                # 名称: (索引, 型号)
                "shoulder_pan": (1, "xl430-w250"),
                "shoulder_lift": (2, "xl430-w250"),
                "elbow_flex": (3, "xl330-m288"),
                "wrist_flex": (4, "xl330-m288"),
                "wrist_roll": (5, "xl330-m288"),
                "gripper": (6, "xl330-m288"),
            },
        ),
    }
    robot_config = KochRobotConfig(leader_arms=leader_arms, follower_arms=follower_arms)
    robot = ManipulatorRobot(robot_config)
    ```

    在实例化过程中覆盖摄像头的示例:
    ```python
    # 定义如何与连接到计算机的两个摄像头通信。
    # 这里，笔记本电脑的网络摄像头和手机（通过USB连接到笔记本电脑）
    # 可以分别使用摄像头索引0和1访问。这些索引可以是
    # 任意的。查看`OpenCVCamera`的文档来找到你自己的摄像头索引。
    cameras = {
        "laptop": OpenCVCamera(camera_index=0, fps=30, width=640, height=480),
        "phone": OpenCVCamera(camera_index=1, fps=30, width=640, height=480),
    }
    robot = ManipulatorRobot(KochRobotConfig(cameras=cameras))
    ```

    机器人实例化后，连接电机总线和摄像头（如有）（必需）:
    ```python
    robot.connect()
    ```

    最高频率遥操作的示例，不需要摄像头:
    ```python
    while True:
        robot.teleop_step()
    ```

    从电机和摄像头（如有）收集最高频率数据的示例:
    ```python
    while True:
        observation, action = robot.teleop_step(record_data=True)
    ```

    使用策略控制机器人的示例:
    ```python
    while True:
        # 使用跟随臂和摄像头捕获观察
        observation = robot.capture_observation()

        # 假设策略已经实例化
        with torch.inference_mode():
            action = policy.select_action(observation)

        # 命令机器人移动
        robot.send_action(action)
    ```

    断开连接的示例（这不是强制性的，因为对象被删除时我们会断开连接）:
    ```python
    robot.disconnect()
    ```
    """

    def __init__(
        self,
        config: ManipulatorRobotConfig,
    ):
        self.config = config
        self.robot_type = self.config.type
        self.calibration_dir = Path(self.config.calibration_dir)
        self.leader_arms = make_motors_buses_from_configs(self.config.leader_arms)
        self.follower_arms = make_motors_buses_from_configs(self.config.follower_arms)
        self.cameras = make_cameras_from_configs(self.config.cameras)
        self.is_connected = False
        self.logs = {}

    def get_motor_names(self, arm: dict[str, MotorsBus]) -> list:
        return [f"{arm}_{motor}" for arm, bus in arm.items() for motor in bus.motors]

    @property
    def camera_features(self) -> dict:
        cam_ft = {}
        for cam_key, cam in self.cameras.items():
            key = f"observation.images.{cam_key}"
            cam_ft[key] = {
                "shape": (cam.height, cam.width, cam.channels),
                "names": ["height", "width", "channels"],
                "info": None,
            }
        return cam_ft

    @property
    def motor_features(self) -> dict:
        action_names = self.get_motor_names(self.leader_arms)
        state_names = self.get_motor_names(self.leader_arms)
        return {
            "action": {
                "dtype": "float32",
                "shape": (len(action_names),),
                "names": action_names,
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (len(state_names),),
                "names": state_names,
            },
        }

    @property
    def features(self):
        return {**self.motor_features, **self.camera_features}

    @property
    def has_camera(self):
        return len(self.cameras) > 0

    @property
    def num_cameras(self):
        return len(self.cameras)

    @property
    def available_arms(self):
        available_arms = []
        for name in self.follower_arms:
            arm_id = get_arm_id(name, "follower")
            available_arms.append(arm_id)
        for name in self.leader_arms:
            arm_id = get_arm_id(name, "leader")
            available_arms.append(arm_id)
        return available_arms

    def connect(self):
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                "ManipulatorRobot已连接。请勿重复运行`robot.connect()`。"
            )

        if not self.leader_arms and not self.follower_arms and not self.cameras:
            raise ValueError(
                "ManipulatorRobot没有任何要连接的设备。请参阅类文档字符串中的使用示例。"
            )

        # 连接机械臂
        for name in self.follower_arms:
            print(f"正在连接{name}跟随臂。")
            self.follower_arms[name].connect()
        for name in self.leader_arms:
            print(f"正在连接{name}主导臂。")
            self.leader_arms[name].connect()

        if self.robot_type in ["koch", "koch_bimanual", "aloha"]:
            from lerobot.common.robot_devices.motors.dynamixel import TorqueMode
        elif self.robot_type in ["so100", "moss", "lekiwi"]:
            from lerobot.common.robot_devices.motors.feetech import TorqueMode

        # 我们假设在连接时，机械臂处于休息位置，可以安全地禁用扭矩
        # 以运行校准和/或设置机器人预设配置。
        for name in self.follower_arms:
            self.follower_arms[name].write("Torque_Enable", TorqueMode.DISABLED.value)
        for name in self.leader_arms:
            self.leader_arms[name].write("Torque_Enable", TorqueMode.DISABLED.value)

        self.activate_calibration()

        # 设置机器人预设（例如Koch v1.1的主导机械臂夹具中的扭矩）
        if self.robot_type in ["koch", "koch_bimanual"]:
            self.set_koch_robot_preset()
        elif self.robot_type == "aloha":
            self.set_aloha_robot_preset()
        elif self.robot_type in ["so100", "moss", "lekiwi"]:
            self.set_so100_robot_preset()

        # 在跟随臂的所有电机上启用扭矩
        for name in self.follower_arms:
            print(f"激活{name}跟随臂上的扭矩。")
            self.follower_arms[name].write("Torque_Enable", 1)

        if self.config.gripper_open_degree is not None:
            if self.robot_type not in ["koch", "koch_bimanual"]:
                raise NotImplementedError(
                    f"{self.robot_type}不支持手柄中的位置和电流控制，这是设置夹具打开所必需的。"
                )
            # 将主导臂设置为扭矩模式，夹具电机设置为一个角度。这样可以挤压夹具
            # 并让它自行弹回到打开位置。
            for name in self.leader_arms:
                self.leader_arms[name].write("Torque_Enable", 1, "gripper")
                self.leader_arms[name].write("Goal_Position", self.config.gripper_open_degree, "gripper")

        # 检查两个机械臂是否可以被读取
        for name in self.follower_arms:
            self.follower_arms[name].read("Present_Position")
        for name in self.leader_arms:
            self.leader_arms[name].read("Present_Position")

        # 连接摄像头
        for name in self.cameras:
            self.cameras[name].connect()

        self.is_connected = True

    def activate_calibration(self):
        """校准后，所有电机都在人类可解释的范围内工作。
        旋转以度数表示，标称范围为[-180, 180]，
        线性运动（如Aloha的夹具）的标称范围为[0, 100]。
        """

        def load_or_run_calibration_(name, arm, arm_type):
            arm_id = get_arm_id(name, arm_type)
            arm_calib_path = self.calibration_dir / f"{arm_id}.json"

            if arm_calib_path.exists():
                with open(arm_calib_path) as f:
                    calibration = json.load(f)
            else:
                # TODO(rcadene): 如果校准文件不可用，在__init__中显示警告
                print(f"缺少校准文件'{arm_calib_path}'")

                if self.robot_type in ["koch", "koch_bimanual", "aloha"]:
                    from lerobot.common.robot_devices.robots.dynamixel_calibration import run_arm_calibration

                    calibration = run_arm_calibration(arm, self.robot_type, name, arm_type)

                elif self.robot_type in ["so100", "moss", "lekiwi"]:
                    from lerobot.common.robot_devices.robots.feetech_calibration import (
                        run_arm_manual_calibration,
                    )

                    calibration = run_arm_manual_calibration(arm, self.robot_type, name, arm_type)

                print(f"校准完成！保存校准文件'{arm_calib_path}'")
                arm_calib_path.parent.mkdir(parents=True, exist_ok=True)
                with open(arm_calib_path, "w") as f:
                    json.dump(calibration, f)

            return calibration

        for name, arm in self.follower_arms.items():
            calibration = load_or_run_calibration_(name, arm, "follower")
            arm.set_calibration(calibration)
        for name, arm in self.leader_arms.items():
            calibration = load_or_run_calibration_(name, arm, "leader")
            arm.set_calibration(calibration)

    def set_koch_robot_preset(self):
        def set_operating_mode_(arm):
            from lerobot.common.robot_devices.motors.dynamixel import TorqueMode

            if (arm.read("Torque_Enable") != TorqueMode.DISABLED.value).any():
                raise ValueError("要运行设置机器人预设，所有电机上的扭矩必须被禁用。")

            # 对除夹具外的所有电机使用"扩展位置模式"，因为在关节模式下伺服
            # 不能旋转超过360度（从0到4095）。组装机械臂时可能会出现一些错误，
            # 你可能最终在关键点处有一个位置为0或4095的伺服。参见[
            # https://emanual.robotis.com/docs/en/dxl/x/x_series/#operating-mode11]
            all_motors_except_gripper = [name for name in arm.motor_names if name != "gripper"]
            if len(all_motors_except_gripper) > 0:
                # 4对应于Koch电机上的扩展位置
                arm.write("Operating_Mode", 4, all_motors_except_gripper)

            # 对夹具使用"基于电流的位置控制"，以受到电流限制的限制。
            # 对于跟随臂夹具，这意味着它可以抓住一个物体而不会用力过猛，即使
            # 它的目标位置是完全抓取（两个夹具手指被命令联合并接触）。
            # 对于主导臂夹具，这意味着我们可以将它用作物理触发器，因为我们可以用手指
            # 强制使它移动，当我们释放力时，它会回到原来的目标位置。
            # 5对应于Koch夹具电机"xl330-m077, xl330-m288"上的电流控制位置
            arm.write("Operating_Mode", 5, "gripper")

        for name in self.follower_arms:
            set_operating_mode_(self.follower_arms[name])

            # 设置更好的PID值以缩小记录状态和动作之间的差距
            # TODO(rcadene): 实现一个自动程序为每个电机设置最佳PID值
            self.follower_arms[name].write("Position_P_Gain", 1500, "elbow_flex")
            self.follower_arms[name].write("Position_I_Gain", 0, "elbow_flex")
            self.follower_arms[name].write("Position_D_Gain", 600, "elbow_flex")

        if self.config.gripper_open_degree is not None:
            for name in self.leader_arms:
                set_operating_mode_(self.leader_arms[name])

                # 在主导臂的夹具上启用扭矩，并将其移动到45度，
                # 这样我们可以将其用作触发器来关闭跟随臂的夹具。
                self.leader_arms[name].write("Torque_Enable", 1, "gripper")
                self.leader_arms[name].write("Goal_Position", self.config.gripper_open_degree, "gripper")

    def set_aloha_robot_preset(self):
        def set_shadow_(arm):
            # 为肩部和肘部设置辅助/影子ID。这些关节有两个电机。
            # 因此，如果只需要其中一个移动到特定位置，
            # 另一个将跟随。这是为了避免损坏电机。
            if "shoulder_shadow" in arm.motor_names:
                shoulder_idx = arm.read("ID", "shoulder")
                arm.write("Secondary_ID", shoulder_idx, "shoulder_shadow")

            if "elbow_shadow" in arm.motor_names:
                elbow_idx = arm.read("ID", "elbow")
                arm.write("Secondary_ID", elbow_idx, "elbow_shadow")

        for name in self.follower_arms:
            set_shadow_(self.follower_arms[name])

        for name in self.leader_arms:
            set_shadow_(self.leader_arms[name])

        for name in self.follower_arms:
            # 按照Trossen Robotics的建议设置131的速度限制
            self.follower_arms[name].write("Velocity_Limit", 131)

            # 对除夹具外的所有电机使用"扩展位置模式"，因为在关节模式下伺服
            # 不能旋转超过360度（从0到4095）。组装机械臂时可能会出现一些错误，
            # 你可能最终在关键点处有一个位置为0或4095的伺服。参见[
            # https://emanual.robotis.com/docs/en/dxl/x/x_series/#operating-mode11]
            all_motors_except_gripper = [
                name for name in self.follower_arms[name].motor_names if name != "gripper"
            ]
            if len(all_motors_except_gripper) > 0:
                # 4对应于Aloha电机上的扩展位置
                self.follower_arms[name].write("Operating_Mode", 4, all_motors_except_gripper)

            # 对跟随臂夹具使用"基于电流的位置控制"，以受到电流限制的限制。
            # 它可以抓住一个物体而不会用力过猛，即使
            # 它的目标位置是完全抓取（两个夹具手指被命令联合并接触）。
            # 5对应于Aloha夹具跟随臂"xm430-w350"上的电流控制位置
            self.follower_arms[name].write("Operating_Mode", 5, "gripper")

            # 注意：我们无法在主导臂夹具上启用扭矩，因为"xc430-w150"没有
            # 电流控制位置模式。

        if self.config.gripper_open_degree is not None:
            warnings.warn(
                f"`gripper_open_degree`被设置为{self.config.gripper_open_degree}，但Aloha预期为None",
                stacklevel=1,
            )

    def set_so100_robot_preset(self):
        for name in self.follower_arms:
            # Mode=0表示位置控制
            self.follower_arms[name].write("Mode", 0)
            # 将P_Coefficient设置为较低值以避免抖动（默认为32）
            self.follower_arms[name].write("P_Coefficient", 16)
            # 将I_Coefficient和D_Coefficient设置为默认值0和32
            self.follower_arms[name].write("I_Coefficient", 0)
            self.follower_arms[name].write("D_Coefficient", 32)
            # 关闭写入锁，使Maximum_Acceleration写入EPROM地址，
            # 这对于Maximum_Acceleration在重启后生效是必需的。
            self.follower_arms[name].write("Lock", 0)
            # 将Maximum_Acceleration设置为254以加速电机的加速和减速。
            # 注意：此配置不在官方STS3215内存表中
            self.follower_arms[name].write("Maximum_Acceleration", 254)
            self.follower_arms[name].write("Acceleration", 254)

    def teleop_step(
        self, record_data=False
    ) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "ManipulatorRobot未连接。你需要运行`robot.connect()`。"
            )

        # 准备将主导臂的位置分配给跟随臂
        leader_pos = {}
        for name in self.leader_arms:
            before_lread_t = time.perf_counter()
            leader_pos[name] = self.leader_arms[name].read("Present_Position")
            leader_pos[name] = torch.from_numpy(leader_pos[name])
            self.logs[f"read_leader_{name}_pos_dt_s"] = time.perf_counter() - before_lread_t

        # 向跟随臂发送目标位置
        follower_goal_pos = {}
        for name in self.follower_arms:
            before_fwrite_t = time.perf_counter()
            goal_pos = leader_pos[name]

            # 当目标位置与当前位置相距太远时进行限制。
            # 由于从跟随臂读取数据，预期帧率较低。
            if self.config.max_relative_target is not None:
                present_pos = self.follower_arms[name].read("Present_Position")
                present_pos = torch.from_numpy(present_pos)
                goal_pos = ensure_safe_goal_position(goal_pos, present_pos, self.config.max_relative_target)

            # 在record_data=True时使用
            follower_goal_pos[name] = goal_pos

            goal_pos = goal_pos.numpy().astype(np.float32)
            self.follower_arms[name].write("Goal_Position", goal_pos)
            self.logs[f"write_follower_{name}_goal_pos_dt_s"] = time.perf_counter() - before_fwrite_t

        # 当不需要记录数据时提前退出
        if not record_data:
            return

        # TODO(rcadene): 添加速度和其他信息
        # 读取跟随臂位置
        follower_pos = {}
        for name in self.follower_arms:
            before_fread_t = time.perf_counter()
            follower_pos[name] = self.follower_arms[name].read("Present_Position")
            follower_pos[name] = torch.from_numpy(follower_pos[name])
            self.logs[f"read_follower_{name}_pos_dt_s"] = time.perf_counter() - before_fread_t

        # 通过连接跟随臂当前位置创建状态
        state = []
        for name in self.follower_arms:
            if name in follower_pos:
                state.append(follower_pos[name])
        state = torch.cat(state)

        # 通过连接跟随臂目标位置创建动作
        action = []
        for name in self.follower_arms:
            if name in follower_goal_pos:
                action.append(follower_goal_pos[name])
        action = torch.cat(action)

        # 从摄像头捕获图像
        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            images[name] = torch.from_numpy(images[name])
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

        # 填充输出字典
        obs_dict, action_dict = {}, {}
        obs_dict["observation.state"] = state
        action_dict["action"] = action
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = images[name]

        return obs_dict, action_dict

    def capture_observation(self):
        """返回的观察没有批次维度。"""
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "ManipulatorRobot未连接。你需要运行`robot.connect()`。"
            )

        # 读取跟随臂位置
        follower_pos = {}
        for name in self.follower_arms:
            before_fread_t = time.perf_counter()
            follower_pos[name] = self.follower_arms[name].read("Present_Position")
            follower_pos[name] = torch.from_numpy(follower_pos[name])
            self.logs[f"read_follower_{name}_pos_dt_s"] = time.perf_counter() - before_fread_t

        # 通过连接跟随臂当前位置创建状态
        state = []
        for name in self.follower_arms:
            if name in follower_pos:
                state.append(follower_pos[name])
        state = torch.cat(state)

        # 从摄像头捕获图像
        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            images[name] = torch.from_numpy(images[name])
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

        # 填充输出字典并格式化为pytorch
        obs_dict = {}
        obs_dict["observation.state"] = state
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = images[name]
        return obs_dict

    def send_action(self, action: torch.Tensor) -> torch.Tensor:
        """命令跟随臂移动到目标关节配置。

        根据配置参数`max_relative_target`，相对动作幅度可能被限制。
        在这种情况下，发送的动作与原始动作不同。
        因此，该函数始终返回实际发送的动作。

        参数:
            action: 包含跟随臂连接目标位置的张量。
        """
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "ManipulatorRobot未连接。你需要运行`robot.connect()`。"
            )

        from_idx = 0
        to_idx = 0
        action_sent = []
        for name in self.follower_arms:
            # 通过分割动作向量获取每个跟随臂的目标位置
            to_idx += len(self.follower_arms[name].motor_names)
            goal_pos = action[from_idx:to_idx]
            from_idx = to_idx

            # 当目标位置与当前位置相距太远时进行限制。
            # 由于从跟随臂读取数据，预期帧率较低。
            if self.config.max_relative_target is not None:
                present_pos = self.follower_arms[name].read("Present_Position")
                present_pos = torch.from_numpy(present_pos)
                goal_pos = ensure_safe_goal_position(goal_pos, present_pos, self.config.max_relative_target)

            # 保存张量以连接并返回
            action_sent.append(goal_pos)

            # 向每个跟随臂发送目标位置
            goal_pos = goal_pos.numpy().astype(np.float32)
            self.follower_arms[name].write("Goal_Position", goal_pos)

        return torch.cat(action_sent)

    def print_logs(self):
        pass
        # TODO(aliberts): 将机器人特定的日志逻辑移到这里

    def disconnect(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "ManipulatorRobot未连接。你需要在断开连接前运行`robot.connect()`。"
            )

        for name in self.follower_arms:
            self.follower_arms[name].disconnect()

        for name in self.leader_arms:
            self.leader_arms[name].disconnect()

        for name in self.cameras:
            self.cameras[name].disconnect()

        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()
