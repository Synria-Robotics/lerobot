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

"""Alicia-D机械臂的实现。"""

import logging
import time

import numpy as np
import torch

# 导入Alicia-D SDK
try:
    from alicia_duo_sdk.controller import get_default_session, ControlApi
except ImportError:
    logging.warning("未找到Alicia-D SDK。请确保已正确安装`alicia_duo_sdk`包。")
    ControlApi = None

from lerobot.common.robot_devices.cameras.utils import make_cameras_from_configs
from lerobot.common.robot_devices.robots.configs import AliciaDuoRobotConfig
from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError, RobotDeviceNotConnectedError


class AliciaDuoRobot:
    """Alicia-D机械臂的控制类实现。
    
    这个类包装了Alicia-D SDK的ControlApi，提供了与LeRobot框架兼容的接口。
    它支持基本的连接、断开连接、读取状态和发送动作等功能。
    
    实例化示例:
    ```python
    robot = AliciaDuoRobot(AliciaDuoRobotConfig())
    ```
    
    在实例化过程中覆盖端口和波特率的示例:
    ```python
    robot = AliciaDuoRobot(AliciaDuoRobotConfig(port="/dev/ttyUSB0", baudrate=921600))
    ```
    """
    
    def __init__(self, config: AliciaDuoRobotConfig, enable_online_smooth=True):
        """初始化Alicia-D机械臂控制器。
        
        Args:
            config: Alicia-D机械臂配置
        """
        self.leader_arms = {}
        self.follower_arms = {}

        self.config = config
        self.robot_type = self.config.type  # 从配置类中获取类型名称
        
        # 保存串口参数
        self.port = self.config.port
        self.baudrate = self.config.baudrate
        self.debug_mode = self.config.debug_mode
        
        # 摄像头
        self.cameras = make_cameras_from_configs(self.config.cameras)
        
        # 连接状态
        self.is_connected = False
        
        # 日志
        self.logs = {}
        
        self.enable_online_smooth = enable_online_smooth
        
        # 创建控制器
        if ControlApi is not None:
            self.session = get_default_session(baudrate=self.baudrate)
            self.controller = ControlApi(session=self.session)
        else:
            self.controller = None
            if not self.config.mock:
                logging.error("无法创建ControlApi。请确保已安装Alicia-D SDK。")
                
        if enable_online_smooth:
            self.controller.startOnlineSmoothing(
                command_rate_hz=200,
                max_joint_velocity_rad_s=2.5,
                max_joint_accel_rad_s2=1,
                max_gripper_velocity_rad_s=1.5,
                max_gripper_accel_rad_s2=10.0,
            )
        
        # 关节数量：6个关节+1个夹爪
        self.joint_count = 6
        self.has_gripper = True
        
        logging.info("已初始化Alicia-D机械臂控制器")
    
    @property
    def features(self):
        """定义观察空间和动作空间的特征。"""
        # 获取电机特征
        motor_features = self.motor_features
        
        # 获取摄像头特征
        cam_features = self.camera_features
        
        # 合并所有特征并返回
        return {**motor_features, **cam_features}
    
    @property
    def motor_features(self) -> dict:
        """返回电机/关节的特征描述。
        
        Returns:
            电机特征字典，包含动作和状态特征
        """
        # 创建关节和夹爪名称列表
        joint_names = [f"joint{i+1}" for i in range(self.joint_count)]
        
        # 动作名称（关节+夹爪）
        action_names = joint_names.copy()
        if self.has_gripper:
            action_names.append("gripper")
        
        # 状态名称（与动作相同）
        state_names = action_names.copy()
        
        # 创建并返回特征字典
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
    def camera_features(self) -> dict:
        """返回摄像头的特征描述。
        
        Returns:
            摄像头特征字典，键是观测名称，值是特征描述
        """
        cam_features = {}
        for cam_key, cam in self.cameras.items():
            key = f"observation.images.{cam_key}"
            cam_features[key] = {
                "shape": (cam.height, cam.width, cam.channels),
                "names": ["height", "width", "channels"],
                "info": None,
            }
        return cam_features

    @property
    def has_camera(self):
        """是否有摄像头。"""
        return len(self.cameras) > 0
    
    @property
    def num_cameras(self):
        """摄像头数量。"""
        return len(self.cameras)
    
    @property
    def available_arms(self):
        """可用的机械臂列表（兼容接口）。"""
        return ["main"]  # Alicia-D只有一个机械臂
    
    def connect(self):
        """连接到机械臂。"""
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                "Alicia-D机械臂已连接。请勿重复运行`robot.connect()`。"
            )
        
        if self.config.mock:
            logging.info("使用模拟模式，不实际连接硬件")
            self.is_connected = True
            return
        
        if self.controller is None:
            raise RobotDeviceNotConnectedError(
                "ArmController未初始化。请确保已安装Alicia-D SDK。"
            )
        
        # # 连接到机械臂
        # logging.info("正在连接到Alicia-D机械臂...")
        # if not self.controller.connect():
        #     raise RobotDeviceNotConnectedError("无法连接到Alicia-D机械臂。请检查连接。")
        
        # 连接摄像头（如果有）
        for name in self.cameras:
            logging.info(f"正在连接{name}摄像头...")
            self.cameras[name].connect()
        
        self.is_connected = True
        logging.info("Alicia-D机械臂连接成功")
    
    def run_calibration(self):
        """空的校准方法实现，满足Robot协议要求。
        
        由于用户已经在外部完成校准，此方法不执行任何操作。
        """
        logging.info("Alicia-D机械臂已在外部校准，无需进行内部校准。")
        pass
    
    def teleop_step(self, record_data=False):
        """执行一步遥操作，可选择记录数据。
        
        Args:
            record_data: 是否记录数据
        
        Returns:
            如果record_data为True，返回(观察，动作)元组；否则返回None
        """
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "AliciaDuoRobot未连接。你需要运行`robot.connect()`。"
            )
        
        # 读取当前状态
        joint_rad = self.controller.get_joints()
        gripper_rad = self.controller.get_gripper()
        
        # 如果不需要记录数据，则提前返回
        if not record_data:
            return None
        
        # 创建观察字典（状态+图像）
        obs_dict = {}
        
        # 关节角度和夹爪角度组合为状态
        joint_angles = torch.tensor(joint_rad, dtype=torch.float32)
        gripper_angle = torch.tensor([gripper_rad], dtype=torch.float32)
        combined_state = torch.cat([joint_angles, gripper_angle])
        obs_dict["observation.state"] = combined_state
        
        # 读取摄像头图像（如果有）
        for name, cam in self.cameras.items():
            frame = cam.async_read()
            obs_dict[f"observation.images.{name}"] = torch.from_numpy(frame)
        
        # 动作与状态相同（因为这是记录模式，实际动作就是当前状态）
        action_dict = {"action": combined_state}
        
        return obs_dict, action_dict
    
    def capture_observation(self):
        """捕获当前观察（状态+图像）。
        
        Returns:
            观察字典
        """
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "AliciaDuoRobot未连接。你需要运行`robot.connect()`。"
            )
        
        # 读取当前状态
        joint_rad = self.controller.get_joints()
        gripper_rad = self.controller.get_gripper()
        
        # 创建观察字典
        obs_dict = {}
        
        # 关节角度和夹爪角度组合为状态
        joint_angles = torch.tensor(joint_rad, dtype=torch.float32)
        gripper_angle = torch.tensor([gripper_rad], dtype=torch.float32)
        combined_state = torch.cat([joint_angles, gripper_angle])
        obs_dict["observation.state"] = combined_state
        
        # 读取摄像头图像（如果有）
        for name, cam in self.cameras.items():
            frame = cam.async_read()
            obs_dict[f"observation.images.{name}"] = torch.from_numpy(frame)
        
        return obs_dict
    
    def send_action(self, action: torch.Tensor) -> torch.Tensor:
        """发送动作到机械臂。
        
        Args:
            action: 包含关节角度和夹爪角度的动作张量
        
        Returns:
            实际发送的动作张量
        """
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "AliciaDuoRobot未连接。你需要运行`robot.connect()`。"
            )
        
        # 从动作张量中提取关节角度和夹爪角度
        if len(action) == self.joint_count + 1:  # 6个关节 + 1个夹爪
            joint_angles = action[:self.joint_count].tolist()
            gripper_angle = action[-1].item()
        else:
            # 如果动作张量形状不符合预期，提供警告
            if len(action) < self.joint_count:
                logging.warning(f"动作张量太短：期望至少{self.joint_count}个关节，实际{len(action)}个")
                # 补充缺失关节值为0
                joint_angles = action.tolist() + [0.0] * (self.joint_count - len(action))
                gripper_angle = None
            else:
                # 关节数量足够，但没有夹爪
                joint_angles = action[:self.joint_count].tolist()
                gripper_angle = None
        
        # 应用安全限制（如果配置了max_relative_target）
        if self.config.max_relative_target is not None:
            # 读取当前关节位置
            joint_rad = self.controller.get_joints()
            gripper_rad = self.controller.get_gripper()
            current_joint_angles = joint_rad
            
            # 限制关节移动范围
            safe_joint_angles = []
            for i, (current, target) in enumerate(zip(current_joint_angles, joint_angles)):
                max_delta = self.config.max_relative_target
                if isinstance(max_delta, list):
                    max_delta = max_delta[i] if i < len(max_delta) else max_delta[-1]
                
                delta = target - current
                if abs(delta) > max_delta:
                    safe_target = current + (max_delta if delta > 0 else -max_delta)
                    logging.warning(f"关节{i+1}移动幅度过大，已限制: {delta:.4f} -> {max_delta:.4f}")
                else:
                    safe_target = target
                safe_joint_angles.append(safe_target)
            
            joint_angles = safe_joint_angles
        
        # 发送命令到机械臂
        if self.enable_online_smooth:
            self.controller.setJointTargetOnline(joint_angles)
            self.controller.setGripperTargetOnline(gripper_angle)
        else:
            self.controller.joint_controller.set_joint_angles(joint_angles)
            self.controller.joint_controller.set_gripper(gripper_angle)

        # 返回实际发送的动作
        if gripper_angle is not None:
            return torch.tensor(joint_angles + [gripper_angle], dtype=torch.float32)
        else:
            return torch.tensor(joint_angles, dtype=torch.float32)
    
    def disconnect(self):
        """断开与机械臂的连接。"""
        if not self.is_connected:
            return
        
        logging.info("正在断开Alicia-D机械臂连接...")
        
        # 断开摄像头连接
        for name, cam in self.cameras.items():
            try:
                cam.disconnect()
                logging.info(f"已断开{name}摄像头")
            except Exception as e:
                logging.error(f"断开{name}摄像头时出错: {e}")
        
        # 断开机械臂连接
        if not self.config.mock and self.controller is not None:
            if self.enable_online_smooth:
                self.controller.stopOnlineSmoothing()
            self.session.joint_controller.disconnect()
            logging.info("已断开机械臂连接")
        
        self.is_connected = False
    
    def __del__(self):
        """析构函数，确保断开连接。"""
        if getattr(self, "is_connected", False):
            self.disconnect()
    
    def print_logs(self):
        """打印日志信息。"""
        for key, value in self.logs.items():
            print(f"{key}: {value}") 