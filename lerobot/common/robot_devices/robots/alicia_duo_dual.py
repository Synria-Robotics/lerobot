"""Dual Alicia-D机械臂的实现。"""

import logging
import time
from typing import Dict, Any

import numpy as np
import torch

# 导入Alicia-D SDK
try:
    from alicia_duo_sdk.controller import get_default_session, ControlApi
except ImportError:
    logging.warning("未找到Alicia-D SDK。请确保已正确安装`alicia_duo_sdk`包。")
    ControlApi = None

from lerobot.common.robot_devices.cameras.utils import make_cameras_from_configs
from lerobot.common.robot_devices.robots.configs import AliciaDuoDualRobotConfig
from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError, RobotDeviceNotConnectedError


class AliciaDuoDualRobot:
    """双Alicia-D机械臂的控制类实现。
    
    这个类管理两个Alicia-D机械臂，提供统一的接口进行数据记录和控制。
    """
    
    def __init__(self, config: AliciaDuoDualRobotConfig, enable_online_smooth=True):
        """初始化双Alicia-D机械臂控制器。
        
        Args:
            config: 双Alicia-D机械臂配置
            enable_online_smooth: 是否启用在线平滑控制
        """
        self.config = config
        self.robot_type = self.config.type
        self.enable_online_smooth = enable_online_smooth
        
        # 创建两个机械臂控制器
        self.robots = {}
        for arm_name, arm_config in config.arms.items():
            session = get_default_session(baudrate=arm_config["baudrate"], port=arm_config["port"])
            if ControlApi is not None:
                self.robots[arm_name] = ControlApi(session=session)
            else:
                self.robots[arm_name] = None
                if not self.config.mock:
                    logging.error(f"无法创建{arm_name}的ControlApi。请确保已安装Alicia-D SDK。")

            if self.enable_online_smooth and self.robots[arm_name] is not None:
                self.robots[arm_name].startOnlineSmoothing(
                    command_rate_hz=200,
                    max_joint_velocity_rad_s=2.5,
                    max_joint_accel_rad_s2=1,
                    max_gripper_velocity_rad_s=1.5,
                    max_gripper_accel_rad_s2=10.0,
                )
        
        # 摄像头
        self.cameras = make_cameras_from_configs(self.config.cameras)
        
        # 连接状态
        self.is_connected = False
        
        # 关节数量：每个机械臂6个关节+1个夹爪
        self.joint_count_per_arm = 6
        self.has_gripper = True
        
        logging.info("已初始化双Alicia-D机械臂控制器")
    
    @property
    def features(self):
        """定义观察空间和动作空间的特征。"""
        motor_features = self.motor_features
        cam_features = self.camera_features
        return {**motor_features, **cam_features}
    
    @property
    def motor_features(self) -> dict:
        """返回双机械臂的电机/关节特征描述。"""
        # 为每个机械臂创建关节名称
        all_action_names = []
        all_state_names = []
        
        for arm_name in self.config.arms.keys():
            joint_names = [f"{arm_name}_joint{i+1}" for i in range(self.joint_count_per_arm)]
            if self.has_gripper:
                joint_names.append(f"{arm_name}_gripper")
            
            all_action_names.extend(joint_names)
            all_state_names.extend(joint_names)
        
        return {
            "action": {
                "dtype": "float32",
                "shape": (len(all_action_names),),
                "names": all_action_names,
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (len(all_state_names),),
                "names": all_state_names,
            },
        }

    @property
    def camera_features(self) -> dict:
        """返回摄像头的特征描述。"""
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
        return len(self.cameras) > 0
    
    @property
    def num_cameras(self):
        return len(self.cameras)
    
    @property
    def available_arms(self):
        return list(self.config.arms.keys())
    
    def connect(self):
        """连接到所有机械臂。"""
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                "双Alicia-D机械臂已连接。请勿重复运行`robot.connect()`。"
            )
        
        if self.config.mock:
            logging.info("使用模拟模式，不实际连接硬件")
            self.is_connected = True
            return
        
        # 连接所有机械臂
        for arm_name, controller in self.robots.items():
            if controller is None:
                raise RobotDeviceNotConnectedError(
                    f"{arm_name}的ControlApi未初始化。请确保已安装Alicia-D SDK。"
                )
            
            # 注释掉实际的机械臂连接代码，与alicia_duo.py保持一致
            # logging.info(f"正在连接到{arm_name}机械臂...")
            # if not controller.connect():
            #     raise RobotDeviceNotConnectedError(f"无法连接到{arm_name}机械臂。请检查连接。")
        
        # 连接摄像头
        for name in self.cameras:
            logging.info(f"正在连接{name}摄像头...")
            self.cameras[name].connect()
        
        self.is_connected = True
        logging.info("双Alicia-D机械臂连接成功")
    
    def run_calibration(self):
        """空的校准方法实现。"""
        logging.info("双Alicia-D机械臂已在外部校准，无需进行内部校准。")
        pass
    
    def teleop_step(self, record_data=False):
        """执行一步遥操作，可选择记录数据。"""
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "AliciaDuoDualRobot未连接。你需要运行`robot.connect()`。"
            )
        
        if not record_data:
            return None
        
        # 读取所有机械臂的状态
        all_states = []
        for arm_name, controller in self.robots.items():
            joint_rad = controller.get_joints()
            gripper_rad = controller.get_gripper()
            joint_angles = torch.tensor(joint_rad, dtype=torch.float32)
            gripper_angle = torch.tensor([gripper_rad], dtype=torch.float32)
            arm_state = torch.cat([joint_angles, gripper_angle])
            all_states.append(arm_state)
        
        # 合并所有状态
        combined_state = torch.cat(all_states)
        
        # 创建观察字典
        obs_dict = {"observation.state": combined_state}
        
        # 读取摄像头图像
        for name, cam in self.cameras.items():
            frame = cam.async_read()
            obs_dict[f"observation.images.{name}"] = torch.from_numpy(frame)
        
        # 动作与状态相同
        action_dict = {"action": combined_state}
        
        return obs_dict, action_dict
    
    def capture_observation(self):
        """捕获当前观察（状态+图像）。"""
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "AliciaDuoDualRobot未连接。你需要运行`robot.connect()`。"
            )
        
        # 读取所有机械臂的状态
        all_states = []
        for arm_name, controller in self.robots.items():
            joint_rad = controller.get_joints()
            gripper_rad = controller.get_gripper()
            joint_angles = torch.tensor(joint_rad, dtype=torch.float32)
            gripper_angle = torch.tensor([gripper_rad], dtype=torch.float32)
            arm_state = torch.cat([joint_angles, gripper_angle])
            all_states.append(arm_state)
        
        # 合并所有状态
        combined_state = torch.cat(all_states)
        
        # 创建观察字典
        obs_dict = {"observation.state": combined_state}
        
        # 读取摄像头图像
        for name, cam in self.cameras.items():
            frame = cam.async_read()
            obs_dict[f"observation.images.{name}"] = torch.from_numpy(frame)
        
        return obs_dict
    
    def send_action(self, action: torch.Tensor) -> torch.Tensor:
        """发送动作到所有机械臂。"""
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "AliciaDuoDualRobot未连接。你需要运行`robot.connect()`。"
            )
        
        # 计算每个机械臂的动作维度
        action_dim_per_arm = self.joint_count_per_arm + (1 if self.has_gripper else 0)
        expected_total_dim = len(self.robots) * action_dim_per_arm
        
        if len(action) != expected_total_dim:
            logging.warning(f"动作张量维度不匹配：期望{expected_total_dim}，实际{len(action)}")
        
        # 分割动作到各个机械臂
        arm_names = list(self.robots.keys())
        sent_actions = []
        
        for i, (arm_name, controller) in enumerate(self.robots.items()):
            start_idx = i * action_dim_per_arm
            end_idx = start_idx + action_dim_per_arm
            arm_action = action[start_idx:end_idx]
            
            # 提取关节角度和夹爪角度
            joint_angles = arm_action[:self.joint_count_per_arm].tolist()
            gripper_angle = arm_action[-1].item() if self.has_gripper else None
            
            # 发送到机械臂
            if self.enable_online_smooth:
                controller.setJointTargetOnline(joint_angles)
                controller.setGripperTargetOnline(gripper_angle)
            else:
                controller.joint_controller.set_joint_angles(joint_angles)
                controller.joint_controller.set_gripper(gripper_angle)
            sent_actions.append(arm_action)
        
        return torch.cat(sent_actions)
    
    def disconnect(self):
        """断开所有连接。"""
        if not self.is_connected:
            return
        
        logging.info("正在断开双Alicia-D机械臂连接...")
        
        # 断开摄像头
        for name, cam in self.cameras.items():
            try:
                cam.disconnect()
                logging.info(f"已断开{name}摄像头")
            except Exception as e:
                logging.error(f"断开{name}摄像头时出错: {e}")
        
        # 断开机械臂
        if not self.config.mock:
            for arm_name, controller in self.robots.items():
                if controller is not None:
                    if self.enable_online_smooth:
                        controller.stopOnlineSmoothing()
                    controller.session.joint_controller.disconnect()
                    logging.info(f"已断开{arm_name}机械臂")
        
        self.is_connected = False
    
    def __del__(self):
        """析构函数。"""
        if getattr(self, "is_connected", False):
            self.disconnect()