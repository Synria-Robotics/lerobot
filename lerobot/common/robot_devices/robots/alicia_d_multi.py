"""Dual Alicia-D机械臂的实现。"""

import logging
import time

import numpy as np
import torch

# 导入Alicia-D SDK
try:
    import alicia_d_sdk
except ImportError:
    logging.warning("未找到Alicia-D SDK。请确保已正确安装`alicia_d_sdk`包。")
    alicia_d_sdk = None

from lerobot.common.robot_devices.cameras.utils import make_cameras_from_configs
from lerobot.common.robot_devices.robots.configs import AliciaDMultiRobotConfig
from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError, RobotDeviceNotConnectedError


class AliciaDMultiRobot:
    """双Alicia-D机械臂的控制类实现。
    
    这个类管理两个Alicia-D机械臂，提供统一的接口进行数据记录和控制。
    """
    
    def __init__(self, config: AliciaDMultiRobotConfig, enable_online_smooth=True):
        """初始化双Alicia-D机械臂控制器。
        
        Args:
            config: 双Alicia-D机械臂配置
            enable_online_smooth: 是否启用在线平滑控制（新SDK已内置平滑功能）
        """
        self.config = config
        self.robot_type = self.config.type
        self.enable_online_smooth = enable_online_smooth
        
        # 创建两个机械臂控制器
        self.robots = {}
        if alicia_d_sdk is not None:
            for arm_name, arm_config in config.arms.items():
                # 使用新的 create_robot 函数创建机器人实例
                # 注意：新 SDK 不再需要单独的 session，直接使用 robot 对象
                self.robots[arm_name] = alicia_d_sdk.create_robot(
                    port=arm_config["port"],
                    baudrate=arm_config["baudrate"],
                    robot_version="v5_6",  # 默认版本，可根据配置调整
                    gripper_type="50mm",   # 默认夹爪类型，可根据配置调整
                    debug_mode=arm_config.get("debug_mode", False)
                )
                # 新 SDK 不再支持 startOnlineSmoothing，平滑功能已内置
        else:
            for arm_name in config.arms.keys():
                self.robots[arm_name] = None
                if not self.config.mock:
                    logging.error(f"无法创建{arm_name}的机器人实例。请确保已安装Alicia-D SDK。")
        
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
                    f"{arm_name}的机器人实例未初始化。请确保已安装Alicia-D SDK。"
                )
            
            logging.info(f"正在连接到{arm_name}机械臂...")
            if not controller.connect():
                raise RobotDeviceNotConnectedError(f"无法连接到{arm_name}机械臂。请检查连接。")
        
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
                "AliciaDMultiRobot未连接。你需要运行`robot.connect()`。"
            )
        
        # 读取所有机械臂的状态
        all_states = []
        for arm_name, controller in self.robots.items():
            joint_rad = controller.get_joints()  # 返回弧度值
            gripper_value = controller.get_gripper()  # 返回 0-100 的值
            joint_angles = torch.tensor(joint_rad, dtype=torch.float32)
            gripper_angle = torch.tensor([gripper_value], dtype=torch.float32)  # 0-100 范围
            arm_state = torch.cat([joint_angles, gripper_angle])
            all_states.append(arm_state)
        
        # 如果不需要记录数据，则提前返回
        if not record_data:
            return None
        
        # 合并所有状态
        combined_state = torch.cat(all_states)
        
        # 创建观察字典
        obs_dict = {"observation.state": combined_state}
        
        # 读取摄像头图像
        for name, cam in self.cameras.items():
            frame = cam.async_read()
            obs_dict[f"observation.images.{name}"] = torch.from_numpy(frame)
        
        # 动作与状态相同（因为这是记录模式，实际动作就是当前状态）
        action_dict = {"action": combined_state}
        
        return obs_dict, action_dict
    
    def capture_observation(self):
        """捕获当前观察（状态+图像）。"""
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "AliciaDMultiRobot未连接。你需要运行`robot.connect()`。"
            )
        
        # 读取所有机械臂的状态
        all_states = []
        for arm_name, controller in self.robots.items():
            joint_rad = controller.get_joints()  # 返回弧度值
            gripper_value = controller.get_gripper()  # 返回 0-100 的值
            joint_angles = torch.tensor(joint_rad, dtype=torch.float32)
            gripper_angle = torch.tensor([gripper_value], dtype=torch.float32)  # 0-100 范围
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
                "AliciaDMultiRobot未连接。你需要运行`robot.connect()`。"
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
            if len(arm_action) == self.joint_count_per_arm + 1:  # 6个关节 + 1个夹爪
                joint_angles = arm_action[:self.joint_count_per_arm].tolist()
                gripper_angle = arm_action[-1].item()
            else:
                # 如果动作张量形状不符合预期，提供警告
                if len(arm_action) < self.joint_count_per_arm:
                    logging.warning(f"{arm_name}动作张量太短：期望至少{self.joint_count_per_arm}个关节，实际{len(arm_action)}个")
                    # 补充缺失关节值为0
                    joint_angles = arm_action.tolist() + [0.0] * (self.joint_count_per_arm - len(arm_action))
                    gripper_angle = None
                else:
                    # 关节数量足够，但没有夹爪
                    joint_angles = arm_action[:self.joint_count_per_arm].tolist()
                    gripper_angle = None
            
            # 应用安全限制（如果配置了max_relative_target）
            if self.config.max_relative_target is not None:
                # 读取当前关节位置
                joint_rad = controller.get_joints()  # 返回弧度值
                current_joint_angles = joint_rad
                # 注意：gripper 值（0-100）不需要安全限制，因为范围固定
                
                # 限制关节移动范围
                safe_joint_angles = []
                for j, (current, target) in enumerate(zip(current_joint_angles, joint_angles)):
                    max_delta = self.config.max_relative_target
                    if isinstance(max_delta, list):
                        # 对于双臂，需要考虑每个机械臂的索引
                        joint_idx = i * self.joint_count_per_arm + j
                        max_delta = max_delta[joint_idx] if joint_idx < len(max_delta) else max_delta[-1] if max_delta else None
                    # 如果max_delta是float/int，直接使用
                    
                    if max_delta is not None:
                        delta = target - current
                        if abs(delta) > max_delta:
                            safe_target = current + (max_delta if delta > 0 else -max_delta)
                            logging.warning(f"{arm_name}关节{j+1}移动幅度过大，已限制: {delta:.4f} -> {max_delta:.4f}")
                        else:
                            safe_target = target
                        safe_joint_angles.append(safe_target)
                    else:
                        safe_joint_angles.append(target)
                
                joint_angles = safe_joint_angles
            
            # 发送命令到机械臂
            # 新 SDK 使用统一的 set_joint_target 和 set_gripper_target 方法
            controller.set_joint_target(target_joints=joint_angles, joint_format='rad')
            if gripper_angle is not None:
                # 新 SDK 的 set_gripper_target 接受 value 参数（0-100 范围）
                # gripper_angle 已经是 0-100 范围的值（来自 get_gripper() 返回）
                controller.set_gripper_target(value=float(gripper_angle), wait_for_completion=False)
            
            # 记录实际发送的动作
            if gripper_angle is not None:
                sent_actions.append(torch.tensor(joint_angles + [gripper_angle], dtype=torch.float32))
            else:
                sent_actions.append(torch.tensor(joint_angles, dtype=torch.float32))
        
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
                    controller.disconnect()
                    logging.info(f"已断开{arm_name}机械臂")
        
        self.is_connected = False
    
    def __del__(self):
        """析构函数。"""
        if getattr(self, "is_connected", False):
            self.disconnect()