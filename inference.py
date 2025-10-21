#!/usr/bin/env python
"""
Alicia-D 机器人在线推理脚本（简化版）

用法:
    python inference.py
"""

import time
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.robots.alicia_d.config_alicia_d import AliciaDConfig
from lerobot.robots.alicia_d.alicia_d import AliciaD

# ============ 配置 ============
POLICY_PATH = "/home/ubuntu/vla/outputs/checkpoints/last/pretrained_model"
INFERENCE_TIME_S = 60  # 推理时长（秒）
FPS = 30  # 控制频率
USE_CAMERAS = True  # 是否使用摄像头
EXECUTE = True  # 是否实际下发到机器人
STANDARD_SHAPE = (480, 640)  # (height, width)
# ============================


def busy_wait(wait_time):
    """等待指定时间"""
    if wait_time > 0:
        time.sleep(wait_time)


def main():
    print("🤖 Alicia-D 机器人推理启动")
    print("=" * 50)
    
    # 检查设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  设备: {device}")
    
    # 创建机器人配置
    print("\n🔧 正在连接机器人...")
    if USE_CAMERAS:
        cameras = AliciaDConfig.default_cameras_config()
    else:
        cameras = {}
    
    config = AliciaDConfig(
        port="/dev/ttyUSB0",
        baudrate=1_000_000,
        cameras=cameras,
        execute_motion=EXECUTE,
        # robot_version="v5_6",
        # gripper_type="50mm",
    )
    
    # 连接机器人
    robot = AliciaD(config)
    robot.connect()
    print("✅ 机器人连接成功")
    
    # 加载策略
    print(f"\n🧠 正在加载模型: {POLICY_PATH}")
    policy = DiffusionPolicy.from_pretrained(POLICY_PATH)
    policy.to(device)
    policy.eval()
    print("✅ 模型加载成功")
    
    print(f"\n▶️  开始推理 (时长: {INFERENCE_TIME_S}s, 频率: {FPS}Hz)")
    print("=" * 50)
    
    # 推理循环
    for step in range(INFERENCE_TIME_S * FPS):
        start_time = time.perf_counter()
        
        # 读取观测
        observation = robot.get_observation()
        
        # 构建状态向量 (关节位置 + 夹爪位置)
        state_vector = []
        for i in range(1, 7):  # joint1 到 joint6
            state_vector.append(observation[f"joint{i}.pos"])
        state_vector.append(observation["gripper.pos"])  # 夹爪位置
        
        # 添加状态向量到观测中
        observation["observation.state"] = torch.tensor(state_vector, dtype=torch.float32)
        
        # 为没有摄像头的策略模型提供虚拟图像数据
        if not USE_CAMERAS:
            # 创建虚拟图像数据 (480, 640, 3) - RGB格式，然后转换为 (3, 480, 640)
            dummy_image_hwc = torch.zeros(480, 640, 3, dtype=torch.float32)
            dummy_image_chw = dummy_image_hwc.permute(2, 0, 1)  # 转换为 (C, H, W)
            observation["observation.images.wrist"] = dummy_image_chw
            observation["observation.images.front"] = dummy_image_chw
        
        # 处理观测为pytorch格式
        for name in observation:
            if "image" in name or name in ["wrist", "front"]:
                # 首先将numpy数组转换为PyTorch张量
                if isinstance(observation[name], np.ndarray):
                    observation[name] = torch.from_numpy(observation[name]).float()
                
                # 图像处理 - 确保正确的形状
                if observation[name].dim() == 3:  # (C, H, W) 或 (H, W, C) 格式
                    if observation[name].shape[0] == 3:  # 已经是 (C, H, W) 格式
                        observation[name] = observation[name].type(torch.float32) / 255
                    else:  # (H, W, C) 格式，需要转换
                        observation[name] = observation[name].type(torch.float32) / 255
                        observation[name] = observation[name].permute(2, 0, 1).contiguous()  # 转换为 (C, H, W)
                elif observation[name].dim() == 4:  # 已经是 (B, C, H, W) 格式
                    observation[name] = observation[name].type(torch.float32) / 255
                else:
                    # 如果已经是 (C, H, W) 格式，直接处理
                    observation[name] = observation[name].type(torch.float32) / 255
                
                # 确保图像是 (C, H, W) 格式，然后调整大小
                if observation[name].dim() == 3:
                    c, h, w = observation[name].shape
                    if (h, w) != STANDARD_SHAPE:
                        observation[name] = F.interpolate(
                            observation[name].unsqueeze(0),
                            size=STANDARD_SHAPE,
                            mode="bilinear",
                            align_corners=False
                        ).squeeze(0)
                elif observation[name].dim() == 4:
                    b, c, h, w = observation[name].shape
                    if (h, w) != STANDARD_SHAPE:
                        observation[name] = F.interpolate(
                            observation[name],
                            size=STANDARD_SHAPE,
                            mode="bilinear",
                            align_corners=False
                        )
        
        # 重命名摄像头数据以匹配策略模型期望的键名
        if "wrist" in observation:
            observation["observation.images.wrist"] = observation.pop("wrist")
        if "front" in observation:
            observation["observation.images.front"] = observation.pop("front")
        
        # 处理所有观测数据
        for name in observation:
            # 只对张量数据调用unsqueeze，标量数据转换为张量
            if isinstance(observation[name], torch.Tensor):
                if observation[name].dim() == 1:  # 一维张量，需要添加batch维度
                    observation[name] = observation[name].unsqueeze(0)
                elif observation[name].dim() == 3:  # 三维张量 (C, H, W)，需要添加batch维度
                    observation[name] = observation[name].unsqueeze(0)
                # 如果已经是4维 (B, C, H, W) 或2维 (B, F)，不需要添加维度
            else:
                # 将标量转换为张量
                observation[name] = torch.tensor(observation[name], dtype=torch.float32).unsqueeze(0)
            
            observation[name] = observation[name].to(device)
        
        # 调试：打印观测数据的形状
        if step == 0:  # 只在第一步打印
            print("🔍 观测数据形状:")
            for name, data in observation.items():
                if isinstance(data, torch.Tensor):
                    print(f"  {name}: {data.shape}")
                else:
                    print(f"  {name}: {type(data)} = {data}")
        
        # 推理动作
        with torch.inference_mode():
            action = policy.select_action(observation)
        action = action.squeeze(0).to("cpu")
        
        if step % 30 == 0:  # 每秒打印一次
            print(f"Step {step:4d} | Action: {action[:7].numpy()}")
        
        # 发送动作到机器人
        action_dict = {
            'joint1.pos': float(action[0]),
            'joint2.pos': float(action[1]),
            'joint3.pos': float(action[2]),
            'joint4.pos': float(action[3]),
            'joint5.pos': float(action[4]),
            'joint6.pos': float(action[5]),
            'gripper.pos': float(action[6]),
        }
        robot.send_action(action_dict)
        
        # 控制频率
        dt_s = time.perf_counter() - start_time
        busy_wait(1 / FPS - dt_s)
    
    print("\n" + "=" * 50)
    print(f"✅ 推理完成，总步数: {INFERENCE_TIME_S * FPS}")
    robot.disconnect()


if __name__ == "__main__":
    main()
