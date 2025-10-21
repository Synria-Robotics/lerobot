#!/usr/bin/env python
"""
Alicia-D 机器人在线推理脚本（简化版）

用法:
    python inference.py
"""

import time
import torch
import torch.nn.functional as F
from pathlib import Path

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.robots.alicia_d.config_alicia_d import AliciaDConfig
from lerobot.robots.alicia_d.alicia_d import AliciaD

# ============ 配置 ============
POLICY_PATH = "/home/ubuntu/vla/outputs/checkpoints/last/pretrained_model"
INFERENCE_TIME_S = 60  # 推理时长（秒）
FPS = 30  # 控制频率
USE_CAMERAS = False  # 是否使用摄像头
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
        robot_version="v5_6",
        gripper_type="50mm",
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
        
        # 处理观测为pytorch格式
        for name in observation:
            if "image" in name or name in ["wrist", "front"]:
                # 图像处理
                observation[name] = observation[name].type(torch.float32) / 255
                observation[name] = observation[name].permute(2, 0, 1).contiguous()
                c, h, w = observation[name].shape
                if (h, w) != STANDARD_SHAPE:
                    observation[name] = F.interpolate(
                        observation[name].unsqueeze(0),
                        size=STANDARD_SHAPE,
                        mode="bilinear",
                        align_corners=False
                    ).squeeze(0)
            
            observation[name] = observation[name].unsqueeze(0)
            observation[name] = observation[name].to(device)
        
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
