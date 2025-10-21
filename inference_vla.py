#!/usr/bin/env python
"""
Alicia-D 机器人 VLA（Vision-Language-Action）在线推理脚本

VLA 模型需要视觉输入和语言指令，能够根据自然语言命令控制机器人。

用法:
    python inference_vla.py
"""

import time
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.robots.alicia_d.config_alicia_d import AliciaDConfig
from lerobot.robots.alicia_d.alicia_d import AliciaD
from lerobot.utils.constants import (
    OBS_LANGUAGE_TOKENS,
    OBS_LANGUAGE_ATTENTION_MASK,
)

# ============ 配置 ============
POLICY_PATH = "/home/ubuntu/vla/lerobot/smolvla_base"  # VLA 模型路径
TASK_INSTRUCTION = "pick up the cube"  # 任务指令（英文）
INFERENCE_TIME_S = 60  # 推理时长（秒）
FPS = 30  # 控制频率
USE_CAMERAS = True  # VLA 必须使用摄像头
EXECUTE = True  # 是否实际下发到机器人
STANDARD_SHAPE = (224, 224)  # VLA 模型通常使用 224x224 图像
# ============================


def busy_wait(wait_time):
    """等待指定时间"""
    if wait_time > 0:
        time.sleep(wait_time)


def main():
    print("🤖 Alicia-D VLA 机器人推理启动")
    print("=" * 50)
    print(f"📝 任务指令: {TASK_INSTRUCTION}")
    print("=" * 50)
    
    # 检查设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  设备: {device}")
    
    # VLA 必须使用摄像头
    if not USE_CAMERAS:
        print("⚠️  VLA 模型需要视觉输入，强制启用摄像头")
    
    # 创建机器人配置
    print("\n🔧 正在连接机器人...")
    cameras = AliciaDConfig.default_cameras_config()
    camera_names_in_order = list(cameras.keys())
    
    config = AliciaDConfig(
        port="/dev/ttyUSB0",
        baudrate=1_000_000,
        cameras=cameras,
        execute_motion=EXECUTE,
    )
    
    # 连接机器人
    robot = AliciaD(config)
    robot.connect()
    print("✅ 机器人连接成功")
    
    # 加载 VLA 策略
    print(f"\n🧠 正在加载 VLA 模型: {POLICY_PATH}")
    try:
        policy = SmolVLAPolicy.from_pretrained(POLICY_PATH)
        policy.to(device)
        policy.eval()
        print("✅ VLA 模型加载成功")
    except Exception as e:
        print(f"❌ VLA 模型加载失败: {e}")
        print("💡 提示: 请确保模型路径正确，且模型是 VLA 类型")
        robot.disconnect()
        return

    # 读取策略期望的图像键（例如 observation.images.camera1/2/...）
    try:
        expected_image_keys = list(policy.config.image_features.keys())
    except Exception:
        expected_image_keys = []

    # 预编语言指令（转 tokens 与 attention mask）
    try:
        tokenizer = policy.model.vlm_with_expert.processor.tokenizer
    except Exception:
        tokenizer = None
    if tokenizer is None:
        print("⚠️ 未找到 tokenizer，将跳过语言输入。")
        lang_input_ids = None
        lang_attn_mask = None
    else:
        enc = tokenizer(
            TASK_INSTRUCTION,
            return_tensors="pt",
            padding=False,
            truncation=True,
        )
        lang_input_ids = enc["input_ids"].to(device)
        lang_attn_mask = enc["attention_mask"].to(device)
    
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
        
        # 处理图像观测
        for name in observation:
            if "image" in name or name in ["wrist", "front", "top"]:
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
                
                # 确保图像是 (C, H, W) 格式，然后调整大小到 VLA 标准尺寸
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
        
        # 将来自 AliciaD 的相机键（按 config 顺序）映射到策略期望键名
        # 先收集已处理好的图像（按相机配置顺序）
        processed_images = []
        for cam_name in camera_names_in_order:
            if cam_name in observation:
                processed_images.append(observation[cam_name])

        # 逐个映射到 expected_image_keys
        for idx, key in enumerate(expected_image_keys):
            if idx < len(processed_images):
                observation[key] = processed_images[idx]
        
        # 处理所有观测数据，添加 batch 维度
        for name in observation:
            if isinstance(observation[name], torch.Tensor):
                if observation[name].dim() == 1:  # 一维张量，需要添加batch维度
                    observation[name] = observation[name].unsqueeze(0)
                elif observation[name].dim() == 3:  # 三维张量 (C, H, W)，需要添加batch维度
                    observation[name] = observation[name].unsqueeze(0)
            else:
                # 将标量转换为张量
                observation[name] = torch.tensor(observation[name], dtype=torch.float32).unsqueeze(0)
            
            observation[name] = observation[name].to(device)
        
        # 注入语言指令 tokens 与 attention mask
        if lang_input_ids is not None and lang_attn_mask is not None:
            observation[OBS_LANGUAGE_TOKENS] = lang_input_ids
            observation[OBS_LANGUAGE_ATTENTION_MASK] = lang_attn_mask

        # 添加语言指令到观测中
        # VLA 模型通常需要 'task' 或 'instruction' 键
        observation["task"] = TASK_INSTRUCTION
        
        # 调试：打印观测数据的形状（仅第一步）
        if step == 0:
            print("📊 观测数据形状:")
            for name, data in observation.items():
                if isinstance(data, torch.Tensor):
                    print(f"  {name}: {data.shape}")
                elif isinstance(data, str):
                    print(f"  {name}: '{data}'")
                else:
                    print(f"  {name}: {type(data)}")
        
        # VLA 推理动作
        with torch.inference_mode():
            try:
                action = policy.select_action(observation)
            except Exception as e:
                print(f"❌ 推理失败: {e}")
                if step == 0:
                    print("💡 可能的原因：")
                    print("  1. 模型输入格式不匹配")
                    print("  2. 缺少必要的观测数据")
                    print("  3. 语言指令格式不正确")
                break
        
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

