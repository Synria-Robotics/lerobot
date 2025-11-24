#!/usr/bin/env python
"""
一键运行：Alicia-D + SmolVLA（同步推理，直接输出关节角）

- 直接运行：python inference_vla.py
- 去掉了异步推理（不再启动 gRPC server/client）
- 观测包含：夹爪状态、关节角、相机图像、task 文本
- 策略输出：7 维关节角 [joint1..joint6, gripper]
- 直接调用 SDK 关节控制接口完成执行
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# 兼容未安装本仓库到环境时，直接从源码导入
try:
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    from lerobot.robots.alicia_d.config_alicia_d import AliciaDConfig
    from lerobot.robots.alicia_d.alicia_d import AliciaD
except Exception:
    sys.path.append(str(Path(__file__).resolve().parent / "src"))
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    from lerobot.robots.alicia_d.config_alicia_d import AliciaDConfig
    from lerobot.robots.alicia_d.alicia_d import AliciaD


# ===== 固定配置（可按需修改） =====
FPS = 30
TASK_TEXT = "pick up the cube"
PRETRAINED = "/home/ubuntu/vla/outputs/checkpoints/last/pretrained_model"  # 可改为本地目录

# 实机执行动作
EXECUTE_MOTION = True


def resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    mps_ok = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    if mps_ok:
        return "mps"
    return "cpu"


def unset_offline_env() -> None:
    # 如需从 Hub 在线加载权重，确保未强制离线
    for k in ("TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE"):
        if k in os.environ:
            os.environ.pop(k, None)


def make_alicia_config() -> AliciaDConfig:
    # 直接使用 config_alicia_d.py 中的默认摄像头配置
    return AliciaDConfig(
        execute_motion=EXECUTE_MOTION,
        port=None,  # SDK 自行扫描；若你明确串口可在此填写
        baudrate=1_000_000,
    )


def _to_chw_float_tensor(img_np: np.ndarray) -> torch.Tensor:
    # 输入 HxWxC uint8/float -> 输出 CxHxW float32 in [0,1]
    t = torch.from_numpy(img_np)
    if t.ndim == 3 and t.shape[-1] == 3:  # HWC
        t = t.permute(2, 0, 1).contiguous()
    t = t.float()
    if t.max() > 1.0:
        t = t / 255.0
    return t


def main() -> int:
    unset_offline_env()

    # 1) 连接 Alicia-D（读相机 + SDK 控制）
    alicia_cfg = make_alicia_config()
    robot = AliciaD(alicia_cfg)
    robot.connect()

    # 2) 加载 SmolVLA（预训练）并创建预/后处理器
    device = resolve_device()
    print(f"检测到设备: {device}")
    print(f"CUDA可用: {torch.cuda.is_available()}, GPU数量: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"当前CUDA设备: {torch.cuda.current_device()}, 设备名称: {torch.cuda.get_device_name(0)}")
    
    policy_cls = get_policy_class("smolvla")
    policy = policy_cls.from_pretrained(PRETRAINED)
    policy.to(device)
    policy.eval()  # 设置为评估模式
    
    # 验证模型是否在GPU上
    try:
        first_param = next(policy.parameters())
        print(f"模型参数设备: {first_param.device}")
    except StopIteration:
        print("警告: 模型没有可训练参数")
    
    # 让处理器与模型同设备运行
    preproc, postproc = make_pre_post_processors(
        policy.config,
        pretrained_path=PRETRAINED,
        preprocessor_overrides={"device_processor": {"device": device}},
        postprocessor_overrides={"device_processor": {"device": device}},
    )

    print(f"已加载策略: {PRETRAINED} | device={device} | FPS={FPS}")
    try:
        dt_target = 1.0 / max(1, FPS)
        frame_count = 0
        while True:
            t0 = time.perf_counter()
            frame_count += 1

            # 2.1 采集观测（关节角 + 图像 + 任务文本）
            obs_raw = robot.get_observation()
            joint_positions = obs_raw.get("joint_positions")
            gripper_pos = float(obs_raw.get("gripper.pos", 0.0))
            if joint_positions is None or len(joint_positions) < 6:
                print("警告：未获取到关节角，跳过本帧")
                time.sleep(dt_target)
                continue

            wrist_frame = obs_raw.get("wrist")
            top_frame = obs_raw.get("top")
            front_frame = obs_raw.get("front")

            # 至少要有腕部/俯视图像；正面相机可选，但如果也给了就一起用
            if wrist_frame is None or top_frame is None:
                print("警告：主要相机帧缺失（wrist/top），跳过本帧")
                time.sleep(dt_target)
                continue

            state_vec = [float(x) for x in joint_positions] + [gripper_pos]

            # 准备图像（键名与预训练配置保持：observation.images.wrist/front）
            obs_dict: dict[str, torch.Tensor | str] = {
                "observation.state": torch.tensor(state_vec, dtype=torch.float32),
                "observation.images.wrist": _to_chw_float_tensor(wrist_frame),
                "observation.images.top": _to_chw_float_tensor(top_frame),
                "observation.images.front": _to_chw_float_tensor(front_frame),
            #在此处修改需要输入的命令，不想输入就直接使用TASK_TEXT
                "task": TASK_TEXT,
            }
            if front_frame is not None:
                obs_dict["observation.images.front"] = _to_chw_float_tensor(front_frame)

            # 兼容某些预训练配置期望的相机键名（camera1/2/3）
            obs_dict["observation.images.camera1"] = obs_dict["observation.images.wrist"]
            obs_dict["observation.images.camera2"] = obs_dict["observation.images.top"]
            if "observation.images.front" in obs_dict:
                obs_dict["observation.images.camera3"] = obs_dict["observation.images.front"]

            # 2.2 预处理（tokenize/normalize/加 batch/放到设备）
            obs_proc = preproc(obs_dict)
            
            # 调试：只在第一帧检查设备（避免刷屏）
            if frame_count == 1 and isinstance(obs_proc, dict):
                for k, v in obs_proc.items():
                    if isinstance(v, torch.Tensor):
                        print(f"[调试] 预处理后 {k} 设备: {v.device}, shape: {v.shape}, dtype: {v.dtype}")
                # 检查所有tensor设备
                tensor_devices = {k: v.device for k, v in obs_proc.items() if isinstance(v, torch.Tensor)}
                print(f"[调试] 所有预处理tensor设备: {tensor_devices}")

            # 2.3 推理一个动作（关节角）并后处理（反归一化/移到CPU）
            with torch.inference_mode():
                action_tensor = policy.select_action(obs_proc)  # (B, action_dim)
                if frame_count == 1 and isinstance(action_tensor, torch.Tensor):
                    print(f"[调试] 推理输出设备: {action_tensor.device}, shape: {action_tensor.shape}")
            action_processed = postproc(action_tensor).detach()
            action_vec = action_processed.squeeze(0).cpu() if action_processed.ndim > 1 else action_processed.cpu()

            # 期望输出为 7 维关节角（joint1..6, gripper）
            if action_vec.ndim != 1 or action_vec.numel() < 7:
                print(f"动作维度异常：{tuple(action_vec.shape)}，跳过本帧")
                time.sleep(dt_target)
                continue

            joint_targets = [float(v) for v in action_vec[:6]]
            gripper_target = float(action_vec[6].item())

            print(
                "target_joints:",
                [round(val, 4) for val in joint_targets],
                "gripper:",
                round(gripper_target, 3),
            )

            # 2.4 直接下发关节角
            robot.send_action({
                "joint_positions": joint_targets,
                "gripper.pos": gripper_target,
            })

            # 频率控制
            elapsed = time.perf_counter() - t0
            sleep_t = dt_target - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    except KeyboardInterrupt:
        print("\n停止推理，准备断开连接...")
    finally:
        try:
            robot.disconnect()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

