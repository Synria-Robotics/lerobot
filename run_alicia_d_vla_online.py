#!/usr/bin/env python
"""
一键运行：Alicia-D + SmolVLA（在线加载）

- 无命令行参数，直接运行即可：python run_alicia_d_vla_online.py
- 会启动本地 PolicyServer（gRPC），随后启动 RobotClient 连接服务器
- 预设为线上 smolvla（Hugging Face Hub），自动选择设备（cuda > mps > cpu）
- 预设相机为 Windows 常见索引（0: wrist, 1: front），如需更改请编辑本文件变量 CAM_Wrist/CAM_Front
- 默认 execute_motion=True（会实际下发动作到硬件）
"""

import os
import sys
import threading
import time
from pathlib import Path

import torch

# 兼容未安装本仓库到环境时，直接从源码导入
try:
    from lerobot.async_inference.configs import PolicyServerConfig, RobotClientConfig
    from lerobot.async_inference.policy_server import serve as serve_policy
    from lerobot.async_inference.robot_client import RobotClient
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.cameras.configs import Cv2Rotation
    from lerobot.robots.alicia_d.config_alicia_d import AliciaDConfig
except Exception:
    sys.path.append(str(Path(__file__).resolve().parent / "src"))
    from lerobot.async_inference.configs import PolicyServerConfig, RobotClientConfig
    from lerobot.async_inference.policy_server import serve as serve_policy
    from lerobot.async_inference.robot_client import RobotClient
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.cameras.configs import Cv2Rotation
    from lerobot.robots.alicia_d.config_alicia_d import AliciaDConfig


# ===== 固定配置（可按需修改后再次运行） =====
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8080
FPS = 20
TASK_TEXT = "pick up the cube"
POLICY_TYPE = "smolvla"
PRETRAINED = "lerobot/smolvla_base"  # 线上模型 ID；亦可改为本地目录

# Windows 常见相机索引（若你的相机顺序不同，请调整为 0/1/2...）
CAM_WRIST_INDEX = 0
CAM_FRONT_INDEX = 1

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
    # 确保允许联网下载权重
    for k in ("TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE"):
        if k in os.environ:
            os.environ.pop(k, None)


def make_alicia_config() -> AliciaDConfig:
    cameras = {
        "wrist": OpenCVCameraConfig(
            index_or_path=CAM_WRIST_INDEX, fps=30, width=640, height=480, rotation=Cv2Rotation.NO_ROTATION
        ),
        "front": OpenCVCameraConfig(
            index_or_path=CAM_FRONT_INDEX, fps=30, width=640, height=480, rotation=Cv2Rotation.NO_ROTATION
        ),
    }
    return AliciaDConfig(
        id="alicia-default",
        cameras=cameras,
        execute_motion=EXECUTE_MOTION,
        port=None,  # 让 SDK 自行扫描；若你明确串口可在此填写
        baudrate=1_000_000,
    )


def start_policy_server_in_background(cfg: PolicyServerConfig) -> threading.Thread:
    t = threading.Thread(target=serve_policy, args=(cfg,), daemon=True)
    t.start()
    return t


def main() -> int:
    unset_offline_env()

    # 1) 启动策略服务器（本地）
    server_cfg = PolicyServerConfig(host=SERVER_HOST, port=SERVER_PORT, fps=FPS)
    server_thread = start_policy_server_in_background(server_cfg)
    # 等待服务器起来
    time.sleep(1.0)

    # 2) 组装 Alicia-D 机器人配置（Windows 相机索引）
    alicia_cfg = make_alicia_config()

    # 3) 组装客户端配置（指向线上 SmolVLA）
    device = resolve_device()
    client_cfg = RobotClientConfig(
        policy_type=POLICY_TYPE,
        pretrained_name_or_path=PRETRAINED,
        robot=alicia_cfg,
        actions_per_chunk=50,
        task=TASK_TEXT,
        server_address=f"{SERVER_HOST}:{SERVER_PORT}",
        policy_device=device,
        chunk_size_threshold=0.5,
        fps=FPS,
        aggregate_fn_name="weighted_average",
        debug_visualize_queue_size=False,
    )

    # 4) 启动客户端并进入控制循环
    client = RobotClient(client_cfg)
    if not client.start():
        print("❌ 无法连接到策略服务器。请检查端口占用或网络。")
        return 1

    print(
        f"✅ 已连接 | device={device} | policy={PRETRAINED} | server={SERVER_HOST}:{SERVER_PORT} | task='{TASK_TEXT}'\n"
        f"   执行动作: {EXECUTE_MOTION} | 相机(wrist,front)=({CAM_WRIST_INDEX},{CAM_FRONT_INDEX}) | FPS={FPS}"
    )

    # 动作接收线程
    action_receiver_thread = threading.Thread(target=client.receive_actions, kwargs={"verbose": True}, daemon=True)
    action_receiver_thread.start()

    try:
        client.control_loop(task=TASK_TEXT, verbose=True)
    except KeyboardInterrupt:
        print("\n⏹ 停止中...")
    finally:
        client.stop()
        action_receiver_thread.join(timeout=2.0)
        print("✅ 已停止")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


