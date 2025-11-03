#!/usr/bin/env python
"""
一键运行：Alicia-D + SmolVLA（同步推理，输出末端位姿 -> IK）

- 直接运行：python inference_vla.py
- 去掉了异步推理（不再启动 gRPC server/client）
- 观测包含：夹爪状态、末端位姿、相机图像、task 文本
- 策略输出：末端位姿 [x, y, z, qx, qy, qz, qw]
- 通过 SDK `set_pose_target` 做逆解并下发到机器人
"""

import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch

# 兼容未安装本仓库到环境时，直接从源码导入
try:
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.cameras.configs import Cv2Rotation
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    from lerobot.robots.alicia_d.config_alicia_d import AliciaDConfig
    from lerobot.robots.alicia_d.alicia_d import AliciaD
except Exception:
    sys.path.append(str(Path(__file__).resolve().parent / "src"))
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.cameras.configs import Cv2Rotation
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    from lerobot.robots.alicia_d.config_alicia_d import AliciaDConfig
    from lerobot.robots.alicia_d.alicia_d import AliciaD


# ===== 固定配置（可按需修改） =====
FPS = 20
TASK_TEXT = "pick up the cube"
PRETRAINED = "lerobot/smolvla_base"  # 可改为本地目录

# Windows 常见相机索引（若你的相机顺序不同，请调整 0/1/2...）
CAM_WRIST_INDEX = 0
CAM_FRONT_INDEX = 2

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


def _normalize_quaternion(q: List[float]) -> List[float]:
    q_np = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q_np)
    if not np.isfinite(n) or n < 1e-8:
        return [0.0, 0.0, 0.0, 1.0]
    q_np = q_np / n
    return q_np.astype(np.float64).tolist()


def main() -> int:
    unset_offline_env()

    # 1) 连接 Alicia-D（读相机 + SDK 控制）
    alicia_cfg = make_alicia_config()
    robot = AliciaD(alicia_cfg)
    robot.connect()

    # 2) 加载 SmolVLA（预训练）并创建预/后处理器
    device = resolve_device()
    policy_cls = get_policy_class("smolvla")
    policy = policy_cls.from_pretrained(PRETRAINED)
    policy.to(device)
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
        while True:
            t0 = time.perf_counter()

            # 2.1 采集观测（构造仅包含 position/images/task 的原始观测）
            _tmp = robot.get_observation()  # 仅用于取相机帧
            pose_info = robot._controller.get_pose()  # 返回包含 position/quaternion_xyzw 的字典
            pos = pose_info.get("position")  # [x,y,z]
            quat = pose_info.get("quaternion_xyzw")  # [qx,qy,qz,qw]
            if pos is None or quat is None:
                print("警告：未获取到末端位姿，跳过本帧")
                time.sleep(dt_target)
                continue

            # 原始 observation：三个键
            obs_raw = {
                "position": {
                    "pose": [float(x) for x in list(pos)],
                    "quaternion": [float(x) for x in list(quat)],
                },
                "images": {
                    "wrist": _tmp.get("wrist"),
                    "front": _tmp.get("front"),
                },
                "task": TASK_TEXT,
            }
            print(obs_raw["position"])
            # 组装 observation.state = [gripper, x, y, z, qx, qy, qz, qw]
            gripper = float(robot._controller.get_gripper() or 0.0)
            state_vec: List[float] = [gripper] + obs_raw["position"]["pose"] + obs_raw["position"]["quaternion"]

            # 准备图像（键名与预训练配置保持：observation.images.wrist/front）
            obs_dict: dict = {
                "observation.state": torch.tensor(state_vec, dtype=torch.float32),
                "observation.images.wrist": _to_chw_float_tensor(obs_raw["images"]["wrist"]),
                "observation.images.front": _to_chw_float_tensor(obs_raw["images"]["front"]),
                "task": obs_raw["task"],
            }

            # 兼容某些预训练配置期望的相机键名（camera1/2/3）
            obs_dict["observation.images.camera1"] = obs_dict["observation.images.wrist"]
            obs_dict["observation.images.camera2"] = obs_dict["observation.images.front"]
            #obs_dict["observation.images.camera3"] = obs_dict["observation.images.front"]

            # 2.2 预处理（tokenize/normalize/加 batch/放到设备）
            obs_proc = preproc(obs_dict)

            # 2.3 推理一个动作（末端位姿）并后处理（反归一化/移到CPU）
            with torch.inference_mode():
                action_tensor = policy.select_action(obs_proc)  # (B, action_dim)
            action_out = postproc(action_tensor).squeeze(0)  # (action_dim,)

            # 期望输出为 7 维位姿（x,y,z,qx,qy,qz,qw）
            if action_out.ndim != 1 or action_out.numel() < 7:
                print(f"动作维度异常：{tuple(action_out.shape)}，跳过本帧")
                time.sleep(dt_target)
                continue

            # 归一化四元数，避免策略输出非单位四元数导致 IK 异常
            raw_pose = action_out[:7].tolist()
            target_pose = raw_pose[:3] + _normalize_quaternion(raw_pose[3:7])

            print(f"target_pose: {target_pose}")
            # 2.4 通过 SDK 做 IK 并执行
            _ik = robot.set_pose_target(
                target_pose=target_pose,
                execute=EXECUTE_MOTION,
                display=False,
            )

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

