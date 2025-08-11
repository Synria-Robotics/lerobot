"""
Replay a recorded dataset on the Alicia Duo robot.

This example loads a LeRobot dataset (e.g., "alicia_demo") and streams each
action to the real robot using Alicia-D-SDK's ControlApi, similar to demo_moveJ.

Safety notes:
- Ensure a safe environment before running. The robot will move through the recorded trajectory.
- Start with a low speed_factor and be ready to E-Stop.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List

import numpy as np
import torch

# Alicia-D SDK
try:
    from alicia_duo_sdk.controller import get_default_session, ControlApi
except ImportError:
    # Fallback for local import if the SDK is not installed but present in the repo
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    sdk_root = repo_root / "Alicia-D-SDK"
    if str(sdk_root) not in sys.path:
        sys.path.append(str(sdk_root))
    from alicia_duo_sdk.controller import get_default_session, ControlApi  # type: ignore

# LeRobot dataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def infer_joint_format(joint_values: np.ndarray | torch.Tensor) -> str:
    """Heuristic: treat as radians unless values clearly exceed radian ranges."""
    values = joint_values.detach().cpu().numpy() if isinstance(joint_values, torch.Tensor) else joint_values
    max_abs = float(np.max(np.abs(values))) if values.size > 0 else 0.0
    # If larger than ~3.6 rad (~206 deg), consider it's actually degrees
    return "deg" if max_abs > 3.6 else "rad"


def gripper_to_deg(value: float) -> int:
    """Convert a gripper scalar to degrees [0, 100] using simple heuristics."""
    v = float(value)
    if 0.0 <= v <= 1.0:
        return int(np.clip(v * 100.0, 0, 100))
    if -1.0 <= v < 0.0:
        return int(np.clip((v + 1.0) * 100.0, 0, 100))
    if -3.7 <= v <= 3.7:  # likely radians
        deg = v * 180.0 / np.pi
        return int(np.clip(deg, 0, 100))
    # Otherwise assume already degrees-like
    return int(np.clip(v, 0, 100))


def replay_episode(
    dataset: LeRobotDataset,
    episode_index: int = 0,
    speed_factor: float = 0.5,
    move_home_before: bool = True,
) -> None:
    session = get_default_session()
    controller = ControlApi(session=session)

    try:
        # if move_home_before:
        #     controller.moveHome()

        # Determine frame range for the episode
        from_idx = dataset.episode_data_index["from"][episode_index].item()
        to_idx = dataset.episode_data_index["to"][episode_index].item()

        # Use timestamps to pace playback; fallback to fps if needed
        use_timestamp_pacing = True
        first_ts = None
        start_time = None

        for idx in range(from_idx, to_idx):
            item = dataset[idx]

            # Extract action: expected shape (7,) -> 6 joints + 1 gripper
            action: torch.Tensor = item["action"].flatten()
            joints, gripper = action[:6], action[6].item()

            # Determine joint unit and send command
            joint_format = infer_joint_format(joints)
            target_joints: List[float] = joints.detach().cpu().tolist()

            if idx == 0:
                controller.moveJ(
                    joint_format=joint_format,
                    target_joints=target_joints,
                    speed_factor=speed_factor,
                    visualize=False
                )
            print(f"idx: {idx}, target_joints: {target_joints}")
            controller.joint_controller.set_joint_angles(
                target_joints
            )

            # controller.moveJ(
            #     joint_format=joint_format,
            #     target_joints=target_joints,
            #     speed_factor=speed_factor,
            #     visualize=False,
            #     n_steps_ref=1,
            #     T_default=0.01,
            # )

            # # Gripper control (best-effort mapping to degrees 0..100)
            # controller.gripper_control(angle_deg=gripper_to_deg(gripper))

            # # Pacing
            # if use_timestamp_pacing:
            #     ts_tensor: torch.Tensor = item["timestamp"].flatten()
            #     ts = float(ts_tensor.item())
            #     if first_ts is None:
            #         first_ts = ts
            #         start_time = time.perf_counter()
            #     # target relative time
            #     elapsed = time.perf_counter() - start_time
            #     target = ts - first_ts
            #     sleep_s = target - elapsed
            #     if sleep_s > 0:
            #         time.sleep(sleep_s)
            # else:
            #     time.sleep(1.0 / float(dataset.fps))
            time.sleep(0.02)

        # Return to home when done
        controller.moveHome()

    except KeyboardInterrupt:
        print("中断：停止回放")
    finally:
        session.joint_controller.disconnect()


def main():
    # Align defaults with examples/1_load_dataset.py
    repo_id = "alicia_demo"
    # Prefer a relative default; replace with your absolute path if needed
    default_root = Path(__file__).resolve().parents[1] / "datasets" / repo_id
    root = str(default_root)

    dataset = LeRobotDataset(repo_id, root, episodes=[0])
    print(f"加载数据集：{repo_id}，fps={dataset.fps}，回放 episode 0，共 {dataset.num_frames} 帧")

    replay_episode(dataset, episode_index=0, speed_factor=0.5, move_home_before=True)


if __name__ == "__main__":
    main()


