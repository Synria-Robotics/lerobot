#!/usr/bin/env python
"""
离线加载本地 SmolVLA 模型并打印成功反馈。

用法:
    python lerobot/inference_vla_outline.py
"""

import sys
import torch

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


# 本地模型路径（请确保包含 config.json 与 model.safetensors 等文件）
POLICY_PATH = "/home/ubuntu/vla/lerobot/smolvla_base"


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        policy = SmolVLAPolicy.from_pretrained(POLICY_PATH, local_files_only=True)
        policy.to(device)
        policy.eval()
    except Exception as e:
        print(f"❌ 本地模型加载失败: {e}")
        print("💡 请确认 POLICY_PATH 指向完整的本地权重目录，并且已禁用从Hub下载。")
        return 1

    print(f"✅ 本地模型加载成功: {POLICY_PATH} | 设备: {device}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


