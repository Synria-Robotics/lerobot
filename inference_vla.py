#!/usr/bin/env python
"""
安全的 SmolVLA 离线推理与策略自检脚本（不连接机器人，不联网）。

功能
- 校验本地策略目录是否完整（含 Git LFS 指针检测）。
- 可选覆盖 VLM 本地路径，避免从 HuggingFace 下载。
- 支持严格离线模式（设置 TRANSFORMERS_OFFLINE / HF_HUB_OFFLINE）。
- 可选 dry-run：仅做装载与最小张量通路检查，不执行真实控制。

用法
    python inference_vla.py \
        --policy_path /home/ubuntu/vla/lerobot/smolvla_base \
        --vlm_path /absolute/path/to/local/smolvlm \
        --offline \
        --dry_run
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import torch
from typing import Dict, Any, Tuple

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors


REQUIRED_POLICY_FILES = [
    "config.json",
    "model.safetensors",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
]


def is_git_lfs_pointer(file_path: Path) -> bool:
    try:
        with open(file_path, "rb") as f:
            head = f.read(64)
        return b"git-lfs.github.com/spec/v1" in head
    except Exception:
        return False


def validate_policy_dir(policy_dir: Path) -> None:
    if not policy_dir.is_dir():
        raise FileNotFoundError(f"策略目录不存在: {policy_dir}")

    missing = [p for p in REQUIRED_POLICY_FILES if not (policy_dir / p).exists()]
    if missing:
        raise FileNotFoundError(f"策略目录缺少必要文件: {missing}")

    # LFS 指针检测
    lfs_suspects = [
        p for p in [
            policy_dir / "model.safetensors",
            policy_dir / "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
            policy_dir / "policy_preprocessor_step_5_normalizer_processor.safetensors",
        ]
        if p.exists() and is_git_lfs_pointer(p)
    ]
    if lfs_suspects:
        names = ", ".join(str(p.name) for p in lfs_suspects)
        raise RuntimeError(
            f"检测到 Git LFS 指针文件 ({names})，实际权重未下载。请先获取真实权重后再试。"
        )


def read_config(policy_dir: Path) -> dict:
    with open(policy_dir / "config.json", "r", encoding="utf-8") as f:
        return json.load(f)


def patch_policy_dir_with_local_vlm(policy_dir: Path, vlm_path: Path) -> Path:
    """复制策略目录到临时目录，并把 config.json 的 vlm_model_name 指向本地路径。"""
    if not vlm_path.is_dir():
        raise FileNotFoundError(f"VLM 本地目录不存在: {vlm_path}")

    tmp_dir = Path(tempfile.mkdtemp(prefix="smolvla_policy_"))
    shutil.copytree(policy_dir, tmp_dir, dirs_exist_ok=True)

    cfg = read_config(tmp_dir)
    cfg["vlm_model_name"] = str(vlm_path)
    # 建议显式加载 VLM 权重
    cfg["load_vlm_weights"] = True

    with open(tmp_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    return tmp_dir


def set_offline_env(enable: bool) -> None:
    if enable:
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")


def looks_like_hf_id(name: str) -> bool:
    return "/" in name and not Path(name).exists()


def run(args: argparse.Namespace) -> int:
    policy_dir = Path(args.policy_path).resolve()
    validate_policy_dir(policy_dir)

    cfg = read_config(policy_dir)
    vlm_name = cfg.get("vlm_model_name", "")

    # 严格离线模式
    set_offline_env(args.offline)

    working_dir = policy_dir
    if args.vlm_path:
        working_dir = patch_policy_dir_with_local_vlm(policy_dir, Path(args.vlm_path).resolve())
    else:
        # 未提供本地 VLM 覆盖，且配置看起来是 Hub ID 时，若 offline 则直接报错，避免误联网
        if args.offline and looks_like_hf_id(vlm_name):
            raise RuntimeError(
                "当前处于离线模式，但 config.json 的 vlm_model_name 看起来是在线模型 ID。"
                " 请通过 --vlm_path 提供本地 VLM 目录。"
            )

    # 仅检查，不实际加载
    if args.check_only:
        print("✅ 自检通过：策略目录结构有效。")
        if working_dir != policy_dir:
            print(f"ℹ️ 已准备覆盖 VLM 的临时目录: {working_dir}")
        return 0

    def _resolve_device(requested: str) -> torch.device:
        req = requested.lower()
        if req == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            mps_ok = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
            if mps_ok:
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(req)

    device = _resolve_device(args.device)

    # 加载策略（强制本地加载）
    policy = SmolVLAPolicy.from_pretrained(str(working_dir), local_files_only=True)
    policy.to(device)
    policy.eval()

    # 构建与服务器一致的预处理/后处理流水线，并对齐设备
    device_override = {"device": str(device)}
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(working_dir),
        preprocessor_overrides={"device_processor": device_override},
        postprocessor_overrides={"device_processor": device_override},
    )

    print("✅ 策略加载成功（本地）")
    print(
        f"Torch env | cuda_available={torch.cuda.is_available()} | mps_available="
        f"{getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available()} | device={device}"
    )

    if args.dry_run:
        # 做一次轻量的张量通路检查，不依赖真实机器人
        # 尽量使用配置中的期望特征来构造最小输入
        image_keys = list(getattr(policy.config, "image_features", {}).keys())
        if not image_keys:
            print("⚠️ 配置未声明 image_features，跳过图像通路测试。")
        obs = {}

        # 伪造一张 3x512x512 的图像（值域 [0,1]）
        if image_keys:
            obs[image_keys[0]] = torch.rand(1, 3, 512, 512, dtype=torch.float32, device=device)

        # 伪造状态向量（长度取 config.max_state_dim）
        max_state = getattr(policy.config, "max_state_dim", 32)
        obs["observation.state"] = torch.zeros(1, max_state, dtype=torch.float32, device=device)

        # 语言指令：若 tokenizer 可用则跳过手动注入，由预处理器处理；否则直接放 task 字符串
        try:
            _ = policy.model.vlm_with_expert.processor.tokenizer  # noqa: F841
            obs["task"] = args.task
        except Exception:
            obs["task"] = args.task

        # 使用与服务器一致的流程：preprocess -> predict_action_chunk -> postprocess
        try:
            obs_pp = preprocessor(obs)
            with torch.inference_mode():
                chunk = policy.predict_action_chunk(obs_pp)
                if chunk.ndim != 3:
                    chunk = chunk.unsqueeze(0)  # (B, T, A)
            # 后处理每个时间步
            B, T, A = chunk.shape
            processed = []
            for i in range(T):
                single = chunk[:, i, :]
                processed.append(postprocessor(single))
            actions = torch.stack(processed, dim=1).squeeze(0).to("cpu")  # (T, A)
            print(f"✅ dry-run 成功 | 原始chunk形状={tuple(chunk.squeeze(0).shape)} | 后处理后形状={tuple(actions.shape)}")
        except Exception as e:
            print(f"❌ dry-run 失败: {e}")
            return 2

        return 0

    # 非 dry-run：执行一次同步推理并打印第一步动作
    try:
        image_keys = list(getattr(policy.config, "image_features", {}).keys())
        obs = {}
        if image_keys:
            obs[image_keys[0]] = torch.rand(1, 3, 224, 224, dtype=torch.float32, device=device)
        max_state = getattr(policy.config, "max_state_dim", 32)
        obs["observation.state"] = torch.zeros(1, max_state, dtype=torch.float32, device=device)
        obs["task"] = args.task

        obs_pp = preprocessor(obs)
        with torch.inference_mode():
            chunk = policy.predict_action_chunk(obs_pp)
            if chunk.ndim != 3:
                chunk = chunk.unsqueeze(0)
        processed = []
        for i in range(chunk.shape[1]):
            processed.append(postprocessor(chunk[:, i, :]))
        actions = torch.stack(processed, dim=1).squeeze(0).to("cpu")
        first = actions[0]
        preview = first.tolist() if first.numel() <= 16 else first[:16].tolist()
        print(f"✅ 同步推理完成 | 第一步动作维度={tuple(first.shape)} | 预览(前16)：{preview}")
        return 0
    except Exception as e:
        print(f"❌ 推理失败: {e}")
        return 3


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Safe SmolVLA offline inference & self-check")
    p.add_argument("--policy_path", type=str, required=True, help="本地 SmolVLA 策略目录（含 config.json/model.safetensors 等）")
    p.add_argument("--vlm_path", type=str, default=None, help="本地 SmolVLM 目录（含 config/tokenizer/model 等），覆盖 config.json 的 vlm_model_name")
    p.add_argument("--offline", action="store_true", help="严格离线模式（禁用一切联网加载）")
    p.add_argument("--dry_run", action="store_true", help="加载策略并做一次最小张量通路检查")
    p.add_argument("--check_only", action="store_true", help="仅做目录与配置自检，不加载模型")
    p.add_argument("--task", type=str, default="pick up the cube", help="语言任务指令")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="推理设备")
    return p


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    try:
        code = run(args)
    except Exception as e:
        print(f"❌ 终止：{e}")
        sys.exit(1)
    sys.exit(code)

