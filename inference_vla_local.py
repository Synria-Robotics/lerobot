#!/usr/bin/env python
"""
仅使用本地文件加载 SmolVLA 策略（支持跳过 VLM，可选 dry-run 检查）。
- 默认严格离线（TRANSFORMERS_OFFLINE / HF_HUB_OFFLINE）
- 若没有本地 VLM 目录，可加 --skip_vlm 开启离线模式并跳过 VLM 加载
- 可选 dry-run：做一次最小张量通路检查，不依赖真实机器人
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import torch

# 兼容在仓库根目录直接运行（未 pip 安装）
try:
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors
except Exception:
    sys.path.append(str(Path(__file__).resolve().parent / "src"))
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

    lfs_suspects = [policy_dir / "model.safetensors"]
    lfs_suspects = [p for p in lfs_suspects if p.exists() and is_git_lfs_pointer(p)]
    if lfs_suspects:
        names = ", ".join(str(p.name) for p in lfs_suspects)
        raise RuntimeError(f"检测到 Git LFS 指针文件 ({names})，实际权重未下载。")


def read_config(policy_dir: Path) -> Dict[str, Any]:
    with open(policy_dir / "config.json", "r", encoding="utf-8") as f:
        return json.load(f)


def write_config(policy_dir: Path, cfg: Dict[str, Any]) -> None:
    with open(policy_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def set_offline_env() -> None:
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")


def looks_like_hf_id(name: str) -> bool:
    return "/" in name and not Path(name).exists()


def patch_policy_dir_with_local_vlm(policy_dir: Path, vlm_path: Path) -> Path:
    if not vlm_path.is_dir():
        raise FileNotFoundError(f"VLM 本地目录不存在: {vlm_path}")
    tmp_dir = Path(tempfile.mkdtemp(prefix="smolvla_policy_"))
    shutil.copytree(policy_dir, tmp_dir, dirs_exist_ok=True)
    cfg = read_config(tmp_dir)
    cfg["vlm_model_name"] = str(vlm_path)
    cfg["load_vlm_weights"] = True
    write_config(tmp_dir, cfg)
    return tmp_dir


def patch_policy_dir_skip_vlm(policy_dir: Path) -> Path:
    """复制到临时目录，并在 config 中开启离线模式以跳过 VLM 加载。"""
    tmp_dir = Path(tempfile.mkdtemp(prefix="smolvla_policy_offline_"))
    shutil.copytree(policy_dir, tmp_dir, dirs_exist_ok=True)
    cfg = read_config(tmp_dir)
    cfg["offline_mode"] = True
    cfg["load_vlm_weights"] = False
    write_config(tmp_dir, cfg)
    return tmp_dir


def resolve_device(requested: str) -> torch.device:
    req = requested.lower()
    if req == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps_ok = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
        if mps_ok:
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(req)


def run(args: argparse.Namespace) -> int:
    # 严格离线
    set_offline_env()

    policy_dir = Path(args.policy_path).resolve()
    validate_policy_dir(policy_dir)

    cfg = read_config(policy_dir)
    vlm_name = cfg.get("vlm_model_name", "")

    working_dir = policy_dir
    if args.skip_vlm:
        working_dir = patch_policy_dir_skip_vlm(policy_dir)
    elif args.vlm_path:
        working_dir = patch_policy_dir_with_local_vlm(policy_dir, Path(args.vlm_path).resolve())
    else:
        # 未显式提供本地 VLM，且配置看起来像 Hub ID 时，离线模式下会联网失败
        if looks_like_hf_id(vlm_name):
            raise RuntimeError(
                "未提供 --vlm_path，且 config.json 的 vlm_model_name 看起来是在线模型 ID；"
                " 如你没有本地 VLM，请加 --skip_vlm 来跳过 VLM 加载。"
            )

    device = resolve_device(args.device)

    # 强制本地加载
    policy = SmolVLAPolicy.from_pretrained(str(working_dir), local_files_only=True)
    policy.to(device)
    policy.eval()

    device_override = {"device": str(device)}
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(working_dir),
        preprocessor_overrides={"device_processor": device_override},
        postprocessor_overrides={"device_processor": device_override},
    )

    print("✅ 策略加载成功（仅本地）")
    print(
        f"Torch | cuda={torch.cuda.is_available()} | mps="
        f"{getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available()} | device={device}"
    )

    if args.dry_run:
        try:
            image_keys = list(getattr(policy.config, "image_features", {}).keys())
            obs: Dict[str, torch.Tensor | str] = {}

            if image_keys:
                obs[image_keys[0]] = torch.rand(1, 3, 512, 512, dtype=torch.float32, device=device)

            max_state = getattr(policy.config, "max_state_dim", 32)
            obs["observation.state"] = torch.zeros(1, max_state, dtype=torch.float32, device=device)
            obs["task"] = args.task

            obs_pp = preprocessor(obs)
            with torch.inference_mode():
                chunk = policy.predict_action_chunk(obs_pp)
                if chunk.ndim != 3:
                    chunk = chunk.unsqueeze(0)  # (B, T, A)

            # 后处理每个时间步
            B, T, A = chunk.shape
            processed = []
            for i in range(T):
                processed.append(postprocessor(chunk[:, i, :]))
            actions = torch.stack(processed, dim=1).squeeze(0).to("cpu")  # (T, A)
            print(f"✅ dry-run 成功 | 原始chunk={tuple(chunk.squeeze(0).shape)} | 后处理后={tuple(actions.shape)}")
        except Exception as e:
            print(f"❌ dry-run 失败: {e}")
            return 2

    return 0


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Load SmolVLA locally (offline) with optional dry-run; support skipping VLM")
    p.add_argument("--policy_path", type=str, required=True, help="本地策略目录（含 config.json/model.safetensors 等）")
    p.add_argument("--vlm_path", type=str, default=None, help="本地 VLM 目录（含 config/tokenizer/model 等），覆盖 config.json 的 vlm_model_name")
    p.add_argument("--skip_vlm", action="store_true", help="不开 VLM：临时将 config.offline_mode=True, load_vlm_weights=False")
    p.add_argument("--dry_run", action="store_true", help="执行一次最小张量通路检查")
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

#!/usr/bin/env python
"""
仅使用本地文件加载 SmolVLA 策略（可选 dry-run 检查）。
- 不联网：默认开启严格离线（TRANSFORMERS_OFFLINE / HF_HUB_OFFLINE）
- 可选覆盖 VLM 本地路径（--vlm_path），避免从 Hub 下载
- 可选 dry-run：做一次最小张量通路检查，不依赖真实机器人
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import torch

# 兼容在仓库根目录直接运行（未 pip 安装）
try:
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors
except Exception:
    sys.path.append(str(Path(__file__).resolve().parent / "src"))
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

    lfs_suspects = [policy_dir / "model.safetensors"]
    lfs_suspects = [p for p in lfs_suspects if p.exists() and is_git_lfs_pointer(p)]
    if lfs_suspects:
        names = ", ".join(str(p.name) for p in lfs_suspects)
        raise RuntimeError(f"检测到 Git LFS 指针文件 ({names})，实际权重未下载。")


def read_config(policy_dir: Path) -> Dict[str, Any]:
    with open(policy_dir / "config.json", "r", encoding="utf-8") as f:
        return json.load(f)


def set_offline_env(enable: bool) -> None:
    if enable:
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")


def looks_like_hf_id(name: str) -> bool:
    return "/" in name and not Path(name).exists()


def patch_policy_dir_with_local_vlm(policy_dir: Path, vlm_path: Path) -> Path:
    if not vlm_path.is_dir():
        raise FileNotFoundError(f"VLM 本地目录不存在: {vlm_path}")
    tmp_dir = Path(tempfile.mkdtemp(prefix="smolvla_policy_"))
    shutil.copytree(policy_dir, tmp_dir, dirs_exist_ok=True)
    cfg = read_config(tmp_dir)
    cfg["vlm_model_name"] = str(vlm_path)
    cfg["load_vlm_weights"] = True
    with open(tmp_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    return tmp_dir


def resolve_device(requested: str) -> torch.device:
    req = requested.lower()
    if req == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps_ok = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
        if mps_ok:
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(req)


def run(args: argparse.Namespace) -> int:
    policy_dir = Path(args.policy_path).resolve()
    validate_policy_dir(policy_dir)

    cfg = read_config(policy_dir)
    vlm_name = cfg.get("vlm_model_name", "")

    # 严格离线（默认启用）
    set_offline_env(True)

    working_dir = policy_dir
    if args.vlm_path:
        working_dir = patch_policy_dir_with_local_vlm(policy_dir, Path(args.vlm_path).resolve())
    else:
        # 未显式提供本地 VLM，且配置看起来像 Hub ID 时，直接在离线模式下报错
        if looks_like_hf_id(vlm_name) and not Path(vlm_name).exists():
            raise RuntimeError("离线模式下，config.json 的 vlm_model_name 看起来是在线模型 ID，请通过 --vlm_path 指定本地目录。")

    device = resolve_device(args.device)

    # 强制本地加载
    policy = SmolVLAPolicy.from_pretrained(str(working_dir), local_files_only=True)
    policy.to(device)
    policy.eval()

    device_override = {"device": str(device)}
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(working_dir),
        preprocessor_overrides={"device_processor": device_override},
        postprocessor_overrides={"device_processor": device_override},
    )

    print("✅ 策略加载成功（仅本地）")
    print(
        f"Torch | cuda={torch.cuda.is_available()} | mps="
        f"{getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available()} | device={device}"
    )

    if args.dry_run:
        try:
            image_keys = list(getattr(policy.config, "image_features", {}).keys())
            obs: Dict[str, torch.Tensor | str] = {}

            if image_keys:
                # 尽量匹配较大的输入，避免下游形状不符
                obs[image_keys[0]] = torch.rand(1, 3, 512, 512, dtype=torch.float32, device=device)

            max_state = getattr(policy.config, "max_state_dim", 32)
            obs["observation.state"] = torch.zeros(1, max_state, dtype=torch.float32, device=device)
            obs["task"] = args.task

            obs_pp = preprocessor(obs)
            with torch.inference_mode():
                chunk = policy.predict_action_chunk(obs_pp)
                if chunk.ndim != 3:
                    chunk = chunk.unsqueeze(0)  # (B, T, A)

            # 后处理每个时间步
            B, T, A = chunk.shape
            processed = []
            for i in range(T):
                processed.append(postprocessor(chunk[:, i, :]))
            actions = torch.stack(processed, dim=1).squeeze(0).to("cpu")  # (T, A)
            print(f"✅ dry-run 成功 | 原始chunk={tuple(chunk.squeeze(0).shape)} | 后处理后={tuple(actions.shape)}")
        except Exception as e:
            print(f"❌ dry-run 失败: {e}")
            return 2

    return 0


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Load SmolVLA locally (offline) with optional dry-run")
    p.add_argument("--policy_path", type=str, required=True, help="本地策略目录（含 config.json/model.safetensors 等）")
    p.add_argument("--vlm_path", type=str, default=None, help="本地 VLM 目录（含 config/tokenizer/model 等），覆盖 config.json 的 vlm_model_name")
    p.add_argument("--dry_run", action="store_true", help="执行一次最小张量通路检查")
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