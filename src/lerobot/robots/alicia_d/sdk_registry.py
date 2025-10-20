#!/usr/bin/env python

import logging
from pathlib import Path
import sys
from typing import Any, Dict, Tuple

logger = logging.getLogger(__name__)


_sdk_mod: Any | None = None
_controllers: Dict[Tuple[str, int, str, str, bool], Any] = {}
_refcounts: Dict[Tuple[str, int, str, str, bool], int] = {}


def _import_sdk() -> Any:
    global _sdk_mod
    if _sdk_mod is not None:
        return _sdk_mod
    try:
        import alicia_d_sdk  # type: ignore

        _sdk_mod = alicia_d_sdk
        return _sdk_mod
    except Exception:
        # try repo-local submodule locations
        try:
            repo_root = Path(__file__).resolve().parents[4]
            for d in [repo_root / "Alicia-D-SDK", repo_root / "lerobot" / "Alicia-D-SDK"]:
                if d.exists() and str(d) not in sys.path:
                    sys.path.insert(0, str(d))
            import alicia_d_sdk  # type: ignore

            _sdk_mod = alicia_d_sdk
            return _sdk_mod
        except Exception as e:
            logger.warning("未找到 Alicia-D SDK。请安装 `alicia_d_sdk` 或初始化子模块 `Alicia-D-SDK`。")
            raise e


def get_controller(
    *,
    port: str | None,
    baudrate: int,
    robot_version: str,
    gripper_type: str,
    debug_mode: bool,
) -> Any:
    """Get or create a shared controller for the given hardware key."""
    port_key = port or ""
    key = (port_key, baudrate, robot_version, gripper_type, debug_mode)

    if key in _controllers:
        _refcounts[key] = _refcounts.get(key, 0) + 1
        return _controllers[key]

    sdk = _import_sdk()
    controller = sdk.create_robot(
        port=port_key,
        baudrate=baudrate,
        robot_version=robot_version,
        gripper_type=gripper_type,
        debug_mode=debug_mode,
    )
    ok = controller.connect()
    if not ok:
        raise RuntimeError("Alicia-D 连接失败，请检查串口与供电。")

    _controllers[key] = controller
    _refcounts[key] = 1
    return controller


def release_controller(
    *,
    port: str | None,
    baudrate: int,
    robot_version: str,
    gripper_type: str,
    debug_mode: bool,
) -> None:
    port_key = port or ""
    key = (port_key, baudrate, robot_version, gripper_type, debug_mode)
    if key not in _controllers:
        return
    _refcounts[key] = max(0, _refcounts.get(key, 0) - 1)
    if _refcounts[key] == 0:
        try:
            logger.info(f"正在断开 Alicia-D 控制器: {key}")
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("SDK disconnect 超时")
            
            # 设置5秒超时
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(5)
            
            _controllers[key].disconnect()
            signal.alarm(0)  # 取消超时
            logger.info("Alicia-D 控制器已成功断开")
        except TimeoutError:
            logger.warning("SDK disconnect 超时，强制继续")
        except Exception:
            logger.exception("断开 Alicia-D 控制器失败")
        finally:
            _controllers.pop(key, None)
            _refcounts.pop(key, None)


