#!/usr/bin/env python

import logging
from pathlib import Path
import sys
from typing import Any

logger = logging.getLogger(__name__)


_sdk_mod: Any | None = None


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
    """Create a new controller instance for the given hardware configuration.
    
    Each call creates a new controller instance, allowing multiple robots to use
    different ports independently.
    """
    port_key = port or ""
    
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
    
    logger.info(f"已创建新的 Alicia-D 控制器实例: port={port_key}, baudrate={baudrate}")
    return controller


def release_controller(
    *,
    port: str | None,
    baudrate: int,
    robot_version: str,
    gripper_type: str,
    debug_mode: bool,
    controller: Any | None = None,
) -> None:
    """Disconnect and release a controller instance.
    
    Args:
        controller: The controller instance to disconnect. If None, this function
                   does nothing (since we no longer track controllers globally).
    """
    if controller is None:
        logger.warning("release_controller 被调用但未提供 controller 实例，跳过断开操作")
        return
    
    try:
        logger.info("正在断开 Alicia-D 控制器")
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("SDK disconnect 超时")
        
        # 设置5秒超时
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(5)
        
        controller.disconnect()
        signal.alarm(0)  # 取消超时
        logger.info("Alicia-D 控制器已成功断开")
    except TimeoutError:
        logger.warning("SDK disconnect 超时，强制继续")
    except Exception:
        logger.exception("断开 Alicia-D 控制器失败")


