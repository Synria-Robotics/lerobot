#!/usr/bin/env python
"""
Alicia-D 机器人在线推理脚本（新版 LeRobot）

支持 ACT 和 Diffusion Policy 的实时推理部署。
使用新版 LeRobot API 和 Alicia-D SDK v6.0.0。

用法:
    # ACT 推理（不实际下发，仅打印）
    python inference.py --policy_type act --policy_path /path/to/act/checkpoint

    # Diffusion Policy 推理并下发到硬件
    python inference.py --policy_type dp --policy_path /path/to/dp/checkpoint --execute

    # 自定义参数
    python inference.py --policy_type act --policy_path /path/to/checkpoint \
        --port /dev/ttyUSB0 --fps 30 --max_duration_s 60 --execute --speed_factor 0.8
"""

import argparse
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

# 新版 LeRobot imports
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.robots.alicia_d.config_alicia_d import AliciaDConfig
from lerobot.robots.alicia_d.alicia_d import AliciaD

# 图像标准尺寸（需与训练时一致）
STANDARD_IMAGE_SHAPE = (480, 640)  # (height, width)

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool) -> None:
    """设置日志"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Alicia-D 在线推理（ACT/Diffusion Policy）")

    # 硬件参数
    parser.add_argument("--port", type=str, default="/dev/ttyUSB0", help="串口设备")
    parser.add_argument("--baudrate", type=int, default=1_000_000, help="串口波特率")
    parser.add_argument("--robot_version", type=str, default="v5_6", help="机器人版本")
    parser.add_argument("--gripper_type", type=str, default="50mm", help="夹爪类型")
    parser.add_argument("--debug_mode", action="store_true", help="SDK 调试模式")

    # 推理参数
    parser.add_argument("--policy_type", choices=["act", "dp"], required=True, help="策略类型: act 或 dp")
    parser.add_argument("--policy_path", type=str, required=True, help="模型权重路径")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="推理设备")
    parser.add_argument("--fps", type=int, default=30, help="控制频率")
    parser.add_argument("--max_duration_s", type=float, default=0.0, help="最大运行时长（秒），0 表示不限制")
    parser.add_argument("--execute", action="store_true", help="是否实际下发到硬件（不加则仅打印）")
    parser.add_argument("--speed_factor", type=float, default=1.0, help="下发速度因子 (0,1]")

    # 摄像头参数
    parser.add_argument("--use_cameras", action="store_true", help="是否使用摄像头（需与训练配置一致）")
    parser.add_argument("--camera_wrist", type=str, default="/dev/video2", help="手腕摄像头设备")
    parser.add_argument("--camera_top", type=str, default="/dev/video0", help="顶部摄像头设备")

    # 其他
    parser.add_argument("--verbose", action="store_true", help="打印调试日志")

    return parser.parse_args()


def load_policy(policy_type: str, policy_path: str, device: str):
    """加载策略模型"""
    policy_path = Path(policy_path).expanduser().resolve()
    
    if not policy_path.exists():
        raise FileNotFoundError(f"模型路径不存在: {policy_path}")
    
    logger.info(f"正在加载 {policy_type.upper()} 模型: {policy_path}")
    
    try:
        if policy_type == "act":
            policy = ACTPolicy.from_pretrained(str(policy_path))
        elif policy_type == "dp":
            policy = DiffusionPolicy.from_pretrained(str(policy_path))
        else:
            raise ValueError(f"不支持的策略类型: {policy_type}")
        
        policy.to(device)
        policy.eval()
        logger.info("✅ 模型加载成功")
        return policy
        
    except Exception as e:
        logger.error(f"❌ 模型加载失败: {e}")
        raise


def make_robot(args: argparse.Namespace) -> AliciaD:
    """创建并连接机器人"""
    logger.info("正在初始化机器人...")
    
    # 创建机器人配置
    if args.use_cameras:
        from lerobot.cameras.opencv.config_opencv import OpenCVCameraConfig, Cv2Rotation
        cameras = {
            "wrist": OpenCVCameraConfig(
                index_or_path=args.camera_wrist,
                fps=30,
                width=640,
                height=480,
                rotation=Cv2Rotation.ROTATE_90
            ),
            "top": OpenCVCameraConfig(
                index_or_path=args.camera_top,
                fps=30,
                width=640,
                height=480,
                rotation=Cv2Rotation.ROTATE_90
            ),
        }
    else:
        cameras = {}
    
    config = AliciaDConfig(
        port=args.port,
        baudrate=args.baudrate,
        cameras=cameras,
        execute_motion=args.execute,  # 是否实际执行动作
        auto_detect_cameras=False,  # 不自动探测，使用指定配置
    )
    
    # 动态设置额外参数
    if hasattr(config, 'robot_version'):
        config.robot_version = args.robot_version
    if hasattr(config, 'gripper_type'):
        config.gripper_type = args.gripper_type
    if hasattr(config, 'debug_mode'):
        config.debug_mode = args.debug_mode
    
    robot = AliciaD(config)
    robot.connect()
    
    logger.info("✅ 机器人连接成功")
    return robot


def process_observation(obs: dict, device: str, use_cameras: bool) -> dict[str, torch.Tensor]:
    """
    处理观测数据为模型输入格式
    
    Args:
        obs: 原始观测数据 {
            'joint1.pos': float, 'joint2.pos': float, ..., 'gripper.pos': float,
            'wrist': np.ndarray, 'top': np.ndarray  # 可选
        }
        device: torch设备
        use_cameras: 是否使用摄像头
    
    Returns:
        batch: 模型输入格式 {
            'observation.state': Tensor[1, 7],
            'observation.images.wrist': Tensor[1, 3, H, W],  # 可选
            'observation.images.top': Tensor[1, 3, H, W],     # 可选
        }
    """
    batch = {}
    
    # 处理状态（关节+夹爪）
    state_keys = ['joint1.pos', 'joint2.pos', 'joint3.pos', 'joint4.pos', 
                  'joint5.pos', 'joint6.pos', 'gripper.pos']
    state = np.array([obs[k] for k in state_keys], dtype=np.float32)
    batch['observation.state'] = torch.from_numpy(state).unsqueeze(0).to(device)
    
    # 处理图像
    if use_cameras:
        for cam_key in ['wrist', 'top']:
            if cam_key in obs:
                img = obs[cam_key]  # numpy array [H, W, C]
                
                # 转换为 float32 并归一化到 [0, 1]
                img = img.astype(np.float32) / 255.0
                
                # 转换为 torch tensor 并调整维度 [H, W, C] -> [C, H, W]
                img_tensor = torch.from_numpy(img).permute(2, 0, 1)
                
                # 调整尺寸到标准大小
                c, h, w = img_tensor.shape
                if (h, w) != STANDARD_IMAGE_SHAPE:
                    img_tensor = F.interpolate(
                        img_tensor.unsqueeze(0),
                        size=STANDARD_IMAGE_SHAPE,
                        mode="bilinear",
                        align_corners=False
                    ).squeeze(0)
                
                # 添加 batch 维度并移到设备
                batch[f'observation.images.{cam_key}'] = img_tensor.unsqueeze(0).to(device)
    
    return batch


def busy_wait(dt_s: float) -> None:
    """等待指定时间"""
    if dt_s > 0:
        time.sleep(dt_s)


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)
    
    # 打印配置信息
    logger.info("=" * 60)
    logger.info("Alicia-D 在线推理启动")
    logger.info("=" * 60)
    logger.info(f"策略类型: {args.policy_type.upper()}")
    logger.info(f"模型路径: {args.policy_path}")
    logger.info(f"推理设备: {args.device}")
    logger.info(f"控制频率: {args.fps} Hz")
    logger.info(f"是否下发: {'是' if args.execute else '否（仅打印）'}")
    logger.info(f"使用摄像头: {'是' if args.use_cameras else '否'}")
    logger.info("=" * 60)
    
    # 加载策略
    try:
        policy = load_policy(args.policy_type, args.policy_path, args.device)
    except Exception as e:
        logger.error(f"策略加载失败: {e}")
        return 1
    
    # 创建并连接机器人
    robot: Optional[AliciaD] = None
    try:
        robot = make_robot(args)
    except Exception as e:
        logger.error(f"机器人初始化失败: {e}")
        return 1
    
    # 设置中断信号处理
    def handle_sigint(_sig, _frame):
        logger.info("\n捕获到中断信号，正在安全关闭...")
        if robot is not None:
            robot.disconnect()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, handle_sigint)
    
    # 重置策略状态
    policy.reset()
    
    # 推理主循环
    period = 1.0 / args.fps
    t0 = time.perf_counter()
    num_steps = 0
    
    logger.info("开始在线推理，按 Ctrl+C 停止...")
    logger.info("=" * 60)
    
    try:
        while True:
            loop_start = time.perf_counter()
            
            # 读取机器人观测
            try:
                obs = robot.get_observation()
            except Exception as e:
                logger.error(f"读取观测失败: {e}")
                break
            
            # 处理观测为模型输入格式
            try:
                batch = process_observation(obs, args.device, args.use_cameras)
            except Exception as e:
                logger.error(f"处理观测失败: {e}")
                break
            
            # 模型推理
            try:
                with torch.inference_mode():
                    action = policy.select_action(batch)  # [1, action_dim] or [action_dim]
                
                # 确保是 1D tensor
                if action.ndim > 1:
                    action = action.squeeze(0)
                action = action.cpu()
                
            except Exception as e:
                logger.error(f"推理失败: {e}")
                break
            
            # 构造动作字典
            action_dict = {
                'joint1.pos': float(action[0]),
                'joint2.pos': float(action[1]),
                'joint3.pos': float(action[2]),
                'joint4.pos': float(action[3]),
                'joint5.pos': float(action[4]),
                'joint6.pos': float(action[5]),
                'gripper.pos': float(action[6]),
            }
            
            # 发送动作（根据 config.execute_motion 决定是否实际执行）
            try:
                robot.send_action(action_dict)
                
                if not args.execute:
                    # 仅打印动作
                    logger.info(f"Step {num_steps:4d} | Action: {np.round(action.numpy(), 4).tolist()}")
                
            except Exception as e:
                logger.error(f"发送动作失败: {e}")
                break
            
            num_steps += 1
            
            # 频率控制
            dt = time.perf_counter() - loop_start
            busy_wait(period - dt)
            
            # 检查是否达到最大时长
            if args.max_duration_s > 0 and (time.perf_counter() - t0) >= args.max_duration_s:
                logger.info(f"达到最大运行时长 {args.max_duration_s}s，停止推理")
                break
    
    except KeyboardInterrupt:
        logger.info("\n用户中断")
    
    finally:
        if robot is not None:
            robot.disconnect()
            logger.info("=" * 60)
            logger.info(f"推理结束，累计步数: {num_steps}")
            logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
