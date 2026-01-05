#!/usr/bin/env python
"""Evaluate a trained policy (ACT, Diffusion, etc.) on bimanual Alicia-D robot arms.

This script loads a trained policy and runs it on the bimanual robot for evaluation.
Optionally records evaluation episodes to a dataset.

Supports all policy types: ACT, Diffusion, TDMPC, VQBeT, etc.

Example usage for ACT:
```shell
python examples/alicia/eval_alicia_arms.py \
    --policy.path=outputs/train/act_bimanual_grab_cube/checkpoints/last/pretrained_model \
    --robot.type=bi_alicia_d_follower \
    --robot.left_arm_port=/dev/ttyACM1 \
    --robot.right_arm_port=/dev/ttyACM0 \
    --robot.cameras='{
        right_wrist: {type: opencv, index_or_path: /dev/video10, width: 640, height: 480, fps: 30},
        left_wrist: {type: opencv, index_or_path: /dev/video18, width: 640, height: 480, fps: 30},
        top: {type: opencv, index_or_path: /dev/video24, width: 640, height: 480, fps: 30},
        front: {type: opencv, index_or_path: /dev/video12, width: 640, height: 480, fps: 30}
    }' \
    --policy.device=cuda \
    --task="Grab the cloth with both arms" \
    --record_eval=false \
    --duration=120 \
    --fps=10
```

Example usage for Diffusion:
```shell
CUDA_VISIBLE_DEVICES=1 python examples/alicia/eval_alicia_arms.py \
    --policy.path=outputs/train/diffusion_bimanual_fold_cloth/checkpoints/last/pretrained_model \
    --robot.type=bi_alicia_d_follower \
    --robot.left_arm_port=/dev/ttyACM1 \
    --robot.right_arm_port=/dev/ttyACM0 \
    --robot.cameras='{
        right_wrist: {type: opencv, index_or_path: /dev/video10, width: 640, height: 480, fps: 30},
        left_wrist: {type: opencv, index_or_path: /dev/video18, width: 640, height: 480, fps: 30},
        top: {type: opencv, index_or_path: /dev/video24, width: 640, height: 480, fps: 30},
        front: {type: opencv, index_or_path: /dev/video12, width: 640, height: 480, fps: 30}
    }' \
    --policy.device=cuda \
    --task="Grab the cloth with both arms" \
    --record_eval=false \
    --duration=120 \
    --fps=10
```

Copyright (c) 2025 Synria Robotics Co., Ltd.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.processor import make_default_processors
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    bi_alicia_d_follower,
    make_robot_from_config,
)
from lerobot.scripts.lerobot_record import record_loop
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import init_keyboard_listener, is_headless
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    """Configuration for policy evaluation on bimanual robot."""

    # Policy configuration
    policy: PreTrainedConfig | None = None

    # Robot configuration
    robot: RobotConfig | None = None

    # Evaluation parameters
    task: str = field(default="", metadata={"help": "Task description for the policy"})
    duration: float = 120.0  # Duration to run evaluation per episode (seconds)
    fps: float = 10.0  # Action execution frequency (Hz)
    num_episodes: int = 5  # Number of evaluation episodes
    record_eval: bool = False  # Whether to record evaluation episodes to a dataset
    eval_dataset_repo_id: str = "temp/eval_not_saved"  # Dataset repo ID (required for features, even if not recording)
    display_data: bool = True  # Display observations and actions in rerun

    def __post_init__(self):
        # HACK: We parse again the cli args here to get the pretrained path if there was one.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            # Resolve the path to ensure it's recognized as a local directory
            # Use resolve() to follow symlinks (e.g., "latest" -> actual checkpoint number)
            policy_path_obj = Path(policy_path)
            
            # Check if path exists (before resolving symlinks)
            if not policy_path_obj.exists():
                # Try to provide helpful error message
                parent_dir = policy_path_obj.parent
                suggestions = []
                if parent_dir.exists():
                    # List available checkpoints
                    try:
                        available = [d.name for d in parent_dir.iterdir() if d.is_dir()]
                        if available:
                            suggestions.append(f"Available checkpoints: {', '.join(sorted(available)[:5])}")
                    except Exception:
                        pass
                
                error_msg = f"Policy path does not exist: {policy_path_obj.resolve()}"
                if suggestions:
                    error_msg += f"\n{suggestions[0]}"
                error_msg += "\nPlease provide a valid path to the pretrained model directory."
                raise ValueError(error_msg)
            
            # Resolve symlinks (e.g., "latest" -> "050000")
            policy_path_resolved = policy_path_obj.resolve()
            
            if not policy_path_resolved.is_dir():
                raise ValueError(
                    f"Policy path is not a directory: {policy_path_resolved}. "
                    "Please provide a path to a directory containing the pretrained model."
                )
            
            # Check if config.json exists in the directory
            config_file = policy_path_resolved / "config.json"
            if not config_file.exists():
                raise ValueError(
                    f"config.json not found in {policy_path_resolved}. "
                    "Please ensure this is a valid pretrained model directory."
                )
            
            policy_path = str(policy_path_resolved)
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path
        else:
            raise ValueError("Policy path is required. Use --policy.path=<path>")

        # Validate that robot configuration is provided
        if self.robot is None:
            raise ValueError("Robot configuration must be provided")

        # Note: eval_dataset_repo_id is always required (for features structure)
        # but episodes are only saved when record_eval=true

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]


@parser.wrap()
def eval_policy(cfg: EvalConfig):
    """Main entry point for policy evaluation."""
    init_logging()
    logger.info("Starting evaluation...")

    # Load dataset metadata from training config
    dataset = None
    policy_path = Path(cfg.policy.pretrained_path)
    train_config_path = policy_path / "train_config.json"
    
    if train_config_path.exists():
        with open(train_config_path, "r") as f:
            train_config = json.load(f)
            if "dataset" in train_config and "repo_id" in train_config["dataset"]:
                dataset = LeRobotDataset(
                    repo_id=train_config["dataset"]["repo_id"],
                    root=train_config["dataset"].get("root"),
                )
                logger.info(f"Loaded dataset metadata from {dataset.repo_id}")
    
    if dataset is None:
        raise ValueError(
            "Could not load dataset metadata. Please ensure the policy was trained with a dataset "
            "and train_config.json exists in the policy directory."
        )

    # Create robot
    logger.info(f"Initializing robot: {cfg.robot.type}")
    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    logger.info("Robot connected")

    # Load policy using factory function (supports all policy types: ACT, Diffusion, etc.)
    logger.info(f"Loading policy from {cfg.policy.pretrained_path}")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
        rename_map=cfg.rename_map if hasattr(cfg, 'rename_map') else None,
    )
    policy.eval()
    logger.info(f"Policy loaded and set to eval mode (type: {cfg.policy.type})")

    # Create processors
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    # Build Policy Processors (following reference examples)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        dataset_stats=dataset.meta.stats,
        # The inference device is automatically set to match the detected hardware
        preprocessor_overrides={"device_processor": {"device": str(cfg.policy.device)}},
    )
    logger.info("Processors loaded")

    # Always create a dataset for features (needed by record_loop)
    # Dataset is always created but episodes are only saved when record_eval=true
    action_features = hw_to_dataset_features(robot.action_features, ACTION)
    obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR)
    dataset_features = {**action_features, **obs_features}

    logger.info(f"Creating dataset for features: {cfg.eval_dataset_repo_id}")
    if cfg.record_eval:
        logger.info("Recording evaluation episodes enabled")
    else:
        logger.info("Recording evaluation episodes disabled (dataset created for features only)")
    
    # Always use videos=True if we have camera features (required by dataset structure)
    # But only start image writer if recording
    has_video_features = any("image" in key or "video" in key for key in dataset_features.keys())
    
    eval_dataset = LeRobotDataset.create(
        repo_id=cfg.eval_dataset_repo_id,
        fps=cfg.fps,
        features=dataset_features,
        robot_type=robot.name,
        use_videos=has_video_features,  # Required if we have camera features
        image_writer_threads=4 if cfg.record_eval else 0,  # Only write images if recording
    )
    logger.info("Dataset created")

    # Initialize keyboard listener and rerun visualization
    listener, events = init_keyboard_listener()
    if cfg.display_data:
        init_rerun(session_name="bi_alicia_d_evaluate")

    if not robot.is_connected:
        raise ValueError("Robot is not connected!")

    logger.info("Starting evaluation loop...")
    recorded_episodes = 0
    
    try:
        while recorded_episodes < cfg.num_episodes and not events["stop_recording"]:
            log_say(f"Running inference, evaluation episode {recorded_episodes + 1} of {cfg.num_episodes}")

            # Main evaluation loop using record_loop
            # Always pass dataset when using policy (needed for features structure)
            record_loop(
                robot=robot,
                events=events,
                fps=cfg.fps,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                dataset=eval_dataset,  # Always pass dataset (features needed even if not recording)
                control_time_s=cfg.duration,
                single_task=cfg.task,
                display_data=cfg.display_data,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
            )

            # Reset the environment if not stopping or re-recording
            if not events["stop_recording"] and (
                (recorded_episodes < cfg.num_episodes - 1) or events["rerecord_episode"]
            ):
                log_say("Reset the environment")
                record_loop(
                    robot=robot,
                    events=events,
                    fps=cfg.fps,
                    control_time_s=cfg.duration,
                    single_task=cfg.task,
                    display_data=cfg.display_data,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                )

            if events["rerecord_episode"]:
                log_say("Re-record episode")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                if cfg.record_eval:
                    eval_dataset.clear_episode_buffer()
                continue

            # Save episode if recording
            if cfg.record_eval:
                eval_dataset.save_episode()
            recorded_episodes += 1

    finally:
        # Clean up
        log_say("Stop evaluation")
        
        if robot.is_connected:
            robot.disconnect()
            logger.info("Robot disconnected")

        if not is_headless() and listener:
            listener.stop()

        if cfg.record_eval:
            eval_dataset.finalize()
            eval_dataset.push_to_hub()
            logger.info("Evaluation dataset saved and pushed to hub")

        logger.info("Evaluation finished")


def main():
    """Entry point for the evaluation script."""
    register_third_party_plugins()
    eval_policy()


if __name__ == "__main__":
    main()
