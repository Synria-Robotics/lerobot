# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import datetime as dt
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Type

import draccus
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

from lerobot.common import envs
from lerobot.common.optim import OptimizerConfig
from lerobot.common.optim.schedulers import LRSchedulerConfig
from lerobot.common.utils.hub import HubMixin
from lerobot.configs import parser
from lerobot.configs.default import DatasetConfig, EvalConfig, WandBConfig
from lerobot.configs.policies import PreTrainedConfig

TRAIN_CONFIG_NAME = "train_config.json"  # 训练配置文件名


@dataclass
class TrainPipelineConfig(HubMixin):
    dataset: DatasetConfig
    env: envs.EnvConfig | None = None
    policy: PreTrainedConfig | None = None
    # 将 `dir` 设置为您希望保存所有运行输出的位置。如果您使用相同的 `dir` 值运行另一个训练会话，
    # 除非您将 `resume` 设置为 true，否则其内容将被覆盖。
    output_dir: Path | None = None
    job_name: str | None = None
    # 将 `resume` 设置为 true 以恢复之前的运行。为了使其正常工作，您需要确保
    # `dir` 是现有运行的目录，其中至少包含一个检查点。
    # 请注意，恢复运行时，默认行为是使用检查点中的配置，
    # 而不管恢复时训练命令提供了什么。
    resume: bool = False
    # `seed` 用于训练（例如：模型初始化、数据集打乱）
    # 以及评估环境。
    seed: int | None = 1000
    # 数据加载器的工作线程数。
    num_workers: int = 8
    batch_size: int = 32
    steps: int = 400_000
    eval_freq: int = 2000
    log_freq: int = 100
    save_checkpoint: bool = True
    # 检查点每 `save_freq` 次训练迭代后以及最后一次训练步骤后保存。
    save_freq: int = 5000
    use_policy_training_preset: bool = True
    optimizer: OptimizerConfig | None = None
    scheduler: LRSchedulerConfig | None = None
    eval: EvalConfig = field(default_factory=EvalConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)

    def __post_init__(self):
        self.checkpoint_path = None

    def validate(self):
        # HACK: 我们在这里再次解析 cli 参数以获取预训练路径（如果存在）。
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            # 仅加载策略配置
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path
        elif self.resume:
            # 整个训练配置已加载，我们只需要获取检查点目录
            config_path = parser.parse_arg("config_path")
            if not config_path:
                raise ValueError(
                    f"恢复运行时需要 config_path。请指定 {TRAIN_CONFIG_NAME} 的路径"
                )
            if not Path(config_path).resolve().exists():
                raise NotADirectoryError(
                    f"{config_path=} 预期为本地路径。"
                    "目前不支持从 hub 恢复。"
                )
            policy_path = Path(config_path).parent
            self.policy.pretrained_path = policy_path
            self.checkpoint_path = policy_path.parent

        if not self.job_name:
            if self.env is None:
                self.job_name = f"{self.policy.type}"
            else:
                self.job_name = f"{self.env.type}_{self.policy.type}"

        if not self.resume and isinstance(self.output_dir, Path) and self.output_dir.is_dir():
            raise FileExistsError(
                f"输出目录 {self.output_dir} 已存在且 resume 为 {self.resume}。"
                f"请更改您的输出目录，以免 {self.output_dir} 被覆盖。"
            )
        elif not self.output_dir:
            now = dt.datetime.now()
            train_dir = f"{now:%Y-%m-%d}/{now:%H-%M-%S}_{self.job_name}"
            self.output_dir = Path("outputs/train") / train_dir

        if isinstance(self.dataset.repo_id, list):
            raise NotImplementedError("LeRobotMultiDataset 当前未实现。")

        if not self.use_policy_training_preset and (self.optimizer is None or self.scheduler is None):
            raise ValueError("未使用策略预设时必须设置优化器和调度器。")
        elif self.use_policy_training_preset and not self.resume:
            self.optimizer = self.policy.get_optimizer_preset()
            self.scheduler = self.policy.get_scheduler_preset()

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """这使解析器能够使用 `--policy.path=local/dir` 从策略加载配置"""
        return ["policy"]

    def to_dict(self) -> dict:
        return draccus.encode(self)

    def _save_pretrained(self, save_directory: Path) -> None:
        with open(save_directory / TRAIN_CONFIG_NAME, "w") as f, draccus.config_type("json"):
            draccus.dump(self, f, indent=4)

    @classmethod
    def from_pretrained(
        cls: Type["TrainPipelineConfig"],
        pretrained_name_or_path: str | Path,
        *,
        force_download: bool = False,
        resume_download: bool = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        **kwargs,
    ) -> "TrainPipelineConfig":
        model_id = str(pretrained_name_or_path)
        config_file: str | None = None
        if Path(model_id).is_dir():
            if TRAIN_CONFIG_NAME in os.listdir(model_id):
                config_file = os.path.join(model_id, TRAIN_CONFIG_NAME)
            else:
                print(f"在 {Path(model_id).resolve()} 中未找到 {TRAIN_CONFIG_NAME}")
        elif Path(model_id).is_file():
            config_file = model_id
        else:
            try:
                config_file = hf_hub_download(
                    repo_id=model_id,
                    filename=TRAIN_CONFIG_NAME,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
            except HfHubHTTPError as e:
                raise FileNotFoundError(
                    f"在 HuggingFace Hub 的 {model_id} 中未找到 {TRAIN_CONFIG_NAME}"
                ) from e

        cli_args = kwargs.pop("cli_args", [])
        cfg = draccus.parse(cls, config_file, args=cli_args)

        return cfg
