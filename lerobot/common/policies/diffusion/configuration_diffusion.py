#!/usr/bin/env python

# Copyright 2024 Columbia Artificial Intelligence, Robotics Lab,
# and The HuggingFace Inc. team. All rights reserved.
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
from dataclasses import dataclass, field

from lerobot.common.optim.optimizers import AdamConfig
from lerobot.common.optim.schedulers import DiffuserSchedulerConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode


@PreTrainedConfig.register_subclass("diffusion")
@dataclass
class DiffusionConfig(PreTrainedConfig):
    """DiffusionPolicy 的配置类。

    默认配置适用于使用 PushT 进行训练，提供本体感受和单摄像头观测。

    您最可能需要更改的参数是那些依赖于环境/传感器的参数。
    这些参数是：`input_shapes` 和 `output_shapes`。

    关于输入和输出的说明：
        - "observation.state" 是必需的输入键。
        - 以下两者之一：
            - 至少需要一个以 "observation.image" 开头的键作为输入。
              和/或
            - 需要 "observation.environment_state" 键作为输入。
        - 如果有多个以 "observation.image" 开头的键，它们将被视为多个摄像头视图。
          目前我们仅支持所有图像具有相同的形状。
        - "action" 是必需的输出键。

    Args:
        n_obs_steps: 传递给策略的观测的环境步数（取当前步和之前的额外步数）。
        horizon: Diffusion 模型动作预测大小，详见 `DiffusionPolicy.select_action`。
        n_action_steps: 在一次策略调用中在环境中运行的动作步数。
            更多细节请参见 `DiffusionPolicy.select_action`。0
        normalization_mapping: 一个字典，键表示模态（"VISUAL", "STATE", "ACTION"），
            值指定要应用的归一化模式 (`NormalizationMode.MEAN_STD` 或 `NormalizationMode.MIN_MAX`)。
            训练时用于归一化输入和目标动作，推理时用于归一化输入和反归一化输出动作。
        drop_n_last_frames: 在为训练采样数据帧时，丢弃每个 episode 末尾的 N 帧。
            这可以避免在训练 Diffusion Policy 时使用过多的填充帧，从而提高性能。
            默认值是根据 `horizon`, `n_action_steps`, `n_obs_steps` 计算得出的。
        vision_backbone: 用于编码图像的 torchvision resnet 主干网络的名称 (例如 "resnet18")。
        crop_shape: (H, W) 形状，用于在视觉主干网络的预处理步骤中裁剪图像。必须适合图像大小。
            如果为 None，则不进行裁剪。
        crop_is_random: 在训练时裁剪是否应随机（在评估模式下始终是中心裁剪）。
        pretrained_backbone_weights: 用于初始化主干网络的 torchvision 预训练权重名称
            (例如 "ResNet18_Weights.IMAGENET1K_V1") 或 `None` (不加载预训练权重)。
        use_group_norm: 是否在视觉主干网络中使用组归一化替换批归一化。
            如果使用预训练权重，则不能设置为 `True`。
        spatial_softmax_num_keypoints: 在视觉特征提取后，Spatial Softmax 层输出的关键点数量。
        use_separate_rgb_encoder_per_camera: 对于多摄像头输入，是否为每个摄像头使用独立的视觉编码器。
        down_dims: Diffusion U-Net 编码器（下采样路径）中每个阶段的特征维度。
            元组的长度决定了下采样的次数。
        kernel_size: Diffusion U-Net 中 1D 卷积层的卷积核大小。
        n_groups: Diffusion U-Net 中 Group Normalization 使用的组数。
        diffusion_step_embed_dim: 用于编码 Diffusion 时间步的嵌入向量维度。
        use_film_scale_modulation: 在 U-Net 的 FiLM (Feature-wise Linear Modulation) 条件化中，
            除了偏置调制外，是否也使用尺度调制。
        noise_scheduler_type: 要使用的噪声调度器的名称。支持的选项：["DDPM", "DDIM"]。
        num_train_timesteps: 前向扩散过程（加噪过程）的总步数。
        beta_schedule: 扩散过程中 beta 值的调度策略名称 (例如 "linear", "squaredcos_cap_v2")。
        beta_start: 扩散过程中 beta 值的起始值。
        beta_end: 扩散过程中 beta 值的结束值。
        prediction_type: Diffusion U-Net 预测的目标类型 ("epsilon" - 预测噪声，或 "sample" - 预测去噪后的样本)。
        clip_sample: 在推理（反向扩散）的每个步骤中，是否将预测的样本裁剪到
            `[-clip_sample_range, +clip_sample_range]` 范围内。
        clip_sample_range: 如果 `clip_sample` 为 `True`，则指定裁剪范围的大小。
        num_inference_steps: 推理时使用的反向扩散（去噪）步数。如果为 `None`，则默认为 `num_train_timesteps`。
            通常可以设置比训练步数少的值以加速推理。
        do_mask_loss_for_padding: 在计算损失时，是否屏蔽掉由于数据集边界效应而产生的填充动作
            （标记为 `action_is_pad=True` 的部分）。
        optimizer_lr: 优化器的基础学习率 (Adam)。用于策略训练预设。
        optimizer_betas: Adam 优化器的 beta 参数 (通常是 `(beta1, beta2)`)。用于策略训练预设。
        optimizer_eps: Adam 优化器的 epsilon 参数，用于数值稳定性。用于策略训练预设。
        optimizer_weight_decay: 优化器的权重衰减（L2 正则化）系数。用于策略训练预设。
        scheduler_name: 学习率调度器的名称 (例如 "cosine")。用于策略训练预设。
        scheduler_warmup_steps: 学习率调度器的预热步数。用于策略训练预设。
    """

    # 输入/输出结构。
    n_obs_steps: int = 2
    horizon: int = 16
    n_action_steps: int = 8

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # 原始实现不为最后 7 个步骤采样帧，
    # 这避免了过多的填充并导致改进的训练结果。
    drop_n_last_frames: int = 7  # horizon - n_action_steps - n_obs_steps + 1

    # 架构/建模。
    # 视觉主干网络。
    vision_backbone: str = "resnet18"
    crop_shape: tuple[int, int] | None = (216, 288)
    crop_is_random: bool = True
    pretrained_backbone_weights: str | None = None
    use_group_norm: bool = True
    spatial_softmax_num_keypoints: int = 32
    use_separate_rgb_encoder_per_camera: bool = False
    # Unet。
    down_dims: tuple[int, ...] = (256, 512, 1024)
    kernel_size: int = 5
    n_groups: int = 8
    diffusion_step_embed_dim: int = 128
    use_film_scale_modulation: bool = True
    # 噪声调度器。
    noise_scheduler_type: str = "DDIM"
    num_train_timesteps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    prediction_type: str = "epsilon"
    clip_sample: bool = True
    clip_sample_range: float = 1.0

    # 推理
    num_inference_steps: int | None = 8

    # 损失计算
    do_mask_loss_for_padding: bool = False

    # 训练预设
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500

    def __post_init__(self):
        super().__post_init__()

        """输入验证（非详尽）。"""
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` 必须是 ResNet 变体之一。得到 {self.vision_backbone}。"
            )

        supported_prediction_types = ["epsilon", "sample"]
        if self.prediction_type not in supported_prediction_types:
            raise ValueError(
                f"`prediction_type` 必须是 {supported_prediction_types} 之一。得到 {self.prediction_type}。"
            )
        supported_noise_schedulers = ["DDPM", "DDIM"]
        if self.noise_scheduler_type not in supported_noise_schedulers:
            raise ValueError(
                f"`noise_scheduler_type` 必须是 {supported_noise_schedulers} 之一。"
                f"得到 {self.noise_scheduler_type}。"
            )

        # 检查 horizon 大小和 U-Net 下采样是否兼容。
        # U-Net 在每个阶段下采样 2 倍。
        downsampling_factor = 2 ** len(self.down_dims)
        if self.horizon % downsampling_factor != 0:
            raise ValueError(
                "horizon 应为下采样因子（由 `len(down_dims)` 决定）的整数倍。"
                f"得到 {self.horizon=} 和 {self.down_dims=}"
            )

    def get_optimizer_preset(self) -> AdamConfig:
        return AdamConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        return DiffuserSchedulerConfig(
            name=self.scheduler_name,
            num_warmup_steps=self.scheduler_warmup_steps,
        )

    def validate_features(self) -> None:
        if len(self.image_features) == 0 and self.env_state_feature is None:
            raise ValueError("您必须在输入中提供至少一个图像或环境状态。")

        if self.crop_shape is not None:
            for key, image_ft in self.image_features.items():
                if self.crop_shape[0] > image_ft.shape[1] or self.crop_shape[1] > image_ft.shape[2]:
                    raise ValueError(
                        f"`crop_shape` 应适合图像形状。`crop_shape` 得到 {self.crop_shape}，"
                        f"`{key}` 得到 {image_ft.shape}。"
                    )

        # 检查所有输入图像是否具有相同的形状。
        first_image_key, first_image_ft = next(iter(self.image_features.items()))
        for key, image_ft in self.image_features.items():
            if image_ft.shape != first_image_ft.shape:
                raise ValueError(
                    f"`{key}` 与 `{first_image_key}` 不匹配，但我们期望所有图像形状都匹配。"
                )

    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None

