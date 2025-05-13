#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
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

from lerobot.common.optim.optimizers import AdamWConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode


@PreTrainedConfig.register_subclass("act")
@dataclass
class ACTConfig(PreTrainedConfig):
    """Action Chunking Transformers (ACT) 策略的配置类。

    默认配置适用于训练双臂 Aloha 任务，如 "insertion" 或 "transfer"。

    您最可能需要更改的参数是那些依赖于环境/传感器的参数，
    即 `input_shapes` 和 `output_shapes`。

    关于输入和输出的说明：
        - 以下两者至少满足其一：
            - 至少有一个以 "observation.image" 开头的键作为输入。
              和/或
            - "observation.environment_state" 键作为输入。
        - 如果有多个以 "observation.images." 开头的键，它们将被视为多个摄像头视图。
          目前仅支持所有图像具有相同形状。
        - 可能在没有 "observation.state" 键（用于本体感受机器人状态）的情况下工作。
        - "action" 是必需的输出键。

    Args:
        n_obs_steps (int): 传递给策略的观测数据包含多少个环境时间步（包括当前步和之前的步）。
        chunk_size (int): 动作预测“块”的大小（以环境步数为单位）。
        n_action_steps (int): 单次调用策略时在环境中执行的动作步数。
            该值不应大于块大小。例如，如果块大小为 100，您可以将其设置为 50。
            这意味着模型预测 100 步的动作，在环境中执行 50 步，并丢弃另外 50 步。
        input_shapes (dict): 定义策略输入数据形状的字典。键表示输入数据名称，
            值是表示相应数据维度的列表。例如，"observation.image" 指的是来自
            摄像头的输入，维度为 [3, 96, 96]，表示它有三个颜色通道和 96x96 的分辨率。
            重要的是，`input_shapes` 不包括批次维度或时间维度。
        output_shapes (dict): 定义策略输出数据形状的字典。键表示输出数据名称，
            值是表示相应数据维度的列表。例如，"action" 指的是 [14] 的输出形状，
            表示 14 维动作。重要的是，`output_shapes` 不包括批次维度或时间维度。
        normalization_mapping (dict): 指定不同模态（"VISUAL", "STATE", "ACTION"）的输入/输出数据的归一化方法。
            键代表模态，值指定要应用的归一化模式。
            两种可用模式是 "mean_std"（减去均值并除以标准差）和 "min_max"（重新缩放到 [-1, 1] 范围）。
            训练时用于归一化输入和目标动作，推理时用于归一化输入和反归一化输出动作。
        vision_backbone (str): 用于编码图像的 torchvision resnet 骨干网络名称。
        pretrained_backbone_weights (str | None): 用于初始化骨干网络的 torchvision 预训练权重。
            `None` 表示不使用预训练权重。
        replace_final_stride_with_dilation (bool): 是否将 ResNet 最后一个 stage 的 2x2 步幅卷积替换为膨胀卷积。
            这可以增加输出特征图的分辨率。
        pre_norm (bool): 是否在 Transformer 块中使用 Pre-Normalization（即在自注意力和前馈网络之前应用 Layer Normalization）。
        dim_model (int): Transformer 块的主要隐藏层维度。
        n_heads (int): Transformer 块中多头注意力机制的头数。
        dim_feedforward (int): Transformer 块中前馈网络的中间层维度。
        feedforward_activation (str): Transformer 块前馈网络中使用的激活函数（"relu", "gelu", "glu"）。
        n_encoder_layers (int): 主要 Transformer 编码器的层数。
        n_decoder_layers (int): 主要 Transformer 解码器的层数。
            注意：虽然原始 ACT 实现为 7，但代码中存在一个 bug，导致只有第一层被使用。
            这里我们通过将其设置为 1 来匹配原始实现。
            请参阅此问题 https://github.com/tonyzhaozh/act/issues/25#issue-2258740521。
        use_vae (bool): 是否在训练中使用变分自编码器（VAE）目标。
            这会引入另一个 Transformer 作为 VAE 的编码器（不要与 Transformer 编码器混淆 - 请参阅策略类中的文档）。
            如果为 `True`，会额外训练一个 VAE 编码器，并将 KL 散度损失加入总损失中。
        latent_dim (int): VAE 的潜空间维度（如果 `use_vae` 为 `True`）。
        n_vae_encoder_layers (int): VAE 编码器（这是一个独立的 Transformer 编码器）的层数（如果 `use_vae` 为 `True`）。
        temporal_ensemble_coeff (float | None): 用于时间集成（Temporal Ensembling）的指数加权系数。
            默认为 None，表示不使用时间集成。使用此功能时，`n_action_steps` 必须为 1，
            因为需要在每个步骤进行推理以形成集成。有关集成工作原理的更多信息，请参阅 `ACTTemporalEnsembler`。
            该系数控制历史预测动作在集成时的权重衰减速度。
        dropout (float): Transformer 层中使用的 Dropout 比率，用于正则化，防止过拟合。
        kl_weight (float): 如果启用了变分目标（`use_vae` 为 `True`），则用于 KL 散度损失项的权重。
            损失计算方式为：`reconstruction_loss + kl_weight * kld_loss`。
        optimizer_lr (float): 优化器的基础学习率（应用于除骨干网络外的模型参数）。
        optimizer_weight_decay (float): 优化器的权重衰减（L2 正则化）系数。
        optimizer_lr_backbone (float): 专门为视觉骨干网络（如 ResNet）设置的学习率。
    """

    # 输入/输出结构 (Input / output structure)
    n_obs_steps: int = 1  # 输入给策略的观测数据包含多少个历史时间步（包括当前步）。注意：当前实现验证要求此值为 1。
    chunk_size: int = 100  # 策略在一次前向传播中预测的动作序列（"块"）的长度（以环境步数为单位）。
    n_action_steps: int = 100  # 模型预测一个动作块后，在环境中实际执行的动作步数。该值必须小于或等于 `chunk_size`。

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,  # 视觉输入的归一化方法
            "STATE": NormalizationMode.MEAN_STD,  # 状态输入的归一化方法
            "ACTION": NormalizationMode.MEAN_STD,  # 动作输出/目标的归一化方法
        }
    ) # 指定不同模态（"VISUAL", "STATE", "ACTION"）的输入/输出数据的归一化方法。

    # 架构 (Architecture)
    # 视觉骨干网络 (Vision backbone)
    vision_backbone: str = "resnet18"  # 用于提取图像特征的 ResNet 骨干网络名称。
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1" # 视觉骨干网络的预训练权重名称或 None。
    replace_final_stride_with_dilation: bool = False # 是否将 ResNet 最后一个 stage 的 2x2 步幅卷积替换为膨胀卷积。

    # Transformer 层 (Transformer layers)
    pre_norm: bool = False  # 是否在 Transformer 块中使用 Pre-Normalization。
    dim_model: int = 512  # Transformer 模型的主要隐藏层维度。
    n_heads: int = 8  # Transformer 中多头注意力机制的头数。
    dim_feedforward: int = 3200  # Transformer 块中前馈网络的中间层维度。
    feedforward_activation: str = "relu"  # Transformer 前馈网络中使用的激活函数。
    n_encoder_layers: int = 4  # 主要 Transformer 编码器的层数。
    # 注意：虽然原始 ACT 实现为 7，但代码中存在一个 bug，导致只有第一层被使用。
    # 这里我们通过将其设置为 1 来匹配原始实现。
    # 请参阅此问题 https://github.com/tonyzhaozh/act/issues/25#issue-2258740521。
    n_decoder_layers: int = 1  # 主要 Transformer 解码器的层数。

    # VAE (Variational Autoencoder)
    use_vae: bool = True  # 是否在训练中使用变分自编码器（VAE）目标。
    latent_dim: int = 32  # VAE 潜空间的维度（如果 `use_vae` 为 `True`）。
    n_vae_encoder_layers: int = 4  # VAE 编码器的层数（如果 `use_vae` 为 `True`）。

    # 推理 (Inference)
    # 注意：在启用时间集成时，ACT 中使用的值为 0.01。
    temporal_ensemble_coeff: float | None = None # 时间集成的指数加权系数。None 表示禁用。使用时 `n_action_steps` 必须为 1。

    # 训练和损失计算 (Training and loss computation)
    dropout: float = 0.1  # Transformer 层中使用的 Dropout 比率。
    kl_weight: float = 10.0  # KL 散度损失项在总损失中的权重（如果 `use_vae` 为 `True`）。

    # 训练预设 (Training preset)
    optimizer_lr: float = 1e-5  # 优化器的基础学习率（非骨干网络部分）。
    optimizer_weight_decay: float = 1e-4  # 优化器的权重衰减系数。
    optimizer_lr_backbone: float = 1e-5  # 视觉骨干网络的学习率。

    def __post_init__(self):
        """执行输入验证（非详尽）。"""
        super().__post_init__()

        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` 必须是 ResNet 变体之一。得到 {self.vision_backbone}。"
            )
        if self.temporal_ensemble_coeff is not None and self.n_action_steps > 1:
            raise NotImplementedError(
                "使用时间集成时，`n_action_steps` 必须为 1。这是因为策略需要在每个步骤被查询以计算集成动作。"
            )
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"块大小是每次模型调用执行动作步数的上限。`n_action_steps` 得到 {self.n_action_steps}，`chunk_size` 得到 {self.chunk_size}。"
            )
        if self.n_obs_steps != 1:
            # TODO(rcadene): 尚未处理多观测步。
            raise ValueError(
                f"尚未处理多观测步。得到 `nobs_steps={self.n_obs_steps}`"
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        """获取预设的 AdamW 优化器配置。"""
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> None:
        """获取预设的学习率调度器配置（当前为 None）。"""
        return None

    def validate_features(self) -> None:
        """验证输入特征是否满足要求。"""
        if not self.image_features and not self.env_state_feature:
            raise ValueError("必须在输入中提供至少一个图像或环境状态。")

    @property
    def observation_delta_indices(self) -> None:
        """获取观测增量索引（当前未使用，返回 None）。"""
        return None

    @property
    def action_delta_indices(self) -> list:
        """获取动作增量索引（当前为块中的所有索引）。"""
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        """获取奖励增量索引（当前未使用，返回 None）。"""
        return None
