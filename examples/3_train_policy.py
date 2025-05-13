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

"""此脚本演示了如何在 PushT 环境中训练扩散策略。

使用此脚本训练模型后，您可以尝试在
examples/2_evaluate_pretrained_policy.py 中对其进行评估。
"""

from pathlib import Path

import torch

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.configs.types import FeatureType


def main():
    # 创建一个目录来存储训练检查点。
    output_directory = Path("outputs/train/example_pusht_diffusion")
    output_directory.mkdir(parents=True, exist_ok=True)

    # # 选择您的设备
    device = torch.device("cuda")

    # 离线训练步数（在此示例中，我们仅进行离线训练。）
    # 根据您的偏好进行调整。需要 5000 步才能获得值得评估的结果。
    training_steps = 5000
    log_freq = 1

    # 从头开始（即不从预训练策略开始）时，我们需要在创建策略之前指定 2 件事：
    #   - 输入/输出形状：以正确调整策略的大小
    #   - 数据集统计信息：用于输入/输出的归一化和反归一化
    dataset_metadata = LeRobotDatasetMetadata("lerobot/pusht")
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    # 策略使用配置类进行初始化，在本例中为 `DiffusionConfig`。对于此示例，
    # 我们将仅使用默认值，因此除了输入/输出特征之外，无需传递任何参数。
    cfg = DiffusionConfig(input_features=input_features, output_features=output_features)

    # 我们现在可以使用此配置和数据集统计信息来实例化我们的策略。
    policy = DiffusionPolicy(cfg, dataset_stats=dataset_metadata.stats)
    policy.train()
    policy.to(device)

    # 另一个策略-数据集交互是与 delta_timestamps。每个策略都期望给定数量的帧，
    # 这对于输入、输出和奖励（如果有）可能不同。
    delta_timestamps = {
        "observation.image": [i / dataset_metadata.fps for i in cfg.observation_delta_indices],
        "observation.state": [i / dataset_metadata.fps for i in cfg.observation_delta_indices],
        "action": [i / dataset_metadata.fps for i in cfg.action_delta_indices],
    }

    # 在这种情况下，使用 Diffusion Policy 的标准配置，它等效于此：
    delta_timestamps = {
        # 加载当前帧之前 -0.1 秒的先前图像和状态，
        # 然后加载对应于 0.0 秒的当前图像和状态。
        "observation.image": [-0.1, 0.0],
        "observation.state": [-0.1, 0.0],
        # 加载先前的动作 (-0.1)，要执行的下一个动作 (0.0)，
        # 以及 14 个未来动作，间隔为 0.1 秒。所有这些动作都将
        # 用于监督策略。
        "action": [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
    }

    # 然后我们可以使用这些 delta_timestamps 配置来实例化数据集。
    dataset = LeRobotDataset("lerobot/pusht", delta_timestamps=delta_timestamps)

    # 然后我们为离线训练创建优化器和数据加载器。
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=4,
        batch_size=64,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

    # 运行训练循环。
    step = 0
    done = False
    while not done:
        for batch in dataloader:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            loss, _ = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % log_freq == 0:
                print(f"步数：{step} 损失：{loss.item():.3f}")
            step += 1
            if step >= training_steps:
                done = True
                break

    # 保存策略检查点。
    policy.save_pretrained(output_directory)


if __name__ == "__main__":
    main()
