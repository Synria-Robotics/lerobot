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

"""
此脚本演示了如何使用 `LeRobotDataset` 类处理来自 Hugging Face 的机器人数据集。
它说明了如何加载数据集、操作数据集以及应用适用于 PyTorch 中机器学习任务的转换。

此脚本包含的功能：
- 查看数据集的元数据并探索其属性。
- 从 hub 加载现有数据集或其子集。
- 按 episode 编号访问帧。
- 使用高级数据集功能，例如基于时间戳的帧选择。
- 演示与 PyTorch DataLoader 的兼容性以进行批处理。

该脚本以如何使用 PyTorch 的 DataLoader 进行批处理数据的示例结束。
"""

from pprint import pprint

import torch
from huggingface_hub import HfApi

import lerobot
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

# 我们自己移植了许多现有的数据集，使用此命令查看列表：
# print("可用数据集列表：")
# pprint(lerobot.available_datasets)

# 您还可以使用 hub api 浏览社区在 hub 上创建/移植的数据集：
# hub_api = HfApi()
# repo_ids = [info.id for info in hub_api.list_datasets(task_categories="robotics", tags=["LeRobot"])]
# pprint(repo_ids)

# 或者直接在您的网络浏览器中浏览它们：
# https://huggingface.co/datasets?other=LeRobot

# 让我们以这个为例
repo_id = "alicia_demo"
root = "D:\\Github\\Synria-Robotics\\lerobot\\datasets\\alicia_demo"

# 我们可以查看并获取其元数据以了解更多信息：
ds_meta = LeRobotDatasetMetadata(repo_id, root)


# 通过仅实例化此类，您可以快速访问有关数据集内容和结构的有用信息，
# 而无需下载实际数据（仅元数据文件——这些文件是轻量级的）。
print(f"总 episode 数：{ds_meta.total_episodes}")
print(f"每个 episode 的平均帧数：{ds_meta.total_frames / ds_meta.total_episodes:.3f}")
print(f"数据收集期间使用的每秒帧数：{ds_meta.fps}")
print(f"机器人类型：{ds_meta.robot_type}")
print(f"用于访问相机图像的键：{ds_meta.camera_keys}\n")

print("任务：")
print(ds_meta.tasks)
print("特征：")
pprint(ds_meta.features)

# 您还可以通过简单地打印对象来获得简短摘要：
print("打印全部信息")
print(ds_meta)

# 然后您可以从 hub 加载实际的数据集。
# 加载任何 episode 子集：
dataset = LeRobotDataset(repo_id, root, episodes=[0])

# 查看您有多少帧：
print(f"选定的 episodes：{dataset.episodes}")
print(f"选定的 episode 数量：{dataset.num_episodes}")
print(f"选定的帧数：{dataset.num_frames}")

# 或者简单地加载整个数据集：
dataset = LeRobotDataset(repo_id, root)
print(f"选定的 episode 数量：{dataset.num_episodes}")
print(f"选定的帧数：{dataset.num_frames}")

# 先前的元数据类包含在数据集的 'meta' 属性中：
print("先前的元数据类包含在数据集的 'meta' 属性中：")
print(dataset.meta)

# LeRobotDataset 实际上包装了一个底层的 Hugging Face 数据集
# （有关更多信息，请参阅 https://huggingface.co/docs/datasets）。
print(dataset.hf_dataset)

# LeRobot 数据集还继承了 PyTorch 数据集，因此您可以执行您熟悉和喜爱的所有操作，
# 例如迭代数据集。
# __getitem__ 迭代数据集的帧。由于我们的数据集也按 episode 结构化，
# 您可以使用 episode_data_index 访问任何 episode 的帧索引。在这里，我们访问与第一个 episode 关联的帧索引：
episode_index = 0
from_idx = dataset.episode_data_index["from"][episode_index].item()
to_idx = dataset.episode_data_index["to"][episode_index].item()

# 然后我们从第一个相机获取所有图像帧：
camera_key = dataset.meta.camera_keys[0]
frames = [dataset[idx][camera_key] for idx in range(from_idx, to_idx)]

# 数据集返回的对象都是 torch.Tensors
print(type(frames[0]))
print(frames[0].shape)

# 由于我们使用的是 pytorch，因此形状采用 pytorch 的 channel-first 约定 (c, h, w)。
# 我们可以将此形状与该特征的可用信息进行比较
pprint(dataset.features[camera_key])
# 特别是：
print(dataset.features[camera_key]["shape"])
# 形状为 (h, w, c)，这是一种更通用的格式。

# 对于许多机器学习应用程序，我们需要加载过去观察的历史记录或未来动作的轨迹。
# 我们的数据集可以使用与当前加载帧的时间戳差异来加载每个键/模态的先前和未来帧。例如：
delta_timestamps = {
    # 加载 4 张图像：当前帧之前 1 秒、之前 500 毫秒、之前 200 毫秒和当前帧
    camera_key: [-1, -0.5, -0.20, 0],
    # 加载 6 个状态向量：当前帧之前 1.5 秒、之前 1 秒、... 200 毫秒、100 毫秒和当前帧
    "observation.state": [-1.5, -1, -0.5, -0.20, -0.10, 0],
    # 加载 64 个动作向量：当前帧、未来 1 帧、未来 2 帧、... 未来 63 帧
    "action": [t / dataset.fps for t in range(64)],
}
# 请注意，在任何情况下，这些 delta_timestamps 值都需要是 (1/fps) 的倍数，以便添加到任何时间戳后，
# 您仍然可以获得有效的时间戳。

dataset = LeRobotDataset(repo_id, root, delta_timestamps=delta_timestamps)
print("\n图像形状:", dataset[0][camera_key].shape)  # (4, c, h, w)
print("状态形状:", dataset[0]['observation.state'].shape)  # (6, c)
print("动作形状:", dataset[0]['action'].shape, "\n")  # (64, c)

# 最后，我们的数据集与 PyTorch dataloaders 和 samplers 完全兼容，因为它们只是 PyTorch 数据集。
dataloader = torch.utils.data.DataLoader(
    dataset,
    num_workers=0,
    batch_size=32,
    shuffle=True,
)

for batch in dataloader:
    print("batch 图像形状:", batch[camera_key].shape)  # (32, 4, c, h, w)
    print("batch 状态形状:", batch['observation.state'].shape)  # (32, 6, c)
    print("batch 动作形状:", batch['action'].shape)  # (32, 64, c)
    break
