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
此脚本演示了如何评估来自 HuggingFace Hub 或本地训练输出目录的预训练策略。
对于后者，您可能需要先运行 examples/3_train_policy.py。

它需要安装 'gym_pusht' 模拟环境。通过运行以下命令进行安装：
```bash
pip install -e ".[pusht]"
```
"""

from pathlib import Path

import gym_pusht  # noqa: F401
import gymnasium as gym
import imageio
import numpy
import torch

from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

# 创建一个目录来存储评估视频
output_directory = Path("outputs/eval/example_pusht_diffusion")
output_directory.mkdir(parents=True, exist_ok=True)

# 选择您的设备
device = "cuda"

# 提供 [hugging face 仓库 id](https://huggingface.co/lerobot/diffusion_pusht)：
pretrained_policy_path = "lerobot/diffusion_pusht"
# 或者本地 outputs/train 文件夹的路径。
# pretrained_policy_path = Path("outputs/train/example_pusht_diffusion")

policy = DiffusionPolicy.from_pretrained(pretrained_policy_path)

# 初始化评估环境以渲染两种观察类型：
# 场景图像和智能体的状态/位置。环境
# 也会在 300 次交互/步骤后自动停止运行。
env = gym.make(
    "gym_pusht/PushT-v0",
    obs_type="pixels_agent_pos",
    max_episode_steps=300,
)

# 我们可以验证策略期望的特征形状与环境
# 产生的观察结果的形状是否匹配
print(policy.config.input_features)
print(env.observation_space)

# 同样，我们可以检查策略产生的动作是否与环境
# 期望的动作匹配
print(policy.config.output_features)
print(env.action_space)

# 重置策略和环境以准备 rollout
policy.reset()
numpy_observation, info = env.reset(seed=42)

# 准备收集每个奖励和 episode 的所有帧，
# 从初始状态到最终状态。
rewards = []
frames = []

# 渲染初始状态的帧
frames.append(env.render())

step = 0
done = False
while not done:
    # 为在 Pytorch 中运行的策略准备观察结果
    state = torch.from_numpy(numpy_observation["agent_pos"])
    image = torch.from_numpy(numpy_observation["pixels"])

    # 转换为 float32，图像从通道优先 [0,255]
    # 转换为通道最后 [0,1]
    state = state.to(torch.float32)
    image = image.to(torch.float32) / 255
    image = image.permute(2, 0, 1)

    # 将数据张量从 CPU 发送到 GPU
    state = state.to(device, non_blocking=True)
    image = image.to(device, non_blocking=True)

    # 添加额外的（空）批次维度，这是前向传递策略所必需的
    state = state.unsqueeze(0)
    image = image.unsqueeze(0)

    # 创建策略输入字典
    observation = {
        "observation.state": state,
        "observation.image": image,
    }

    # 根据当前观察预测下一个动作
    with torch.inference_mode():
        action = policy.select_action(observation)

    # 为环境准备动作
    numpy_action = action.squeeze(0).to("cpu").numpy()

    # 在环境中执行一步并接收新的观察结果
    numpy_observation, reward, terminated, truncated, info = env.step(numpy_action)
    print(f"步骤={step} 奖励={reward} 终止={terminated}")

    # 跟踪所有奖励和帧
    rewards.append(reward)
    frames.append(env.render())

    # 当达到成功状态（即 terminated 为 True）时，
    # 或达到最大迭代次数（即 truncated 为 True）时，rollout 被视为完成
    done = terminated | truncated | done
    step += 1

if terminated:
    print("成功！")
else:
    print("失败！")

# 获取环境的速度（即其每秒帧数）。
fps = env.metadata["render_fps"]

# 将所有帧编码为 mp4 视频。
video_path = output_directory / "rollout.mp4"
imageio.mimsave(str(video_path), numpy.stack(frames), fps=fps)

print(f"评估视频保存在 '{video_path}'。")
