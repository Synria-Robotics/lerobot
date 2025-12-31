# Alicia-D LeRobot
[English Version](README.md) | [中文版](README_CN.md) | [官方淘宝店](https://g84gtpygdv6trpvdhcsy0kfr73avcip.taobao.com/shop/view_shop.htm?appUid=RAzN8HWKU5B7MfX6JjEWgkuNfftNVbnrjbjx6fPjY9KqXB46Rvy&spm=a21n57.1.hoverItem.2) | [Alicia-D 产品手册](https://docs.sparklingrobo.com/)
<p align="center"><img src="./media/readme/Alicia_D_v5_5.jpg" width="500" /></p>



<p align="center">
  <img alt="LeRobot, Hugging Face Robotics Library" src="./media/readme/lerobot-logo-thumbnail.png" width="100%">
</p>
<div align="center">

[![Tests](https://github.com/huggingface/lerobot/actions/workflows/nightly.yml/badge.svg?branch=main)](https://github.com/huggingface/lerobot/actions/workflows/nightly.yml?query=branch%3Amain)
[![Python versions](https://img.shields.io/pypi/pyversions/lerobot)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/huggingface/lerobot/blob/main/LICENSE)
[![Status](https://img.shields.io/pypi/status/lerobot)](https://pypi.org/project/lerobot/)
[![Version](https://img.shields.io/pypi/v/lerobot)](https://pypi.org/project/lerobot/)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-v2.1-ff69b4.svg)](https://github.com/huggingface/lerobot/blob/main/CODE_OF_CONDUCT.md)

</div>



**LeRobot** 旨在为 PyTorch 中的真实世界机器人提供模型、数据集和工具。目标是降低入门门槛，让每个人都能为共享数据集和预训练模型做出贡献并从中受益。

🤗标准化了从低成本机械臂（SO-100）到人形机器人的各种平台的控制。

🤗 标准化的、可扩展的 LeRobotDataset 格式（Parquet + MP4 或图像），托管在 Hugging Face Hub 上，支持大规模机器人数据集的高效存储、流式传输和可视化。

🤗 已证明可迁移到真实世界的最新策略，随时可用于训练和部署。

🤗 对开源生态系统的全面支持，以普及物理 AI。

## 快速开始

请参考[基础使用文档](./docs/Alicia_D_Usage_CN.md)

## 机器人与控制

<div align="center">
  <img src="./media/readme/robots_control_video.webp" width="640px" alt="Reachy 2 Demo">
</div>

LeRobot 提供了一个统一的 `Robot` 类接口，将控制逻辑与硬件细节解耦。它支持广泛的机器人和遥操作设备。

```python
from lerobot.robots.myrobot import MyRobot

# 连接到机器人
robot = MyRobot(config=...)
robot.connect()

# 读取观测并发送动作
obs = robot.get_observation()
action = model.select_action(obs)
robot.send_action(action)
```

**支持的硬件：** Alicia-D、SO100、LeKiwi、Koch、HopeJR、OMX、EarthRover、Reachy2、游戏手柄、键盘、手机、OpenARM、Unitree G1。

虽然这些设备已原生集成到 LeRobot 代码库中，但该库设计为可扩展的。您可以轻松实现 Robot 接口，利用 LeRobot 的数据收集、训练和可视化工具，用于您自己的自定义机器人。

有关详细的硬件设置指南，请参阅[硬件文档](https://huggingface.co/docs/lerobot/integrate_hardware)。


## 模型

LeRobot 在纯 PyTorch 中实现了最新的策略，涵盖模仿学习、强化学习和视觉-语言-动作（VLA）模型，更多模型即将推出。它还为您提供了工具来检测和检查训练过程。

<p align="center">
  <img alt="Gr00t Architecture" src="./media/readme/VLA_architecture.jpg" width="640px">
</p>

通过命令行训练策略：

```bash
lerobot-train \
  --policy=act \
  --dataset.repo_id=lerobot/aloha_mobile_cabinet
```

| 类别                   | 模型                                                                                                                                                                 |
| -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **模仿学习**     | [ACT](./docs/source/policy_act_README.md), [Diffusion](./docs/source/policy_diffusion_README.md), [VQ-BeT](./docs/source/policy_vqbet_README.md)                       |
| **强化学习** | [HIL-SERL](./docs/source/hilserl.mdx), [TDMPC](./docs/source/policy_tdmpc_README.md) & QC-FQL（即将推出）                                                            |
| **VLA 模型**            | [Pi0.5](./docs/source/pi05.mdx), [GR00T N1.5](./docs/source/policy_groot_README.md), [SmolVLA](./docs/source/policy_smolvla_README.md), [XVLA](./docs/source/xvla.mdx) |

与硬件类似，您可以轻松实现自己的策略，并利用 LeRobot 的数据收集、训练和可视化工具，将模型分享到 HF Hub。

有关详细的策略设置指南，请参阅[策略文档](https://huggingface.co/docs/lerobot/bring_your_own_policies)。

## 推理与评估

使用统一的评估脚本在仿真或真实硬件上评估您的策略。LeRobot 支持标准基准测试，如 **LIBERO**、**MetaWorld** 等更多即将推出。

```bash
# 在 LIBERO 基准测试上评估策略
lerobot-eval \
  --policy.path=lerobot/pi0_libero_finetuned \
  --env.type=libero \
  --env.task=libero_object \
  --eval.n_episodes=10
```

通过遵循 [EnvHub 文档](https://huggingface.co/docs/lerobot/envhub)，了解如何实现自己的仿真环境或基准测试，并从 HF Hub 分发。

## 资源

- **[文档](https://huggingface.co/docs/lerobot/index)：** 教程和 API 的完整指南。
- **[Discord](https://discord.gg/3gxM6Avj)：** 加入 `LeRobot` 服务器与社区讨论。
- **[X](https://x.com/LeRobotHF)：** 在 X 上关注我们，了解最新动态。
- **[机器人学习教程](https://huggingface.co/spaces/lerobot/robot-learning-tutorial)：** 使用 LeRobot 学习机器人学习的免费实践课程。

## 引用

如果您在研究中使用 LeRobot，请引用：

```bibtex
@misc{cadene2024lerobot,
    author = {Cadene, Remi and Alibert, Simon and Soare, Alexander and Gallouedec, Quentin and Zouitine, Adil and Palma, Steven and Kooijmans, Pepijn and Aractingi, Michel and Shukor, Mustafa and Aubakirova, Dana and Russi, Martino and Capuano, Francesco and Pascal, Caroline and Choghari, Jade and Moss, Jess and Wolf, Thomas},
    title = {LeRobot: State-of-the-art Machine Learning for Real-World Robotics in Pytorch},
    howpublished = "\url{https://github.com/huggingface/lerobot}",
    year = {2024}
}
```

如果您在研究中使用 Alicia-D LeRobot 集成，请同时引用：

```bibtex
@software{synria2025aliciadlerobot,
    title = {Alicia-D LeRobot Integration: Robot Learning Framework for Alicia-D Robotic Arms},
    author = {Synria Robotics Team},
    year = {2025},
    publisher = {Synria Robotics Co., Ltd.},
    url = {https://github.com/Synria-Robotics/lerobot},
    version = {6.1.0-beta1}
}
```

