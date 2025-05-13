本教程介绍了如何在 LeRobot 中使用 [Aloha 和 Aloha 2 固定式机器人](https://www.trossenrobotics.com/aloha-stationary)。

## 设置

请遵循 [Trossen Robotics 的文档](https://docs.trossenrobotics.com/aloha_docs/2.0/getting_started/stationary/hardware_setup.html) 来设置硬件并将 4 个机械臂和 4 个摄像头连接到您的计算机。


## 安装 LeRobot

在您的计算机上：

1. [安装 Miniconda](https://docs.anaconda.com/miniconda/#quick-command-line-install)：
```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
```

2. 重启 shell 或运行 `source ~/.bashrc`

3. 为 lerobot 创建并激活一个新的 conda 环境
```bash
conda create -y -n lerobot python=3.10 && conda activate lerobot
```

4. 克隆 LeRobot：
```bash
git clone https://github.com/huggingface/lerobot.git ~/lerobot
```

5. 当使用 `miniconda` 时，在您的环境中安装 `ffmpeg`：
```bash
conda install ffmpeg -c conda-forge
```

6. 安装 LeRobot 以及 Aloha 电机 (dynamixel) 和摄像头 (intelrealsense) 的依赖项：
```bash
cd ~/lerobot && pip install -e ".[dynamixel, intelrealsense]"
```

## 遥操作

**/!\ 为了安全，请阅读此内容 /!\**
遥操作是指手动操作主机械臂来移动从机械臂。重要的是：
1. 确保您的主机械臂与从机械臂处于相同的位置，这样从机械臂就不会移动过快以匹配主机械臂。
2. 我们的代码假定您的机器人是按照 Trossen Robotics 的说明组装的。这使我们可以跳过校准，因为我们使用了 `.cache/calibration/aloha_default` 中预定义的校准文件。如果您更换了电机，请确保遵循 Trossen Robotics 的确切说明。

通过运行以下代码，您可以开始您的第一次 **安全** 遥操作：

> **注意：** 要可视化数据，请启用 `--control.display_data=true`。这将使用 `rerun` 流式传输数据。

```bash
python lerobot/scripts/control_robot.py \
  --robot.type=aloha \
  --robot.max_relative_target=5 \
  --control.type=teleoperate
```

通过添加 `--robot.max_relative_target=5`，我们覆盖了 [`AlohaRobotConfig`](lerobot/common/robot_devices/robots/configs.py) 中定义的 `max_relative_target` 的默认值。为了更安全，预期值为 `5` 以限制移动幅度，但遥操作不会很平滑。当您感到有信心时，可以通过在命令行中添加 `--robot.max_relative_target=null` 来禁用此限制：
```bash
python lerobot/scripts/control_robot.py \
  --robot.type=aloha \
  --robot.max_relative_target=null \
  --control.type=teleoperate
```

## 录制数据集

熟悉遥操作后，您可以使用 Aloha 录制您的第一个数据集。

如果您想使用 Hugging Face Hub 的功能上传数据集，并且之前没有这样做过，请确保您已使用写访问令牌登录，该令牌可以从 [Hugging Face 设置](https://huggingface.co/settings/tokens) 生成：
```bash
huggingface-cli login --token ${HUGGINGFACE_TOKEN} --add-to-git-credential
```

将您的 Hugging Face 仓库名称存储在一个变量中以运行这些命令：
```bash
HF_USER=$(huggingface-cli whoami | head -n 1)
echo $HF_USER
```

录制 2 个片段并将您的数据集上传到 Hub：
```bash
python lerobot/scripts/control_robot.py \
  --robot.type=aloha \
  --robot.max_relative_target=null \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="抓住一个乐高积木并将其放入箱子中。" \
  --control.repo_id=${HF_USER}/aloha_test \
  --control.tags='["tutorial"]' \
  --control.warmup_time_s=5 \
  --control.episode_time_s=30 \
  --control.reset_time_s=30 \
  --control.num_episodes=2 \
  --control.push_to_hub=true
```

## 可视化数据集

如果您使用 `--control.push_to_hub=true` 将数据集上传到 Hub，您可以通过复制粘贴您的仓库 ID [在线可视化您的数据集](https://huggingface.co/spaces/lerobot/visualize_dataset)，仓库 ID 由以下命令给出：
```bash
echo ${HF_USER}/aloha_test
```

如果您没有使用 `--control.push_to_hub=false` 上传，您也可以在本地使用以下命令进行可视化：
```bash
python lerobot/scripts/visualize_dataset_html.py \
  --repo-id ${HF_USER}/aloha_test
```

## 回放片段

**/!\ 为了安全，请阅读此内容 /!\**
回放是指自动重播在给定数据集片段中记录的动作序列（即电机的目标位置）。确保机器人当前的初始位置与片段中的初始位置相似，这样您的从机械臂就不会移动过快以到达第一个目标位置。为了安全起见，您可能希望如上所述在命令行中添加 `--robot.max_relative_target=5`。

现在尝试在您的机器人上回放第一个片段：
```bash
python lerobot/scripts/control_robot.py \
  --robot.type=aloha \
  --robot.max_relative_target=null \
  --control.type=replay \
  --control.fps=30 \
  --control.repo_id=${HF_USER}/aloha_test \
  --control.episode=0
```

## 训练策略

要训练控制机器人的策略，请使用 [`python lerobot/scripts/train.py`](../lerobot/scripts/train.py) 脚本。需要一些参数。这是一个示例命令：
```bash
python lerobot/scripts/train.py \
  --dataset.repo_id=${HF_USER}/aloha_test \
  --policy.type=act \
  --output_dir=outputs/train/act_aloha_test \
  --job_name=act_aloha_test \
  --policy.device=cuda \
  --wandb.enable=true
```

让我们解释一下：
1. 我们使用 `--dataset.repo_id=${HF_USER}/aloha_test` 提供了数据集作为参数。
2. 我们使用 `policy.type=act` 提供了策略。这将从 [`configuration_act.py`](../lerobot/common/policies/act/configuration_act.py) 加载配置。重要的是，此策略将自动适应已保存在数据集中的机器人电机状态、电机动作和摄像头的数量（例如 `laptop` 和 `phone`）。
4. 我们提供了 `policy.device=cuda`，因为我们在 Nvidia GPU 上进行训练，但您可以使用 `policy.device=mps` 在 Apple 芯片上进行训练。
5. 我们提供了 `wandb.enable=true` 以使用 [Weights and Biases](https://docs.wandb.ai/quickstart) 来可视化训练图。这是可选的，但如果您使用它，请确保通过运行 `wandb login` 登录。

有关 `train` 脚本的更多信息，请参阅上一个教程：[`examples/4_train_policy_with_script.md`](../examples/4_train_policy_with_script.md)

训练应该需要几个小时。您将在 `outputs/train/act_aloha_test/checkpoints` 中找到检查点。

## 评估您的策略

您可以使用 [`lerobot/scripts/control_robot.py`](../lerobot/scripts/control_robot.py) 中的 `record` 函数，但需要输入策略检查点。例如，运行此命令以录制 10 个评估片段：
```bash
python lerobot/scripts/control_robot.py \
  --robot.type=aloha \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="抓住一个乐高积木并将其放入箱子中。" \
  --control.repo_id=${HF_USER}/eval_act_aloha_test \
  --control.tags='["tutorial"]' \
  --control.warmup_time_s=5 \
  --control.episode_time_s=30 \
  --control.reset_time_s=30 \
  --control.num_episodes=10 \
  --control.push_to_hub=true \
  --control.policy.path=outputs/train/act_aloha_test/checkpoints/last/pretrained_model \
  --control.num_image_writer_processes=1
```

如您所见，这与之前用于录制训练数据集的命令几乎相同。有两点不同：
1. 有一个额外的 `--control.policy.path` 参数，指示您的策略检查点的路径（例如 `outputs/train/eval_act_aloha_test/checkpoints/last/pretrained_model`）。如果您已将模型检查点上传到 Hub，也可以使用模型仓库（例如 `${HF_USER}/act_aloha_test`）。
2. 数据集的名称以 `eval` 开头，以反映您正在运行推理（例如 `${HF_USER}/eval_act_aloha_test`）。
3. 我们使用 `--control.num_image_writer_processes=1` 而不是默认值 (`0`)。在我们的计算机上，使用专用进程将 4 个摄像头的图像写入磁盘，可以在推理期间达到恒定的 30 fps。您可以随意探索 `--control.num_image_writer_processes` 的不同值。

## 更多

请参阅此[之前的教程](https://github.com/huggingface/lerobot/blob/main/examples/7_get_started_with_real_robot.md#4-train-a-policy-on-your-data)以获取更深入的解释。

如果您有任何问题或需要帮助，请在 Discord 的 `#aloha-arm` 频道中联系我们。
