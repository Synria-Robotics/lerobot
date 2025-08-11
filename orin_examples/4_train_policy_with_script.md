本教程将解释训练脚本、如何使用它，以及特别是如何配置训练运行所需的一切。
> **注意：** 以下假设您在配备 CUDA GPU 的机器上运行这些命令。如果您没有（或者如果您使用的是 Mac），您可以分别添加 `--policy.device=cpu` (`--policy.device=mps`)。但是，请注意代码在 CPU 上执行速度要慢得多。


## 训练脚本

LeRobot 在 [`lerobot/scripts/train.py`](../lerobot/scripts/train.py) 提供了一个训练脚本。在高层次上，它执行以下操作：

- 初始化/加载后续步骤的配置。
- 实例化一个数据集。
- （可选）实例化与该数据集对应的模拟环境。
- 实例化一个策略。
- 运行标准的训练循环，包括前向传播、反向传播、优化步骤，以及偶尔的日志记录、评估（在环境上评估策略）和检查点保存。

## 配置系统概述

在训练脚本中，主函数 `train` 期望一个 `TrainPipelineConfig` 对象：
```python
# train.py
@parser.wrap()
def train(cfg: TrainPipelineConfig):
```

您可以检查在 [`lerobot/configs/train.py`](../lerobot/configs/train.py) 中定义的 `TrainPipelineConfig`（它有大量注释，旨在作为理解任何选项的参考）

运行脚本时，命令行输入会通过 `@parser.wrap()` 装饰器进行解析，并自动生成此类的实例。在底层，这是通过 [Draccus](https://github.com/dlwh/draccus) 完成的，这是一个专门用于此目的的工具。如果您熟悉 Hydra，Draccus 同样可以从配置文件（.json、.yaml）加载配置，并通过命令行输入覆盖它们的值。与 Hydra 不同，这些配置是通过数据类在代码中预定义的，而不是完全在配置文件中定义。这允许更严格的序列化/反序列化、类型化，并直接在代码中将配置作为对象进行操作，而不是作为字典或命名空间（这在 IDE 中启用了诸如自动完成、跳转到定义等良好功能）。

让我们看一个简化的例子。在其他属性中，训练配置具有以下属性：
```python
@dataclass
class TrainPipelineConfig:
    dataset: DatasetConfig
    env: envs.EnvConfig | None = None
    policy: PreTrainedConfig | None = None
```
其中 `DatasetConfig` 例如定义如下：
```python
@dataclass
class DatasetConfig:
    repo_id: str
    episodes: list[int] | None = None
    video_backend: str = "pyav"
```

这创建了一个层次关系，例如，假设我们有一个 `TrainPipelineConfig` 的 `cfg` 实例，我们可以使用 `cfg.dataset.repo_id` 访问 `repo_id` 的值。
从命令行，我们可以使用非常相似的语法 `--dataset.repo_id=仓库/id` 来指定此值。

默认情况下，每个字段都采用数据类中指定的默认值。如果字段没有默认值，则需要从命令行或配置文件中指定——其路径也在命令行中给出（下面会详细介绍）。在上面的示例中，`dataset` 字段没有默认值，这意味着必须指定它。


## 从 CLI 指定值

假设我们想在 [pusht](https://huggingface.co/datasets/lerobot/pusht) 数据集上训练 [扩散策略](../lerobot/common/policies/diffusion)，并使用 [gym_pusht](https://github.com/huggingface/gym-pusht) 环境进行评估。执行此操作的命令如下所示：
```bash
python lerobot/scripts/train.py \
    --dataset.repo_id=lerobot/pusht \
    --policy.type=diffusion \
    --env.type=pusht
```

让我们分解一下：
- 要指定数据集，我们只需要指定它在 hub 上的 `repo_id`，这是 `DatasetConfig` 中唯一必需的参数。其余字段具有默认值，在这种情况下我们对这些默认值感到满意，因此我们只需添加选项 `--dataset.repo_id=lerobot/pusht`。
- 要指定策略，我们可以使用 `--policy` 附加 `.type` 来选择扩散策略。这里，`.type` 是一个特殊参数，允许我们选择继承自 `draccus.ChoiceRegistry` 并且已使用 `register_subclass()` 方法装饰的配置类。要更好地理解此功能，请查看此 [Draccus 演示](https://github.com/dlwh/draccus?tab=readme-ov-file#more-flexible-configuration-with-choice-types)。在我们的代码中，我们主要使用此机制来选择策略、环境、机器人和一些其他组件，如优化器。可供选择的策略位于 [lerobot/common/policies](../lerobot/common/policies)
- 类似地，我们使用 `--env.type=pusht` 选择环境。不同的环境配置可在 [`lerobot/common/envs/configs.py`](../lerobot/common/envs/configs.py) 中找到

让我们看另一个例子。假设您一直在 [lerobot/aloha_sim_insertion_human](https://huggingface.co/datasets/lerobot/aloha_sim_insertion_human) 上训练 [ACT](../lerobot/common/policies/act)，并使用 [gym-aloha](https://github.com/huggingface/gym-aloha) 环境进行评估：
```bash
python lerobot/scripts/train.py \
    --policy.type=act \
    --dataset.repo_id=lerobot/aloha_sim_insertion_human \
    --env.type=aloha \
    --output_dir=outputs/train/act_aloha_insertion
```
> 注意我们添加了 `--output_dir` 来明确指定将此运行的输出（检查点、训练状态、配置等）写入何处。这不是强制性的，如果您不指定它，将根据当前日期和时间、env.type 和 policy.type 创建一个默认目录。这通常看起来像 `outputs/train/2025-01-24/16-10-05_aloha_act`。

我们现在想在另一个任务上为 aloha 训练一个不同的策略。我们将更改数据集并使用 [lerobot/aloha_sim_transfer_cube_human](https://huggingface.co/datasets/lerobot/aloha_sim_transfer_cube_human) 代替。当然，我们还需要更改环境的任务以匹配这个其他任务。
查看 [`AlohaEnv`](../lerobot/common/envs/configs.py) 配置，默认任务是 `"AlohaInsertion-v0"`，这对应于我们在上面命令中训练的任务。[gym-aloha](https://github.com/huggingface/gym-aloha?tab=readme-ov-file#description) 环境也有 `AlohaTransferCube-v0` 任务，这对应于我们想要训练的这个其他任务。综合起来，我们可以使用以下命令在这个不同的任务上训练这个新策略：
```bash
python lerobot/scripts/train.py \
    --policy.type=act \
    --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
    --env.type=aloha \
    --env.task=AlohaTransferCube-v0 \
    --output_dir=outputs/train/act_aloha_transfer
```

## 从配置文件加载

现在，假设我们想要重现刚才的运行。该运行在其检查点中生成了一个 `train_config.json` 文件，该文件序列化了它使用的 `TrainPipelineConfig` 实例：
```json
{
    "dataset": {
        "repo_id": "lerobot/aloha_sim_transfer_cube_human",
        "episodes": null,
        ...
    },
    "env": {
        "type": "aloha",
        "task": "AlohaTransferCube-v0",
        "fps": 50,
        ...
    },
    "policy": {
        "type": "act",
        "n_obs_steps": 1,
        ...
    },
    ...
}
```

然后我们可以简单地使用以下命令从此文件加载配置值：
```bash
python lerobot/scripts/train.py \
    --config_path=outputs/train/act_aloha_transfer/checkpoints/last/pretrained_model/ \
    --output_dir=outputs/train/act_aloha_transfer_2
```
`--config_path` 也是一个特殊参数，允许从本地配置文件初始化配置。它可以指向包含 `train_config.json` 的目录，也可以直接指向配置文件本身。

与 Hydra 类似，如果我们愿意，我们仍然可以在 CLI 中覆盖一些参数，例如：
```bash
python lerobot/scripts/train.py \
    --config_path=outputs/train/act_aloha_transfer/checkpoints/last/pretrained_model/ \
    --output_dir=outputs/train/act_aloha_transfer_2
    --policy.n_action_steps=80
```
> 注意：虽然 `--output_dir` 通常不是必需的，但在这种情况下我们需要指定它，因为它否则会从 `train_config.json` 中获取值（即 `outputs/train/act_aloha_transfer`）。为了防止意外删除以前运行的检查点，如果您尝试写入现有目录，我们会引发错误。恢复运行时情况并非如此，这是您接下来将学到的内容。

`--config_path` 也可以接受 hub 上包含 `train_config.json` 文件的仓库的 repo_id，例如运行：
```bash
python lerobot/scripts/train.py --config_path=lerobot/diffusion_pusht
```
将使用与训练 [lerobot/diffusion_pusht](https://huggingface.co/lerobot/diffusion_pusht) 相同的配置开始训练运行。


## 恢复训练

能够恢复训练运行在它因任何原因崩溃或中止的情况下非常重要。我们将在这里演示如何做到这一点。

让我们重用上一次运行的命令并添加一些更多选项：
```bash
python lerobot/scripts/train.py \
    --policy.type=act \
    --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
    --env.type=aloha \
    --env.task=AlohaTransferCube-v0 \
    --log_freq=25 \
    --save_freq=100 \
    --output_dir=outputs/train/run_resumption
```

在这里，我们注意将日志频率和检查点频率设置为较低的数字，以便我们可以展示恢复。您应该能够看到一些日志记录并在 1 分钟内（取决于硬件）获得第一个检查点。等待第一个检查点发生，您应该在终端中看到类似这样的一行：
```
信息 2025-01-24 16:10:56 ts/train.py:263 在步骤 100 后检查点策略
```
现在让我们通过终止进程（按 `ctrl`+`c`）来模拟崩溃。然后我们可以简单地从最后一个可用的检查点恢复此运行：
```bash
python lerobot/scripts/train.py \
    --config_path=outputs/train/run_resumption/checkpoints/last/pretrained_model/ \
    --resume=true
```
您应该从日志记录中看到您的训练从中断的地方继续。

您可能想要恢复运行的另一个原因仅仅是扩展训练并添加更多训练步骤。训练步骤的数量由选项 `--steps` 设置，默认为 100 000。
您可以使用以下命令将上一次运行的步骤数加倍：
```bash
python lerobot/scripts/train.py \
    --config_path=outputs/train/run_resumption/checkpoints/last/pretrained_model/ \
    --resume=true \
    --steps=200000
```

## 运行的输出
在输出目录中，将有一个名为 `checkpoints` 的文件夹，其结构如下：
```bash
outputs/train/run_resumption/checkpoints
├── 000100  # 训练步骤 100 的 checkpoint_dir
│   ├── pretrained_model/
│   │   ├── config.json  # 策略配置
│   │   ├── model.safetensors  # 策略权重
│   │   └── train_config.json  # 训练配置
│   └── training_state/
│       ├── optimizer_param_groups.json  # 优化器参数组
│       ├── optimizer_state.safetensors  # 优化器状态
│       ├── rng_state.safetensors  # rng 状态
│       ├── scheduler_state.json  # 调度器状态
│       └── training_step.json  # 训练步骤
├── 000200
└── last -> 000200  # 指向最后一个可用检查点的符号链接
```

## 微调预训练策略

除了 Draccus 中当前的功能外，我们还为策略添加了一个特殊的 `.path` 参数，它允许您像使用 `PreTrainedPolicy.from_pretrained()` 一样加载策略。在这种情况下，`path` 可以是包含检查点的本地目录，也可以是指向 hub 上预训练策略的 repo_id。

例如，我们可以将在 aloha 传输任务上预训练的[策略](https://huggingface.co/lerobot/act_aloha_sim_transfer_cube_human)微调到 aloha 插入任务上。我们可以通过以下方式实现：
```bash
python lerobot/scripts/train.py \
    --policy.path=lerobot/act_aloha_sim_transfer_cube_human \
    --dataset.repo_id=lerobot/aloha_sim_insertion_human \
    --env.type=aloha \
    --env.task=AlohaInsertion-v0
```

这样做时，请记住微调数据集的特征必须与预训练策略的输入/输出特征相匹配。

## 典型的日志和指标

当您开始训练过程时，您将首先在终端中看到打印出的完整配置。您可以检查它以确保您正确配置了运行。最终配置也将与检查点一起保存。

之后，您将看到如下训练日志：
```
信息 2024-08-14 13:35:12 ts/train.py:192 步数:0 样本:64 回合:1 轮次:0.00 损失:1.112 梯度范数:15.387 学习率:2.0e-07 更新秒数:1.738 数据秒数:4.774
```
或评估日志：
```
信息 2024-08-14 13:38:45 ts/train.py:226 步数:100 样本:6K 回合:52 轮次:0.25 ∑奖励:20.693 成功率:0.0% 评估秒数:120.266
```

如果 `wandb.enable` 设置为 `true`，这些日志也将保存在 wandb 中。以下是一些缩写的含义：
- `smpl`：训练期间看到的样本数。
- `ep`：训练期间看到的回合数。一个回合包含一个完整操作任务中的多个样本。
- `epch`：所有唯一样本被看到的次数（轮次）。
- `grdn`：梯度范数。
- `∑rwrd`：计算每个评估回合中的奖励总和，然后取它们的平均值。
- `success`：评估回合的平均成功率。奖励和成功率通常不同，除非在稀疏奖励设置中，其中仅当任务成功完成时 reward=1。
- `eval_s`：在环境中评估策略的时间，以秒为单位。
- `updt_s`：更新网络参数的时间，以秒为单位。
- `data_s`：加载一批数据的时间，以秒为单位。

一些指标对于初始性能分析很有用。例如，如果您通过 `nvidia-smi` 命令发现当前 GPU 利用率较低，并且 `data_s` 有时过高，您可能需要修改批处理大小或数据加载工作进程数以加速数据加载。我们还推荐使用 [pytorch profiler](https://github.com/huggingface/lerobot?tab=readme-ov-file#improve-your-code-with-profiling) 进行详细的性能探测。

## 简而言之

我们将在这里总结本教程中要记住的主要用例。

#### 从头开始训练策略 – CLI
```bash
python lerobot/scripts/train.py \
    --policy.type=act \  # <- 选择 'act' 策略
    --env.type=pusht \  # <- 选择 'pusht' 环境
    --dataset.repo_id=lerobot/pusht  # <- 在此数据集上训练
```

#### 从头开始训练策略 - 配置文件 + CLI
```bash
python lerobot/scripts/train.py \
    --config_path=path/to/pretrained_model \  # <- 也可以是 repo_id
    --policy.n_action_steps=80  # <- 您仍然可以覆盖值
```

#### 恢复/继续训练运行
```bash
python lerobot/scripts/train.py \
    --config_path=checkpoint/pretrained_model/ \
    --resume=true \
    --steps=200000  # <- 您可以更改一些训练参数
```

#### 微调
```bash
python lerobot/scripts/train.py \
    --policy.path=lerobot/act_aloha_sim_transfer_cube_human \  # <- 也可以是检查点的本地路径
    --dataset.repo_id=lerobot/aloha_sim_insertion_human \
    --env.type=aloha \
    --env.task=AlohaInsertion-v0
```

---

现在您已经了解了如何训练策略的基础知识，您可能想知道如何将这些知识应用于实际机器人，或者如何记录您自己的数据集并在您的特定任务上训练策略？
如果是这样，请转到下一个教程 [`7_get_started_with_real_robot.md`](./7_get_started_with_real_robot.md)。

或者在此期间，祝您训练愉快！ 🤗
