# Huggingface Lerobot 超详细技术文档

## 目录

1. 简介
2. 安装与环境准备
3. 主要命令与参数详解
   - 3.1 机器人控制与数据采集（control_robot.py）
   - 3.2 策略训练（train.py）
   - 3.3 策略评估（eval.py）
4. 机器人与数据集配置参数详解
   - 4.1 机器人配置
   - 4.2 摄像头配置
   - 4.3 电机配置
   - 4.4 数据集配置
   - 4.5 策略类型详解
   - 4.6 LeRobotDataset内部结构
5. 进阶说明与常见问题

---

## 1. 简介

Lerobot 是 Huggingface 推出的机器人学习与控制库，支持多种主流机器人、仿真环境和模仿/强化学习算法。其设计目标是让机器人 AI 训练、部署、数据采集和模型复现变得极为简单。

---

## 2. 安装与环境准备

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
conda create -y -n lerobot python=3.10
conda activate lerobot
conda install ffmpeg -c conda-forge
pip install -e .
```
如需仿真环境支持（如 aloha、pusht）：
```bash
pip install -e ".[aloha, pusht]"
```

---

## 3. 主要命令与参数详解

### 3.1 机器人控制与数据采集（control_robot.py）

#### 基本用法

```bash
python lerobot/scripts/control_robot.py --robot.type=xuanya_arm --control.type=record --control.fps=30 --control.single_task="机械臂抓取物体任务" --control.repo_id=local/xuanya_dataset_task1 --control.warmup_time_s=5 --control.episode_time_s=10 --control.reset_time_s=5 --control.num_episodes=2 --control.push_to_hub=false
```

#### 主要参数详细说明

- `--robot.type`  
  机器人类型（如：xuanya_arm、aloha、so100、lekiwi等）。决定了后续所有硬件配置、通信协议等。

- `--control.type`  
  控制模式。可选值：
  - `calibrate`：校准机器人
  - `teleoperate`：手动遥控
  - `record`：采集数据集
  - `replay`：回放数据
  - `remote_robot`：远程控制（如 LeKiwi）

- `--control.fps`  
  控制/采集的帧率（每秒帧数）。如 30 表示每秒采集30帧。影响数据集时间分辨率。

- `--control.single_task`  
  本次采集/控制的任务描述（字符串）。如"机械臂抓取物体任务"。用于数据集元数据和后续训练的任务提示。

- `--control.repo_id`  
  数据集标识符。建议格式为 `用户名/数据集名`，如 `local/xuanya_dataset_task1`。用于本地存储和上传到 Huggingface Hub。

- `--control.warmup_time_s`  
  每个回合开始前的预热时间（秒）。用于设备预热、同步等。

- `--control.episode_time_s`  
  每个回合采集/控制的持续时间（秒）。

- `--control.reset_time_s`  
  每个回合结束后的环境重置时间（秒）。

- `--control.num_episodes`  
  总共采集/控制的回合数。

- `--control.push_to_hub`  
  是否将采集到的数据集上传到 Huggingface Hub（true/false）。

- `--control.policy.path`  
  （可选）用于部署的预训练策略路径。可为本地目录或 Huggingface Hub 上的模型 repo id。

- `--control.resume`  
  是否在已有数据集基础上继续采集（true/false）。

- `--control.display_data`  
  是否在屏幕上实时显示所有摄像头画面（true/false）。

- `--control.play_sounds`  
  是否使用语音合成提示事件（true/false）。

- `--control.num_image_writer_processes`  
  处理将帧保存为PNG的子进程数量。0表示仅用线程，≥1表示用子进程+线程。影响高帧率下的IO性能。

- `--control.num_image_writer_threads_per_camera`  
  每个摄像头用于写PNG的线程数。线程数过多可能导致主线程阻塞，过少则可能丢帧。

- `--control.video`  
  是否将数据集帧编码为视频（true/false）。

- `--control.private`  
  上传到 Huggingface Hub 时是否为私有仓库（true/false）。

- `--control.tags`  
  上传到 Hub 时为数据集添加的标签（列表）。

---

### 3.2 策略训练（train.py）

#### 基本用法

```bash
python lerobot/scripts/train.py --dataset.repo_id=local/xuanya_dataset_task1 --dataset.root=/home/ubuntu/.cache/huggingface/lerobot/local/xuanya_dataset_task1 --policy.type=act --output_dir=outputs/train/act_xuanya_task1 --job_name=act_xuanya_task1 --policy.device=cuda --wandb.enable=false
```

#### 主要参数详细说明

- `--dataset.repo_id`  
  数据集标识符。与采集时保持一致。

- `--dataset.root`  
  数据集本地根目录。

- `--policy.type`  
  策略类型。可选如 act、diffusion、tdmpc 等。

- `--output_dir`  
  训练输出目录。所有日志、模型、配置、视频等均保存在此。

- `--job_name`  
  本次训练任务名称。用于区分不同实验。

- `--policy.device`  
  训练所用设备。可选 `cuda`（GPU）、`cpu`。

- `--wandb.enable`  
  是否启用 Weights & Biases 日志记录（true/false）。

- `--resume`  
  是否从已有 checkpoint 恢复训练。

- `--seed`  
  随机种子。影响模型初始化、数据打乱等。

- `--num_workers`  
  数据加载器工作线程数。

- `--batch_size`  
  每次训练的 batch 大小。

- `--steps`  
  总训练步数。

- `--eval_freq`  
  每隔多少步进行一次评估。

- `--log_freq`  
  日志打印频率。

- `--save_checkpoint`  
  是否保存 checkpoint。

- `--save_freq`  
  checkpoint 保存频率。

- `--optimizer`  
  优化器配置（如 Adam、SGD 等）。

- `--scheduler`  
  学习率调度器配置。

- `--eval.n_episodes`  
  每次评估的回合数。

- `--eval.batch_size`  
  评估时的并行环境数。

- `--eval.use_async_envs`  
  评估时是否使用异步环境。

- `--wandb.project`  
  wandb 项目名。

- `--wandb.entity`  
  wandb 用户/团队名。

- `--wandb.notes`  
  wandb 备注。

- `--wandb.run_id`  
  wandb 运行ID。

- `--wandb.mode`  
  wandb 模式（online/offline/disabled）。

---

### 3.3 策略评估（eval.py）

#### 基本用法

```bash
python lerobot/scripts/eval.py --policy.path=outputs/train/act_xuanya_task1/checkpoints/last/pretrained_model --env.type=pusht --eval.batch_size=10 --eval.n_episodes=10 --policy.use_amp=false --policy.device=cuda
```

#### 主要参数详细说明

- `--policy.path`  
  预训练策略路径（本地或Hub）。

- `--env.type`  
  环境类型（如 pusht、aloha、xarm 等）。

- `--eval.batch_size`  
  并行评估环境数。

- `--eval.n_episodes`  
  评估回合数。

- `--policy.use_amp`  
  是否使用自动混合精度。

- `--policy.device`  
  评估所用设备。

- `--output_dir`  
  评估输出目录。

- `--job_name`  
  评估任务名。

- `--seed`  
  随机种子。

---

## 4. 机器人与数据集配置参数详解

### 4.1 机器人配置（以xuanya_arm为例）

- `robot_type`  
  机器人类型，固定为 "xuanya_arm"。

- `calibration_dir`  
  校准文件目录。

- `max_relative_target`  
  运动指令的最大相对幅度限制，保证安全。

- `serial_port`  
  机械臂串口号，留空自动搜索。

- `baudrate`  
  串口波特率，默认921600。

- `debug_mode`  
  是否开启调试模式。

- `leader_arms`  
  主控臂配置（一般为空，SDK自动管理）。

- `follower_arms`  
  从控臂配置（一般为空，SDK自动管理）。

- `mock`  
  是否为模拟模式。

### 4.2 摄像头配置

- `camera_index`  
  摄像头索引或设备路径。

- `fps`  
  帧率。

- `width`/`height`  
  分辨率。

- `color_mode`  
  颜色模式（rgb/bgr）。

- `rotation`  
  旋转角度。

- `mock`  
  是否为模拟摄像头。

### 4.3 电机配置

- `port`  
  电机串口号。

- `motors`  
  电机字典，键为电机名，值为（ID, 型号）元组。

- `mock`  
  是否为模拟电机。

### 4.4 数据集配置

- `repo_id`  
  数据集标识符。

- `root`  
  本地根目录。

- `episodes`  
  指定加载的回合索引列表。

- `image_transforms`  
  图像增强配置。

- `revision`  
  数据集版本。

- `use_imagenet_stats`  
  是否使用ImageNet均值方差归一化。

- `video_backend`  
  视频解码后端（如 pyav、torchcodec）。

### 4.5 策略类型详解

Lerobot支持多种先进的策略类型，每种都有其特定的优势和应用场景：

1. **ACT (Action Chunking Transformer)**
   - 核心特点：基于Transformer架构的模仿学习方法，能够一次性预测长序列的动作
   - 主要优势：优秀的长期规划能力和高精度动作生成
   - 关键参数：
     - `chunk_size`：动作预测"块"的大小（默认100步）
     - `vision_backbone`：视觉骨干网络（如resnet18）
     - `dim_model`：Transformer隐藏层维度（默认512）
     - `use_vae`：是否使用变分自编码器目标（默认True）
   - 适用场景：双臂操作、精细抓取等高精度任务

2. **Diffusion Policy**
   - 核心特点：基于扩散模型的策略学习方法
   - 主要优势：处理不确定性和多模态任务能力强，生成轨迹平滑
   - 关键参数：
     - `n_diffusion_steps`：扩散过程的步数
     - `hidden_dim`：隐藏层维度
     - `beta_schedule`：噪声调度方式
   - 适用场景：需要精确但灵活轨迹的操作任务

3. **TDMPC (Temporal Difference Model Predictive Control)**
   - 核心特点：结合模型预测控制和强化学习的混合方法
   - 主要优势：在长期规划与短期执行间取得平衡，可适应环境变化
   - 关键参数：
     - `horizon`：预测规划的时间范围（默认5）
     - `discount`：折扣因子（默认0.9）
     - `uncertainty_regularizer_coeff`：不确定性正则化系数
   - 适用场景：需要实时调整的动态环境

4. **VQ-BeT (Vector Quantized Behavior Transformer)**
   - 核心特点：基于向量量化和Transformer的行为生成方法
   - 主要优势：将连续行为空间离散化，能有效处理复杂任务
   - 关键参数：
     - `vqvae_n_embed`：码本大小（默认16）
     - `vqvae_embedding_dim`：嵌入维度（默认256）
     - `gpt_n_layer`：GPT层数（默认8）
   - 适用场景：需要学习复杂行为序列的任务

5. **PI0 (Physical Intelligence Zero)**
   - 核心特点：基于视觉-语言-动作流的通用机器人控制模型
   - 主要优势：利用大型语言模型和视觉模型的迁移学习能力
   - 关键参数：
     - `tokenizer_max_length`：文本标记最大长度（默认48）
     - `resize_imgs_with_padding`：图像处理尺寸（默认224x224）
     - `adapt_to_pi_aloha`：是否适配PI内部运行时（默认False）
   - 适用场景：需要语言指令理解的通用机器人控制

6. **PI0FAST**
   - 核心特点：PI0的优化版本，专注于实时性能
   - 主要优势：在保持PI0核心功能的同时提高推理速度
   - 关键参数：与PI0相似，但有额外的性能优化参数
   - 适用场景：需要低延迟响应的实时控制场景

### 4.6 LeRobotDataset内部结构

`LeRobotDataset`是Lerobot的核心数据集类，提供了一种灵活且统一的方式来处理各种机器人数据。其内部结构设计如下：

1. **基本组成**
   - `hf_dataset`：基于Hugging Face的数据集，内部使用Arrow/Parquet格式高效存储
   - `episode_data_index`：包含每个回合起始和结束索引的索引表
   - `stats`：数据统计信息，包括每个特征的最大值、最小值、均值和标准差
   - `info`：数据集元数据，包括版本、帧率、编码设置等
   - `videos_dir`：视频文件存储目录
   - `camera_keys`：所有摄像头特征的键名列表

2. **数据格式**
   - **状态数据**：如机器人关节角度，存储为浮点数数组
   - **图像数据**：存储为视频文件或PNG图像序列
   - **动作数据**：控制命令，通常是浮点数数组
   - **元数据**：每帧的时间戳、回合索引、帧索引等

3. **核心特性**
   - **时间索引**：通过`delta_timestamps`参数可以检索相对于当前帧的历史或未来帧
   - **标准化**：内置的统计信息用于数据标准化
   - **自动下载**：从Hugging Face Hub自动下载和缓存数据
   - **灵活查询**：支持按回合、时间戳或帧索引检索数据

4. **数据加载示例**：
   ```python
   # 加载数据集
   dataset = LeRobotDataset("lerobot/aloha_static_coffee")
   
   # 获取单个帧
   frame = dataset[0]
   
   # 获取当前帧和相对于它的历史帧
   frames = dataset.get_item_with_relative_frames(
       0, delta_timestamps={"observation.image": [-1, -0.5, 0]}
   )
   ```

5. **数据存储结构**
   - 元数据：JSON文件存储在`meta/`文件夹
   - 视频数据：按回合组织的MP4文件存储在`videos/`文件夹
   - 表格数据：Parquet格式存储在`data/`文件夹
   - 统计信息：存储在`meta/stats.json`中

6. **特有功能**
   - **视频/图像转换**：支持将图像序列编码为视频以节省空间
   - **异步IO**：多进程图像写入以提高高帧率采集性能
   - **Hub集成**：无缝上传/下载到Hugging Face Hub
   - **缓存管理**：自动处理本地缓存以提高性能

---

## 5. 进阶说明与常见问题

- **如何自定义机器人？**  
  参考 `lerobot/common/robot_devices/robots/configs.py`，仿照已有类添加新机器人配置。

- **如何自定义策略？**  
  参考 `lerobot/configs/policies.py`，实现新的 PreTrainedConfig 子类，并注册到 ChoiceRegistry。

- **如何上传模型/数据集到 Huggingface Hub？**  
  使用 `save_pretrained` 方法或命令行参数 `--push_to_hub=true`。

- **如何恢复训练？**  
  加上 `--resume=true` 并指定 `--output_dir` 为已有训练目录。

- **如何可视化数据集？**  
  使用 `python lerobot/scripts/visualize_dataset.py --repo-id=xxx --episode-index=0`。

---

如需进一步定制或遇到问题，建议查阅 `examples/` 目录下的相关 markdown 教程，或加入官方 Discord 社区交流。

--- 