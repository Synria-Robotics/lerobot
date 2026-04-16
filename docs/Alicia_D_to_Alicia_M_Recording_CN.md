# Alicia-D 示教 Alicia-M 新手使用说明

这份文档是写给第一次接触 `LeRobot`、`Alicia-D leader` 和 `Alicia-M follower` 的用户的。

如果你之前没有录过数据、没有训过模型，也不知道 `/dev/ttyACM0`、`/dev/video5`、`repo_id`、`rename_map` 是什么，可以直接按这份文档一步一步做。

这份文档会带你完成 3 件事：

1. 用 `Alicia-D leader` 示教 `Alicia-M follower`，录制一个本地数据集。
2. 用录好的数据训练一个 `ACT` 模型。
3. 把训练好的模型重新加载到 `Alicia-M` 上做真机推理。

文档后半部分也保留了 `pi0` / `pi05` 的用法，但如果你是第一次上手，建议先跑通 `ACT`。

---

## 目录

- [1. 先看这个：第一次上手最短流程](#1-先看这个第一次上手最短流程)
- [2. 先理解几个最重要的名词](#2-先理解几个最重要的名词)
- [3. 安装环境](#3-安装环境)
- [4. 连接硬件并确认端口](#4-连接硬件并确认端口)
- [5. 第一次录制数据](#5-第一次录制数据)
- [6. 正式录制数据集](#6-正式录制数据集)
- [7. ACT 训练](#7-act-训练)
- [8. pi0 / pi05 训练](#8-pi0--pi05-训练)
- [9. 使用训练好的模型做真机推理](#9-使用训练好的模型做真机推理)
- [10. 键盘快捷键](#10-键盘快捷键)
- [11. 最常见问题与排查方法](#11-最常见问题与排查方法)

---

## 1. 先看这个：第一次上手最短流程

如果你现在只想先跑通一遍，不想先看所有解释，那么可以按下面这条路线走：

1. 安装好 Python 环境、`alicia_d_sdk`、`alicia_m_sdk` 和 `LeRobot`。
2. 查出 3 个设备号：
   - `Alicia-M` 的串口，例如 `/dev/ttyACM1`
   - `Alicia-D` 的串口，例如 `/dev/ttyACM0`
   - 摄像头设备，例如 `/dev/video5`
3. 先录制 1 个 episode，确认机械臂能动、相机有图像、不会报错。
4. 确认没问题后，把 `--dataset.num_episodes` 改大，正式采集数据。
5. 用录好的数据先训练 `ACT`。
6. 用 `checkpoints/last/pretrained_model` 做 1 个 episode 的真机推理。

第一次使用时，强烈建议这样做：

- 先只接 1 个摄像头，不要一开始就上多相机。
- 先只录 `1` 个 episode 做自检，不要直接录几十个。
- 录制和推理时都加上 `--play_sounds=false`，这样就算系统没有 `spd-say` 也不会报错。
- 先使用 `--robot.control_mode=mit --robot.use_interpolation=false`。
- 先使用 `--robot.speed=50` 这样的低速值。
- 先把 `--dataset.push_to_hub=false`，全部只保存在本地。

如果你是第一次训练模仿学习模型，推荐顺序是：

1. 先录 5 到 20 个干净的演示 episode。
2. 先训练 `ACT`。
3. 确认 `ACT` 可以推理后，再考虑 `pi0` / `pi05`。

---

## 2. 先理解几个最重要的名词

这一节很重要。很多新手报错，其实不是命令不会写，而是不知道这些参数在说什么。

### 2.1 Leader 和 Follower 是什么

- `Alicia-D leader`
  你手里拿着做示教的机械臂。你动它，它就产生动作信号。

- `Alicia-M follower`
  真正执行动作的机械臂。程序会把 leader 的动作转发给它。

简单理解：

- `leader` 负责“告诉系统你想怎么动”
- `follower` 负责“真的动起来”

### 2.2 什么叫“计算机中介控制”

在本文档对应的场景里，`Alicia-D` 和 `Alicia-M` 不是靠一根专用控制线直接互连，而是都接到同一台电脑上。

动作链路是这样的：

1. 电脑从 `Alicia-D leader` 读取当前关节状态。
2. 电脑把这些关节值映射成 `Alicia-M follower` 的关节格式。
3. 电脑再把动作发送给 `Alicia-M follower`。
4. 同时电脑从 `Alicia-M` 和相机读取观测。
5. 程序把“观测 + 动作 + 任务文本”写入数据集。

所以你会看到命令里有两个很关键的参数：

- `--teleop.directly_controls_robot=false`
- `--teleop.target_follower_type=alicia_m`

它们的含义分别是：

- `--teleop.directly_controls_robot=false`
  表示 leader 不是通过硬件线直接控制 follower，而是通过电脑中转。

- `--teleop.target_follower_type=alicia_m`
  表示 leader 的动作要先映射成 `Alicia-M` 的关节定义。

如果这两个参数没配对，命令有时也能跑起来，但动作语义往往是错的。

### 2.3 `/dev/ttyACM0` 和 `/dev/video5` 是什么

在 Ubuntu 里：

- `/dev/ttyACM0`、`/dev/ttyACM1` 这类名字通常表示 USB 串口设备
- `/dev/video0`、`/dev/video5` 这类名字通常表示摄像头设备

它们不是固定不变的。

今天 `Alicia-M` 可能是 `/dev/ttyACM1`，明天也可能变成 `/dev/ttyACM0`。

所以你不能死记一个设备号，必须每次根据实际连接情况确认。

### 2.4 `repo_id` 和 `root` 是什么

很多新手会搞混这两个参数。

- `--dataset.repo_id`
  是数据集的逻辑名字，看起来像 `用户名/数据集名`。
  即使你不上传到 Hugging Face，也通常还是要填它。

- `--dataset.root`
  是数据集在你电脑上的保存目录。

例如：

```bash
--dataset.repo_id=ubuntu/alicia_d_to_alicia_m_test
--dataset.root=/home/ubuntu/Data/LerobotData/alicia_dm
```

它们的意思是：

- 这个数据集的名字叫 `ubuntu/alicia_d_to_alicia_m_test`
- 它的本地文件保存在 `/home/ubuntu/Data/LerobotData/alicia_dm`

### 2.5 Episode 是什么

`episode` 可以理解为“一次完整演示”。

例如你要演示“把一个物体从左边拿到右边”，那从开始录制到这次任务结束，这整段过程就是一个 episode。

命令里经常会看到：

- `--dataset.num_episodes=10`
- `--dataset.episode_time_s=30`
- `--dataset.reset_time_s=15`

含义是：

- 录制 10 次演示
- 每次演示最长录制 30 秒
- 每次演示结束后，留 15 秒给你把场景和机械臂复位

### 2.6 什么是 `rename_map`

这是训练和推理里最容易把新手搞懵的一个概念。

简单理解：

- 你的数据集里，相机图像有一个名字
- 模型也期待一个图像名字
- 如果两个名字不一样，就要用 `rename_map` 做“改名映射”

例如：

- 你的数据集相机键名是 `front`
- 那么数据集里的图像特征通常会叫 `observation.images.front`

如果某个预训练模型期待的是：

- `observation.images.base_0_rgb`

那训练时就要告诉程序：

```bash
--rename_map='{"observation.images.front":"observation.images.base_0_rgb"}'
```

注意：

- `lerobot-train` 使用的是 `--rename_map`
- `lerobot-record --policy.path=...` 做推理时，使用的是 `--dataset.rename_map`

---

## 3. 安装环境

如果你已经有能正常运行的 `lerobot` 环境，可以跳到下一节。

### 3.1 前置条件

建议准备：

- Ubuntu Linux
- Python 3.10
- Conda 或虚拟环境
- 已安装 `alicia_d_sdk`
- 已安装 `alicia_m_sdk`
- 如果要训练模型，建议有 CUDA GPU

### 3.2 创建 Python 环境

```bash
conda create -n lerobot python=3.10
conda activate lerobot
```

如果你已经有现成环境，保证后面的命令都在同一个环境里运行即可。

### 3.3 确认 Alicia SDK 可以导入

运行：

```bash
python -c "import alicia_d_sdk, alicia_m_sdk; print('Alicia SDKs OK')"
```

如果你看到：

```text
Alicia SDKs OK
```

说明两个 SDK 都已经装好。

如果这里报错：

- 不要继续执行 `lerobot-record`
- 先把 SDK 安装问题解决

因为后面的录制、训练、推理都依赖这两个 SDK。

### 3.4 安装 LeRobot

假设你的仓库在：

```bash
/home/ubuntu/lerobot
```

则执行：

```bash
cd /home/ubuntu/lerobot
pip install -e .
```

`-e` 表示“可编辑安装”。

它的好处是：

- 你改了仓库里的 Python 文件后，通常不用重新安装
- 代码改动会直接生效

### 3.5 验证命令可以调用

```bash
lerobot-record --help
lerobot-train --help
lerobot-find-port
```

如果这些命令能正常打印帮助信息或设备信息，说明安装基本正常。

### 3.6 如果系统没有 `spd-say`

有些 Ubuntu 环境没有安装 `spd-say`，这会导致录制结束时报类似错误：

```text
FileNotFoundError: [Errno 2] No such file or directory: 'spd-say'
```

现在仓库代码已经做了兼容，但对新手来说，最稳妥的做法仍然是：

- 在录制和推理命令后面都加上 `--play_sounds=false`

这样即使系统没有语音播报程序，也不会影响主流程。

---

## 4. 连接硬件并确认端口

这一节非常重要。很多录制失败都是因为串口和摄像头设备号填错了。

### 4.1 正确的物理连接方式

本文档对应的连接方式是：

1. `Alicia-D leader` 用 USB 线连接到电脑。
2. `Alicia-M follower` 用 USB 线连接到电脑。
3. 摄像头连接到电脑。
4. 不使用 leader 到 follower 的硬件直连控制线。

### 4.2 先学会识别串口

最简单的方法不是“猜”，而是“拔插对比”。

建议按这个顺序做：

1. 先拔掉 `Alicia-D` 和 `Alicia-M`。
2. 运行：

```bash
ls /dev/ttyACM*
```

3. 只插上 `Alicia-M`，再运行一次：

```bash
ls /dev/ttyACM*
```

4. 新出现的那个设备，通常就是 `Alicia-M` 的串口。
5. 再插上 `Alicia-D`，再运行一次：

```bash
ls /dev/ttyACM*
```

6. 新增的另一个设备，就是 `Alicia-D` 的串口。

例如：

- 插 `Alicia-M` 前没有任何 `ttyACM`
- 插上后出现 `/dev/ttyACM1`
- 再插 `Alicia-D` 后出现 `/dev/ttyACM0`

那么这一次就应当写成：

- `--robot.port=/dev/ttyACM1`
- `--teleop.port=/dev/ttyACM0`

如果你懒得用拔插法，也可以试：

```bash
lerobot-find-port
```

但对新手来说，拔插法最直观，也最不容易搞错。

### 4.3 再学会识别摄像头设备

摄像头也建议用“拔插对比法”。

1. 先拔掉摄像头。
2. 运行：

```bash
ls /dev/video*
```

3. 插上摄像头，再运行一次：

```bash
ls /dev/video*
```

4. 新出现的 `/dev/videoX` 通常就是这只摄像头暴露出来的设备节点。

注意：

- 一只 USB 摄像头有时会同时暴露多个 `/dev/video*`
- 其中不一定每个节点都真的能正常出图像
- 有的节点可能只是元数据节点，或者某种不适合当前读取方式的流

所以只看到设备号还不够，你最好再执行：

```bash
lerobot-find-cameras opencv
```

它会列出当前能检测到的 OpenCV 摄像头。

如果你的系统里一只摄像头对应了多个节点，比如 `/dev/video4` 和 `/dev/video5`，你需要实际测试哪个能正常读图。

对很多新手来说，最稳妥的方法是：

1. 先用 `lerobot-find-cameras opencv` 看看有哪些节点
2. 再把你准备写进命令的那个节点单独拿来试
3. 如果录制时报“读不到图像”“分辨率设置失败”“Bad file descriptor”，就换另一个视频节点

### 4.4 Linux 串口权限

如果设备存在，但打开串口时报权限错误，可以把当前用户加入 `dialout` 组：

```bash
sudo usermod -a -G dialout $USER
```

执行后通常需要：

- 重新登录系统
- 或者至少重新开一个终端会话

### 4.5 第一次上手推荐先设置 5 个变量

把下面 5 行中的值改成你自己的实际设备和路径：

```bash
ALICIA_M_PORT=/dev/ttyACM1
ALICIA_D_PORT=/dev/ttyACM0
CAMERA_DEV=/dev/video5
DATA_ROOT=/home/ubuntu/Data/LerobotData/alicia_dm
DATASET_ID=ubuntu/alicia_d_to_alicia_m_test
```

后面命令里的参数就都可以直接替换成这几个变量，能减少你手抄错设备号的概率。

---

## 5. 第一次录制数据

这一节的目标不是“录很多数据”，而是“先确认整条链路是通的”。

第一次录制时，请只做这件事：

- 录 `1` 个 episode
- 只用 `1` 个相机
- 不上传
- 不追求速度
- 先确认整个系统不报错

### 5.1 第一次录制前的检查清单

正式敲命令前，请确认：

- `Alicia-D leader` 已连接到电脑
- `Alicia-M follower` 已连接到电脑
- 摄像头已连接到电脑
- 你已经确认了 `Alicia-M` 的串口号
- 你已经确认了 `Alicia-D` 的串口号
- 你已经确认了相机设备号
- 当前终端已经执行过 `conda activate lerobot`
- 工作空间周围没有会被机械臂碰到的物体
- 第一次测试时，机械臂附近有人看护

### 5.2 推荐的新手测试命令

把下面命令中的设备号和路径替换成你自己的值：

```bash
lerobot-record \
    --robot.type=alicia_m_follower \
    --robot.port=$ALICIA_M_PORT \
    --robot.version=v1_1 \
    --robot.control_aim=operation \
    --robot.control_mode=mit \
    --robot.use_interpolation=false \
    --robot.speed=50 \
    --robot.max_relative_target=8 \
    --robot.cameras="{front: {type: opencv, index_or_path: $CAMERA_DEV, width: 640, height: 480, fps: 30}}" \
    --robot.id=alicia_m \
    --teleop.type=alicia_d_leader \
    --teleop.port=$ALICIA_D_PORT \
    --teleop.gripper_type=50mm \
    --teleop.directly_controls_robot=false \
    --teleop.target_follower_type=alicia_m \
    --teleop.id=alicia_d \
    --dataset.repo_id=$DATASET_ID \
    --dataset.root=$DATA_ROOT \
    --dataset.num_episodes=1 \
    --dataset.single_task="Teleoperate Alicia-M with Alicia-D leader" \
    --dataset.episode_time_s=30 \
    --dataset.reset_time_s=15 \
    --display_data=true \
    --dataset.push_to_hub=false \
    --play_sounds=false
```

如果你不想用环境变量，也可以直接把变量展开成具体值，例如：

```bash
--robot.port=/dev/ttyACM1
--teleop.port=/dev/ttyACM0
--robot.cameras="{front: {type: opencv, index_or_path: /dev/video5, width: 640, height: 480, fps: 30}}"
--dataset.root=/home/ubuntu/Data/LerobotData/alicia_dm
--dataset.repo_id=ubuntu/alicia_d_to_alicia_m_test
```

### 5.3 这条命令每一部分是什么意思

下面只解释新手最容易困惑的部分。

#### 机器人部分

- `--robot.type=alicia_m_follower`
  表示被控机械臂是 `Alicia-M follower`。

- `--robot.port=$ALICIA_M_PORT`
  这是 `Alicia-M` 的串口，不是 `Alicia-D` 的。

- `--robot.control_mode=mit`
  使用 SDK 的 `MIT` 控制模式。

- `--robot.use_interpolation=false`
  表示在 `MIT` 模式下直接使用 SDK 的 MIT PD 控制，不再额外走固件插值。
  对示教录制来说，这通常更接近“你怎么动 leader，follower 就怎么跟”。

- `--robot.speed=50`
  这是比较保守的低速值。第一次测试建议不要设太高。

- `--robot.max_relative_target=8`
  每一步动作变化量的安全裁剪上限。
  你可以先理解成“限制 follower 不要因为单步目标跳太大而猛冲”。

- `--robot.cameras="{front: {...}}"`
  这里配置了 1 个相机。
  相机的名字叫 `front`，因此写进数据集后的图像特征名通常就是：

```text
observation.images.front
```

这点在训练和推理时很重要。

#### 示教器部分

- `--teleop.type=alicia_d_leader`
  表示示教设备是 `Alicia-D leader`。

- `--teleop.port=$ALICIA_D_PORT`
  这是 `Alicia-D` 的串口。

- `--teleop.directly_controls_robot=false`
  表示动作经过电脑中转，而不是 leader 通过硬件线直接控制 follower。

- `--teleop.target_follower_type=alicia_m`
  表示 leader 的动作会被映射成 `Alicia-M` 的关节定义。

#### 数据集部分

- `--dataset.repo_id=$DATASET_ID`
  数据集逻辑名字。

- `--dataset.root=$DATA_ROOT`
  本地保存路径。

- `--dataset.num_episodes=1`
  这里只录 1 个 episode，目的是先排查问题。

- `--dataset.single_task="Teleoperate Alicia-M with Alicia-D leader"`
  这是任务文本。
  建议录制、训练、推理阶段都尽量保持一致。

- `--dataset.push_to_hub=false`
  不上传，只保存在本地。

#### 其他部分

- `--display_data=true`
  用于显示当前观测和动作，方便你判断是否正常。

- `--play_sounds=false`
  禁用语音播报，避免因为系统缺少 `spd-say` 而报错。

### 5.4 运行后你应该看到什么

如果一切正常，通常你会看到类似下面几类信息：

- 机械臂模型加载成功
- 串口连接成功
- 相机连接成功
- 进入录制流程

在行为上，你应当看到：

- `Alicia-M follower` 会跟随 `Alicia-D leader` 的动作
- 相机图像能显示出来
- 录制完成后，程序不会抛 traceback

### 5.5 怎样算“第一次录制成功”

第一次录制不要求数据很多，只要满足下面这些，就算成功：

- 命令没有报 Python traceback
- follower 能响应 leader 的动作
- 相机图像能正常显示或被正常采集
- 录制结束后，`$DATA_ROOT` 目录下出现了数据文件

### 5.6 第一次录制失败时，不要急着改很多参数

新手最常犯的错误是：

- 一看到报错，就同时改串口、改相机、改分辨率、改模型、改控制模式

这会让你根本不知道是哪一项修好了，或者是哪一项又引入了新问题。

正确做法是：

1. 只保留 1 个相机
2. 只录 1 个 episode
3. 每次只改 1 个东西
4. 改完后重新测试

---

## 6. 正式录制数据集

当你已经成功跑通上一节的 1 个 episode 测试后，就可以开始正式采集了。

### 6.1 正式采集前的建议

第一次正式采集时，建议：

- 先采 5 到 20 个 episode
- 每个 episode 时长先控制在 20 到 60 秒
- 尽量让演示动作稳定、干净、可重复
- 宁可少录一点，也不要录很多失败示例

对模仿学习来说，数据质量通常比数据数量更重要。

### 6.2 一个正式采集命令示例

```bash
lerobot-record \
    --robot.type=alicia_m_follower \
    --robot.port=$ALICIA_M_PORT \
    --robot.version=v1_1 \
    --robot.control_aim=operation \
    --robot.control_mode=mit \
    --robot.use_interpolation=false \
    --robot.speed=50 \
    --robot.max_relative_target=8 \
    --robot.cameras="{front: {type: opencv, index_or_path: $CAMERA_DEV, width: 640, height: 480, fps: 30}}" \
    --robot.id=alicia_m \
    --teleop.type=alicia_d_leader \
    --teleop.port=$ALICIA_D_PORT \
    --teleop.gripper_type=50mm \
    --teleop.directly_controls_robot=false \
    --teleop.target_follower_type=alicia_m \
    --teleop.id=alicia_d \
    --dataset.repo_id=$DATASET_ID \
    --dataset.root=$DATA_ROOT \
    --dataset.num_episodes=10 \
    --dataset.single_task="Teleoperate Alicia-M with Alicia-D leader" \
    --dataset.episode_time_s=30 \
    --dataset.reset_time_s=15 \
    --display_data=true \
    --dataset.push_to_hub=false \
    --play_sounds=false
```

和上一节相比，这里最主要只是把：

```bash
--dataset.num_episodes=1
```

改成了：

```bash
--dataset.num_episodes=10
```

### 6.3 数据采集质量建议

下面这些建议对新手非常有用：

- 每个 episode 尽量只做一种明确任务
- 尽量让相机视角稳定，不要频繁碰摄像头
- 机械臂起始位姿尽量一致
- 如果某次演示明显失败，宁可重录，不要留进数据集
- 如果操作任务很短，也不要为了凑时长故意拖很久

### 6.4 什么时候需要 `--resume=true`

如果你已经录了一部分数据，后来中断了，想在原数据集上继续追加新的 episode，可以使用：

```bash
lerobot-record \
    ... \
    --resume=true
```

恢复录制时，下面这些东西要和原数据集保持一致：

- `--dataset.repo_id`
- `--dataset.root`
- 机器人类型
- 相机键名
- 相机分辨率
- 相机帧率
- 数据集整体特征结构

如果这些不一致，程序通常会报：

```text
Dataset metadata compatibility check failed
```

### 6.5 怎样快速检查数据有没有真的录到

录制完成后，你至少应该检查两件事：

1. `--dataset.root` 对应的目录下是否出现了新文件
2. 录制过程中是否确实有机械臂动作和相机图像

如果目录里什么都没有，或者程序一开始就报错退出，那就不是“录得不好”，而是“根本没录成功”。

---

## 7. ACT 训练

如果你是第一次训练模仿学习模型，推荐先从 `ACT` 开始。

原因很简单：

- 它比较适合做本地数据集的第一轮验证
- 用你自己的数据从零开始训练时，通常不需要先去适配复杂的预训练相机键名
- 整体上比大模型微调更容易定位问题

### 7.1 训练前先确认这几件事

开始训练前，请确认：

- 你已经成功录到数据
- 数据集路径写对了
- 训练环境里能看到 GPU
- 你知道自己录制时的相机键名是什么

如果你录制时使用的是：

```bash
--robot.cameras="{front: {...}}"
```

那么训练数据中的图像键一般就是：

```text
observation.images.front
```

### 7.2 推荐的新手 ACT 训练命令

```bash
lerobot-train \
    --dataset.repo_id=$DATASET_ID \
    --dataset.root=$DATA_ROOT \
    --dataset.video_backend=pyav \
    --policy.type=act \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --output_dir=outputs/train/act_alicia_d_to_alicia_m \
    --job_name=act_alicia_d_to_alicia_m \
    --batch_size=4 \
    --steps=50000 \
    --save_freq=5000 \
    --log_freq=100 \
    --eval_freq=0
```

如果你的 GPU 显存比较大，可以再尝试把：

```bash
--batch_size=4
```

提高到：

```bash
--batch_size=8
```

如果报显存不足，就往更小调，例如：

```bash
--batch_size=2
```

### 7.3 这条训练命令怎么理解

- `--dataset.repo_id`
  训练时要读取哪个数据集。

- `--dataset.root`
  数据集在本地的实际目录。

- `--dataset.video_backend=pyav`
  对真实机器人视频数据，通常建议显式写成 `pyav`，兼容性更稳。

- `--policy.type=act`
  表示从零创建一个 `ACT` 策略。

- `--policy.device=cuda`
  用 GPU 训练。
  如果你没有 GPU，也可以试 `cpu`，但会非常慢。

- `--policy.push_to_hub=false`
  不把模型上传到 Hugging Face。

- `--output_dir=outputs/train/act_alicia_d_to_alicia_m`
  训练输出目录。

- `--job_name=act_alicia_d_to_alicia_m`
  训练任务名字。

- `--save_freq=5000`
  每 5000 步保存一次检查点。

- `--eval_freq=0`
  关闭训练时的环境评估。
  对真机离线数据训练来说，第一次上手这样最省事。

### 7.4 训练启动后你会看到什么

正常情况下，你会看到：

- 训练配置打印出来
- 数据集开始加载
- 模型开始前向和反向训练
- loss 等日志不断刷新
- 每隔一段时间保存一次 checkpoint

### 7.5 训练结果保存在哪里

训练输出通常在：

```bash
outputs/train/act_alicia_d_to_alicia_m
```

里面通常会有：

- 配置文件
- 日志
- `checkpoints/`

而推理时最常用的路径通常是：

```bash
outputs/train/act_alicia_d_to_alicia_m/checkpoints/last/pretrained_model
```

如果你记不住全部结构，只记住这条经验也够用：

- 推理时优先找 `checkpoints/last/pretrained_model`

### 7.6 什么时候 ACT 需要 `rename_map`

如果你是：

- 用自己的 Alicia-M 数据集
- 从零训练 `ACT`
- 录制和训练都使用同一套相机键名

那通常不需要 `rename_map`。

最常见情况是：

- 录制时相机名叫 `front`
- 那么数据集键名就是 `observation.images.front`
- `ACT` 训练时直接沿用这个键名

只有在下面这种情况下，你才需要考虑 `rename_map`：

- 你录制时叫 `front`
- 但模型配置里期待的是别的图像键名

### 7.7 恢复 ACT 训练

如果训练中断了，或者你想继续在已有 checkpoint 上接着训，可以用：

```bash
lerobot-train \
    --config_path=/home/ubuntu/lerobot/outputs/train/act_alicia_d_to_alicia_m/checkpoints/last/pretrained_model \
    --resume=true \
    --steps=80000 \
    --save_freq=5000 \
    --log_freq=100
```

也可以把 `--config_path` 直接写到 `train_config.json`。

请注意：

- `--resume=true` 不能漏
- `--config_path` 不要只指到 `.../checkpoints/050000` 这种根目录
- 通常应该指向 `pretrained_model` 目录，或者它里面的 `train_config.json`

### 7.8 ACT 训练失败时先查什么

新手第一次训练报错，优先排查下面几类问题：

1. 数据集路径写错
2. GPU 不可用
3. 视频解码后端有问题
4. 批次太大导致显存不足

最稳妥的保守配置是：

```bash
--dataset.video_backend=pyav
--batch_size=2
--eval_freq=0
```

---

## 8. pi0 / pi05 训练

这一节是进阶内容。

如果你还没跑通 `ACT`，不建议一上来就先弄 `pi0` 或 `pi05`。

因为对新手来说，大模型微调比 `ACT` 更容易遇到这些问题：

- 模型下载相关问题
- 显存压力更大
- 图像键名不匹配
- 训练配置更复杂

### 8.1 什么时候适合开始试 `pi0` / `pi05`

建议在下面条件满足后再试：

- 你已经成功录到数据
- 你已经成功训练过 `ACT`
- 你已经知道自己的图像键名是什么
- 你知道如何处理 `Feature mismatch`

### 8.2 PI0 训练命令

```bash
lerobot-train \
    --dataset.repo_id=$DATASET_ID \
    --dataset.root=$DATA_ROOT \
    --dataset.video_backend=pyav \
    --policy.path=lerobot/pi0_base \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --output_dir=outputs/train/pi0_alicia_d_to_alicia_m \
    --job_name=pi0_alicia_d_to_alicia_m \
    --batch_size=1 \
    --steps=20000 \
    --save_freq=5000 \
    --log_freq=100 \
    --eval_freq=0
```

### 8.3 PI0.5 训练命令

在 LeRobot CLI 里，`PI0.5` 的名字通常写成 `pi05`：

```bash
lerobot-train \
    --dataset.repo_id=$DATASET_ID \
    --dataset.root=$DATA_ROOT \
    --dataset.video_backend=pyav \
    --policy.path=lerobot/pi05_base \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --output_dir=outputs/train/pi05_alicia_d_to_alicia_m \
    --job_name=pi05_alicia_d_to_alicia_m \
    --batch_size=1 \
    --steps=20000 \
    --save_freq=5000 \
    --log_freq=100 \
    --eval_freq=0
```

### 8.4 `Feature mismatch` 是什么意思

这类错误通常长得像：

```text
Feature mismatch between dataset/environment and policy config
```

它的本质意思是：

- 数据集里的特征名字
- 当前模型期待的特征名字

对不上。

对单相机的 Alicia-M 数据来说，如果你录制时使用：

```bash
--robot.cameras="{front: {...}}"
```

那么数据集图像键通常是：

```text
observation.images.front
```

而某些预训练模型可能期待：

```text
observation.images.base_0_rgb
```

这时你就需要训练时加：

```bash
--rename_map='{"observation.images.front":"observation.images.base_0_rgb"}'
```

### 8.5 训练和推理阶段的 `rename_map` 写法不同

这点非常容易混淆，请务必记住：

- 训练时：

```bash
--rename_map='{"observation.images.front":"observation.images.base_0_rgb"}'
```

- 推理时：

```bash
--dataset.rename_map='{"observation.images.front":"observation.images.base_0_rgb"}'
```

训练和推理的参数名不一样。

### 8.6 恢复 `pi0` / `pi05` 训练

和 `ACT` 一样，恢复训练时也是使用：

```bash
--config_path=...
--resume=true
```

例如：

```bash
lerobot-train \
    --config_path=/home/ubuntu/lerobot/outputs/train/pi0_alicia_d_to_alicia_m/checkpoints/last/pretrained_model \
    --resume=true \
    --steps=40000
```

---

## 9. 使用训练好的模型做真机推理

推理就是：

- 不再用人手实时控制 follower
- 而是让模型自己输出动作，让 `Alicia-M` 去执行

### 9.1 推理前的安全建议

第一次真机推理前，请务必注意：

- 先清空机械臂周围可能碰撞的物体
- 先让任务场景尽量简单
- 第一次只跑 `1` 个 episode
- 第一次推理时一定要有人在旁边看护
- 一旦动作异常，立刻停止

### 9.2 一个很容易踩坑的规则：评估数据集名必须以 `eval_` 开头

当你使用：

```bash
lerobot-record --policy.path=...
```

时，程序会把这次运行视为“评估 / 推理”。

这时 `repo_id` 里的数据集名字必须以 `eval_` 开头。

合法示例：

- `temp/eval_act_alicia_m`
- `ubuntu/eval_pi0_alicia_m`

不合法示例：

- `temp/act_eval`
- `ubuntu/pi0_test`

也就是说，斜杠后面的那一段名字必须以 `eval_` 开头。

### 9.3 ACT 推理命令

假设你的 `ACT` 训练输出在：

```bash
outputs/train/act_alicia_d_to_alicia_m
```

那么最常用的推理路径是：

```bash
outputs/train/act_alicia_d_to_alicia_m/checkpoints/last/pretrained_model
```

完整命令示例：

```bash
lerobot-record \
    --policy.path=outputs/train/act_alicia_d_to_alicia_m/checkpoints/last/pretrained_model \
    --policy.device=cuda \
    --robot.type=alicia_m_follower \
    --robot.port=$ALICIA_M_PORT \
    --robot.version=v1_1 \
    --robot.control_aim=operation \
    --robot.control_mode=mit \
    --robot.use_interpolation=false \
    --robot.speed=50 \
    --robot.max_relative_target=8 \
    --robot.cameras="{front: {type: opencv, index_or_path: $CAMERA_DEV, width: 640, height: 480, fps: 30}}" \
    --robot.id=alicia_m \
    --teleop.type=alicia_d_leader \
    --teleop.port=$ALICIA_D_PORT \
    --teleop.gripper_type=50mm \
    --teleop.directly_controls_robot=false \
    --teleop.target_follower_type=alicia_m \
    --teleop.id=alicia_d \
    --dataset.repo_id=temp/eval_act_alicia_m \
    --dataset.root=/home/ubuntu/Data/LerobotData/alicia_dm_eval_act \
    --dataset.num_episodes=1 \
    --dataset.single_task="Teleoperate Alicia-M with Alicia-D leader" \
    --dataset.episode_time_s=30 \
    --dataset.reset_time_s=15 \
    --display_data=true \
    --dataset.push_to_hub=false \
    --play_sounds=false
```

### 9.4 为什么推理还要写 `teleop.*`

很多新手看到推理命令里还带着 `teleop.*`，会困惑：

- “不是已经让模型控制了吗，为什么还要写 leader 的参数？”

原因是：

- 主录制阶段，动作来自模型
- 但在 episode 之间的 reset 阶段，你仍然可以借助 leader 做人工复位

如果你非常确定自己不需要这个功能，也可以去掉整组 `teleop.*` 参数。

但对新手来说，保留它们更稳妥。

### 9.5 ACT 推理时一般需不需要 `rename_map`

如果满足下面条件：

- 你用自己的 Alicia-M 数据训练的 `ACT`
- 训练和推理时使用的是同样的相机键名
- 相机数量和分辨率没变

那么通常不需要 `--dataset.rename_map`。

最典型的情况是：

- 训练时使用 `front`
- 推理时也还是 `front`

这时直接跑就行。

### 9.6 PI0 推理命令

```bash
lerobot-record \
    --policy.path=outputs/train/pi0_alicia_d_to_alicia_m/checkpoints/last/pretrained_model \
    --policy.device=cuda \
    --robot.type=alicia_m_follower \
    --robot.port=$ALICIA_M_PORT \
    --robot.version=v1_1 \
    --robot.control_aim=operation \
    --robot.control_mode=mit \
    --robot.use_interpolation=false \
    --robot.speed=50 \
    --robot.max_relative_target=8 \
    --robot.cameras="{front: {type: opencv, index_or_path: $CAMERA_DEV, width: 640, height: 480, fps: 30}}" \
    --robot.id=alicia_m \
    --teleop.type=alicia_d_leader \
    --teleop.port=$ALICIA_D_PORT \
    --teleop.gripper_type=50mm \
    --teleop.directly_controls_robot=false \
    --teleop.target_follower_type=alicia_m \
    --teleop.id=alicia_d \
    --dataset.repo_id=temp/eval_pi0_alicia_m \
    --dataset.root=/home/ubuntu/Data/LerobotData/alicia_dm_eval_pi0 \
    --dataset.num_episodes=1 \
    --dataset.single_task="Teleoperate Alicia-M with Alicia-D leader" \
    --dataset.episode_time_s=30 \
    --dataset.reset_time_s=15 \
    --display_data=true \
    --dataset.push_to_hub=false \
    --play_sounds=false
```

### 9.7 PI0.5 推理命令

```bash
lerobot-record \
    --policy.path=outputs/train/pi05_alicia_d_to_alicia_m/checkpoints/last/pretrained_model \
    --policy.device=cuda \
    --robot.type=alicia_m_follower \
    --robot.port=$ALICIA_M_PORT \
    --robot.version=v1_1 \
    --robot.control_aim=operation \
    --robot.control_mode=mit \
    --robot.use_interpolation=false \
    --robot.speed=50 \
    --robot.max_relative_target=8 \
    --robot.cameras="{front: {type: opencv, index_or_path: $CAMERA_DEV, width: 640, height: 480, fps: 30}}" \
    --robot.id=alicia_m \
    --teleop.type=alicia_d_leader \
    --teleop.port=$ALICIA_D_PORT \
    --teleop.gripper_type=50mm \
    --teleop.directly_controls_robot=false \
    --teleop.target_follower_type=alicia_m \
    --teleop.id=alicia_d \
    --dataset.repo_id=temp/eval_pi05_alicia_m \
    --dataset.root=/home/ubuntu/Data/LerobotData/alicia_dm_eval_pi05 \
    --dataset.num_episodes=1 \
    --dataset.single_task="Teleoperate Alicia-M with Alicia-D leader" \
    --dataset.episode_time_s=30 \
    --dataset.reset_time_s=15 \
    --display_data=true \
    --dataset.push_to_hub=false \
    --play_sounds=false
```

### 9.8 推理阶段如果也遇到图像键名不匹配

如果推理时报特征不匹配，就和训练时一样，需要改名映射。

例如：

```bash
--dataset.rename_map='{"observation.images.front":"observation.images.base_0_rgb"}'
```

再强调一遍：

- 训练用 `--rename_map`
- 推理用 `--dataset.rename_map`

### 9.9 怎样判断推理是否成功

推理成功通常表现为：

- 命令正常启动
- 模型成功加载
- `Alicia-M` 会按照模型输出动作运动
- 整个 episode 结束时不会抛 traceback

如果机械臂完全不动，优先排查：

- `--policy.path` 是否写对
- 模型是否真的训练完成
- 训练和推理的相机键名是否一致

---

## 10. 键盘快捷键

录制和推理过程中，常用快捷键如下：

| 快捷键 | 操作 | 说明 |
|------|------|------|
| `←` 左箭头 | 重新录制当前回合 | 清空当前回合缓冲区并重新开始 |
| `→` 右箭头 | 提前结束当前阶段 | 可用于提前结束录制或 reset 阶段 |
| `ESC` | 停止整个会话 | 退出录制或推理 |

如果你在无显示器环境、远程终端环境，或者系统不支持全局按键监听，快捷键可能不可用。

---

## 11. 最常见问题与排查方法

这一节按新手最常遇到的问题来写。

### 11.1 `ImportError: Alicia-D SDK is not available`

或者：

```text
ImportError: Alicia-M SDK is not available
```

说明当前 Python 环境里导不到 SDK。

先执行：

```bash
python -c "import alicia_d_sdk, alicia_m_sdk; print('OK')"
```

如果这条命令都不通过，就先不要继续跑 `lerobot-record`。

### 11.2 串口号写错了

症状通常是：

- 机械臂连接失败
- 命令一开始就报串口相关错误

最稳妥的排查方法：

1. 拔掉 `Alicia-D` 和 `Alicia-M`
2. 用 `ls /dev/ttyACM*` 看一遍
3. 只插 `Alicia-M` 再看一遍
4. 只插 `Alicia-D` 再看一遍

不要猜设备号。

### 11.3 摄像头号写错了

症状通常是：

- 相机初始化失败
- 读帧失败
- `Bad file descriptor`
- 分辨率设置失败

排查建议：

1. 先执行：

```bash
ls /dev/video*
```

2. 再执行：

```bash
lerobot-find-cameras opencv
```

3. 如果一只相机对应多个 `/dev/video*`，逐个测试哪个节点能正常出图

经验上：

- 同一只摄像头暴露出的多个视频节点里，不一定每个都能用
- 有时 `/dev/video4` 不行，但 `/dev/video5` 可以

### 11.4 系统没有 `spd-say`

如果出现：

```text
FileNotFoundError: [Errno 2] No such file or directory: 'spd-say'
```

最简单的解决办法就是在命令里加：

```bash
--play_sounds=false
```

### 11.5 `Feature mismatch between dataset/environment and policy config`

这类错误几乎都和图像键名不匹配有关。

先想清楚两件事：

1. 你的数据集图像键名是什么
2. 模型期望的图像键名是什么

如果你录制时相机名叫 `front`，那么数据集图像键通常就是：

```text
observation.images.front
```

如果模型期待的是别的名字，就需要：

- 训练时用 `--rename_map`
- 推理时用 `--dataset.rename_map`

### 11.6 评估数据集命名不对

如果你在推理时使用了：

```bash
lerobot-record --policy.path=...
```

那 `repo_id` 里的数据集名必须以 `eval_` 开头。

例如：

```bash
temp/eval_act_alicia_m
```

是对的，而：

```bash
temp/act_alicia_m_eval
```

通常是不对的。

### 11.7 恢复训练失败

常见原因有两个：

1. `--config_path` 指错了
2. 忘了加 `--resume=true`

优先使用这种写法：

```bash
--config_path=/home/ubuntu/lerobot/outputs/train/act_alicia_d_to_alicia_m/checkpoints/last/pretrained_model
--resume=true
```

### 11.8 CUDA 显存不足

如果你看到：

```text
CUDA out of memory
```

优先做下面几件事：

1. 把 `--batch_size` 降低
2. 关闭其他占 GPU 的程序
3. 使用 `--dataset.video_backend=pyav`

对新手来说，最保守的起点是：

```bash
--batch_size=1
```

或者：

```bash
--batch_size=2
```

### 11.9 输出目录已存在

如果训练时报：

```text
FileExistsError: Output directory ... already exists
```

说明你正试图在一个已有输出目录上启动“新的训练”。

处理方式有两种：

1. 换一个新的 `--output_dir`
2. 如果你是想接着之前的训练继续，就用 `--resume=true` 和正确的 `--config_path`

### 11.10 相机和训练时不一致

如果训练时你用的是：

- 1 个相机
- 键名叫 `front`
- 分辨率 640x480

那推理时尽量也保持：

- 还是 1 个相机
- 还是 `front`
- 还是 640x480

否则你就很可能遇到：

- 特征不匹配
- 模型输入分布变化太大
- 推理效果明显变差

### 11.11 一句话排障原则

如果你现在已经报错了，请按这个顺序排查：

1. 先确认环境和 SDK
2. 再确认串口号
3. 再确认摄像头号
4. 再确认录制能不能先跑通 1 个 episode
5. 再去考虑训练和推理

不要在“还没录通数据”的时候就同时改训练参数、模型类型和推理命令。

---

## 最后给新手的建议

如果你完全没有经验，最推荐的路线只有 3 步：

1. 先录 1 个 episode，确认系统通。
2. 再录 10 个左右干净 episode。
3. 先训练 `ACT`，再做 1 个 episode 的真机推理。

只要这 3 步能通，后面的 `pi0`、`pi05`、多相机、更多任务，都会容易很多。
