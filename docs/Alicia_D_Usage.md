# Alicia-D Robot Integration with LeRobot

This guide provides comprehensive instructions for using Alicia-D robotic arms with the LeRobot framework for dataset recording and policy training.

## Table of Contents

- [Installation](#installation)
- [Hardware Setup](#hardware-setup)
- [Dataset Recording](#dataset-recording)
- [Policy Training](#policy-training)
- [Policy Evaluation](#policy-evaluation)
- [Keyboard Shortcuts](#keyboard-shortcuts)
- [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites

- Python 3.10
- Conda (recommended) or virtual environment
- Ubuntu Linux (recommended) or compatible Linux distribution

### Step 1: Create Conda Environment

```bash
conda create -n lerobot python=3.10
conda activate lerobot
```

### Step 2: Install Alicia-D SDK

```bash
# Create workspace directory
mkdir -p alicia_lerobot
cd alicia_lerobot
```

### Step 3: Install LeRobot

```bash
# Clone and install LeRobot
git clone https://github.com/Synria-Robotics/lerobot.git -b v6.1.0
cd lerobot
pip install -e .
```

### Step 4: Verify Installation

```bash
# Test that commands are available
lerobot-record --help
lerobot-train --help
```

---

## Hardware Setup

### Connection Requirements

1. **Follower Arm(s)**: Connect the Type-C USB cable from the follower arm(s) to your computer
2. **Leader Arm(s)**: Leader arms are connected to follower arms via hardware control wire (no computer connection needed)
3. **Cameras**: Connect cameras to your computer

### Port Detection

Use the following command to identify available serial ports:

```bash
lerobot-find-port
```

Common port locations:
- Linux: `/dev/ttyACM0`, `/dev/ttyACM1`, `/dev/ttyUSB0`
- Check camera devices: `ls /dev/video*`

---

## Dataset Recording

### Overview

Alicia-D leader arms can control follower arms in two modes:

1. **Direct Hardware Control (Default)**: Leader arms directly control follower arms via hardware control wire, bypassing the computer. During recording, the system:
   - Reads joint positions from the follower arm (which reflects leader commands)
   - Captures camera images
   - Records actions based on follower observations (since leader directly controls follower)

2. **Computer-Mediated Control**: Actions are sent through the computer from teleoperator to robot. This mode is useful when:
   - Leader and follower are not physically connected via hardware wire
   - You want to add processing/filtering of actions before sending to robot
   - Testing or debugging scenarios

The control mode is controlled by the `--teleop.directly_controls_robot` parameter (default: `true`).

### Single Arm Configuration

**Command:**

```bash
lerobot-record \
    --robot.type=alicia_d_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.cameras="{laptop: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30}}" \
    --robot.id=black \
    --teleop.type=alicia_d_leader \
    --teleop.id=leader_arm \
    --dataset.repo_id=ubuntu/grab-cube-dataset \
    --dataset.root=/home/ubuntu/Data/LerobotData \
    --dataset.num_episodes=10 \
    --dataset.single_task="Grab the cube" \
    --dataset.episode_time_s=60 \
    --dataset.reset_time_s=30 \
    --display_data=true \
    --dataset.push_to_hub=false
```

**Key Parameters:**
- `--robot.port`: Serial port of the follower arm (use `lerobot-find-port` to detect)
- `--teleop.directly_controls_robot`: Control mode (default: `true`). Set to `false` for computer-mediated control (requires `--teleop.port`)
- `--dataset.repo_id`: Dataset repository identifier (format: `username/dataset-name`)
- `--dataset.num_episodes`: Number of episodes to record

**Computer-Mediated Control:** Add `--teleop.directly_controls_robot=false --teleop.port=/dev/ttyACM0` to the command above.

### Dual Arm (Bimanual) Configuration

**Command:**

```bash
lerobot-record \
    --robot.type=bi_alicia_d_follower \
    --robot.left_arm_port=/dev/ttyACM0 \
    --robot.right_arm_port=/dev/ttyACM1 \
    --robot.cameras='{
        camera1: {type: opencv, index_or_path: /dev/video10, width: 640, height: 480, fps: 30},
        camera2: {type: opencv, index_or_path: /dev/video18, width: 640, height: 480, fps: 30},
        camera3: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30}
    }' \
    --robot.id=bimanual_follower \
    --teleop.type=bi_alicia_d_leader \
    --teleop.id=bimanual_leader \
    --dataset.repo_id=ubuntu/bimanual-grab-cube-dataset \
    --dataset.root=/home/ubuntu/Data/LerobotData/test2 \
    --dataset.num_episodes=20 \
    --dataset.single_task="Grab the cloth with both arms" \
    --dataset.episode_time_s=60 \
    --dataset.reset_time_s=30 \
    --dataset.chunks_size=10 \
    --dataset.data_files_size_in_mb=50 \
    --dataset.video_files_size_in_mb=100 \
    --display_data=true \
    --dataset.push_to_hub=false
```

**Key Parameters:**
- `--robot.left_arm_port` / `--robot.right_arm_port`: Serial ports for follower arms
- `--teleop.directly_controls_robot`: Control mode (default: `true`). Set to `false` for computer-mediated control (requires `--teleop.left_arm_port` and `--teleop.right_arm_port`)

**Computer-Mediated Control:** Add `--teleop.directly_controls_robot=false --teleop.left_arm_port=/dev/ttyACM2 --teleop.right_arm_port=/dev/ttyACM3` to the command above.

### Resuming Recording

To continue recording on an existing dataset:

```bash
lerobot-record \
    ... \
    --resume=true
```

This will:
- Load the existing dataset
- Continue from the last episode
- Maintain dataset compatibility

**Note:** Ensure your robot configuration matches the original recording setup.

---

## Policy Training

### Overview

Train imitation learning policies (ACT, Diffusion Policy, etc.) on your recorded datasets.

### ACT Policy Training

**Basic Command:**

```bash
lerobot-train \
    --dataset.repo_id=ubuntu/bimanual-grab-cube-dataset \
    --dataset.root=/home/ubuntu/Data/LerobotData/test2 \
    --dataset.video_backend=pyav \
    --policy.type=act \
    --policy.push_to_hub=false \
    --output_dir=outputs/train/act_bimanual_grab_cube \
    --job_name=act_bimanual_grab_cube \
    --policy.device=cuda \
    --wandb.enable=true \
    --wandb.project=alicia-d-bimanual \
    --steps=50000 \
    --batch_size=8 \
    --save_freq=5000 \
    --log_freq=100 \
    --eval_freq=5000
```

**Note:** If you encounter CUDA out of memory errors, reduce `--batch_size` (try 4, 8, or 16). For bimanual setups with multiple cameras, smaller batch sizes are often necessary.

### Resuming Training

To resume from a checkpoint, add `--config_path` pointing to the checkpoint directory (or `train_config.json` file):

```bash
lerobot-train \
    --config_path=/home/ubuntu/Alicia/lerobot/outputs/train/act_bimanual_grab_cube/checkpoints/050000 \
    --dataset.repo_id=ubuntu/bimanual-grab-cube-dataset \
    --dataset.root=/home/ubuntu/Data/LerobotData/test2 \
    --dataset.video_backend=pyav \
    --policy.type=act \
    --policy.push_to_hub=false \
    --output_dir=outputs/train/act_bimanual_grab_cube \
    --job_name=act_bimanual_grab_cube \
    --policy.device=cuda \
    --wandb.enable=true \
    --wandb.project=alicia-d-bimanual \
    --steps=100000 \
    --batch_size=8 \
    --save_freq=5000 \
    --log_freq=100 \
    --eval_freq=5000
```

**Note:** Use absolute paths for `--config_path`. You can change training parameters (e.g., `--steps`) when resuming.

### Diffusion Policy Training

**Command:**

```bash
lerobot-train \
    --dataset.repo_id=ubuntu/bimanual-grab-cube-dataset \
    --dataset.root=/home/ubuntu/Data/LerobotData/test2 \
    --dataset.video_backend=pyav \
    --policy.type=diffusion \
    --policy.push_to_hub=false \
    --output_dir=outputs/train/dp_bimanual_grab_cube \
    --job_name=dp_bimanual_grab_cube \
    --policy.device=cuda \
    --wandb.enable=true \
    --wandb.project=alicia-d-bimanual \
    --steps=50000 \
    --batch_size=8 \
    --save_freq=5000 \
    --log_freq=100 \
    --eval_freq=5000
```

**Note:** Diffusion Policy typically requires more memory than ACT. Start with `--batch_size=4` or `--batch_size=8` and increase if memory allows. To resume, add `--config_path` as shown in the ACT example above.

### Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--dataset.repo_id` | Dataset repository ID | Required |
| `--dataset.root` | Local dataset path (faster loading) | Cache directory |
| `--dataset.video_backend` | Video decoder: `pyav` or `torchcodec` | Auto-detect |
| `--policy.type` | Policy type: `act`, `diffusion`, etc. | Required |
| `--policy.device` | Device: `cuda` or `cpu` | `cpu` |
| `--policy.push_to_hub` | Push model to Hugging Face Hub after training | `true` |
| `--steps` | Number of training steps | 50000 |
| `--batch_size` | Batch size (reduce if CUDA OOM: try 4, 8, or 16) | 32 |
| `--save_freq` | Checkpoint save frequency | 5000 |
| `--log_freq` | Logging frequency | 100 |
| `--eval_freq` | Evaluation frequency (0 to disable) | 5000 |

### Video Backend Selection

The `--dataset.video_backend` parameter selects the video decoder:

- **`pyav`** (Recommended): Stable, works with system FFmpeg, more compatible
- **`torchcodec`**: Faster but requires specific FFmpeg library versions

If you encounter FFmpeg library errors, use `--dataset.video_backend=pyav`.

### Hugging Face Hub Configuration

To push datasets or models to the Hugging Face Hub, you need to authenticate first.

#### Step 1: Install Hugging Face CLI (if not already installed)

```bash
pip install huggingface_hub
```

#### Step 2: Create Hugging Face Account (if needed)

If you don't have a Hugging Face account yet:

1. Go to https://huggingface.co/join
2. Sign up with your email or GitHub account
3. **Choose account type:**
   - **Personal Account** (default): Free tier, suitable for individual projects and research
   - **Classroom Organization**: For educational institutions and classrooms (free, requires verification)
   - **Non-profit Organization**: For registered non-profit organizations (free, requires verification)

**For most users:** A personal account is sufficient and provides free access to:
- Unlimited public repositories (datasets and models)
- Private repositories (limited number on free tier)
- All basic Hub features needed for LeRobot

**For educational use:** If you're part of a school/university, consider creating a Classroom organization for:
- Centralized workspace for students
- Collaborative datasets and models
- Educational resources and demos

**For non-profit organizations:** If you're a registered non-profit, you can apply for non-profit status for:
- Enhanced collaboration features
- Priority support
- Additional resources

#### Step 3: Login to Hugging Face Hub

```bash
hf auth login
```

This will prompt you to:
1. Enter your Hugging Face token (get it from https://huggingface.co/settings/tokens)
2. Choose whether to save the token to your git credentials

**Getting a Hugging Face Token:**
1. Go to https://huggingface.co/settings/tokens
2. Click "New token"
3. **Select token type:**
   - **Read/Write token** (Recommended): Simple and sufficient for most users. Provides full read and write access to your repositories.
   - **Fine-grained token** (Advanced): More secure with granular permissions. Use if you need to restrict access to specific repositories or resources.
4. Copy the token
5. Paste it when prompted by `hf auth login`

**Recommendation:** For LeRobot usage (pushing datasets and models), a **Read/Write token** is recommended as it's simpler and provides all necessary permissions. Use fine-grained tokens only if you need specific access restrictions for security purposes.

#### Step 4: Verify Authentication

```bash
hf whoami
```

This should display your Hugging Face username if authentication is successful.

#### Alternative: Using Environment Variable

Instead of `hf auth login`, you can set the token as an environment variable:

```bash
export HF_TOKEN="your_token_here"
```

Or add it to your `~/.bashrc` or `~/.zshrc`:

```bash
echo 'export HF_TOKEN="your_token_here"' >> ~/.bashrc
source ~/.bashrc
```

### Pushing Datasets to Hub

When recording datasets with `--dataset.push_to_hub=true`:

```bash
lerobot-record \
    --robot.type=bi_alicia_d_follower \
    --robot.left_arm_port=/dev/ttyACM1 \
    --robot.right_arm_port=/dev/ttyACM0 \
    --robot.cameras='{
        right_wrist: {type: opencv, index_or_path: /dev/video10, width: 640, height: 480, fps: 30},
        left_wrist: {type: opencv, index_or_path: /dev/video18, width: 640, height: 480, fps: 30},
        top: {type: opencv, index_or_path: /dev/video24, width: 640, height: 480, fps: 30},
        front: {type: opencv, index_or_path: /dev/video12, width: 640, height: 480, fps: 30}
    }' \
    --robot.id=bimanual_follower \
    --teleop.type=bi_alicia_d_leader \
    --teleop.id=bimanual_leader \
    --dataset.repo_id=ubuntu/bimanual-grab-cube-dataset \
    --dataset.root=/home/ubuntu/Data/LerobotData/cloth1 \
    --dataset.num_episodes=10 \
    --dataset.single_task="Grab the cloth with both arms" \
    --dataset.episode_time_s=48 \
    --dataset.reset_time_s=5 \
    --display_data=true \
    --dataset.push_to_hub=true \
    --dataset.private=false
```

**Key Parameters:**
- `--dataset.repo_id`: Repository ID in format `username/dataset-name` (e.g., `ubuntu/bimanual-grab-cube-dataset`)
- `--dataset.push_to_hub`: Set to `true` to push after recording completes
- `--dataset.private`: Set to `true` for private repositories, `false` for public (default: `false`)
- `--dataset.tags`: Optional list of tags for the dataset (e.g., `--dataset.tags="['robotics', 'bimanual', 'manipulation']"`)

**Note:** The dataset is always saved locally first, then pushed to the Hub after recording completes.

### Disabling Model Upload to Hub

By default, LeRobot attempts to push trained models to the Hugging Face Hub after training completes. If you don't want to upload models (e.g., for local-only training), set:

```bash
--policy.push_to_hub=false
```

**Note:** If `push_to_hub=true` (default), you must either:
- Have Hugging Face authentication configured (`hf auth login`)
- Or set `--policy.push_to_hub=false` to avoid authentication errors

Models are always saved locally in the `output_dir` directory regardless of this setting.

---

## Policy Evaluation

### Overview

After training a policy, you can evaluate it on the real robot using the evaluation script. The evaluation script loads a trained policy checkpoint and runs it on the robot, optionally recording evaluation episodes to a dataset.

### Single Arm Evaluation

**Command:**

```bash
python examples/alicia/eval_alicia_arms.py \
    --policy.path=outputs/train/act_grab_cube/checkpoints/last/pretrained_model \
    --robot.type=alicia_d_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.cameras="{front: {type: opencv, index_or_path: /dev/video12, width: 640, height: 480, fps: 30}}" \
    --policy.device=cuda \
    --task="Grab the cube" \
    --duration=120 \
    --fps=10 \
    --num_episodes=5 \
    --record_eval=false
```

**Key Parameters:**
- `--policy.path`: Path to the trained policy checkpoint directory (e.g., `outputs/train/act_grab_cube/checkpoints/last/pretrained_model` or `outputs/train/act_grab_cube/checkpoints/050000/pretrained_model`)
- `--robot.port`: Serial port of the follower arm
- `--task`: Task description (should match the task used during training)
- `--duration`: Duration of each evaluation episode in seconds
- `--fps`: Action execution frequency (Hz)
- `--num_episodes`: Number of evaluation episodes to run
- `--record_eval`: Whether to record evaluation episodes to a dataset (`true` or `false`)
- `--eval_dataset_repo_id`: Dataset repository ID for recording evaluation episodes (required if `record_eval=true`)

### Dual Arm (Bimanual) Evaluation

**Command:**

```bash
python examples/alicia/eval_alicia_arms.py \
    --policy.path=outputs/train/act_bimanual_grab_cube/checkpoints/last/pretrained_model \
    --robot.type=bi_alicia_d_follower \
    --robot.left_arm_port=/dev/ttyACM1 \
    --robot.right_arm_port=/dev/ttyACM0 \
    --robot.cameras='{
        right_wrist: {type: opencv, index_or_path: /dev/video10, width: 640, height: 480, fps: 30},
        left_wrist: {type: opencv, index_or_path: /dev/video18, width: 640, height: 480, fps: 30},
        top: {type: opencv, index_or_path: /dev/video24, width: 640, height: 480, fps: 30},
        front: {type: opencv, index_or_path: /dev/video12, width: 640, height: 480, fps: 30}
    }' \
    --policy.device=cuda \
    --task="Grab and handover the red cube to the other arm" \
    --duration=120 \
    --fps=10 \
    --num_episodes=5 \
    --record_eval=false
```

**Key Parameters:**
- `--robot.left_arm_port` / `--robot.right_arm_port`: Serial ports for follower arms
- `--robot.cameras`: Camera configuration (should match training setup)

### Recording Evaluation Episodes

To record evaluation episodes for later analysis:

```bash
python examples/alicia/eval_alicia_arms.py \
    --policy.path=outputs/train/act_bimanual_grab_cube/checkpoints/last/pretrained_model \
    --robot.type=bi_alicia_d_follower \
    --robot.left_arm_port=/dev/ttyACM1 \
    --robot.right_arm_port=/dev/ttyACM0 \
    --robot.cameras='{
        right_wrist: {type: opencv, index_or_path: /dev/video10, width: 640, height: 480, fps: 30},
        left_wrist: {type: opencv, index_or_path: /dev/video18, width: 640, height: 480, fps: 30},
        top: {type: opencv, index_or_path: /dev/video24, width: 640, height: 480, fps: 30},
        front: {type: opencv, index_or_path: /dev/video12, width: 640, height: 480, fps: 30}
    }' \
    --policy.device=cuda \
    --task="Grab and handover the red cube to the other arm" \
    --duration=120 \
    --fps=10 \
    --num_episodes=5 \
    --record_eval=true \
    --eval_dataset_repo_id=ubuntu/eval_bimanual_grab_cube
```

**Note:** When `record_eval=true`, the evaluation episodes are saved to the specified dataset repository and can be pushed to the Hugging Face Hub for analysis.

### Evaluation Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--policy.path` | Path to trained policy checkpoint directory | Required |
| `--robot.type` | Robot type: `alicia_d_follower` or `bi_alicia_d_follower` | Required |
| `--robot.port` | Serial port (single arm) | Required |
| `--robot.left_arm_port` / `--robot.right_arm_port` | Serial ports (bimanual) | Required |
| `--robot.cameras` | Camera configuration | Required |
| `--policy.device` | Device: `cuda` or `cpu` | `cpu` |
| `--task` | Task description (should match training) | `""` |
| `--duration` | Duration per episode (seconds) | `120.0` |
| `--fps` | Action execution frequency (Hz) | `10.0` |
| `--num_episodes` | Number of evaluation episodes | `5` |
| `--record_eval` | Record evaluation episodes to dataset | `false` |
| `--eval_dataset_repo_id` | Dataset repo ID for recording | `"temp/eval_not_saved"` |
| `--display_data` | Display observations/actions in rerun | `true` |

### Policy Checkpoint Paths

The `--policy.path` parameter accepts:
- **Local checkpoint directory**: `outputs/train/act_bimanual_grab_cube/checkpoints/last/pretrained_model` (symlink to latest checkpoint)
- **Specific checkpoint**: `outputs/train/act_bimanual_grab_cube/checkpoints/050000/pretrained_model` (specific checkpoint number)
- **Hugging Face Hub model**: `username/model_name` (if model was pushed to hub)

**Note:** Use `last` to automatically use the latest checkpoint, or specify a checkpoint number (e.g., `050000`) to use a specific checkpoint.

### Tips for Evaluation

1. **Match Training Configuration**: Ensure robot configuration (ports, cameras) matches the training setup
2. **Task Description**: Use the same task description as during training for best results
3. **FPS Consistency**: Use the same FPS as training (typically 10 Hz for ACT policies)
4. **Visualization**: Set `--display_data=true` to visualize policy behavior in rerun
5. **Recording**: Set `--record_eval=true` to save evaluation episodes for analysis

---

## Keyboard Shortcuts

During dataset recording, the following keyboard shortcuts are available:

| Shortcut | Action | Description |
|----------|--------|-------------|
| **← (Left Arrow)** | Re-record episode | Clears current episode buffer and restarts recording for the same episode number |
| **→ (Right Arrow)** | Exit early | Exits the current loop (recording or reset phase) |
| **ESC** | Stop recording | Stops the entire data recording session |

### Re-recording Episodes

If an episode goes wrong:

1. Press **← (Left Arrow)** during or after recording
2. The episode buffer is cleared
3. Recording restarts for the same episode number
4. The episode counter does not increment

This allows you to discard bad episodes and re-record them without affecting the total episode count.

---

## Troubleshooting

### Common Issues

#### 1. FFmpeg/TorchCodec Library Error

**Error:** `RuntimeError: Could not load libtorchcodec`

**Solution:** Use `pyav` backend:
```bash
--dataset.video_backend=pyav
```

#### 2. Port Not Found

**Error:** `ConnectionError: Failed to connect to Alicia-D robot`

**Solution:**
- Check port with: `lerobot-find-port`
- Verify USB cable connection
- Check permissions: `sudo usermod -a -G dialout $USER` (logout/login required)

#### 3. Camera Not Detected

**Error:** Camera initialization fails

**Solution:**
- List cameras: `ls /dev/video*`
- Check camera permissions
- Verify camera is not used by another process

#### 4. Dataset Compatibility Error

**Error:** `ValueError: Dataset metadata compatibility check failed`

**Solution:**
- Ensure robot configuration matches original recording
- Check FPS, features, and robot type match

#### 5. Hugging Face Hub Authentication Error

**Error:** `401 Client Error: Unauthorized for url: https://huggingface.co/api/repos/create`

**Solution:** Authenticate with Hugging Face Hub:
```bash
# Install huggingface_hub if not already installed
pip install huggingface_hub

# Login to Hugging Face Hub
hf auth login
```

Enter your Hugging Face token when prompted (get it from https://huggingface.co/settings/tokens).

**Alternative:** Set token as environment variable:
```bash
export HF_TOKEN="your_token_here"
```

**To disable uploads instead:**
```bash
--dataset.push_to_hub=false  # For datasets
--policy.push_to_hub=false   # For models
```

#### 6. CUDA Out of Memory Error

**Error:** `torch.OutOfMemoryError: CUDA out of memory`

**Solution:** Reduce batch size:
```bash
--batch_size=8  # or try 4 or 16
```

**Additional memory optimization tips:**
- For bimanual setups with multiple cameras, start with `--batch_size=4` or `--batch_size=8`
- Clear GPU cache: `torch.cuda.empty_cache()` (if modifying code)
- Reduce image resolution in dataset recording (e.g., 320x240 instead of 640x480)
- Use gradient accumulation to maintain effective batch size with smaller batches
- Close other GPU-intensive applications

#### 7. Teleoperator Connection Issues

**Error:** `DeviceNotConnectedError` or actions not being sent to robot

**Solution:** 
- **Hardware wire connected (default):** Use `--teleop.directly_controls_robot=true` (or omit)
- **Not physically connected:** Use `--teleop.directly_controls_robot=false` and specify `--teleop.port` (single arm) or `--teleop.left_arm_port`/`--teleop.right_arm_port` (bimanual)

### Getting Help

- **Official Documentation:** [LeRobot Documentation](https://huggingface.co/docs/lerobot/il_robots#record-a-dataset)
- **GitHub Issues:** [LeRobot Issues](https://github.com/huggingface/lerobot/issues)
- **Discord:** [LeRobot Discord](https://discord.gg/3gxM6Avj)

---

## Additional Resources

- [Alicia-D Product Manual](https://docs.sparklingrobo.com/)
- [LeRobot Policy Documentation](https://huggingface.co/docs/lerobot/bring_your_own_policies)
- [LeRobot Hardware Integration Guide](https://huggingface.co/docs/lerobot/integrate_hardware)
