# Alicia-D Robot Integration with LeRobot

This guide provides comprehensive instructions for using Alicia-D robotic arms with the LeRobot framework for dataset recording and policy training.

## Table of Contents

- [Installation](#installation)
- [Hardware Setup](#hardware-setup)
- [Dataset Recording](#dataset-recording)
- [Policy Training](#policy-training)
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

# Clone and install Alicia-D SDK
git clone https://github.com/Synria-Robotics/Alicia-D-SDK.git -b v6.1.0
cd Alicia-D-SDK
pip install -e .
cd ..
```

### Step 3: Install LeRobot

```bash
# Clone and install LeRobot
git clone https://github.com/Synria-Robotics/lerobot.git -b v6.1.0-beta1
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
3. **Cameras**: Connect USB cameras to your computer

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

Alicia-D leader arms directly control follower arms via hardware control wire, bypassing the computer. During recording, the system:
- Reads joint positions from the follower arm (which reflects leader commands)
- Captures camera images
- Records actions based on follower observations (since leader directly controls follower)

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

**Parameters:**
- `--robot.port`: Serial port of the follower arm (use `lerobot-find-port` to detect)
- `--robot.cameras`: Camera configuration dictionary
- `--dataset.repo_id`: Dataset repository identifier (format: `username/dataset-name`)
- `--dataset.root`: Local directory to save dataset (optional, defaults to cache)
- `--dataset.num_episodes`: Number of episodes to record
- `--dataset.episode_time_s`: Duration of each episode in seconds
- `--dataset.reset_time_s`: Time for environment reset between episodes

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

**Additional Parameters:**
- `--robot.left_arm_port`: Serial port for left follower arm
- `--robot.right_arm_port`: Serial port for right follower arm
- `--dataset.chunks_size`: Maximum number of files per chunk directory (default: 1000)
- `--dataset.data_files_size_in_mb`: Maximum size for data parquet files in MB (default: 100)
- `--dataset.video_files_size_in_mb`: Maximum size for video files in MB (default: 200)

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
    --batch_size=32 \
    --save_freq=5000 \
    --log_freq=100 \
    --eval_freq=5000
```

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
    --batch_size=32 \
    --save_freq=5000 \
    --log_freq=100 \
    --eval_freq=5000
```

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
| `--batch_size` | Batch size | 32 |
| `--save_freq` | Checkpoint save frequency | 5000 |
| `--log_freq` | Logging frequency | 100 |
| `--eval_freq` | Evaluation frequency (0 to disable) | 5000 |

### Video Backend Selection

The `--dataset.video_backend` parameter selects the video decoder:

- **`pyav`** (Recommended): Stable, works with system FFmpeg, more compatible
- **`torchcodec`**: Faster but requires specific FFmpeg library versions

If you encounter FFmpeg library errors, use `--dataset.video_backend=pyav`.

### Disabling Model Upload to Hub

By default, LeRobot attempts to push trained models to the Hugging Face Hub after training completes. If you don't want to upload models (e.g., for local-only training), set:

```bash
--policy.push_to_hub=false
```

**Note:** If `push_to_hub=true` (default), you must either:
- Have Hugging Face authentication configured (`huggingface-cli login`)
- Or set `--policy.push_to_hub=false` to avoid authentication errors

Models are always saved locally in the `output_dir` directory regardless of this setting.

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

**Solution:** Disable model upload to Hub:
```bash
--policy.push_to_hub=false
```

Alternatively, authenticate with Hugging Face:
```bash
huggingface-cli login
```

### Getting Help

- **Official Documentation:** [LeRobot Documentation](https://huggingface.co/docs/lerobot/il_robots#record-a-dataset)
- **GitHub Issues:** [LeRobot Issues](https://github.com/huggingface/lerobot/issues)
- **Discord:** [LeRobot Discord](https://discord.gg/3gxM6Avj)

---

## Additional Resources

- [Alicia-D Product Manual](https://docs.sparklingrobo.com/)
- [LeRobot Policy Documentation](https://huggingface.co/docs/lerobot/bring_your_own_policies)
- [LeRobot Hardware Integration Guide](https://huggingface.co/docs/lerobot/integrate_hardware)
