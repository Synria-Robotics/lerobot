
import subprocess

cmd = [
    "python", "/home/ubuntu/alerobot/lerobot/scripts/train.py",
    "--policy.type=diffusion",
    "--dataset.repo_id=/home/ubuntu/Github/alicia_recorder/ledata/training_standardized",
    "--output_dir=outputs/train/dp_alicia_duo_grasp"
    # "--batch_size=4",
    # "--steps=5"
]

subprocess.run(cmd, check=True)