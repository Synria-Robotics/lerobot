

# Installation


Create conda environment
```
conda create -n lerobot python=3.10
conda activate lerobot
```


Enter the path to the repositories destination:

1. install Alicia-D-SDK
```
mkdir -p alicia_lerobot
cd alicia_lerobot
git clone https://github.com/Synria-Robotics/Alicia-D-SDK.git -b v6.1.0
cd Alicia-D-SDK
pip install -e .
```

2. install lerobot
```
cd .. # back to alicia_lerobot
git clone https://github.com/Synria-Robotics/lerobot.git -v6.1.0-beta1
cd lerobot
pip install -e .
```






# Record Dataset


## For single Alicia-D follower

```
lerobot-record \
    --robot.type=alicia_d_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.cameras="{laptop: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30}}" \
    --robot.id=black \
    --teleop.type=alicia_d_leader \
    --teleop.id=leader_arm \
    --dataset.repo_id=ubuntu/grab-cube-dataset \
    --dataset.num_episodes=2 \
    --dataset.single_task="Grab the cube" \
    --display_data=true \
    --dataset.push_to_hub=false
```

