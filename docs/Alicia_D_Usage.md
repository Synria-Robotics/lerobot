

# Installation


Create a conda environment:
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

Plug the type-C wire(s) to follower arm(s), connecting it(them) with the computer.
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

Replace the robot port and camera index without your own.



## For dual Alicia-D followers

```
lerobot-record \
    --robot.type=bi_alicia_d_follower \
    --robot.left_arm_port=/dev/ttyACM0 \
    --robot.right_arm_port=/dev/ttyACM1 \
    --robot.cameras='{
        camera1: {type: opencv, index_or_path: /dev/video10, width: 640, height: 480, fps: 30},
        camera2: {type: opencv, index_or_path: /dev/video16, width: 640, height: 480, fps: 30},
        camera3: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30}
    }' \
    --robot.id=bimanual_follower \
    --teleop.type=bi_alicia_d_leader \
    --teleop.id=bimanual_leader \
    --dataset.repo_id=ubuntu/bimanual-grab-cube-dataset \
    --dataset.num_episodes=2 \
    --dataset.single_task="Grab the cube with both arms" \
    --display_data=true \
    --dataset.push_to_hub=false
```

Replace the robot port and camera index without your own.
