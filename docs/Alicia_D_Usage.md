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

