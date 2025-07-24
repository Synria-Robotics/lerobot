import time
import torch
import torch.nn.functional as F

from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.common.robot_devices.robots.utils import make_robot_config, make_robot_from_config


STANDARD_SHAPE = (480, 640)  # (height, width)

def busy_wait(wait_time):
    if wait_time > 0:
        time.sleep(wait_time)

def main():
    inference_time_s = 60
    fps = 30

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载机器人
    config = make_robot_config(
        robot_type="alicia_duo",
        port="/dev/ttyUSB0",  # 或留空自动搜索
        baudrate=921600
    )
    robot = make_robot_from_config(config)
    robot.connect()

    # 加载策略
    ckpt_path = "outputs/train/dp_alicia_duo_grasp/checkpoints/100000/pretrained_model"
    policy = DiffusionPolicy.from_pretrained(ckpt_path)
    policy.to(device)
    policy.eval()

    for _ in range(inference_time_s * fps):
        start_time = time.perf_counter()

        # 读取观测
        observation = robot.capture_observation()

        # 处理观测为pytorch格式
        for name in observation:
            if "image" in name:
                observation[name] = observation[name].type(torch.float32) / 255
                observation[name] = observation[name].permute(2, 0, 1).contiguous()
                c, h, w = observation[name].shape
                if (h, w) != STANDARD_SHAPE:
                    observation[name] = F.interpolate(
                        observation[name].unsqueeze(0),
                        size=STANDARD_SHAPE,
                        mode="bilinear",
                        align_corners=False
                    ).squeeze(0)

            observation[name] = observation[name].unsqueeze(0)
            observation[name] = observation[name].to(device)

        # 推理动作
        with torch.inference_mode():
            action = policy.select_action(observation)
        action = action.squeeze(0).to("cpu")

        robot.send_action(action)
        # time.sleep(2)
        # 控制频率
        dt_s = time.perf_counter() - start_time
        busy_wait(1 / fps - dt_s)

    robot.disconnect()

if __name__ == "__main__":
    main()