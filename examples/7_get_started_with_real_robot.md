# 实操机器人入门指南

本教程将指导您完成设置和训练神经网络以自主控制真实机器人的过程。

**您将学习的内容：**
1. 如何订购和组装机器人。
2. 如何连接、配置和校准机器人。
3. 如何记录和可视化您的数据集。
4. 如何使用您的数据训练策略并准备评估。
5. 如何评估您的策略并可视化结果。

通过以下步骤，您将能够以高成功率复制诸如拾取乐高积木并将其放入箱中的任务，如[此视频](https://x.com/RemiCadene/status/1814680760592572934)所示。

本教程专门为经济实惠的[Koch v1.1](https://github.com/jess-moss/koch-v1-1)机器人设计，但它包含了额外信息，通过更改一些配置可以轻松适应各种类型的机器人，如[Aloha双臂机器人](https://aloha-2.github.io)。Koch v1.1由一个引导臂和一个跟随臂组成，每个臂有6个电机。它可以与一个或多个摄像头配合使用来记录场景，这些摄像头作为机器人的视觉传感器。

在数据收集阶段，您将通过移动引导臂来控制跟随臂。这一过程被称为"远程操作"。这种技术用于收集机器人轨迹。之后，您将训练神经网络模仿这些轨迹，并部署该网络以使您的机器人能够自主运行。

如果您在教程的任何步骤中遇到问题，请随时在[Discord](https://discord.com/invite/s3KuuzsPFb)上寻求帮助，或者通过创建问题或拉取请求与我们一起迭代改进本教程。谢谢！

## 1. 订购并组装Koch v1.1

按照[Koch v1.1 Github页面](https://github.com/jess-moss/koch-v1-1)提供的采购和组装说明进行操作。这将指导您设置引导臂和跟随臂，如下图所示。

<div style="text-align:center;">
  <img src="../media/tutorial/koch_v1_1_leader_follower.webp?raw=true" alt="Koch v1.1引导臂和跟随臂" title="Koch v1.1引导臂和跟随臂" width="50%">
</div>

想要了解组装过程的可视化演示，您可以参考[此视频教程](https://youtu.be/8nQIg9BwwTk)。

## 2. 配置电机，校准手臂，远程操作您的Koch v1.1

首先，通过运行以下命令之一安装Koch v1.1等使用dynamixel电机构建的机器人所需的额外依赖项（确保已安装gcc）。

使用`pip`：
```bash
pip install -e ".[dynamixel]"
```

使用`poetry`：
```bash
poetry sync --extras "dynamixel"
```

使用`uv`：
```bash
uv sync --extra "dynamixel"
```

现在您可以将5V电源插入引导臂（较小的那个）的电机总线，因为它的所有电机只需要5V。

然后将12V电源插入跟随臂的电机总线。它有两个需要12V的电机，其余的将通过电压转换器以5V供电。

最后，通过USB将两个臂都连接到您的计算机。请注意，USB不提供任何电源，两个臂都需要插入各自的电源才能被您的计算机检测到。

现在您已经准备好首次配置电机，详情如下节所述。在接下来的部分中，您将通过在交互式Python会话中运行一些Python代码，或者将其复制粘贴到Python文件中，来了解我们的类和函数。

如果您已经首次配置了电机，可以通过直接运行远程操作脚本来简化流程（详情在教程后面部分）：

> **注意：** 要可视化数据，请启用`--control.display_data=true`。这将使用`rerun`流式传输数据。

```bash
python lerobot/scripts/control_robot.py \
  --robot.type=koch \
  --control.type=teleoperate
```

它将自动：
1. 识别任何缺失的校准并启动校准程序。
2. 连接机器人并开始远程操作。

### a. 使用DynamixelMotorsBus控制您的电机

您可以使用[`DynamixelMotorsBus`](../lerobot/common/robot_devices/motors/dynamixel.py)与连接到相应USB总线的电机链进行通信。这个类利用Python [Dynamixel SDK](https://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_sdk/sample_code/python_read_write_protocol_2_0/#python-read-write-protocol-20)来简化对电机的读写操作。

**首次配置您的电机**

您需要依次拔掉每个电机并运行一个命令来识别电机。电机将保存自己的标识，所以您只需执行一次。首先拔掉所有电机。

先从引导臂开始，因为它的所有电机都是同一类型。插入引导臂的第一个电机，然后运行此脚本将其ID设置为1。
```bash
python lerobot/scripts/configure_motor.py \
  --port /dev/tty.usbmodem58760432961 \
  --brand dynamixel \
  --model xl330-m288 \
  --baudrate 1000000 \
  --ID 1
```

然后拔掉第一个电机，插入第二个电机并将其ID设置为2。
```bash
python lerobot/scripts/configure_motor.py \
  --port /dev/tty.usbmodem58760432961 \
  --brand dynamixel \
  --model xl330-m288 \
  --baudrate 1000000 \
  --ID 2
```

对所有电机重复此过程，直到ID 6。

跟随臂的过程几乎相同，但跟随臂有两种类型的电机。对于前两个电机，确保将模型设置为`xl430-w250`。_重要提示：配置跟随电机需要插拔电源。确保为XL330使用5V电源，为XL430使用12V电源！_

当所有电机都正确配置后，您就可以按照原始视频所示，将它们以菊花链方式全部连接起来。

**实例化DynamixelMotorsBus**

首先，为每个手臂创建两个[`DynamixelMotorsBus`](../lerobot/common/robot_devices/motors/dynamixel.py)实例，使用它们相应的USB端口（例如`DynamixelMotorsBus(port="/dev/tty.usbmodem575E0031751")`）。

要找到每个手臂的正确端口，运行实用程序脚本两次：
```bash
python lerobot/scripts/find_motors_bus_port.py
```

识别引导臂端口时的示例输出（例如，Mac上的`/dev/tty.usbmodem575E0031751`，或者Linux上可能是`/dev/ttyACM0`）：
```
Finding all available ports for the MotorBus.
['/dev/tty.usbmodem575E0032081', '/dev/tty.usbmodem575E0031751']
Remove the usb cable from your DynamixelMotorsBus and press Enter when done.

[...断开引导臂连接并按Enter...]

The port of this DynamixelMotorsBus is /dev/tty.usbmodem575E0031751
Reconnect the usb cable.
```

识别跟随臂端口时的示例输出（例如，`/dev/tty.usbmodem575E0032081`，或者Linux上可能是`/dev/ttyACM1`）：
```
Finding all available ports for the MotorBus.
['/dev/tty.usbmodem575E0032081', '/dev/tty.usbmodem575E0031751']
Remove the usb cable from your DynamixelMotorsBus and press Enter when done.

[...断开跟随臂连接并按Enter...]

The port of this DynamixelMotorsBus is /dev/tty.usbmodem575E0032081
Reconnect the usb cable.
```

故障排除：在Linux上，您可能需要通过运行以下命令来授予USB端口访问权限：
```bash
sudo chmod 666 /dev/tty.usbmodem575E0032081
sudo chmod 666 /dev/tty.usbmodem575E0031751
```

*列出和配置电机*

接下来，您需要列出每个手臂的电机，包括它们的名称、索引和型号。最初，每个电机都分配了出厂默认索引`1`。由于每个电机在连接到公共总线的链上时需要唯一的索引才能正常工作，您需要分配不同的索引。建议使用升序索引顺序，从`1`开始（例如，`1, 2, 3, 4, 5, 6`）。这些索引将在首次连接期间保存在每个电机的持久内存中。

要为电机分配索引，在交互式Python会话中运行以下代码。将`port`值替换为您之前识别的值：
```python
from lerobot.common.robot_devices.motors.configs import DynamixelMotorsBusConfig
from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsBus

leader_config = DynamixelMotorsBusConfig(
    port="/dev/tty.usbmodem575E0031751",
    motors={
        # 名称: (索引, 型号)
        "shoulder_pan": (1, "xl330-m077"),
        "shoulder_lift": (2, "xl330-m077"),
        "elbow_flex": (3, "xl330-m077"),
        "wrist_flex": (4, "xl330-m077"),
        "wrist_roll": (5, "xl330-m077"),
        "gripper": (6, "xl330-m077"),
    },
)

follower_config = DynamixelMotorsBusConfig(
    port="/dev/tty.usbmodem575E0032081",
    motors={
        # 名称: (索引, 型号)
        "shoulder_pan": (1, "xl430-w250"),
        "shoulder_lift": (2, "xl430-w250"),
        "elbow_flex": (3, "xl330-m288"),
        "wrist_flex": (4, "xl330-m288"),
        "wrist_roll": (5, "xl330-m288"),
        "gripper": (6, "xl330-m288"),
    },
)

leader_arm = DynamixelMotorsBus(leader_config)
follower_arm = DynamixelMotorsBus(follower_config)
```

重要提示：现在您已经有了端口，请更新[`KochRobotConfig`](../lerobot/common/robot_devices/robots/configs.py)。您会找到类似以下内容：
```python
@RobotConfig.register_subclass("koch")
@dataclass
class KochRobotConfig(ManipulatorRobotConfig):
    calibration_dir: str = ".cache/calibration/koch"
    # `max_relative_target`限制相对位置目标向量的幅度，出于安全目的。
    # 设置为正标量以对所有电机使用相同的值，或设置为与跟随臂电机数量相同长度的列表。
    max_relative_target: int | None = None

    leader_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": DynamixelMotorsBusConfig(
                port="/dev/tty.usbmodem585A0085511", <-- 在此更新
                motors={
                    # 名称: (索引, 型号)
                    "shoulder_pan": [1, "xl330-m077"],
                    "shoulder_lift": [2, "xl330-m077"],
                    "elbow_flex": [3, "xl330-m077"],
                    "wrist_flex": [4, "xl330-m077"],
                    "wrist_roll": [5, "xl330-m077"],
                    "gripper": [6, "xl330-m077"],
                },
            ),
        }
    )

    follower_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": DynamixelMotorsBusConfig(
                port="/dev/tty.usbmodem585A0076891", <-- 在此更新
                motors={
                    # 名称: (索引, 型号)
                    "shoulder_pan": [1, "xl430-w250"],
                    "shoulder_lift": [2, "xl430-w250"],
                    "elbow_flex": [3, "xl330-m288"],
                    "wrist_flex": [4, "xl330-m288"],
                    "wrist_roll": [5, "xl330-m288"],
                    "gripper": [6, "xl330-m288"],
                },
            ),
        }
    )
```

**连接和配置您的电机**

在开始使用电机之前，您需要配置它们以确保正确通信。当您首次连接电机时，[`DynamixelMotorsBus`](../lerobot/common/robot_devices/motors/dynamixel.py)会自动检测当前电机索引（出厂设置为`1`）与指定索引（例如`1, 2, 3, 4, 5, 6`）之间的任何不匹配。这会触发配置程序，要求您拔掉电源线和电机，然后从最靠近总线的电机开始依次重新连接每个电机。

有关配置程序的可视化指南，请参考[配置程序视频教程](https://youtu.be/U78QQ9wCdpY)。

要连接和配置引导臂，请在与本教程前面相同的Python交互式会话中运行以下代码：
```python
leader_arm.connect()
```

当您第一次连接引导臂时，您可能会看到类似这样的输出：
```
Read failed due to communication error on port /dev/tty.usbmodem575E0032081 for group_key ID_shoulder_pan_shoulder_lift_elbow_flex_wrist_flex_wrist_roll_gripper: [TxRxResult] There is no status packet!

/!\ A configuration issue has been detected with your motors:
If this is the first time you are using these motors, press enter to configure your motors... but before verify that all the cables are connected the proper way. If you find an issue, before making a modification, kill the python process, unplug the power cord to not damage the motors, rewire correctly, then plug the power again and relaunch the script.

Motor indices detected: {9600: [1]}

1. Unplug the power cord
2. Plug/unplug minimal number of cables to only have the first 1 motor(s) (['shoulder_pan']) connected.
3. Re-plug the power cord
Press Enter to continue...

*Follow the procedure*

Setting expected motor indices: [1, 2, 3, 4, 5, 6]
```

配置好引导臂后，通过运行以下命令为跟随臂重复此过程：
```python
follower_arm.connect()
```

恭喜！两个臂现在都已正确配置和连接。将来您不需要再次进行配置程序。

**故障排除**：

如果配置过程失败，您可能需要通过Dynamixel Wizard进行配置。

已知故障模式：
- 调用`arm.connect()`在Ubuntu 22上引发`OSError: No motor found, but one new motor expected. Verify power cord is plugged in and retry`。

步骤：
1. 访问https://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_wizard2/#connect-dynamixel。
2. 按照网页第3部分的软件安装说明进行操作。
3. 启动软件。
4. 在`Tools` > `Options` > `Scan`菜单下配置设备扫描选项。仅选中协议2.0，仅选择感兴趣的USB端口标识符，选择所有波特率，将ID范围设置为`[0, 10]`。_虽然此步骤并非严格必要，但它极大地加快了扫描速度_。
5. 对于每个电机依次：
  - 断开驱动板的电源。
  - 将**仅**要配置的电机连接到驱动板，确保将其与任何其他电机断开连接。
  - 重新连接驱动板的电源。
  - 从软件菜单中选择`Device` > `Scan`并让扫描运行。应该会出现一个设备。
  - 如果设备旁边有星号(*)，表示固件确实已过期。从软件菜单中，选择`Tools` > `Firmware Update`。按照提示操作。
  - 主面板应该有一个带有设备各种参数的表格（参考网页第5部分）。选择带有`ID`的行，然后通过在右下角面板上选择并单击`Save`来设置所需的ID。
  - 就像您对ID所做的那样，也将`Baud Rate`设置为1 Mbps。
6. 检查所有操作是否正确：
   - 按最终配置重新接线臂并给两者供电。
   - 扫描设备。所有12个电机都应该出现。
   - 逐个选择电机并移动手臂。检查右上角附近的图形指示器是否显示移动。

** Dynamixel XL430-W250电机有一个常见问题，在从Mac和Windows Dynamixel Wizard2应用程序升级固件后，电机变得无法被发现。当这种情况发生时，需要进行固件恢复（选择`DYNAMIXEL Firmware Recovery`并按照提示操作）。有两种已知的解决方法可以进行此固件重置：
  1) 在Linux机器上安装Dynamixel Wizard并完成固件恢复
  2) 使用Dynamixel U2D2在Windows或Mac上执行重置。此U2D2可以在[这里](https://www.robotis.us/u2d2/)购买。
  对于任一解决方案，打开DYNAMIXEL Wizard 2.0并选择适当的端口。此时您可能无法在GUI中看到电机。选择`Firmware Recovery`，仔细选择正确的型号，并等待过程完成。最后，重新扫描以确认固件恢复成功。

**使用DynamixelMotorsBus读写**

要熟悉`DynamixelMotorsBus`如何与电机通信，您可以先从读取数据开始。在同一交互式Python会话中复制粘贴此代码：
```python
leader_pos = leader_arm.read("Present_Position")
follower_pos = follower_arm.read("Present_Position")
print(leader_pos)
print(follower_pos)
```

预期输出可能如下所示：
```
array([2054,  523, 3071, 1831, 3049, 2441], dtype=int32)
array([2003, 1601,   56, 2152, 3101, 2283], dtype=int32)
```

尝试将手臂移动到各种位置，观察值如何变化。

现在让我们尝试通过复制粘贴此代码来启用跟随臂的扭矩：
```python
from lerobot.common.robot_devices.motors.dynamixel import TorqueMode

follower_arm.write("Torque_Enable", TorqueMode.ENABLED.value)
```

扭矩启用后，跟随臂将锁定在当前位置。在扭矩启用时不要尝试手动移动手臂，因为这可能会损坏电机。

现在，为了更熟悉读写，让我们通过复制粘贴以下示例代码以编程方式移动手臂：
```python
# 获取当前位置
position = follower_arm.read("Present_Position")

# 将第一个电机（shoulder_pan）位置+10步
position[0] += 10
follower_arm.write("Goal_Position", position)

# 所有电机位置-30步
position -= 30
follower_arm.write("Goal_Position", position)

# 夹持器+30步
position[-1] += 30
follower_arm.write("Goal_Position", position[-1], "gripper")
```

当您完成实验后，可以尝试禁用扭矩，但确保握住机器人以防止其跌落：
```python
follower_arm.write("Torque_Enable", TorqueMode.DISABLED.value)
```

最后，断开手臂连接：
```python
leader_arm.disconnect()
follower_arm.disconnect()
```

或者，您可以拔掉电源线，这将自动禁用扭矩并断开电机连接。

*/!\ 警告*：这些电机容易过热，尤其是在扭矩启用或长时间插电的情况下。使用后请拔掉电源。

### b. 使用ManipulatorRobot远程操作Koch v1.1

**实例化ManipulatorRobot**

在远程操作机器人之前，您需要使用先前定义的`leader_config`和`follower_config`来实例化[`ManipulatorRobot`](../lerobot/common/robot_devices/robots/manipulator.py)。

对于Koch v1.1机器人，我们只有一个引导臂，所以我们将其称为`"main"`并定义为`leader_arms={"main": leader_config}`。我们对跟随臂做同样的操作。对于其他机器人（如Aloha），可能有两对引导和跟随臂，您可以这样定义：`leader_arms={"left": left_leader_config, "right": right_leader_config}`。跟随臂也是如此。


运行以下代码来实例化您的机械手机器人：
```python
from lerobot.common.robot_devices.robots.configs import KochRobotConfig
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot

robot_config = KochRobotConfig(
  leader_arms={"main": leader_config},
  follower_arms={"main": follower_config},
  cameras={},  # 我们现在不使用任何相机
)
robot = ManipulatorRobot(robot_config)
```

`KochRobotConfig`用于设置相关设置和校准过程。例如，我们激活了Koch v1.1引导臂的夹持器的扭矩，并将其定位在40度角以用作触发器。

对于[Aloha双臂机器人](https://aloha-2.github.io)，我们会使用`AlohaRobotConfig`来设置不同的设置，如影子关节（肩部、肘部）的次要ID。特定于Aloha，LeRobot附带存储在`.cache/calibration/aloha_default`中的默认校准文件。假设电机已正确组装，预计Aloha不需要手动校准步骤。

**校准并连接ManipulatorRobot**

接下来，您需要校准Koch机器人，以确保当引导臂和跟随臂处于相同物理位置时，它们具有相同的位置值。这种校准至关重要，因为它允许在一个Koch机器人上训练的神经网络可以在另一个上工作。

当您首次连接机器人时，[`ManipulatorRobot`](../lerobot/common/robot_devices/robots/manipulator.py)将检测是否缺少校准文件并触发校准程序。在此过程中，您将被引导将每个臂移动到三个不同的位置。

以下是您将移动跟随臂到的位置：

| 1. 零位置                                                                                                                                              | 2. 旋转位置                                                                                                                                               | 3. 休息位置                                                                                                                                              |
| ----------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| <img src="../media/koch/follower_zero.webp?raw=true" alt="Koch v1.1跟随臂零位置" title="Koch v1.1跟随臂零位置" style="width:100%;"> | <img src="../media/koch/follower_rotated.webp?raw=true" alt="Koch v1.1跟随臂旋转位置" title="Koch v1.1跟随臂旋转位置" style="width:100%;"> | <img src="../media/koch/follower_rest.webp?raw=true" alt="Koch v1.1跟随臂休息位置" title="Koch v1.1跟随臂休息位置" style="width:100%;"> |

以下是引导臂对应的位置：

| 1. 零位置                                                                                                                                         | 2. 旋转位置                                                                                                                                            | 3. 休息位置                                                                                                                                         |
| ------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| <img src="../media/koch/leader_zero.webp?raw=true" alt="Koch v1.1引导臂零位置" title="Koch v1.1引导臂零位置" style="width:100%;"> | <img src="../media/koch/leader_rotated.webp?raw=true" alt="Koch v1.1引导臂旋转位置" title="Koch v1.1引导臂旋转位置" style="width:100%;"> | <img src="../media/koch/leader_rest.webp?raw=true" alt="Koch v1.1引导臂休息位置" title="Koch v1.1引导臂休息位置" style="width:100%;"> |

您可以查看[校准程序的视频教程](https://youtu.be/8drnU9uRY24)了解更多详情。

在校准过程中，我们会计算您的电机自首次使用以来完成的360度全旋转次数。这就是为什么我们要求您移动到这个任意的"零"位置。我们并不真正"设置"零位置，所以您不需要精确。计算这些"偏移量"以将电机值移到0周围后，我们需要评估每个电机的旋转方向，这可能有所不同。这就是为什么我们要求您将所有电机旋转到大约90度，以测量值是否呈负向或正向变化。

最后，休息位置确保在校准后跟随臂和引导臂大致对齐，防止在开始远程操作时出现可能损坏电机的突然运动。

重要的是，一旦校准，所有Koch机器人在接收到命令时都会移动到相同的位置（例如零位置和旋转位置）。

运行以下代码来校准并连接您的机器人：
```python
robot.connect()
```

输出将如下所示：
```
Connecting main follower arm
Connecting main leader arm

Missing calibration file '.cache/calibration/koch/main_follower.json'
Running calibration of koch main follower...
Move arm to zero position
[...]
Move arm to rotated position
[...]
Move arm to rest position
[...]
Calibration is done! Saving calibration file '.cache/calibration/koch/main_follower.json'

Missing calibration file '.cache/calibration/koch/main_leader.json'
Running calibration of koch main leader...
Move arm to zero position
[...]
Move arm to rotated position
[...]
Move arm to rest position
[...]
Calibration is done! Saving calibration file '.cache/calibration/koch/main_leader.json'
```

*验证校准*

校准完成后，您可以检查引导臂和跟随臂的位置以确保它们匹配。如果校准成功，位置应该非常相似。

运行此代码以获取度数表示的位置：
```python
leader_pos = robot.leader_arms["main"].read("Present_Position")
follower_pos = robot.follower_arms["main"].read("Present_Position")

print(leader_pos)
print(follower_pos)
```

示例输出：
```
array([-0.43945312, 133.94531, 179.82422, -18.984375, -1.9335938, 34.541016], dtype=float32)
array([-0.58723712, 131.72314, 174.98743, -16.872612, 0.786213, 35.271973], dtype=float32)
```

这些值以度为单位，使它们更易于解释和调试。校准过程中使用的零位置大致对应于每个电机的0度，旋转位置大致对应于每个电机的90度。

**远程操作Koch v1.1**

您可以通过读取引导臂的位置并将其作为目标位置发送给跟随臂，轻松远程操作您的机器人。

要以大约200Hz的频率远程操作机器人30秒，请运行以下代码：
```python
import tqdm
seconds = 30
frequency = 200
for _ in tqdm.tqdm(range(seconds*frequency)):
  leader_pos = robot.leader_arms["main"].read("Present_Position")
  robot.follower_arms["main"].write("Goal_Position", leader_pos)
```

*使用`teleop_step`进行远程操作*

或者，您可以使用[`ManipulatorRobot`](../lerobot/common/robot_devices/robots/manipulator.py)中的`teleop_step`方法远程操作机器人。

运行此代码进行远程操作：
```python
for _ in tqdm.tqdm(range(seconds*frequency)):
  robot.teleop_step()
```

*在远程操作过程中记录数据*

远程操作对于记录数据特别有用。您可以使用`teleop_step(record_data=True)`返回跟随臂的位置作为`"observation.state"`和引导臂的位置作为`"action"`。此函数还将numpy数组转换为PyTorch张量。如果您正在使用具有两个引导臂和两个跟随臂的机器人（如Aloha），则位置会被连接起来。

运行以下代码，查看慢慢移动引导臂如何影响观察和动作：
```python
leader_pos = robot.leader_arms["main"].read("Present_Position")
follower_pos = robot.follower_arms["main"].read("Present_Position")
observation, action = robot.teleop_step(record_data=True)

print(follower_pos)
print(observation)
print(leader_pos)
print(action)
```

*异步帧录制*

此外，`teleop_step`可以异步记录来自多个相机的帧，并将它们作为`"observation.images.CAMERA_NAME"`包含在观察字典中。这个特性将在下一部分详细介绍。

*断开机器人连接*

完成后，请务必通过运行以下命令断开机器人连接：
```python
robot.disconnect()
```

或者，您可以拔掉电源线，这也会禁用扭矩。

*/!\ 警告*：这些电机容易过热，尤其是在扭矩下或长时间插电的情况下。使用后请拔掉电源。

根据您之前学习的内容，您现在可以轻松地记录一个包含单个回合状态和动作的数据集。您可以使用 `busy_wait` 来控制远程操作的速度，并以固定的 `fps`（每秒帧数）进行记录。

尝试以下代码，以60 fps的速度记录30秒：
```python
import time
from lerobot.scripts.control_robot import busy_wait

record_time_s = 30
fps = 60

states = []
actions = []
for _ in range(record_time_s * fps):
  start_time = time.perf_counter()
  observation, action = robot.teleop_step(record_data=True)

  states.append(observation["observation.state"])
  actions.append(action["action"])

  dt_s = time.perf_counter() - start_time
  busy_wait(1 / fps - dt_s)

# 注意，观察和动作可在RAM中获取，但
# 您可以使用pickle/hdf5或我们优化的格式
# `LeRobotDataset`将它们存储在磁盘上。下面会详细介绍。
```

重要的是，仍有许多实用工具尚未涵盖。例如，如果您有摄像头，您需要将图像保存到磁盘以避免内存不足，并在线程中进行以避免减慢与机器人的通信速度。此外，您需要将数据存储在为训练和网络共享而优化的格式中，如[`LeRobotDataset`](../lerobot/common/datasets/lerobot_dataset.py)。下一节将详细介绍这些内容。

### a. 使用 `record` 函数

您可以使用[`lerobot/scripts/control_robot.py`](../lerobot/scripts/control_robot.py)中的`record`函数来实现高效的数据记录。它包含许多记录实用工具：
1. 摄像头的帧在线程中保存到磁盘，并在每个回合记录结束时编码为视频。
2. 摄像头的视频流显示在窗口中，以便您可以验证它们。
3. 数据使用[`LeRobotDataset`](../lerobot/common/datasets/lerobot_dataset.py)格式存储，并推送到您的Hugging Face页面（除非提供了`--control.push_to_hub=false`）。
4. 在记录过程中进行检查点保存，因此如果出现任何问题，您可以通过再次运行相同的命令并加上`--control.resume=true`来恢复记录。如果您想从头开始记录，则需要手动删除数据集目录。
5. 使用命令行参数设置数据记录流程：
   - `--control.warmup_time_s=10` 定义在开始数据收集前的预热秒数。它允许机器人设备预热和同步（默认为10秒）。
   - `--control.episode_time_s=60` 定义每个回合数据记录的秒数（默认为60秒）。
   - `--control.reset_time_s=60` 定义每个回合后重置环境的秒数（默认为60秒）。
   - `--control.num_episodes=50` 定义要记录的回合数量（默认为50）。
6. 在数据记录过程中使用键盘按键控制流程：
   - 在回合记录过程中任何时候按右箭头 `->` 可提前停止并进入重置阶段。在重置过程中同样可以提前停止并进入下一回合记录。
   - 在回合记录或重置过程中任何时候按左箭头 `<-` 可提前停止，取消当前回合，并重新记录。
   - 在回合记录过程中任何时候按Esc键 `ESC` 可提前结束会话并直接进入视频编码和数据集上传阶段。
7. 与 `teleoperate` 类似，您也可以使用命令行覆盖任何设置。

在尝试 `record` 之前，如果您想将数据集推送到hub，请确保您已使用具有写访问权限的令牌登录，该令牌可以从[Hugging Face设置](https://huggingface.co/settings/tokens)生成：
```bash
huggingface-cli login --token ${HUGGINGFACE_TOKEN} --add-to-git-credential
```
同时，将您的Hugging Face仓库名称存储在一个变量中（例如 `cadene` 或 `lerobot`）。例如，运行以下命令使用您的Hugging Face用户名作为仓库：
```bash
HF_USER=$(huggingface-cli whoami | head -n 1)
echo $HF_USER
```
如果您不想推送到hub，请使用 `--control.push_to_hub=false`。

现在运行以下命令记录2个回合：
```bash
python lerobot/scripts/control_robot.py \
  --robot.type=koch \
  --control.type=record \
  --control.single_task="抓取一个乐高积木并将其放入箱中。" \
  --control.fps=30 \
  --control.repo_id=${HF_USER}/koch_test \
  --control.tags='["tutorial"]' \
  --control.warmup_time_s=5 \
  --control.episode_time_s=30 \
  --control.reset_time_s=30 \
  --control.num_episodes=2 \
  --control.push_to_hub=true
```

这将把您的数据集本地写入 `~/.cache/huggingface/lerobot/{repo-id}`（例如 `data/cadene/koch_test`）并将其推送到hub上的 `https://huggingface.co/datasets/{HF_USER}/{repo-id}`。您的数据集将自动标记为 `LeRobot`，以便社区可以轻松找到它，您还可以添加自定义标签（在本例中为 `tutorial`）。

您可以通过搜索 `LeRobot` 标签在hub上查找其他LeRobot数据集：https://huggingface.co/datasets?other=LeRobot

您将看到许多行出现，如下所示：
```
INFO 2024-08-10 15:02:58 ol_robot.py:219 dt:33.34 (30.0hz) dtRlead: 5.06 (197.5hz) dtWfoll: 0.25 (3963.7hz) dtRfoll: 6.22 (160.7hz) dtRlaptop: 32.57 (30.7hz) dtRphone: 33.84 (29.5hz)
```
它包含：
- `2024-08-10 15:02:58`，即调用print函数的日期和时间，
- `ol_robot.py:219`，即调用print函数的文件名末尾和行号（`lerobot/scripts/control_robot.py` 第 `219` 行）。
- `dt:33.34 (30.0hz)`，即前一次调用`robot.teleop_step(record_data=True)`与当前调用之间花费的"时间增量"或毫秒数，以及相关频率（33.34毫秒等于30.0 Hz）；注意我们使用 `--fps 30` 所以我们期望频率为30.0 Hz；当一个步骤花费更多时间时，该行会显示为黄色。
- `dtRlead: 5.06 (197.5hz)`，即读取引导臂当前位置的时间增量。
- `dtWfoll: 0.25 (3963.7hz)`，即在跟随臂上写入目标位置的时间增量；写入是异步的，所以比读取花费的时间更少。
- `dtRfoll: 6.22 (160.7hz)`，即读取跟随臂当前位置的时间增量。
- `dtRlaptop:32.57 (30.7hz)`，即在异步运行的线程中从笔记本电脑摄像头捕获图像的时间增量。
- `dtRphone:33.84 (29.5hz)`，即在异步运行的线程中从手机摄像头捕获图像的时间增量。

故障排除：
- 在Linux上，如果在数据记录过程中左右箭头键和Esc键没有任何效果，请确保您已设置`$DISPLAY`环境变量。参见[pynput限制](https://pynput.readthedocs.io/en/latest/limitations.html#linux)。

数据记录结束时，您的数据集将上传到您的Hugging Face页面（例如 https://huggingface.co/datasets/cadene/koch_test），您可以通过运行以下命令获取该页面：
```bash
echo https://huggingface.co/datasets/${HF_USER}/koch_test
```

### b. 记录数据集的建议

一旦您熟悉了数据记录，就可以创建更大的数据集用于训练了。一个良好的起始任务是在不同位置抓取物体并将其放入箱中。我们建议至少记录50个回合，每个位置10个回合。保持摄像头固定，并在整个记录过程中保持一致的抓取行为。

在接下来的部分中，您将训练您的神经网络。在实现可靠的抓取性能后，您可以开始在数据收集过程中引入更多变化，如额外的抓取位置、不同的抓取技术以及改变摄像头位置。

避免过快地添加太多变化，因为这可能会阻碍您的结果。

在未来几个月内，我们计划发布一个用于机器人学习的基础模型。我们预计微调这个模型将增强泛化能力，减少在数据收集过程中对严格一致性的需求。

### c. 可视化所有回合

您可以通过运行以下命令可视化您的数据集：
```bash
python lerobot/scripts/visualize_dataset_html.py \
  --repo-id ${HF_USER}/koch_test
```

注意：如果您的数据集未上传到hugging face hub，您可能需要添加 `--local-files-only 1`。

这将启动一个本地web服务器，如下所示：
<div style="text-align:center;">
  <img src="../media/tutorial/visualize_dataset_html.webp?raw=true" alt="Koch v1.1引导臂和跟随臂" title="Koch v1.1引导臂和跟随臂" width="100%">
</div>

### d. 使用 `replay` 函数在机器人上重放回合

[`lerobot/scripts/control_robot.py`](../lerobot/scripts/control_robot.py)的一个有用功能是 `replay` 函数，它允许在您的机器人上重放您已记录的任何回合或来自其他数据集的回合。此功能帮助您测试机器人动作的可重复性，并评估相同型号机器人之间的可迁移性。

要重放您刚记录的数据集的第一个回合，请运行以下命令：
```bash
python lerobot/scripts/control_robot.py \
  --robot.type=koch \
  --control.type=replay \
  --control.fps=30 \
  --control.repo_id=${HF_USER}/koch_test \
  --control.episode=0
```

您的机器人应该复制类似于您记录的动作。例如，查看[这个视频](https://x.com/RemiCadene/status/1793654950905680090)，我们在来自[Trossen Robotics](https://www.trossenrobotics.com)的Aloha机器人上使用`replay`。

## 4. 基于数据训练策略

### a. 使用 `train` 脚本

要训练用于控制机器人的策略，请使用[`python lerobot/scripts/train.py`](../lerobot/scripts/train.py)脚本。需要几个参数。以下是示例命令：
```bash
python lerobot/scripts/train.py \
  --dataset.repo_id=${HF_USER}/koch_test \
  --policy.type=act \
  --output_dir=outputs/train/act_koch_test \
  --job_name=act_koch_test \
  --policy.device=cuda \
  --wandb.enable=true
```

让我们解释一下：
1. 我们通过 `--dataset.repo_id=${HF_USER}/koch_test` 提供了数据集作为参数。
2. 我们通过 `policy.type=act` 提供了策略。这会从[`configuration_act.py`](../lerobot/common/policies/act/configuration_act.py)加载配置。重要的是，这个策略将自动适应您机器人的电机状态数量、电机动作数量和摄像头（例如`laptop`和`phone`），这些都已保存在您的数据集中。
4. 我们提供了 `policy.device=cuda`，因为我们在Nvidia GPU上训练，但您也可以使用 `policy.device=mps` 在Apple silicon上训练。
5. 我们提供了 `wandb.enable=true` 以使用[Weights and Biases](https://docs.wandb.ai/quickstart)来可视化训练图表。这是可选的，但如果使用，请确保通过运行 `wandb login` 登录。

有关`train`脚本的更多信息，请参阅前面的教程：[`examples/4_train_policy_with_script.md`](../examples/4_train_policy_with_script.md)

### b. （可选）将策略检查点上传到hub

训练完成后，使用以下命令上传最新检查点：
```bash
huggingface-cli upload ${HF_USER}/act_koch_test \
  outputs/train/act_koch_test/checkpoints/last/pretrained_model
```

您也可以上传中间检查点：
```bash
CKPT=010000
huggingface-cli upload ${HF_USER}/act_koch_test_${CKPT} \
  outputs/train/act_koch_test/checkpoints/${CKPT}/pretrained_model
```

## 5. 评估您的策略

现在您有了策略检查点，可以使用[`ManipulatorRobot`](../lerobot/common/robot_devices/robots/manipulator.py)和策略中的方法轻松控制您的机器人。

尝试以下代码，以30 fps的速度运行60秒的推理：
```python
from lerobot.common.policies.act.modeling_act import ACTPolicy

inference_time_s = 60
fps = 30
device = "cuda"  # 注意：在Mac上，使用 "mps" 或 "cpu"

ckpt_path = "outputs/train/act_koch_test/checkpoints/last/pretrained_model"
policy = ACTPolicy.from_pretrained(ckpt_path)
policy.to(device)

for _ in range(inference_time_s * fps):
  start_time = time.perf_counter()

  # 读取跟随者状态并访问摄像头中的帧
  observation = robot.capture_observation()

  # 转换为pytorch格式：通道在前，float32类型，像素范围在[0,1]
  # 带有批次维度
  for name in observation:
    if "image" in name:
      observation[name] = observation[name].type(torch.float32) / 255
      observation[name] = observation[name].permute(2, 0, 1).contiguous()
    observation[name] = observation[name].unsqueeze(0)
    observation[name] = observation[name].to(device)

  # 使用策略基于当前观察计算下一个动作
  action = policy.select_action(observation)
  # 移除批次维度
  action = action.squeeze(0)
  # 移动到cpu（如果尚未移动）
  action = action.to("cpu")
  # 命令机器人移动
  robot.send_action(action)

  dt_s = time.perf_counter() - start_time
  busy_wait(1 / fps - dt_s)
```

### a. 使用我们的 `record` 函数

理想情况下，当使用神经网络控制机器人时，您可能希望记录评估回合，以便稍后可视化它们，甚至像在强化学习中那样对它们进行训练。这基本上对应于记录一个新数据集，但使用神经网络提供动作而不是远程操作。

为此，您可以使用[`lerobot/scripts/control_robot.py`](../lerobot/scripts/control_robot.py)中的`record`函数，但将策略检查点作为输入。例如，运行以下命令记录10个评估回合：
```bash
python lerobot/scripts/control_robot.py \
  --robot.type=koch \
  --control.type=record \
  --control.fps=30 \
  --control.repo_id=${HF_USER}/eval_act_koch_test \
  --control.tags='["tutorial"]' \
  --control.warmup_time_s=5 \
  --control.episode_time_s=30 \
  --control.reset_time_s=30 \
  --control.num_episodes=10 \
  --control.push_to_hub=true \
  --control.policy.path=outputs/train/act_koch_test/checkpoints/last/pretrained_model
```

如您所见，它几乎与之前用于记录训练数据集的命令相同。有两点变化：
1. 有一个额外的 `--control.policy.path` 参数，指示策略检查点的路径（例如 `outputs/train/eval_koch_test/checkpoints/last/pretrained_model`）。如果您将模型检查点上传到hub，您也可以使用模型库（例如 `${HF_USER}/act_koch_test`）。
2. 数据集名称以 `eval` 开头，以反映您正在运行推理（例如 `${HF_USER}/eval_act_koch_test`）。

### b. 事后可视化评估

然后，您可以通过运行与之前相同的命令但使用新的推理数据集作为参数来可视化您的评估数据集：
```bash
python lerobot/scripts/visualize_dataset.py \
  --repo-id ${HF_USER}/eval_act_koch_test
```

## 6. 下一步

加入我们的[Discord](https://discord.com/invite/s3KuuzsPFb)，一起协作收集数据，帮助我们训练完全开源的机器人学习基础模型！

使用您之前学到的知识，您现在可以轻松地记录一个包含单个回合状态和动作的数据集。您可以使用 `busy_wait` 来控制远程操作的速度，并以固定的 `fps`（每秒帧数）进行记录。

尝试以下代码，以60 fps的速度记录30秒：
```python
import time
from lerobot.scripts.control_robot import busy_wait

record_time_s = 30
fps = 60

states = []
actions = []
for _ in range(record_time_s * fps):
  start_time = time.perf_counter()
  observation, action = robot.teleop_step(record_data=True)

  states.append(observation["observation.state"])
  actions.append(action["action"])

  dt_s = time.perf_counter() - start_time
  busy_wait(1 / fps - dt_s)

# 注意，观察和动作可在RAM中获取，但
# 您可以使用pickle/hdf5或我们优化的格式
# `LeRobotDataset`将它们存储在磁盘上。下面会详细介绍。
```

重要的是，仍有许多实用工具尚未涵盖。例如，如果您有摄像头，您需要将图像保存到磁盘以避免内存不足，并在线程中进行以避免减慢与机器人的通信速度。此外，您需要将数据存储在为训练和网络共享而优化的格式中，如[`LeRobotDataset`](../lerobot/common/datasets/lerobot_dataset.py)。下一节将详细介绍这些内容。

### a. 使用 `record` 函数

您可以使用[`lerobot/scripts/control_robot.py`](../lerobot/scripts/control_robot.py)中的`record`函数来实现高效的数据记录。它包含许多记录实用工具：
1. 摄像头的帧在线程中保存到磁盘，并在每个回合记录结束时编码为视频。
2. 摄像头的视频流显示在窗口中，以便您可以验证它们。
3. 数据使用[`LeRobotDataset`](../lerobot/common/datasets/lerobot_dataset.py)格式存储，并推送到您的Hugging Face页面（除非提供了`--control.push_to_hub=false`）。
4. 在记录过程中进行检查点保存，因此如果出现任何问题，您可以通过再次运行相同的命令并加上`--control.resume=true`来恢复记录。如果您想从头开始记录，则需要手动删除数据集目录。
5. 使用命令行参数设置数据记录流程：
   - `--control.warmup_time_s=10` 定义在开始数据收集前的预热秒数。它允许机器人设备预热和同步（默认为10秒）。
   - `--control.episode_time_s=60` 定义每个回合数据记录的秒数（默认为60秒）。
   - `--control.reset_time_s=60` 定义每个回合后重置环境的秒数（默认为60秒）。
   - `--control.num_episodes=50` 定义要记录的回合数量（默认为50）。
6. 在数据记录过程中使用键盘按键控制流程：
   - 在回合记录过程中任何时候按右箭头 `->` 可提前停止并进入重置阶段。在重置过程中同样可以提前停止并进入下一回合记录。
   - 在回合记录或重置过程中任何时候按左箭头 `<-` 可提前停止，取消当前回合，并重新记录。
   - 在回合记录过程中任何时候按Esc键 `ESC` 可提前结束会话并直接进入视频编码和数据集上传阶段。
7. 与 `teleoperate` 类似，您也可以使用命令行覆盖任何设置。

在尝试 `record` 之前，如果您想将数据集推送到hub，请确保您已使用具有写访问权限的令牌登录，该令牌可以从[Hugging Face设置](https://huggingface.co/settings/tokens)生成：
```bash
huggingface-cli login --token ${HUGGINGFACE_TOKEN} --add-to-git-credential
```
同时，将您的Hugging Face仓库名称存储在一个变量中（例如 `cadene` 或 `lerobot`）。例如，运行以下命令使用您的Hugging Face用户名作为仓库：
```bash
HF_USER=$(huggingface-cli whoami | head -n 1)
echo $HF_USER
```
如果您不想推送到hub，请使用 `--control.push_to_hub=false`。

现在运行以下命令记录2个回合：
```bash
python lerobot/scripts/control_robot.py \
  --robot.type=koch \
  --control.type=record \
  --control.single_task="抓取一个乐高积木并将其放入箱中。" \
  --control.fps=30 \
  --control.repo_id=${HF_USER}/koch_test \
  --control.tags='["tutorial"]' \
  --control.warmup_time_s=5 \
  --control.episode_time_s=30 \
  --control.reset_time_s=30 \
  --control.num_episodes=2 \
  --control.push_to_hub=true
```

这将把您的数据集本地写入 `~/.cache/huggingface/lerobot/{repo-id}`（例如 `data/cadene/koch_test`）并将其推送到hub上的 `https://huggingface.co/datasets/{HF_USER}/{repo-id}`。您的数据集将自动标记为 `LeRobot`，以便社区可以轻松找到它，您还可以添加自定义标签（在本例中为 `tutorial`）。

您可以通过搜索 `LeRobot` 标签在hub上查找其他LeRobot数据集：https://huggingface.co/datasets?other=LeRobot

您将看到许多行出现，如下所示：
```
INFO 2024-08-10 15:02:58 ol_robot.py:219 dt:33.34 (30.0hz) dtRlead: 5.06 (197.5hz) dtWfoll: 0.25 (3963.7hz) dtRfoll: 6.22 (160.7hz) dtRlaptop: 32.57 (30.7hz) dtRphone: 33.84 (29.5hz)
```
它包含：
- `2024-08-10 15:02:58`，即调用print函数的日期和时间，
- `ol_robot.py:219`，即调用print函数的文件名末尾和行号（`lerobot/scripts/control_robot.py` 第 `219` 行）。
- `dt:33.34 (30.0hz)`，即前一次调用`robot.teleop_step(record_data=True)`与当前调用之间花费的"时间增量"或毫秒数，以及相关频率（33.34毫秒等于30.0 Hz）；注意我们使用 `--fps 30` 所以我们期望频率为30.0 Hz；当一个步骤花费更多时间时，该行会显示为黄色。
- `dtRlead: 5.06 (197.5hz)`，即读取引导臂当前位置的时间增量。
- `dtWfoll: 0.25 (3963.7hz)`，即在跟随臂上写入目标位置的时间增量；写入是异步的，所以比读取花费的时间更少。
- `dtRfoll: 6.22 (160.7hz)`，即读取跟随臂当前位置的时间增量。
- `dtRlaptop:32.57 (30.7hz)`，即在异步运行的线程中从笔记本电脑摄像头捕获图像的时间增量。
- `dtRphone:33.84 (29.5hz)`，即在异步运行的线程中从手机摄像头捕获图像的时间增量。

故障排除：
- 在Linux上，如果在数据记录过程中左右箭头键和Esc键没有任何效果，请确保您已设置`$DISPLAY`环境变量。参见[pynput限制](https://pynput.readthedocs.io/en/latest/limitations.html#linux)。

数据记录结束时，您的数据集将上传到您的Hugging Face页面（例如 https://huggingface.co/datasets/cadene/koch_test），您可以通过运行以下命令获取该页面：
```bash
echo https://huggingface.co/datasets/${HF_USER}/koch_test
```

### b. 记录数据集的建议

一旦您熟悉了数据记录，就可以创建更大的数据集用于训练了。一个良好的起始任务是在不同位置抓取物体并将其放入箱中。我们建议至少记录50个回合，每个位置10个回合。保持摄像头固定，并在整个记录过程中保持一致的抓取行为。

在接下来的部分中，您将训练您的神经网络。在实现可靠的抓取性能后，您可以开始在数据收集过程中引入更多变化，如额外的抓取位置、不同的抓取技术以及改变摄像头位置。

避免过快地添加太多变化，因为这可能会阻碍您的结果。

在未来几个月内，我们计划发布一个用于机器人学习的基础模型。我们预计微调这个模型将增强泛化能力，减少在数据收集过程中对严格一致性的需求。

### c. 可视化所有回合

您可以通过运行以下命令可视化您的数据集：
```bash
python lerobot/scripts/visualize_dataset_html.py \
  --repo-id ${HF_USER}/koch_test
```

注意：如果您的数据集未上传到hugging face hub，您可能需要添加 `--local-files-only 1`。

这将启动一个本地web服务器，如下所示：
<div style="text-align:center;">
  <img src="../media/tutorial/visualize_dataset_html.webp?raw=true" alt="Koch v1.1引导臂和跟随臂" title="Koch v1.1引导臂和跟随臂" width="100%">
</div>

### d. 使用 `replay` 函数在机器人上重放回合

[`lerobot/scripts/control_robot.py`](../lerobot/scripts/control_robot.py)的一个有用功能是 `replay` 函数，它允许在您的机器人上重放您已记录的任何回合或来自其他数据集的回合。此功能帮助您测试机器人动作的可重复性，并评估相同型号机器人之间的可迁移性。

要重放您刚记录的数据集的第一个回合，请运行以下命令：
```bash
python lerobot/scripts/control_robot.py \
  --robot.type=koch \
  --control.type=replay \
  --control.fps=30 \
  --control.repo_id=${HF_USER}/koch_test \
  --control.episode=0
```

您的机器人应该复制类似于您记录的动作。例如，查看[这个视频](https://x.com/RemiCadene/status/1793654950905680090)，我们在来自[Trossen Robotics](https://www.trossenrobotics.com)的Aloha机器人上使用`replay`。

## 4. 基于数据训练策略

### a. 使用 `train` 脚本

要训练用于控制机器人的策略，请使用[`python lerobot/scripts/train.py`](../lerobot/scripts/train.py)脚本。需要几个参数。以下是示例命令：
```bash
python lerobot/scripts/train.py \
  --dataset.repo_id=${HF_USER}/koch_test \
  --policy.type=act \
  --output_dir=outputs/train/act_koch_test \
  --job_name=act_koch_test \
  --policy.device=cuda \
  --wandb.enable=true
```

让我们解释一下：
1. 我们通过 `--dataset.repo_id=${HF_USER}/koch_test` 提供了数据集作为参数。
2. 我们通过 `policy.type=act` 提供了策略。这会从[`configuration_act.py`](../lerobot/common/policies/act/configuration_act.py)加载配置。重要的是，这个策略将自动适应您机器人的电机状态数量、电机动作数量和摄像头（例如`laptop`和`phone`），这些都已保存在您的数据集中。
4. 我们提供了 `policy.device=cuda`，因为我们在Nvidia GPU上训练，但您也可以使用 `policy.device=mps` 在Apple silicon上训练。
5. 我们提供了 `wandb.enable=true` 以使用[Weights and Biases](https://docs.wandb.ai/quickstart)来可视化训练图表。这是可选的，但如果使用，请确保通过运行 `wandb login` 登录。

有关`train`脚本的更多信息，请参阅前面的教程：[`examples/4_train_policy_with_script.md`](../examples/4_train_policy_with_script.md)

### b. （可选）将策略检查点上传到hub

训练完成后，使用以下命令上传最新检查点：
```bash
huggingface-cli upload ${HF_USER}/act_koch_test \
  outputs/train/act_koch_test/checkpoints/last/pretrained_model
```

您也可以上传中间检查点：
```bash
CKPT=010000
huggingface-cli upload ${HF_USER}/act_koch_test_${CKPT} \
  outputs/train/act_koch_test/checkpoints/${CKPT}/pretrained_model
```

## 5. 评估您的策略

现在您有了策略检查点，可以使用[`ManipulatorRobot`](../lerobot/common/robot_devices/robots/manipulator.py)和策略中的方法轻松控制您的机器人。

尝试以下代码，以30 fps的速度运行60秒的推理：
```python
from lerobot.common.policies.act.modeling_act import ACTPolicy

inference_time_s = 60
fps = 30
device = "cuda"  # 注意：在Mac上，使用 "mps" 或 "cpu"

ckpt_path = "outputs/train/act_koch_test/checkpoints/last/pretrained_model"
policy = ACTPolicy.from_pretrained(ckpt_path)
policy.to(device)

for _ in range(inference_time_s * fps):
  start_time = time.perf_counter()

  # 读取跟随者状态并访问摄像头中的帧
  observation = robot.capture_observation()

  # 转换为pytorch格式：通道在前，float32类型，像素范围在[0,1]
  # 带有批次维度
  for name in observation:
    if "image" in name:
      observation[name] = observation[name].type(torch.float32) / 255
      observation[name] = observation[name].permute(2, 0, 1).contiguous()
    observation[name] = observation[name].unsqueeze(0)
    observation[name] = observation[name].to(device)

  # 使用策略基于当前观察计算下一个动作
  action = policy.select_action(observation)
  # 移除批次维度
  action = action.squeeze(0)
  # 移动到cpu（如果尚未移动）
  action = action.to("cpu")
  # 命令机器人移动
  robot.send_action(action)

  dt_s = time.perf_counter() - start_time
  busy_wait(1 / fps - dt_s)
```

### a. 使用我们的 `record` 函数

理想情况下，当使用神经网络控制机器人时，您可能希望记录评估回合，以便稍后可视化它们，甚至像在强化学习中那样对它们进行训练。这基本上对应于记录一个新数据集，但使用神经网络提供动作而不是远程操作。

为此，您可以使用[`lerobot/scripts/control_robot.py`](../lerobot/scripts/control_robot.py)中的`record`函数，但将策略检查点作为输入。例如，运行以下命令记录10个评估回合：
```bash
python lerobot/scripts/control_robot.py \
  --robot.type=koch \
  --control.type=record \
  --control.fps=30 \
  --control.repo_id=${HF_USER}/eval_act_koch_test \
  --control.tags='["tutorial"]' \
  --control.warmup_time_s=5 \
  --control.episode_time_s=30 \
  --control.reset_time_s=30 \
  --control.num_episodes=10 \
  --control.push_to_hub=true \
  --control.policy.path=outputs/train/act_koch_test/checkpoints/last/pretrained_model
```

如您所见，它几乎与之前用于记录训练数据集的命令相同。有两点变化：
1. 有一个额外的 `--control.policy.path` 参数，指示策略检查点的路径（例如 `outputs/train/eval_koch_test/checkpoints/last/pretrained_model`）。如果您将模型检查点上传到hub，您也可以使用模型库（例如 `${HF_USER}/act_koch_test`）。
2. 数据集名称以 `eval` 开头，以反映您正在运行推理（例如 `${HF_USER}/eval_act_koch_test`）。

### b. 事后可视化评估

然后，您可以通过运行与之前相同的命令但使用新的推理数据集作为参数来可视化您的评估数据集：
```bash
python lerobot/scripts/visualize_dataset.py \
  --repo-id ${HF_USER}/eval_act_koch_test
```

## 6. 下一步

加入我们的[Discord](https://discord.com/invite/s3KuuzsPFb)，一起协作收集数据，帮助我们训练完全开源的机器人学习基础模型！
