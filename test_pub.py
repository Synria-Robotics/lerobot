#!/usr/bin/env python

import rospy
import random
import time
from serial_server_node.msg import ArmJointState

def publish_random_joint_values():
    # 初始化ROS节点
    rospy.init_node('random_joint_publisher', anonymous=True)
    
    # 创建发布者
    pub = rospy.Publisher('/read_lead_arm', ArmJointState, queue_size=10)
    
    # 设置发布频率
    rate = rospy.Rate(30)  # 5Hz，与您的原始命令保持一致
    
    print("开始发布随机关节数据到 /read_lead_arm 话题...")
    
    count = 0
    current_values = {
        'joint1': 0.0,
        'joint2': 0.0,
        'joint3': 0.0,
        'joint4': 0.0,
        'joint5': 22.0,
        'joint6': 0.0,
        'gripper': 0.0
    }
    
    while not rospy.is_shutdown():
        # 创建消息
        msg = ArmJointState()
        
        # 每20次循环随机变化一次数值
        if count % 3 == 0:
            # 随机生成关节角度
            current_values['joint1'] = random.uniform(-45.0, 45.0)
            current_values['joint2'] = random.uniform(-90.0, 90.0)
            current_values['joint3'] = random.uniform(-90.0, 90.0)
            current_values['joint4'] = random.uniform(-90.0, 90.0)
            current_values['joint5'] = random.uniform(-90.0, 90.0)
            current_values['joint6'] = random.uniform(-180.0, 180.0)
            current_values['gripper'] = random.uniform(0.0, 100.0)
        
        # 填充消息头
        msg.header.seq = count
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = ''
        
        # 填充关节数据
        msg.joint1 = current_values['joint1']
        msg.joint2 = current_values['joint2']
        msg.joint3 = current_values['joint3']
        msg.joint4 = current_values['joint4']
        msg.joint5 = current_values['joint5']
        msg.joint6 = current_values['joint6']
        msg.gripper = current_values['gripper']
        msg.time = time.time()
        
        # 发布消息
        pub.publish(msg)
        
        # 打印当前数值
        if count % 30 == 0:
            print(f"发布 #{count}: [j1:{msg.joint1:.1f}, j2:{msg.joint2:.1f}, j3:{msg.joint3:.1f}, "
                  f"j4:{msg.joint4:.1f}, j5:{msg.joint5:.1f}, j6:{msg.joint6:.1f}, g:{msg.gripper:.1f}]")
        
        count += 1
        rate.sleep()

if __name__ == '__main__':
    try:
        publish_random_joint_values()
    except rospy.ROSInterruptException:
        print("程序被中断")
        pass