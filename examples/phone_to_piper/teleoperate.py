# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specif

import time
import logging
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

from lerobot_robot_piper import PiperFollower, PiperFollowerConfig
from lerobot.teleoperators.phone import Phone, PhoneConfig
from lerobot.teleoperators.phone.config_phone import PhoneOS
from lerobot.utils.robot_utils import precise_sleep

FPS = 30
logger = logging.getLogger(__name__)

def main():
    logging.basicConfig(level=logging.INFO)

    # 1. Initialize Robot and Phone Configs
    robot_config = PiperFollowerConfig(
        can_port="can0",  # Adjust to your physical CAN port interface
        use_mit_mode=False,
        speed_rate=10,
    )
    teleop_config = PhoneConfig(phone_os=PhoneOS.ANDROID)

    robot = PiperFollower(robot_config)
    teleop_device = Phone(teleop_config)

    # Connect to both devices
    robot.connect()
    teleop_device.connect()

    if not robot.is_connected or not teleop_device.is_connected:
        raise ValueError("Robot or teleoperator device is not connected!")

    # State variables for relative Cartesian mapping
    enabled_prev = False
    x_init, y_init, z_init = 0.0, 0.0, 0.0
    rot_init = Rotation.identity()

    # Gripper state variables
    gripper_pos = 0.0  # mm
    gripper_speed_factor = 20.0
    dt = 1.0 / FPS

    print("\nStarting teleop loop. Move your phone while holding the touch trigger...")

    try:
        while True:
            t0 = time.perf_counter()

            # Read joint observations (for record/logging)
            joint_obs = robot.get_observation()

            # Read phone controller updates
            phone_obs = teleop_device.get_action()
            if not phone_obs:
                precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
                continue

            enabled = phone_obs["phone.enabled"]
            pos_cal = phone_obs["phone.pos"]  # relative translation vector
            rot_cal = phone_obs["phone.rot"]  # relative rotation
            inputs = phone_obs["phone.raw_inputs"]

            # Handle rising edge of the clutch trigger button (engagement)
            if enabled and not enabled_prev:
                print("Clutch engaged! Capturing starting pose...")
                
                # Fetch starting pose from robot (retry a few times if return is empty/none)
                pose_msg = None
                for _ in range(10):
                    pose_msg = robot.piper.GetArmEndPoseMsgs()
                    if pose_msg is not None and pose_msg.end_pose is not None:
                        if pose_msg.end_pose.X_axis != 0 or pose_msg.end_pose.Y_axis != 0:
                            break
                    time.sleep(0.02)

                if pose_msg is None or pose_msg.end_pose is None:
                    print("Warning: Failed to read end pose from Piper. Skipping this frame.")
                    enabled_prev = False
                    continue

                # Convert SDK micrometers to meters (divide by 10^6)
                x_init = pose_msg.end_pose.X_axis * 1e-6
                y_init = pose_msg.end_pose.Y_axis * 1e-6
                z_init = pose_msg.end_pose.Z_axis * 1e-6

                # Convert SDK millidegrees to degrees (divide by 1000)
                rpy_init = [
                    pose_msg.end_pose.RX_axis / 1000.0,
                    pose_msg.end_pose.RY_axis / 1000.0,
                    pose_msg.end_pose.RZ_axis / 1000.0
                ]
                rot_init = Rotation.from_euler("xyz", rpy_init, degrees=True)

                # Reset target gripper position to current gripper observation
                gripper_pos = joint_obs.get("gripper.pos", 0.0)

            # Map inputs and command the robot
            if enabled:
                # Map phone movement axes to the robot's base coordinate frame
                # WebXR/HEBI phone: pos_cal is [X_right, Y_forward, Z_up] relative translation
                # Standard mapping to robot base frame (+X forward, +Y left, +Z up):
                dx = -pos_cal[1]
                dy = pos_cal[0]
                dz = pos_cal[2]

                # Accumulate the relative motion onto the starting pose
                target_pos = np.array([x_init, y_init, z_init]) + np.array([dx, dy, dz])
                target_rot = Rotation.from_matrix(rot_cal.as_matrix() @ rot_init.as_matrix())
                target_rpy = target_rot.as_euler("xyz", degrees=True)

                # Process gripper speed from button inputs (Button A = open/close, Button B = opposite)
                a = float(inputs.get("reservedButtonA", 0.0))
                b = float(inputs.get("reservedButtonB", 0.0))
                gripper_vel = a - b
                gripper_pos = np.clip(gripper_pos + gripper_vel * gripper_speed_factor * dt, 0.0, 70.0)

                # Scale coordinates/rotations for the SDK
                # (X, Y, Z in 0.001 mm; RX, RY, RZ in 0.001 degrees)
                x_sdk = int(round(target_pos[0] * 1e6))
                y_sdk = int(round(target_pos[1] * 1e6))
                z_sdk = int(round(target_pos[2] * 1e6))
                rx_sdk = int(round(target_rpy[0] * 1000))
                ry_sdk = int(round(target_rpy[1] * 1000))
                rz_sdk = int(round(target_rpy[2] * 1000))

                # Write control mode to MOVE P (Cartesian Position Control) -> 0x00
                robot.piper.MotionCtrl_2(0x01, 0x00, robot_config.speed_rate, 0xAD)
                
                # Command Cartesian coordinates
                robot.piper.EndPoseCtrl(x_sdk, y_sdk, z_sdk, rx_sdk, ry_sdk, rz_sdk)

                # Command gripper (convert mm to 0.001 mm)
                gripper_sdk = int(round(gripper_pos * 1000))
                robot.piper.GripperCtrl(abs(gripper_sdk), robot_config.gripper_effort, 0x01, 0)

            enabled_prev = enabled
            precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))

    except KeyboardInterrupt:
        print("\nExiting teleoperation...")
    finally:
        # Disconnect devices cleanly
        robot.disconnect()
        teleop_device.disconnect()

if __name__ == "__main__":
    main()
