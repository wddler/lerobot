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

# Length of the physical gripper in meters (along local Z-axis of the wrist flange)
# Piper's standard gripper is approx 14.5 cm
TOOL_OFFSET_Z = 0.145

# Cartesian safety bounds [min, max] in meters relative to the robot base
# applied to the TCP (gripper tip). Tabletop is at Z = 0.0, using 0.03 for 3cm clearance.
EE_BOUNDS = {
    "min": np.array([-0.6, -0.6, 0.03]),
    "max": np.array([0.6, 0.6, 0.6]),
}
# Maximum allowed end-effector translation per step (in meters) to rate-limit tracking jumps
MAX_EE_STEP_M = 0.05

def main():
    logging.basicConfig(level=logging.INFO)

    # 1. Initialize Robot and Phone Configs
    robot_config = PiperFollowerConfig(
        can_port="can0",  # Adjust to your physical CAN port interface
        use_mit_mode=False,
        speed_rate=50,
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
    tcp_init_pos = np.zeros(3)
    rot_flange_init = Rotation.identity()
    last_pos = None

    # Read initial observations to get starting gripper position
    joint_obs = robot.get_observation()

    # Gripper state variables
    gripper_pos = joint_obs.get("gripper.pos", 0.0)  # mm
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

            # Process gripper speed from button inputs (Button A = open/close, Button B = opposite)
            # This runs at all times, so the gripper works even when the arm movement is locked (clutch disengaged)
            a = float(inputs.get("reservedButtonA", 0.0))
            b = float(inputs.get("reservedButtonB", 0.0))
            gripper_vel = a - b
            gripper_pos = np.clip(gripper_pos + gripper_vel * gripper_speed_factor * dt, 0.0, 70.0)

            # Command gripper (convert mm to 0.001 mm)
            gripper_sdk = int(round(gripper_pos * 1000))
            robot.piper.GripperCtrl(abs(gripper_sdk), robot_config.gripper_effort, 0x01, 0)

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
                x_flange_init = pose_msg.end_pose.X_axis * 1e-6
                y_flange_init = pose_msg.end_pose.Y_axis * 1e-6
                z_flange_init = pose_msg.end_pose.Z_axis * 1e-6

                # Convert SDK millidegrees to degrees (divide by 1000)
                rpy_flange_init = [
                    pose_msg.end_pose.RX_axis / 1000.0,
                    pose_msg.end_pose.RY_axis / 1000.0,
                    pose_msg.end_pose.RZ_axis / 1000.0
                ]
                rot_flange_init = Rotation.from_euler("xyz", rpy_flange_init, degrees=True)

                # Calculate initial Tool Center Point (TCP) position by offsetting along the local Z-axis
                flange_init_pos = np.array([x_flange_init, y_flange_init, z_flange_init])
                local_z_init = rot_flange_init.as_matrix()[:, 2]
                tcp_init_pos = flange_init_pos + local_z_init * TOOL_OFFSET_Z

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

                # Accumulate the relative motion onto the starting TCP position
                target_tcp_pos = tcp_init_pos + np.array([dx, dy, dz])
                
                # Apply tabletop and workspace bounds safety clipping directly to the TCP (gripper tip)
                target_tcp_pos = np.clip(target_tcp_pos, EE_BOUNDS["min"], EE_BOUNDS["max"])

                # Calculate target orientation
                target_rot = Rotation.from_matrix(rot_cal.as_matrix() @ rot_flange_init.as_matrix())
                target_rpy = target_rot.as_euler("xyz", degrees=True)

                # Convert target TCP position back to required wrist flange coordinates for the SDK
                local_z_target = target_rot.as_matrix()[:, 2]
                target_flange_pos = target_tcp_pos - local_z_target * TOOL_OFFSET_Z

                # Rate-limit jumps on the commanded flange position
                if last_pos is not None:
                    dpos = target_flange_pos - last_pos
                    step_len = np.linalg.norm(dpos)
                    if step_len > MAX_EE_STEP_M:
                        target_flange_pos = last_pos + dpos * (MAX_EE_STEP_M / step_len)
                last_pos = target_flange_pos



                # Scale coordinates/rotations for the SDK
                # (X, Y, Z in 0.001 mm; RX, RY, RZ in 0.001 degrees)
                x_sdk = int(round(target_flange_pos[0] * 1e6))
                y_sdk = int(round(target_flange_pos[1] * 1e6))
                z_sdk = int(round(target_flange_pos[2] * 1e6))
                rx_sdk = int(round(target_rpy[0] * 1000))
                ry_sdk = int(round(target_rpy[1] * 1000))
                rz_sdk = int(round(target_rpy[2] * 1000))

                # Write control mode to MOVE P (Cartesian Position Control) -> 0x00
                robot.piper.MotionCtrl_2(0x01, 0x00, robot_config.speed_rate, 0xAD)
                
                # Command Cartesian coordinates
                robot.piper.EndPoseCtrl(x_sdk, y_sdk, z_sdk, rx_sdk, ry_sdk, rz_sdk)



            else:
                # Reset tracking history when clutch is disengaged
                last_pos = None

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
