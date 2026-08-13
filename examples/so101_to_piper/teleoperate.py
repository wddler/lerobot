# !/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
# See the License for the specific language governing permissions and
# limitations under the License.

"""SO101 leader -> Piper teleoperation in end-effector (Cartesian) space.

The SO101 leader is 5-DOF + gripper; Piper is 6-DOF + gripper, so their joints don't map 1:1.
Instead we run forward kinematics on the SO101 leader to recover its end-effector pose, take the
position + orientation delta since the clutch was last engaged, scale the position part, and feed
that delta straight into Piper's own onboard Cartesian controller (`EndPoseCtrl`) -- the same
mechanism `examples/phone_to_piper` uses, just driven by the SO101 leader instead of a phone.

The orientation delta is applied in the world/base frame (see `MapSOLeaderToRobotAction`), so
tilting/rotating the SO101 leader in a given real-world direction rotates Piper's gripper the same
way, regardless of how each arm's own URDF defines its end-effector-local axes. This assumes both
arms are mounted the same way (e.g. side by side on the same table) -- if they're rotated relative
to each other, the mapping will be off by that same rotation. The SO101's own gripper position
(0-100, analog) is mirrored 1:1 onto Piper's gripper (0-70mm).

Hold SPACE to engage the clutch (Piper tracks the SO101 leader's delta); release it and Piper
freezes in place. Requires an X11 or trusted-macOS session (or Windows) for pynput to capture the
held key -- see `lerobot.utils.keyboard_input` for details. Without one, the clutch never engages
and a warning is logged at startup; Piper's gripper still stays live.
"""

import logging
import time
from pathlib import Path

import numpy as np
from lerobot_robot_piper import PiperFollower, PiperFollowerConfig
from scipy.spatial.transform import Rotation

from lerobot.lerobot_types import RobotAction
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import RobotProcessorPipeline, robot_action_to_transition, transition_to_robot_action
from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
from lerobot.teleoperators.so_leader.so_leader_processor import MapSOLeaderToRobotAction
from lerobot.utils.import_utils import _pynput_available
from lerobot.utils.robot_utils import precise_sleep

FPS = 30
logger = logging.getLogger(__name__)

# Guarded the same way as lerobot.teleoperators.keyboard.teleop_keyboard: importing `pynput.keyboard`
# doesn't just check the package is installed, it also probes for a usable display backend and
# raises if none is found (e.g. no X server / empty DISPLAY, as over a plain SSH session). Catch
# that here so the script degrades to "clutch never engages" instead of crashing at import time.
PYNPUT_AVAILABLE = _pynput_available
keyboard = None
if PYNPUT_AVAILABLE:
    try:
        from pynput import keyboard
    except Exception as e:
        PYNPUT_AVAILABLE = False
        logger.warning(
            "Could not initialize pynput keyboard backend (%s); the SPACE clutch will be "
            "unavailable and Piper's arm will never move (only its gripper will stay live). "
            "Run this from an X11 session (or trusted macOS/Windows) to use the clutch.",
            e,
        )

# Multiplier applied to the SO101 leader's Cartesian position delta before sending it to Piper,
# e.g. 2.0 makes 10cm of SO101 motion move Piper by 20cm.
SCALE_FACTOR = 1.0

# The SO101 URDF+meshes bundled with the phone_to_piper example; reused here to avoid duplicating
# the (multi-MB) mesh assets.
SO101_URDF_PATH = str(
    (Path(__file__).parent / "../phone_to_piper/piper/so101_new_calib.urdf").resolve()
)

# Length of the physical gripper in meters (along local Z-axis of the wrist flange)
# Piper's standard gripper is approx 14.5 cm
TOOL_OFFSET_Z = 0.145

# Cartesian safety bounds [min, max] in meters relative to the robot base, applied to the TCP
# (gripper tip). Tabletop is at Z = 0.0, using 0.03 for 3cm clearance.
EE_BOUNDS = {
    "min": np.array([-0.6, -0.3, 0.01]),
    "max": np.array([0.6, 0.3, 0.6]),
}
# Maximum allowed end-effector translation per step (in meters) to rate-limit tracking jumps
MAX_EE_STEP_M = 0.05


def main():
    logging.basicConfig(level=logging.INFO)

    # 1. Initialize Robot, leader and clutch configs
    robot_config = PiperFollowerConfig(
        can_port="can0",  # Adjust to your physical CAN port interface
        use_mit_mode=False,
        speed_rate=25,
    )
    leader_config = SO101LeaderConfig(port="/dev/ttyACM0")  # Adjust to your SO101 leader's serial port

    robot = PiperFollower(robot_config)
    leader = SO101Leader(leader_config)
    clutch = KeyboardTeleop(KeyboardTeleopConfig())

    # Connect all devices
    robot.connect()
    leader.connect()
    clutch.connect()

    if not robot.is_connected or not leader.is_connected:
        raise ValueError("Robot or SO101 leader is not connected!")

    # 2. Build the SO101 leader forward-kinematics solver and the leader-joints -> Cartesian-delta
    # processor step.
    leader_kinematics_solver = RobotKinematics(
        urdf_path=SO101_URDF_PATH,
        target_frame_name="gripper_frame_link",
        joint_names=list(leader.bus.motors.keys()),
    )

    leader_to_delta = RobotProcessorPipeline[RobotAction, RobotAction](
        steps=[
            MapSOLeaderToRobotAction(
                kinematics=leader_kinematics_solver,
                motor_names=list(leader.bus.motors.keys()),
                scale_factor=SCALE_FACTOR,
            ),
        ],
        to_transition=robot_action_to_transition,
        to_output=transition_to_robot_action,
    )

    # State variables for relative Cartesian mapping
    enabled_prev = False
    tcp_init_pos = np.zeros(3)
    rot_flange_init = Rotation.identity()
    last_pos = None
    dt = 1.0 / FPS

    print("\nStarting teleop loop. Hold SPACE and move the SO101 leader...")

    try:
        while True:
            t0 = time.perf_counter()

            # Read SO101 leader joints and run FK -> Cartesian delta since the clutch was engaged
            leader_action = leader.get_action()
            leader_action["enabled"] = PYNPUT_AVAILABLE and keyboard.Key.space in clutch.get_action()
            delta_action = leader_to_delta(leader_action)

            enabled = delta_action["enabled"]

            # Mirror SO101's 0-100 gripper 1:1 onto Piper's 0-70mm gripper (always active, even
            # when the clutch is disengaged, so the gripper stays responsive while arm motion is
            # locked)
            gripper_mm = np.clip(delta_action["gripper_pos"], 0.0, 100.0) / 100.0 * 70.0
            gripper_sdk = int(round(gripper_mm * 1000))
            robot.piper.GripperCtrl(abs(gripper_sdk), robot_config.gripper_effort, 0x01, 0)

            # Handle rising edge of the clutch (SPACE) engagement
            if enabled and not enabled_prev:
                print("Clutch engaged! Capturing starting pose...")

                # Fetch starting pose from robot (retry a few times if return is empty/none)
                pose_msg = None
                for _ in range(10):
                    pose_msg = robot.piper.GetArmEndPoseMsgs()
                    if (
                        pose_msg is not None
                        and pose_msg.end_pose is not None
                        and (pose_msg.end_pose.X_axis != 0 or pose_msg.end_pose.Y_axis != 0)
                    ):
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

                # Convert SDK millidegrees to degrees (divide by 1000). This is the orientation
                # reference that the SO101 leader's rotation delta gets applied on top of.
                rpy_flange_init = [
                    pose_msg.end_pose.RX_axis / 1000.0,
                    pose_msg.end_pose.RY_axis / 1000.0,
                    pose_msg.end_pose.RZ_axis / 1000.0,
                ]
                rot_flange_init = Rotation.from_euler("xyz", rpy_flange_init, degrees=True)
                print(f"Piper orientation at clutch-engage (RX, RY, RZ deg): {np.round(rpy_flange_init, 1)}")
                if abs(abs(rpy_flange_init[1]) - 90.0) < 15.0:
                    print(
                        "  Warning: RY is close to +/-90deg -- the 'xyz' Euler decomposition used to "
                        "talk to the Piper SDK is singular (gimbal lock) near there, which can make X/Y "
                        "orientation changes scramble into large RX/RZ jumps instead of tracking smoothly. "
                        "If X/Y orientation tracking looks broken, try re-engaging the clutch from a pose "
                        "tilted noticeably away from straight-down."
                    )

                # Calculate initial Tool Center Point (TCP) position by offsetting along the local
                # Z-axis
                flange_init_pos = np.array([x_flange_init, y_flange_init, z_flange_init])
                local_z_init = rot_flange_init.as_matrix()[:, 2]
                tcp_init_pos = flange_init_pos + local_z_init * TOOL_OFFSET_Z

            # Map SO101 delta and command the robot
            if enabled:
                # Accumulate the SO101 leader's Cartesian delta onto the starting TCP position.
                target_tcp_pos = tcp_init_pos + np.array(
                    [delta_action["target_x"], delta_action["target_y"], delta_action["target_z"]]
                )

                # Apply tabletop and workspace bounds safety clipping directly to the TCP (gripper
                # tip)
                target_tcp_pos = np.clip(target_tcp_pos, EE_BOUNDS["min"], EE_BOUNDS["max"])

                # Apply the SO101 leader's rotation delta (since clutch-engage, expressed in the
                # world/base frame -- see MapSOLeaderToRobotAction) on top of Piper's orientation at
                # clutch-engage time. Pre-multiplying applies the delta in world frame: tilt the
                # SO101 leader down and Piper tilts down too, regardless of either arm's own local
                # end-effector axis convention. This assumes both arms are mounted the same way
                # (e.g. side by side on the same table).
                delta_rot = Rotation.from_rotvec(
                    [delta_action["target_wx"], delta_action["target_wy"], delta_action["target_wz"]]
                )
                target_rot = Rotation.from_matrix(delta_rot.as_matrix() @ rot_flange_init.as_matrix())
                target_rpy = target_rot.as_euler("xyz", degrees=True)

                # Convert target TCP position back to required wrist flange coordinates for the
                # SDK, using the *target* orientation (not the fixed reference one)
                local_z = target_rot.as_matrix()[:, 2]
                target_flange_pos = target_tcp_pos - local_z * TOOL_OFFSET_Z

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
                # Reset tracking history when the clutch is disengaged
                last_pos = None

            enabled_prev = enabled
            precise_sleep(max(dt - (time.perf_counter() - t0), 0.0))

    except KeyboardInterrupt:
        print("\nExiting teleoperation...")
    finally:
        # Disconnect devices cleanly
        robot.disconnect()
        leader.disconnect()
        clutch.disconnect()


if __name__ == "__main__":
    main()
