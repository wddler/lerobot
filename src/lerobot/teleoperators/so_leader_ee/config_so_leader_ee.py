#!/usr/bin/env python

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

from dataclasses import dataclass

from ..config import TeleoperatorConfig
from ..so_leader.config_so_leader import SOLeaderTeleopConfig


@TeleoperatorConfig.register_subclass("so101_leader_ee")
@TeleoperatorConfig.register_subclass("so100_leader_ee")
@dataclass(kw_only=True)
class SOLeaderEETeleopConfig(SOLeaderTeleopConfig):
    """SO leader (SO100/SO101), but reporting a Cartesian delta instead of raw joint angles.

    Pairs with a follower whose `send_action()` understands the Cartesian action schema (see
    `SOLeaderEETeleop`/`MapSOLeaderToRobotAction`), e.g. `piper_follower` -- useful when the
    leader and follower have a different number of joints and a direct joint-space mapping
    doesn't apply.
    """

    # Path to the leader's URDF, used for forward kinematics. It is highly recommended to use
    # the SO101 URDF from https://github.com/TheRobotStudio/SO-ARM100 (Simulation/SO101/so101_new_calib.urdf).
    urdf_path: str

    # Name of the end-effector frame in the URDF.
    target_frame_name: str = "gripper_frame_link"

    # Multiplier applied to the leader's Cartesian position delta before it's reported, e.g.
    # 2.0 makes 10cm of leader motion a 20cm follower motion. Orientation is never scaled.
    scale_factor: float = 1.0


SO100LeaderEETeleopConfig = SOLeaderEETeleopConfig
SO101LeaderEETeleopConfig = SOLeaderEETeleopConfig
