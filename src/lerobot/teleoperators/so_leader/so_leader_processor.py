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

from dataclasses import dataclass, field

import numpy as np

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import RobotAction
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import ProcessorStepRegistry, RobotActionProcessorStep
from lerobot.utils.rotation import Rotation


@ProcessorStepRegistry.register("map_so_leader_to_robot_action")
@dataclass
class MapSOLeaderToRobotAction(RobotActionProcessorStep):
    """
    Maps raw SO leader (SO100/SO101) joint angles to a Cartesian delta command.

    The SO leader has no dedicated Cartesian output, so this step runs forward kinematics on its
    joints to recover its end-effector pose, and reports the position and orientation delta since
    the pose latched at the last rising edge of `enabled`. The orientation delta is expressed in
    the *world/base frame* (`R_curr @ R_ref.T`), not the leader's own local frame: applying it as
    `delta @ R_follower_ref` on a follower rotates the follower the same way in absolute space
    (tilt the leader down, the follower tilts down), regardless of how each arm's own URDF happens
    to define its end-effector-local axes. This assumes the leader and follower base frames are
    themselves roughly aligned (both mounted the same way, e.g. side by side on the same table) --
    if they're not, a body-frame delta would need an explicit calibration rotation between the two
    conventions instead, which this step does not attempt. This also sidesteps the SO101 (5-DOF)
    vs. a 6-DOF follower joint-count mismatch, since only a Cartesian delta is produced, not joint
    angles.

    The `enabled` clutch signal is not derived from the leader arm itself (SO leaders have no
    dedicated clutch input); the caller must inject an `"enabled"` key into the action dict (e.g.
    from a held keyboard key) before this step runs.

    The gripper is passed through as an absolute position: the SO leader's gripper is itself an
    analog 0-100 position sensor (not a button), so there's no need to integrate a velocity like
    e.g. the phone teleoperator does.

    Attributes:
        kinematics: Forward-kinematics model for the SO leader.
        motor_names: Leader motor names, in the order the `kinematics` solver was built with
            (`gripper` must be last, matching the SO leader's own motor bus order).
        scale_factor: Multiplier applied to the leader's Cartesian position delta before it's
            reported, e.g. 2.0 makes a 10cm leader motion a 20cm follower motion. Orientation is
            never scaled.
        reference_pos: Internal state: the leader EE position latched at the last rising edge of
            `enabled`.
        reference_rot: Internal state: the leader EE rotation matrix latched at the last rising
            edge of `enabled`.
        _prev_enabled: Internal state used to detect the rising edge of `enabled`.
    """

    kinematics: RobotKinematics
    motor_names: list[str]
    scale_factor: float = 1.0

    reference_pos: np.ndarray | None = field(default=None, init=False, repr=False)
    reference_rot: np.ndarray | None = field(default=None, init=False, repr=False)
    _prev_enabled: bool = field(default=False, init=False, repr=False)

    def action(self, action: RobotAction) -> RobotAction:
        enabled = bool(action.pop("enabled"))

        q = np.array([float(action.pop(f"{name}.pos")) for name in self.motor_names], dtype=float)
        # Gripper is the last motor in the SO leader's bus (see so_leader.py); it doesn't
        # participate in the Cartesian delta, just pass its position straight through.
        gripper_pos = float(q[-1])

        t_curr = self.kinematics.forward_kinematics(q)
        pos_curr = t_curr[:3, 3]
        rot_curr = t_curr[:3, :3]

        if enabled:
            if not self._prev_enabled or self.reference_pos is None:
                self.reference_pos = pos_curr.copy()
                self.reference_rot = rot_curr.copy()
            delta_pos = (pos_curr - self.reference_pos) * self.scale_factor
            # Rotation delta since latch, expressed in the world/base frame (see class docstring).
            delta_rot = rot_curr @ self.reference_rot.T
            delta_rotvec = Rotation.from_matrix(delta_rot).as_rotvec()
        else:
            # Force a fresh latch next time the clutch is engaged.
            self.reference_pos = None
            self.reference_rot = None
            delta_pos = np.zeros(3)
            delta_rotvec = np.zeros(3)

        action["enabled"] = enabled
        action["target_x"] = float(delta_pos[0])
        action["target_y"] = float(delta_pos[1])
        action["target_z"] = float(delta_pos[2])
        action["target_wx"] = float(delta_rotvec[0])
        action["target_wy"] = float(delta_rotvec[1])
        action["target_wz"] = float(delta_rotvec[2])
        action["gripper_pos"] = gripper_pos

        self._prev_enabled = enabled
        return action

    def reset(self):
        """Resets the latched reference pose and the enabled edge-detector."""
        self.reference_pos = None
        self.reference_rot = None
        self._prev_enabled = False

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        for name in self.motor_names:
            features[PipelineFeatureType.ACTION].pop(f"{name}.pos", None)

        for feat in [
            "enabled",
            "target_x",
            "target_y",
            "target_z",
            "target_wx",
            "target_wy",
            "target_wz",
            "gripper_pos",
        ]:
            features[PipelineFeatureType.ACTION][feat] = PolicyFeature(type=FeatureType.ACTION, shape=(1,))

        return features
