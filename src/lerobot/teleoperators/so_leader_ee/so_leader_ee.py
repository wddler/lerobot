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

import logging
from queue import Queue
from typing import Any

from lerobot.model.kinematics import RobotKinematics
from lerobot.utils.decorators import check_if_not_connected
from lerobot.utils.import_utils import _pynput_available

from ..so_leader.so_leader import SOLeader
from ..so_leader.so_leader_processor import MapSOLeaderToRobotAction
from .config_so_leader_ee import SOLeaderEETeleopConfig

logger = logging.getLogger(__name__)

PYNPUT_AVAILABLE = _pynput_available
keyboard = None
if PYNPUT_AVAILABLE:
    try:
        from pynput import keyboard
    except Exception as e:
        PYNPUT_AVAILABLE = False
        logging.info("Could not import pynput keyboard backend: %s", e)


class SOLeaderEETeleop(SOLeader):
    """SO leader (SO100/SO101) reporting a Cartesian delta instead of raw joint angles.

    Reuses `SOLeader`'s motor bus (identical hardware setup, so a follower-model arm used as a
    manual input device works exactly the same way), but `get_action()` runs forward kinematics
    on the read joints and reports a position + orientation delta (see
    `MapSOLeaderToRobotAction`) instead of the joints themselves -- letting it drive a follower
    with a different number of joints (e.g. a 6-DOF Piper) that understands the same Cartesian
    action schema in its own `send_action()`.

    The clutch ("enabled") has no dedicated input on an SO leader, so this teleoperator holds
    its own SPACE-bar keyboard listener (same mechanism as `lerobot.teleoperators.keyboard`):
    hold SPACE to report `enabled=True`. This requires an X11 or trusted-macOS/Windows session
    for `pynput` to capture the held key; otherwise `enabled` is always reported `False` (a
    warning is logged once, at connect time).
    """

    config_class = SOLeaderEETeleopConfig
    name = "so_leader_ee"

    def __init__(self, config: SOLeaderEETeleopConfig):
        super().__init__(config)
        self.config = config

        self.kinematics = RobotKinematics(
            urdf_path=config.urdf_path,
            target_frame_name=config.target_frame_name,
            joint_names=list(self.bus.motors.keys()),
        )
        self._map_step = MapSOLeaderToRobotAction(
            kinematics=self.kinematics,
            motor_names=list(self.bus.motors.keys()),
            scale_factor=config.scale_factor,
        )

        self._listener: Any = None
        self._event_queue: Queue = Queue()
        self._current_pressed: dict = {}

    @property
    def action_features(self) -> dict[str, type]:
        return {
            "enabled": float,
            "target_x": float,
            "target_y": float,
            "target_z": float,
            "target_wx": float,
            "target_wy": float,
            "target_wz": float,
            "gripper_pos": float,
        }

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    def connect(self, calibrate: bool = True) -> None:
        super().connect(calibrate)
        if PYNPUT_AVAILABLE:
            self._listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
            self._listener.start()
        else:
            logger.warning(
                "pynput is unavailable in this environment (no X11/trusted-macOS/Windows session), "
                "so the SPACE clutch cannot be captured. %s will report enabled=False.",
                self,
            )

    def _on_press(self, key) -> None:
        self._event_queue.put((key, True))

    def _on_release(self, key) -> None:
        self._event_queue.put((key, False))

    def _clutch_engaged(self) -> bool:
        while not self._event_queue.empty():
            key, pressed = self._event_queue.get_nowait()
            self._current_pressed[key] = pressed
        if not PYNPUT_AVAILABLE:
            return False
        return bool(self._current_pressed.get(keyboard.Key.space, False))

    @check_if_not_connected
    def get_action(self) -> dict[str, float]:
        action = super().get_action()
        action["enabled"] = self._clutch_engaged()
        return self._map_step.action(action)

    def disconnect(self) -> None:
        if self._listener is not None:
            self._listener.stop()
        super().disconnect()


SO100LeaderEETeleop = SOLeaderEETeleop
SO101LeaderEETeleop = SOLeaderEETeleop
