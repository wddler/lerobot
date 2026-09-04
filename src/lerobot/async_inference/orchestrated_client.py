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
# See the License for the specific language governing permissions and
# limitations under the License.

"""Orchestrated async-inference client.

Same as ``robot_client`` (streams robot observations to a remote
``policy_server`` and executes the action chunks it returns), except the task
instruction is not fixed for the run. A small local HTTP control server lets an
external process — here, the ``vlm-orchestrator`` — swap the instruction and
pause/resume execution while this client stays connected and the policy stays
resident on the GPU server.

Control API (default ``http://127.0.0.1:8791``):

======  ============  ==============================================================
method  path          body / effect
======  ============  ==============================================================
POST    /task         ``{"task": "..."}`` — set the instruction and resume execution
POST    /idle         hold the arm, stop sending observations to the policy
GET     /status       ``{"active": bool, "task": str, "latest_action": int}``
GET     /frame.jpg    most recent frame from ``--frame_camera`` as JPEG
======  ============  ==============================================================

The client boots **idle**: it connects, sends the policy instructions (the
server loads the model once), then waits for the first ``POST /task`` before it
streams anything or moves the arm.

Recording (on by default): every active segment — from one ``POST /task`` to the
next ``/task`` or ``/idle`` — is written as one episode of a ``LeRobotDataset``,
so all rollouts are captured. Disable with ``--record=false``; choose where it
goes with ``--dataset_repo_id`` / ``--dataset_root`` / ``--dataset_push_to_hub``.

Example — GPU box:

```shell
CUDA_VISIBLE_DEVICES=1 python -m lerobot.async_inference.policy_server \
    --host=0.0.0.0 --port=8081
```

Example — robot laptop:

```shell
python -m lerobot.async_inference.orchestrated_client \
    --config_path=src/lerobot/configs/rollout_orchestrated_piperx.toml \
    --control_host=127.0.0.1 --control_port=8791 --frame_camera=top
```
"""

import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from pprint import pformat

import draccus

from lerobot.utils.import_utils import register_third_party_plugins

from .configs import RobotClientConfig
from .helpers import RawObservation
from .robot_client import RobotClient


@dataclass
class OrchestratedClientConfig(RobotClientConfig):
    """``RobotClientConfig`` plus the local control-server and recording settings."""

    control_host: str = field(
        default="127.0.0.1", metadata={"help": "Host/interface for the local control server"}
    )
    control_port: int = field(default=8791, metadata={"help": "Port for the local control server"})
    frame_camera: str = field(
        default="top",
        metadata={"help": "Camera key (from [robot.cameras.*]) served at GET /frame.jpg"},
    )

    record: bool = field(
        default=True, metadata={"help": "Record every rollout segment to a LeRobotDataset"}
    )
    dataset_repo_id: str = field(
        default="",
        metadata={"help": "Dataset repo id 'namespace/name'; blank auto-derives from the checkpoint"},
    )
    dataset_root: str = field(
        default="", metadata={"help": "Local dataset directory; blank uses the LeRobot default cache"}
    )
    dataset_push_to_hub: bool = field(
        default=False, metadata={"help": "Push the recorded dataset to the Hub on shutdown"}
    )


def auto_dataset_repo_id(pretrained_path: str) -> str:
    """Derive a dataset repo id from a checkpoint path, e.g.
    '.../pi05_lora_rabc/080000/pretrained_model' -> 'local/eval_pi05_lora_rabc_080000_<ts>'."""
    parts = [p for p in Path(str(pretrained_path)).parts if p not in ("/", "pretrained_model", "merged")]
    tag = "_".join(parts[-2:]) if len(parts) >= 2 else (parts[-1] if parts else "policy")
    tag = re.sub(r"[^0-9A-Za-z_.-]+", "_", tag).strip("_")
    return f"local/eval_{tag}_{time.strftime('%Y%m%d_%H%M%S')}"


class _RolloutRecorder:
    """Writes each active task segment as one episode of a LeRobotDataset."""

    def __init__(self, robot, fps: int, repo_id: str, root: str, push_to_hub: bool, logger):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.utils.constants import ACTION, OBS_STR
        from lerobot.utils.feature_utils import (
            build_dataset_frame,
            combine_feature_dicts,
            hw_to_dataset_features,
        )

        self._logger = logger
        self._build_frame = build_dataset_frame
        self._OBS, self._ACTION = OBS_STR, ACTION
        self._action_keys = list(robot.action_features)
        self._push_to_hub = push_to_hub

        features = combine_feature_dicts(
            hw_to_dataset_features(robot.observation_features, OBS_STR, use_video=True),
            hw_to_dataset_features(robot.action_features, ACTION, use_video=True),
        )
        n_cams = len(getattr(robot, "cameras", {}) or {})
        self.dataset = LeRobotDataset.create(
            repo_id,
            fps,
            features=features,
            root=root or None,
            robot_type=robot.name,
            use_videos=True,
            image_writer_processes=0,
            image_writer_threads=4 * max(n_cams, 1),
        )
        self._episode_task: str | None = None
        self._episode_frames = 0
        self._logger.info(f"[recorder] writing to {self.dataset.root} (repo_id={repo_id})")

    @property
    def episode_task(self) -> str | None:
        return self._episode_task

    def begin_episode(self, task: str) -> None:
        self._episode_task = task
        self._episode_frames = 0

    def add(self, raw_observation: RawObservation, action: dict | None, task: str) -> None:
        if self._episode_task is None:
            self.begin_episode(task)
        obs_frame = self._build_frame(self.dataset.features, raw_observation, self._OBS)
        action = action or dict.fromkeys(self._action_keys, 0.0)
        act_frame = self._build_frame(self.dataset.features, action, self._ACTION)
        self.dataset.add_frame({**obs_frame, **act_frame, "task": task})
        self._episode_frames += 1

    def end_episode(self) -> None:
        if self._episode_task is None:
            return
        if self._episode_frames > 0:
            self.dataset.save_episode()
            self._logger.info(
                f"[recorder] saved episode: {self._episode_frames} frames, task={self._episode_task!r}"
            )
        self._episode_task = None
        self._episode_frames = 0

    def close(self) -> None:
        self.end_episode()
        try:
            self.dataset.finalize()
        except Exception as e:  # noqa: BLE001
            self._logger.warning(f"[recorder] finalize failed: {e}")
        if self._push_to_hub:
            try:
                self.dataset.push_to_hub()
            except Exception as e:  # noqa: BLE001
                self._logger.warning(f"[recorder] push_to_hub failed: {e}")


class OrchestratedRobotClient(RobotClient):
    """A ``RobotClient`` whose task instruction and active/idle state are driven
    at runtime by an external process over HTTP, recording each segment."""

    def __init__(self, config: OrchestratedClientConfig):
        super().__init__(config)
        self._frame_camera = config.frame_camera

        self._state_lock = threading.Lock()
        self._live_task: str = config.task or ""
        self._active: bool = False

        self._frame_lock = threading.Lock()
        self._latest_frame = None  # np.ndarray (H, W, 3), RGB — lerobot camera convention

        # Recording — only the control-loop thread ever touches self._recorder.
        self._recorder: _RolloutRecorder | None = None
        self._last_action: dict | None = None
        if config.record:
            repo_id = config.dataset_repo_id or auto_dataset_repo_id(config.pretrained_name_or_path)
            try:
                self._recorder = _RolloutRecorder(
                    self.robot,
                    config.fps,
                    repo_id,
                    config.dataset_root,
                    config.dataset_push_to_hub,
                    self.logger,
                )
            except Exception as e:  # noqa: BLE001
                self.logger.warning(f"[recorder] disabled — could not create dataset: {e}")

    # ------------------------------------------------------------------
    # Control API (called from the HTTP server thread)
    # ------------------------------------------------------------------
    def set_task(self, task: str) -> None:
        """Set the instruction and (re)activate execution. Clears any pending
        actions so the new instruction starts from a clean slate."""
        with self._state_lock:
            self._live_task = task
            self._active = True
        self._drain_action_queue()
        self.must_go.set()
        self.logger.info(f"[orchestrator] task set, executing: {task!r}")

    def go_idle(self) -> None:
        """Stop streaming observations and hold the arm in place. The open
        recording episode is closed by the control loop on its next tick."""
        with self._state_lock:
            was_active = self._active
            self._active = False
        self._drain_action_queue()
        if was_active:
            self.logger.info("[orchestrator] idle — holding position")

    def status(self) -> dict:
        with self._state_lock:
            task, active = self._live_task, self._active
        with self.latest_action_lock:
            latest_action = self.latest_action
        return {"active": active, "task": task, "latest_action": latest_action}

    def latest_frame_jpeg(self) -> bytes | None:
        with self._frame_lock:
            frame = None if self._latest_frame is None else self._latest_frame.copy()
        if frame is None:
            return None
        import cv2

        # lerobot cameras hand back RGB; cv2.imencode expects BGR.
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr)
        return buf.tobytes() if ok else None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _current_task(self) -> str:
        with self._state_lock:
            return self._live_task

    def _is_active(self) -> bool:
        with self._state_lock:
            return self._active

    def _drain_action_queue(self) -> None:
        with self.action_queue_lock:
            while not self.action_queue.empty():
                try:
                    self.action_queue.get_nowait()
                except Exception:
                    break

    def _cache_frame(self, raw_observation: RawObservation | None) -> None:
        if not raw_observation:
            return
        frame = raw_observation.get(self._frame_camera)
        if frame is not None:
            with self._frame_lock:
                self._latest_frame = frame

    def _grab_idle_frame(self) -> None:
        """While idle we still refresh the frame so the orchestrator's vision
        checks work, but we never send it to the policy or move the arm."""
        try:
            self._cache_frame(self.robot.get_observation())
        except Exception as e:  # noqa: BLE001 - best effort, keep the loop alive
            self.logger.debug(f"[orchestrator] idle frame grab failed: {e}")

    def _record_tick(self, raw_observation: RawObservation, task: str) -> None:
        try:
            self._recorder.add(raw_observation, self._last_action, task)
        except Exception as e:  # noqa: BLE001
            self.logger.warning(f"[recorder] disabled after error: {e}")
            self._recorder = None

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------
    def control_loop_observation(self, task: str, verbose: bool = False) -> RawObservation:
        # Ignore the caller's fixed `task` — the live one wins — and snapshot
        # the frame on the way through.
        raw_observation = super().control_loop_observation(self._current_task(), verbose)
        self._cache_frame(raw_observation)
        return raw_observation

    def control_loop(self, task: str, verbose: bool = False):
        # Wait at barrier for synchronized start (mirrors RobotClient.control_loop).
        self.start_barrier.wait()
        self.logger.info("Orchestrated control loop starting — idle until first POST /task")

        _performed_action = None
        _captured_observation = None

        while self.running:
            control_loop_start = time.perf_counter()

            if self._is_active():
                live_task = self._current_task()

                # Roll the recording episode when the instruction changes.
                if self._recorder is not None and self._recorder.episode_task != live_task:
                    self._recorder.end_episode()
                    self._recorder.begin_episode(live_task)

                if self.actions_available():
                    _performed_action = self.control_loop_action(verbose)
                    self._last_action = _performed_action

                raw_observation = None
                if self._ready_to_send_observation():
                    raw_observation = self.control_loop_observation(live_task, verbose)
                    _captured_observation = raw_observation
                elif self._recorder is not None:
                    # Not sending this tick, but keep the recording cadence at fps.
                    raw_observation = self.robot.get_observation()
                    self._cache_frame(raw_observation)

                if self._recorder is not None and raw_observation is not None:
                    self._record_tick(raw_observation, live_task)
            else:
                if self._recorder is not None:
                    self._recorder.end_episode()
                self._grab_idle_frame()

            time.sleep(max(0, self.config.environment_dt - (time.perf_counter() - control_loop_start)))

        return _captured_observation, _performed_action


def _make_control_server(client: OrchestratedRobotClient, host: str, port: int) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):  # silence per-request logging
            pass

        def _send_json(self, code: int, obj: dict) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            if self.path == "/status":
                self._send_json(200, client.status())
            elif self.path in ("/frame.jpg", "/frame"):
                jpeg = client.latest_frame_jpeg()
                if jpeg is None:
                    self._send_json(503, {"error": "no frame captured yet"})
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpeg)))
                self.end_headers()
                self.wfile.write(jpeg)
            else:
                self._send_json(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length", 0) or 0)
            raw = self.rfile.read(length) if length else b""
            if self.path == "/task":
                try:
                    task = json.loads(raw)["task"]
                    if not isinstance(task, str) or not task.strip():
                        raise ValueError
                except Exception:
                    self._send_json(400, {"error": 'expected {"task": "<non-empty string>"}'})
                    return
                client.set_task(task)
                self._send_json(200, client.status())
            elif self.path == "/idle":
                client.go_idle()
                self._send_json(200, client.status())
            else:
                self._send_json(404, {"error": "not found"})

    httpd = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=httpd.serve_forever, name="control-server", daemon=True).start()
    return httpd


@draccus.wrap()
def orchestrated_client(cfg: OrchestratedClientConfig):
    logging.info(pformat(cfg.to_dict()))

    client = OrchestratedRobotClient(cfg)
    httpd = _make_control_server(client, cfg.control_host, cfg.control_port)
    client.logger.info(f"Control server listening on http://{cfg.control_host}:{cfg.control_port}")

    if not client.start():
        httpd.shutdown()
        if client._recorder is not None:
            client._recorder.close()
        return

    action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)
    action_receiver_thread.start()

    try:
        client.control_loop(task=cfg.task)
    finally:
        client.stop()
        httpd.shutdown()
        action_receiver_thread.join()
        if client._recorder is not None:
            client._recorder.close()
        client.logger.info("Client stopped")


if __name__ == "__main__":
    register_third_party_plugins()
    orchestrated_client()
