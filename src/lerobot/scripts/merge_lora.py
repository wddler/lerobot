#!/usr/bin/env python

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

"""Merge a LoRA / PEFT adapter checkpoint into its base policy.

The async-inference ``policy_server`` (and any plain ``Policy.from_pretrained``)
loads a *full* checkpoint — ``config.json`` + ``model.safetensors`` — and has no
PEFT code path. A LoRA training run instead writes ``adapter_config.json`` +
``adapter_model.safetensors``, which those loaders can't read (you get
``Repo id must be in the form 'repo_name'...`` because the path falls through to
a Hub lookup).

This script loads the base policy named in the adapter config, applies the
adapter, merges the weights, and writes a normal deployable checkpoint that the
server loads unchanged. Processor configs (``*_processor`` / ``processor.json``)
and ``train_config.json`` are copied over from the adapter dir so
``make_pre_post_processors`` still works.

Usage:
```shell
uv run python -m lerobot.scripts.merge_lora \
    --adapter_path=outputs/train/pi05_lora_rabc/080000/pretrained_model \
    --output_path=outputs/train/pi05_lora_rabc/080000/merged
# then point the server at .../080000/merged
```
"""

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import draccus

from lerobot.configs import PreTrainedConfig
from lerobot.policies import get_policy_class
from lerobot.utils.import_utils import register_third_party_plugins, require_package
from lerobot.utils.utils import init_logging

# Files worth carrying over from the adapter dir if the base dir doesn't have them
# (or the adapter's are newer / training-specific).
_SIDECAR_GLOBS = ("*_processor*", "processor*.json", "*preprocessor*", "*postprocessor*", "train_config.json")


@dataclass
class MergeLoraConfig:
    # Directory (or Hub repo id) holding adapter_config.json + adapter_model.safetensors.
    adapter_path: str
    # Where to write the merged full checkpoint.
    output_path: str
    # Override the base model from the adapter config (path or Hub id) if it moved.
    base_path: str | None = None
    # Device to load/merge on. CPU works and needs no GPU.
    device: str = "cpu"
    # Overwrite output_path if it already exists.
    overwrite: bool = False


@draccus.wrap()
def merge_lora(cfg: MergeLoraConfig):
    init_logging()
    require_package("peft", extra="peft")
    from peft import PeftConfig, PeftModel

    adapter_path = cfg.adapter_path
    output_path = Path(cfg.output_path)
    if output_path.exists():
        if not cfg.overwrite:
            raise FileExistsError(f"{output_path} exists; pass --overwrite to replace it")
        shutil.rmtree(output_path)

    peft_config = PeftConfig.from_pretrained(adapter_path)
    base_path = cfg.base_path or peft_config.base_model_name_or_path
    if not base_path:
        raise ValueError(
            "The adapter config has no base_model_name_or_path — pass --base_path explicitly."
        )
    logging.info(f"Adapter:      {adapter_path}")
    logging.info(f"Base policy:  {base_path}")

    base_cfg = PreTrainedConfig.from_pretrained(base_path)
    base_cfg.device = cfg.device
    # torch.compile / CUDA graphs are a training/serving concern; keep merge plain.
    if hasattr(base_cfg, "compile_model"):
        base_cfg.compile_model = False
    policy_cls = get_policy_class(base_cfg.type)
    logging.info(f"Loading base {base_cfg.type} policy on {cfg.device}...")
    policy = policy_cls.from_pretrained(base_path, config=base_cfg)

    logging.info("Applying adapter and merging weights...")
    merged = PeftModel.from_pretrained(policy, adapter_path).merge_and_unload()

    logging.info(f"Writing merged checkpoint to {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output_path)
    merged.config.save_pretrained(output_path)

    # Carry over processor / train configs so the server can build pre/post processors.
    src_dirs = [Path(adapter_path)]
    base_dir = Path(base_path)
    if base_dir.is_dir():
        src_dirs.append(base_dir)
    for src in src_dirs:
        if not src.is_dir():
            continue
        for pattern in _SIDECAR_GLOBS:
            for f in src.glob(pattern):
                dest = output_path / f.name
                if f.is_file() and not dest.exists():
                    shutil.copy2(f, dest)
                    logging.info(f"  copied {f.name} from {src}")

    logging.info("Done. Point --pretrained_name_or_path / the policy server at:")
    logging.info(f"  {output_path.resolve()}")


if __name__ == "__main__":
    register_third_party_plugins()
    merge_lora()
