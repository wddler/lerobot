lerobot-dataset-viz \
    --repo-id denis/industrial_assembly_kitting_20260820_132543 \
    --episode-index 49


lerobot-record --config_path src/lerobot/configs/record_piperx.toml

lerobot-teleoperate --config_path src/lerobot/configs/teleop_piperx.toml

uv run python -m lerobot.async_inference.robot_client --config_path=rollout_pi05_piperx.toml

# orchestrator-driven rollout: task instruction is set at runtime over HTTP by vlm-orchestrator,
# and every task segment is recorded to a LeRobotDataset (local/eval_<checkpoint>_<ts> by default)
# GPU box:
CUDA_VISIBLE_DEVICES=1 uv run python -m lerobot.async_inference.policy_server --host=0.0.0.0 --port=8081
# robot laptop:
uv run python -m lerobot.async_inference.orchestrated_client --config_path=src/lerobot/configs/rollout_orchestrated_piperx.toml

# LoRA checkpoint -> merge into a full checkpoint the policy server can load (no PEFT loader server-side)
uv run python -m lerobot.scripts.merge_lora \
    --adapter_path=outputs/train/pi05_lora_rabc/080000/pretrained_model \
    --output_path=outputs/train/pi05_lora_rabc/080000/merged

lerobot-edit-dataset \
    --repo_id denis/industrial_assembly_kitting_20260820_132543 \
    --operation.type delete_episodes \
    --operation.episode_indices "[50]"



uv run python episodes_per_task.py --repo-id denis/industrial_assembly_kitting_20260820_132543

v4l2-ctl -d /dev/video2 --set-ctrl=focus_automatic_continuous=0

