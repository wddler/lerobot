#!/bin/bash
#
#SBATCH --job-name="pi05-rabc-kitting"
#SBATCH --partition=gpu-a100
#SBATCH --time=24:00:00
#SBATCH --account=innovation
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=6G
#SBATCH --output=logs/train.%j.out
#SBATCH --error=logs/train.%j.err

# Create log in same location as the executable
mkdir -p logs

set -euo pipefail

module load 2025

# Setup environment variables
export HOME=/home/dzatyagof
export DATA=/scratch/dzatyagof
export HF_HOME=/scratch/dzatyagof/huggingface
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# Variables related to this specific dataset and model
export MODEL_TYPE="pi05"
export DATASET_NAME="industrial_assembly_kitting_20260820_132543"
export DATASET_OWNER="denis"
# Fill in with: ls $HF_HOME/hub/models--lerobot--pi05_base/snapshots/
export PRETRAINED_MODEL="${DATA}/huggingface/hub/models--lerobot--pi05_base/snapshots/<COMMIT_HASH>"

cd "$HOME/lerobot"

# `uv sync` creates a normal venv at .venv/ -- activate it directly (same idea
# as `conda activate lerobot` before) so `python`/`lerobot-train` resolve to
# this project's environment, not whatever $HOME/.venv gets picked up by a
# stray VIRTUAL_ENV set elsewhere (e.g. .bashrc).
source "$HOME/lerobot/.venv/bin/activate"

# Display GPU info and the selected device used in PyTorch
nvidia-smi
python -c "
import sys, torch, datasets
print('venv prefix:', sys.prefix)
print('PyTorch:', torch.__version__, 'CUDA available:', torch.cuda.is_available(),
      'Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
print('datasets:', datasets.__version__)
"

# Dataset (+ its sarm_progress.parquet, generated beforehand) lives under
# $HOME/datasets, not scratch -- must already be there before submitting.
# Note that --policy.compile_model is good for training but for inference you
# need to make sure it is turned off in the resulting "pretrained_model/config.json"
srun --cpu-bind=none lerobot-train \
     --dataset.root="${HOME}/datasets/${DATASET_NAME}" \
     --dataset.repo_id="${DATASET_OWNER}/${DATASET_NAME}" \
     --policy.type="${MODEL_TYPE}" \
     --policy.repo_id="${DATASET_OWNER}/${DATASET_NAME}_${MODEL_TYPE}_rabc_policy" \
     --policy.device=cuda \
     --policy.compile_model=true \
     --policy.pretrained_path="${PRETRAINED_MODEL}" \
     --policy.gradient_checkpointing=true \
     --policy.dtype=bfloat16 \
     --policy.push_to_hub=false \
     --sample_weighting.type=rabc \
     --sample_weighting.head_mode=sparse \
     --sample_weighting.kappa=0.01 \
     --output_dir="${DATA}/outputs/train/${DATASET_NAME}_pi05_rabc" \
     --job_name="${DATASET_NAME}_pi05_rabc" \
     --wandb.enable=true \
     --wandb.project=pi05_rabc \
     --steps=40000 \
     --batch_size=16 \
     --dataset.image_transforms.enable=true

# --- To resume this same run across a second/third 24h submission (needed if
# --steps doesn't finish in one job), resubmit with these two extra flags: ---
#      --resume=true \
#      --config_path="${DATA}/outputs/train/${DATASET_NAME}_pi05_rabc/checkpoints/last/pretrained_model/train_config.json"

# W&B logged offline (no internet on compute node) -- sync from a LOGIN node after the job finishes:
#   wandb sync ${DATA}/outputs/train/${DATASET_NAME}_pi05_rabc/wandb/offline-run-*

# The dataset + output_dir above live on /scratch, which is periodically
# purged. Once training finishes, copy the final checkpoint somewhere
# persistent (e.g. the Project drive) rather than leaving it here indefinitely:
#   rsync -av ${DATA}/outputs/train/${DATASET_NAME}_pi05_rabc /tudelft.net/staff-umbrella/<your-project-dir>/outputs/
