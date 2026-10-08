#!/usr/bin/env python
"""Print how many episodes exist per task in a LeRobotDataset.

Usage:
    uv run python episodes_per_task.py --repo-id denis/my_dataset --root /path/to/dataset
"""

import argparse
import glob
from collections import Counter

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", type=str, required=True)
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Local dataset directory. Defaults to $HF_LEROBOT_HOME/{repo_id}.",
    )
    args = parser.parse_args()

    if args.root is not None:
        root = args.root
    else:
        import os

        hf_lerobot_home = os.environ.get("HF_LEROBOT_HOME", os.path.expanduser("~/.cache/huggingface/lerobot"))
        root = f"{hf_lerobot_home}/{args.repo_id}"

    ep_files = sorted(glob.glob(f"{root}/meta/episodes/chunk-*/file-*.parquet"))
    if not ep_files:
        raise FileNotFoundError(f"No episode metadata found under {root}/meta/episodes/")

    episodes = pd.concat(
        [pd.read_parquet(f)[["episode_index", "tasks"]] for f in ep_files], ignore_index=True
    ).sort_values("episode_index")

    counts = Counter()
    for tasks in episodes["tasks"]:
        for task in tasks:
            counts[task] += 1

    total = len(episodes)
    print(f"{total} episodes total\n")
    for task, count in counts.most_common():
        print(f"{count:4d}  {task}")


if __name__ == "__main__":
    main()
