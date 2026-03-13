from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from tqdm import tqdm


def create_weighted_dataset(
    *,
    repo_name: str,
    fps: int,
    top_shape: tuple[int, ...],
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
    state_dim: int,
    action_dim: int,
) -> LeRobotDataset:
    return LeRobotDataset.create(
        repo_id=repo_name,
        robot_type="so101",
        fps=fps,
        features={
            "observation.images.top_rgb": {
                "dtype": "image",
                "shape": tuple(top_shape),
                "names": ["height", "width", "channel"],
            },
            "observation.images.left_rgb": {
                "dtype": "image",
                "shape": tuple(left_shape),
                "names": ["height", "width", "channel"],
            },
            "observation.images.right_rgb": {
                "dtype": "image",
                "shape": tuple(right_shape),
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (state_dim,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (action_dim,),
                "names": ["actions"],
            },
            "sample_weight": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["sample_weight"],
            },
        },
        image_writer_threads=8,
        image_writer_processes=2,
    )


def load_manifests(manifest_paths: list[Path]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for manifest_path in manifest_paths:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries.extend(manifest["episodes"])
    return sorted(entries, key=lambda x: int(x["global_episode_index"]))


def merge_shards(
    *,
    manifest_paths: list[Path],
    repo_name: str,
    fps: int,
    top_shape: tuple[int, ...],
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
    state_dim: int,
    action_dim: int,
) -> dict[str, int]:
    dataset = create_weighted_dataset(
        repo_name=repo_name,
        fps=fps,
        top_shape=top_shape,
        left_shape=left_shape,
        right_shape=right_shape,
        state_dim=state_dim,
        action_dim=action_dim,
    )

    entries = load_manifests(manifest_paths)
    stats = {"episodes_saved": 0, "frames_saved": 0}

    for entry in tqdm(entries, desc="Merging shards", unit="episode", dynamic_ncols=True):
        bundle_path = Path(entry["bundle_path"])
        prompt = str(entry["prompt"])
        with np.load(bundle_path) as bundle:
            top_images = bundle["top_images"]
            left_images = bundle["left_images"]
            right_images = bundle["right_images"]
            states = bundle["states"]
            actions = bundle["actions"]
            sample_weights = bundle["sample_weights"]

            frame_count = int(states.shape[0])
            for i in range(frame_count):
                dataset.add_frame(
                    {
                        "observation.images.top_rgb": top_images[i],
                        "observation.images.left_rgb": left_images[i],
                        "observation.images.right_rgb": right_images[i],
                        "observation.state": states[i],
                        "actions": actions[i],
                        "sample_weight": sample_weights[i],
                        "task": prompt,
                    }
                )
            dataset.save_episode()

            stats["episodes_saved"] += 1
            stats["frames_saved"] += frame_count

    return stats

