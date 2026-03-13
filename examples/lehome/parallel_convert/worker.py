from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from parallel_convert import common


def _prepare_episode_bundle(
    json_path: Path,
    source_root: Path,
    state_dim: int,
    action_dim: int,
    default_prompt: str,
) -> dict[str, Any]:
    rows = common.load_rows(json_path)
    row0 = rows[0]

    state_dim_found = len(row0["observation.state"])
    action_dim_found = len(row0["action"])
    if state_dim_found != state_dim or action_dim_found != action_dim:
        raise ValueError(
            "shape mismatch "
            f"state_dim={state_dim_found} (expected {state_dim}), "
            f"action_dim={action_dim_found} (expected {action_dim})"
        )

    episode_prompt = str(row0.get("prompt", default_prompt))

    top_images: list[np.ndarray] = []
    left_images: list[np.ndarray] = []
    right_images: list[np.ndarray] = []
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    sample_weights: list[np.ndarray] = []

    for row in rows:
        top_images.append(
            common.load_rgb_image(common.resolve_path(row["observation.image.top_rgb"], root=source_root))
        )
        left_images.append(
            common.load_rgb_image(common.resolve_path(row["observation.image.left_rgb"], root=source_root))
        )
        right_images.append(
            common.load_rgb_image(common.resolve_path(row["observation.image.right_rgb"], root=source_root))
        )
        states.append(np.asarray(row["observation.state"], dtype=np.float32))
        actions.append(np.asarray(row["action"], dtype=np.float32))
        sample_weights.append(np.asarray([float(row.get("weight", 1.0))], dtype=np.float32))

    return {
        "prompt": episode_prompt,
        "num_frames": len(rows),
        "top_images": np.stack(top_images, axis=0),
        "left_images": np.stack(left_images, axis=0),
        "right_images": np.stack(right_images, axis=0),
        "states": np.stack(states, axis=0),
        "actions": np.stack(actions, axis=0),
        "sample_weights": np.stack(sample_weights, axis=0),
    }


def convert_shard(
    *,
    shard_index: int,
    episode_specs: list[tuple[int, str]],
    source_root: str,
    temp_root: str,
    state_dim: int,
    action_dim: int,
    default_prompt: str,
    compress_temp_shards: bool,
) -> dict[str, Any]:
    source_root_path = Path(source_root)
    temp_root_path = Path(temp_root)
    shard_dir = temp_root_path / f"shard_{shard_index:04d}"
    shard_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries: list[dict[str, Any]] = []
    saved_episodes = 0
    saved_frames = 0

    episode_iter = tqdm(
        episode_specs,
        desc=f"shard_{shard_index:04d}",
        unit="episode",
        leave=False,
        dynamic_ncols=True,
    )
    for global_episode_index, json_path_str in episode_iter:
        json_path = Path(json_path_str)
        bundle = _prepare_episode_bundle(json_path, source_root_path, state_dim, action_dim, default_prompt)

        bundle_path = shard_dir / f"episode_{global_episode_index:08d}.npz"
        save_fn = np.savez_compressed if compress_temp_shards else np.savez
        save_fn(
            bundle_path,
            top_images=bundle["top_images"],
            left_images=bundle["left_images"],
            right_images=bundle["right_images"],
            states=bundle["states"],
            actions=bundle["actions"],
            sample_weights=bundle["sample_weights"],
        )

        manifest_entries.append(
            {
                "global_episode_index": global_episode_index,
                "json_path": str(json_path),
                "bundle_path": str(bundle_path),
                "prompt": bundle["prompt"],
                "num_frames": bundle["num_frames"],
            }
        )
        saved_episodes += 1
        saved_frames += int(bundle["num_frames"])
        episode_iter.set_postfix(frames=saved_frames)

    manifest_path = shard_dir / "manifest.json"
    manifest = {
        "shard_index": shard_index,
        "episodes": manifest_entries,
        "episodes_saved": saved_episodes,
        "frames_saved": saved_frames,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "shard_index": shard_index,
        "manifest_path": str(manifest_path),
        "episodes_saved": saved_episodes,
        "frames_saved": saved_frames,
    }
