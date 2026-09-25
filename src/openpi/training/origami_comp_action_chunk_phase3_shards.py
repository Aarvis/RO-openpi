"""Shared format helpers for resumable Origami Phase-3 mixed-speed shards."""

from __future__ import annotations

from bisect import bisect_right
import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import openpi.training.config as _config
import openpi.training.origami_comp_action_chunk_shards as _chunk
import openpi.training.origami_vla_dataset as _dataset


FORMAT_VERSION = "origami_comp_action_chunk_phase3_mixed_speed_v2"
SHARD_MANIFEST_FILENAME = "shard_manifest.json"
COMPLETE_MARKER = "complete.marker"
ROWS_FILENAME = "rows.parquet"
METADATA_FILENAME = "metadata.json"
VIRTUAL_PLAN_COLUMNS = ("physical_row_id", "speed_id", "occurrence_id")


@dataclasses.dataclass(frozen=True)
class ShardSpec:
    split: str
    shard_id: int
    shard_name: str
    shard_dir: Path
    num_rows: int
    episodes: tuple[str, ...]


def phase3_config(config_name: str) -> tuple[_config.TrainConfig, _config.OrigamiCompActionChunkDataConfig]:
    config = _config.get_config(config_name)
    if not isinstance(config.data, _config.OrigamiCompActionChunkDataConfig):
        raise TypeError(f"{config_name!r} is not an Origami action-chunk configuration.")
    if not isinstance(config.data.shard_build, _config.OrigamiCompActionChunkPhase3ShardBuildConfig):
        raise TypeError(f"{config_name!r} does not use the Phase-3 shard-build configuration.")
    return config, config.data


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def drop_tactile_raw_for_virtual_sample(
    settings: _dataset.OrigamiVlaSettings,
    *,
    episode_uid: str,
    frame_position: int,
    stride: int,
    occurrence_id: int,
) -> bool:
    """Return a reproducible raw-tactile dropout decision for one logical sample.

    Physical shards always retain source raw tactile.  This helper belongs in
    the future shard loader, after it has resolved a virtual-plan occurrence.
    Including both stride and occurrence means repeated physical frames are
    independent logical training samples while remaining resume/worker safe.
    """
    probability = float(settings.tactile_raw_input_dropout_prob)
    if probability <= 0.0:
        return False
    if probability >= 1.0:
        return True
    payload = (
        f"{settings.tactile_raw_dropout_seed}:{episode_uid}:{int(frame_position)}:"
        f"{int(stride)}:{int(occurrence_id)}"
    ).encode("utf-8")
    value = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") / float(1 << 64)
    return value < probability


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temp.replace(path)


def close_memmap(array: np.ndarray) -> None:
    mmap = getattr(array, "_mmap", None)
    if mmap is not None:
        mmap.close()


def flush_memmaps(arrays: dict[str, np.ndarray]) -> None:
    for array in arrays.values():
        flush = getattr(array, "flush", None)
        if flush is not None:
            flush()


def phase3_array_specs(
    settings: _dataset.OrigamiVlaSettings,
    mixed_speed: _config.OrigamiMixedSpeedShardConfig,
    *,
    num_rows: int,
    image_size: int,
) -> dict[str, tuple[np.dtype | type, tuple[int, ...]]]:
    specs: dict[str, tuple[np.dtype | type, tuple[int, ...]]] = {
        "state": (np.float32, (num_rows, settings.state_dim)),
        "tactile": (np.float32, (num_rows, settings.tactile_dim)),
        "frame_position": (np.int64, (num_rows,)),
        "frame_index": (np.int64, (num_rows,)),
        "timestamp": (np.float32, (num_rows,)),
        "sample_weight": (np.float32, (num_rows,)),
        "planner_available": (np.bool_, (num_rows,)),
        "planner_state_belief": (np.float32, (num_rows, settings.planner_belief_dim)),
        "planner_progress_transition": (np.float32, (num_rows, settings.planner_progress_dim)),
        "planner_uncertainty": (np.float32, (num_rows, settings.planner_uncertainty_dim)),
        "planner_history_latent": (np.float32, (num_rows, settings.planner_history_dim)),
    }
    for stride in mixed_speed.ordered_strides:
        specs[f"actions_stride_{stride}"] = (
            np.float32,
            (num_rows, mixed_speed.action_horizon, settings.action_dim),
        )
    for image_key in settings.image_modalities:
        specs[f"{_chunk.IMAGE_ARRAY_PREFIX}{image_key}"] = (np.uint8, (num_rows, image_size, image_size, 3))
    if settings.load_tactile_images:
        fingers = int(settings.tactile_deform_grid["rows"]) * int(settings.tactile_deform_grid["cols"])
        specs.update(
            {
                "tactile_deform_images": (
                    np.uint8,
                    (num_rows, fingers, 3, settings.tactile_image_size, settings.tactile_image_size),
                ),
                "tactile_raw_images": (
                    np.uint8,
                    (num_rows, fingers, 3, settings.tactile_image_size, settings.tactile_image_size),
                ),
                "tactile_deform_available": (np.bool_, (num_rows,)),
                "tactile_raw_available": (np.bool_, (num_rows,)),
            }
        )
    return specs


def create_arrays(
    arrays_dir: Path,
    specs: dict[str, tuple[np.dtype | type, tuple[int, ...]]],
    *,
    mode: str,
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for name, (dtype, shape) in specs.items():
        path = arrays_dir / f"{name}.npy"
        if mode == "w+":
            arrays[name] = _chunk.create_memmap(path, dtype=dtype, shape=shape)
        else:
            arrays[name] = np.load(path, mmap_mode=mode)
            if arrays[name].dtype != np.dtype(dtype) or arrays[name].shape != shape:
                raise ValueError(f"Incompatible resumed array {path}: {arrays[name].dtype}/{arrays[name].shape}")
    return arrays


def load_phase3_specs(shard_root: Path, *, split: str, require_complete: bool = True) -> list[ShardSpec]:
    manifest_path = shard_root / SHARD_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT_VERSION:
        raise ValueError(f"Unsupported Phase-3 shard format: {manifest.get('format')!r}")
    specs: list[ShardSpec] = []
    for item in manifest.get("shards", []):
        if str(item.get("split")) != split:
            continue
        directory = shard_root / str(item["relative_dir"])
        if require_complete and not (directory / COMPLETE_MARKER).is_file():
            continue
        specs.append(
            ShardSpec(
                split=split,
                shard_id=int(item["shard_id"]),
                shard_name=str(item["shard_name"]),
                shard_dir=directory,
                num_rows=int(item["num_rows"]),
                episodes=tuple(str(uid) for uid in item["episodes"]),
            )
        )
    specs.sort(key=lambda spec: spec.shard_id)
    if not specs:
        raise RuntimeError(f"No complete Phase-3 {split!r} shards found in {shard_root}.")
    return specs


def physical_rows(rows: pd.DataFrame, episodes: tuple[str, ...]) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for episode_uid in episodes:
        piece = rows.loc[rows["episode_uid"].astype(str) == str(episode_uid)].sort_values("frame_position")
        if piece.empty:
            raise ValueError(f"Shard episode {episode_uid!r} has no manifest rows.")
        pieces.append(piece)
    frame = pd.concat(pieces, axis=0, ignore_index=True)
    if frame.duplicated(subset=["episode_uid", "frame_position"]).any():
        raise ValueError("Phase-3 source manifest contains duplicate episode/frame rows.")
    frame["local_physical_row"] = np.arange(len(frame), dtype=np.int64)
    return frame


def build_virtual_plan(
    rows: pd.DataFrame,
    mixed_speed: _config.OrigamiMixedSpeedShardConfig,
    *,
    shard_id: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Create the compact, batch-size-independent episode-balanced plan."""
    episode_rows = {
        str(uid): group["local_physical_row"].to_numpy(dtype=np.uint32, copy=True)
        for uid, group in rows.groupby("episode_uid", sort=False)
    }
    episodes = tuple(episode_rows)
    if not episodes:
        raise ValueError("Cannot build a virtual plan for an empty shard.")
    entries: list[np.ndarray] = []
    per_speed: dict[str, Any] = {}
    episode_count = len(episodes)

    for stride in mixed_speed.ordered_strides:
        coverage = float(mixed_speed.stride_episode_coverage[stride])
        rng = np.random.default_rng(int(mixed_speed.seed) + int(shard_id) * 1_000_003 + int(stride) * 10_007)
        order = np.asarray(episodes, dtype=object)
        rng.shuffle(order)
        selected_rows: list[np.ndarray] = []
        full_passes = int(np.floor(coverage))
        for pass_index in range(full_passes):
            rotated = np.roll(order, pass_index)
            selected_rows.extend(episode_rows[str(uid)] for uid in rotated.tolist())
        remainder_episode_equivalents = (coverage - full_passes) * episode_count
        whole_episodes = int(np.floor(remainder_episode_equivalents + 1.0e-12))
        fractional_episode = remainder_episode_equivalents - whole_episodes
        # One shuffled order makes the partial component episode-balanced and
        # avoids repeated selections within a speed's fractional pass.
        for uid in order[:whole_episodes].tolist():
            selected_rows.append(episode_rows[str(uid)])
        partial_rows = 0
        if fractional_episode > 1.0e-12:
            uid = str(order[whole_episodes % episode_count])
            candidates = episode_rows[uid]
            partial_rows = int(round(fractional_episode * len(candidates)))
            partial_rows = min(len(candidates), max(0, partial_rows))
            if partial_rows:
                chosen = rng.choice(candidates, size=partial_rows, replace=False)
                selected_rows.append(np.asarray(chosen, dtype=np.uint32))
        selected = (
            np.concatenate(selected_rows, axis=0).astype(np.uint32, copy=False)
            if selected_rows
            else np.empty((0,), dtype=np.uint32)
        )
        speed_column = np.full((len(selected),), int(stride), dtype=np.uint32)
        # A row may recur at a stride because coverage exceeds one complete
        # episode pass. Occurrence IDs are contiguous per (row, stride) and
        # stay attached to their samples through the final plan shuffle.
        occurrence_column = np.empty((len(selected),), dtype=np.uint32)
        occurrences: dict[int, int] = {}
        for index, row_id in enumerate(selected.tolist()):
            occurrence = occurrences.get(int(row_id), 0)
            occurrence_column[index] = occurrence
            occurrences[int(row_id)] = occurrence + 1
        entries.append(np.column_stack((selected, speed_column, occurrence_column)))
        per_speed[str(stride)] = {
            "episode_coverage": coverage,
            "full_episode_passes": full_passes,
            "fractional_episode_equivalents": float(remainder_episode_equivalents),
            "fractional_rows": partial_rows,
            "logical_samples": int(len(selected)),
        }

    plan = np.concatenate(entries, axis=0).astype(np.uint32, copy=False)
    permutation = np.random.default_rng(int(mixed_speed.seed) + int(shard_id) * 1_000_003 + 97).permutation(len(plan))
    plan = plan[permutation]
    return plan, {
        "columns": list(VIRTUAL_PLAN_COLUMNS),
        "dtype": "uint32",
        "logical_samples": int(len(plan)),
        "by_speed": per_speed,
    }


def locate_spec(specs: list[ShardSpec], cumulative_rows: list[int], index: int) -> tuple[ShardSpec, int]:
    shard_index = bisect_right(cumulative_rows, int(index))
    previous = 0 if shard_index == 0 else cumulative_rows[shard_index - 1]
    return specs[shard_index], int(index) - previous
