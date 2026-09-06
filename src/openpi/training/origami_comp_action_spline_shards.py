from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
import dataclasses
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import openpi.training.origami_comp_action_chunk_shards as _chunk_shards
import openpi.training.origami_vla_dataset as _origami_vla_dataset


SHARD_MANIFEST_FILENAME = _chunk_shards.SHARD_MANIFEST_FILENAME
SHARD_PLAN_FILENAME = _chunk_shards.SHARD_PLAN_FILENAME
SHARD_ROWS_FILENAME = _chunk_shards.SHARD_ROWS_FILENAME
SHARD_METADATA_FILENAME = _chunk_shards.SHARD_METADATA_FILENAME
SHARD_COMPLETE_MARKER = _chunk_shards.SHARD_COMPLETE_MARKER
IMAGE_ARRAY_PREFIX = _chunk_shards.IMAGE_ARRAY_PREFIX

ShardSpec = _chunk_shards.ShardSpec
SequentialVideoReader = _chunk_shards.SequentialVideoReader

parse_size_bytes = _chunk_shards.parse_size_bytes
resolve_shard_root = _chunk_shards.resolve_shard_root
settings_from_data_factory = _chunk_shards.settings_from_data_factory
load_shard_manifest = _chunk_shards.load_shard_manifest
load_shard_specs = _chunk_shards.load_shard_specs
resize_with_pad_uint8 = _chunk_shards.resize_with_pad_uint8
resize_tactile_cell = _chunk_shards.resize_tactile_cell
split_tactile_grid = _chunk_shards.split_tactile_grid
create_memmap = _chunk_shards.create_memmap
make_shard_row_order = _chunk_shards.make_shard_row_order
load_rows_for_shard = _chunk_shards.load_rows_for_shard
load_shard_rows_file = _chunk_shards.load_shard_rows_file
_row_bool = _chunk_shards._row_bool
_planner_state_belief_key = _chunk_shards._planner_state_belief_key
_planner_feature_key = _chunk_shards._planner_feature_key
_read_planner_feature = _chunk_shards._read_planner_feature
_read_frame_index = _chunk_shards._read_frame_index
_read_timestamps = _chunk_shards._read_timestamps
_episode_rows_for_physical_write = _chunk_shards._episode_rows_for_physical_write
_safe_makedirs = _chunk_shards._safe_makedirs
_close_memmap = _chunk_shards._close_memmap


@dataclasses.dataclass
class _ShardBundle:
    spec: ShardSpec
    rows: pd.DataFrame
    row_order: np.ndarray | None
    arrays: dict[str, np.ndarray]


def estimate_row_bytes(settings: _origami_vla_dataset.OrigamiVlaSettings, *, image_size: int) -> int:
    if settings.action_source != "spline":
        raise ValueError(
            "Origami comp action-spline shards require "
            f"action_source='spline', got {settings.action_source!r}."
        )
    image_size = int(image_size)
    image_bytes = len(settings.image_modalities) * image_size * image_size * 3
    tactile_fingers = int(settings.tactile_deform_grid["rows"]) * int(settings.tactile_deform_grid["cols"])
    tactile_bytes = 0
    if settings.load_tactile_images:
        tactile_bytes += 2 * tactile_fingers * 3 * int(settings.tactile_image_size) * int(settings.tactile_image_size)
        tactile_bytes += 2  # deform/raw availability masks
    numeric_bytes = 0
    numeric_bytes += settings.state_dim * np.dtype(np.float32).itemsize
    numeric_bytes += settings.tactile_dim * np.dtype(np.float32).itemsize
    numeric_bytes += settings.action_horizon * settings.action_dim * np.dtype(np.float32).itemsize
    numeric_bytes += settings.action_horizon * settings.action_dim * np.dtype(np.bool_).itemsize
    numeric_bytes += np.dtype(np.float32).itemsize
    numeric_bytes += np.dtype(np.int64).itemsize * 2
    numeric_bytes += np.dtype(np.float32).itemsize
    if settings.include_planner_features:
        numeric_bytes += np.dtype(np.bool_).itemsize
        numeric_bytes += settings.planner_belief_dim * np.dtype(np.float32).itemsize
        numeric_bytes += settings.planner_progress_dim * np.dtype(np.float32).itemsize
        numeric_bytes += settings.planner_uncertainty_dim * np.dtype(np.float32).itemsize
        numeric_bytes += settings.planner_history_dim * np.dtype(np.float32).itemsize
    return int(image_bytes + tactile_bytes + numeric_bytes)


def _drop_tactile_raw_input(
    settings: _origami_vla_dataset.OrigamiVlaSettings,
    episode_uid: str,
    frame_position: int,
) -> bool:
    return _chunk_shards._drop_tactile_raw_input(settings, episode_uid, frame_position)


def _drop_tactile_image_input(
    settings: _origami_vla_dataset.OrigamiVlaSettings,
    episode_uid: str,
    frame_position: int,
) -> bool:
    probability = float(getattr(settings, "tactile_image_input_dropout_prob", 0.0))
    if probability <= 0.0:
        return False
    if probability >= 1.0:
        return True
    seed = int(getattr(settings, "tactile_image_dropout_seed", settings.tactile_raw_dropout_seed))
    payload = f"{seed}:{episode_uid}:{int(frame_position)}:all_tactile".encode("utf-8")
    value = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") / float(1 << 64)
    return value < probability


def extract_spline_target_sample(target_archive: Any, sample_index: int) -> tuple[np.ndarray, np.ndarray]:
    archive_keys = set(getattr(target_archive, "files", ()))
    if "local_delta_control_points" in archive_keys:
        control_points = np.asarray(target_archive["local_delta_control_points"], dtype=np.float32)
        sample_offsets = np.asarray(target_archive["control_point_offsets"], dtype=np.int64)
    elif "coefficients" in archive_keys:
        control_points = np.asarray(target_archive["coefficients"], dtype=np.float32)
        sample_offsets = np.asarray(target_archive["coefficient_offsets"], dtype=np.int64)
    else:
        raise KeyError("Spline target archive must contain local_delta_control_points or coefficients.")

    knot_offsets = np.asarray(target_archive["local_knot_offsets"], dtype=np.int64)
    local_knots = np.asarray(target_archive["local_knots"], dtype=np.float32)
    sample_index = int(sample_index)
    cp_start, cp_end = int(sample_offsets[sample_index]), int(sample_offsets[sample_index + 1])
    knot_start, knot_end = int(knot_offsets[sample_index]), int(knot_offsets[sample_index + 1])
    if cp_end <= cp_start:
        raise ValueError(f"Empty control-point slice for sample_index={sample_index}")
    if knot_end <= knot_start:
        raise ValueError(f"Empty knot slice for sample_index={sample_index}")
    return control_points[cp_start:cp_end], local_knots[knot_start:knot_end]


def pack_spline_target(
    target_archive: Any,
    sample_index: int,
    settings: _origami_vla_dataset.OrigamiVlaSettings,
) -> tuple[np.ndarray, np.ndarray]:
    control_points, local_knots = extract_spline_target_sample(target_archive, sample_index)
    span_widths = _origami_vla_dataset.local_knots_to_span_widths(local_knots, settings.degree)
    expected_spans = int(settings.max_span_count)
    if span_widths.shape[0] != expected_spans:
        raise ValueError(
            f"Expected exactly {expected_spans} local spline span widths, got {span_widths.shape[0]} "
            f"for sample_index={sample_index}."
        )
    return _origami_vla_dataset.pack_spline_actions(
        control_points,
        span_widths,
        max_control_points=settings.max_control_points,
        max_span_count=settings.max_span_count,
        action_dim=settings.action_dim,
    )


class OrigamiCompActionSplineShardDataset:
    def __init__(self, settings: _origami_vla_dataset.OrigamiVlaSettings, *, split: str):
        if settings.action_source != "spline":
            raise ValueError(
                "Origami comp action-spline shards require "
                f"action_source='spline', got {settings.action_source!r}."
            )
        self._settings = settings
        self._split = split
        self._shard_root = resolve_shard_root(settings)
        self._shards = load_shard_specs(
            self._shard_root,
            split=split,
            manifest_filename=settings.shard_manifest_name,
            complete_marker_name=settings.shard_complete_marker_name,
            require_complete=settings.shard_require_complete,
        )
        self._cumulative_rows: list[int] = []
        total = 0
        for spec in self._shards:
            total += int(spec.num_rows)
            self._cumulative_rows.append(total)
        self._shard_cache: OrderedDict[int, _ShardBundle] = OrderedDict()

    def __len__(self) -> int:
        return self._cumulative_rows[-1]

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_shard_cache"] = OrderedDict()
        return state

    def close(self) -> None:
        cache = getattr(self, "_shard_cache", None)
        if cache is None:
            return
        while cache:
            _shard_id, bundle = cache.popitem(last=False)
            self._close_shard_bundle(bundle)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _close_shard_bundle(self, bundle: _ShardBundle) -> None:
        close_memmap = _close_memmap
        if not callable(close_memmap):
            return
        for array in bundle.arrays.values():
            close_memmap(array)
        if bundle.row_order is not None:
            close_memmap(bundle.row_order)

    def _enforce_cache_limit(self) -> None:
        while len(self._shard_cache) > int(self._settings.shard_max_cached_shards):
            _shard_id, bundle = self._shard_cache.popitem(last=False)
            self._close_shard_bundle(bundle)

    def _load_shard(self, shard_index: int) -> _ShardBundle:
        cached = self._shard_cache.get(shard_index)
        if cached is not None:
            self._shard_cache.move_to_end(shard_index)
            return cached
        spec = self._shards[shard_index]
        arrays_dir = spec.shard_dir / "arrays"
        arrays: dict[str, np.ndarray] = {
            "state": np.load(arrays_dir / "state.npy", mmap_mode="r"),
            "tactile": np.load(arrays_dir / "tactile.npy", mmap_mode="r"),
            "actions": np.load(arrays_dir / "actions.npy", mmap_mode="r"),
            "action_mask": np.load(arrays_dir / "action_mask.npy", mmap_mode="r"),
            "sample_weight": np.load(arrays_dir / "sample_weight.npy", mmap_mode="r"),
            "frame_position": np.load(arrays_dir / "frame_position.npy", mmap_mode="r"),
            "frame_index": np.load(arrays_dir / "frame_index.npy", mmap_mode="r"),
            "timestamp": np.load(arrays_dir / "timestamp.npy", mmap_mode="r"),
        }
        for image_key in self._settings.image_modalities:
            arrays[f"{IMAGE_ARRAY_PREFIX}{image_key}"] = np.load(
                arrays_dir / f"{IMAGE_ARRAY_PREFIX}{image_key}.npy",
                mmap_mode="r",
            )
        if self._settings.load_tactile_images:
            arrays["tactile_deform_images"] = np.load(arrays_dir / "tactile_deform_images.npy", mmap_mode="r")
            arrays["tactile_raw_images"] = np.load(arrays_dir / "tactile_raw_images.npy", mmap_mode="r")
            arrays["tactile_raw_available"] = np.load(arrays_dir / "tactile_raw_available.npy", mmap_mode="r")
            deform_path = arrays_dir / "tactile_deform_available.npy"
            arrays["tactile_deform_available"] = (
                np.load(deform_path, mmap_mode="r")
                if deform_path.exists()
                else np.ones((spec.num_rows,), dtype=bool)
            )
        if self._settings.include_planner_features:
            arrays["planner_available"] = np.load(arrays_dir / "planner_available.npy", mmap_mode="r")
            arrays["planner_state_belief"] = np.load(arrays_dir / "planner_state_belief.npy", mmap_mode="r")
            arrays["planner_progress_transition"] = np.load(
                arrays_dir / "planner_progress_transition.npy",
                mmap_mode="r",
            )
            arrays["planner_uncertainty"] = np.load(arrays_dir / "planner_uncertainty.npy", mmap_mode="r")
            arrays["planner_history_latent"] = np.load(arrays_dir / "planner_history_latent.npy", mmap_mode="r")
        row_order_path = arrays_dir / "row_order.npy"
        row_order = (
            np.load(row_order_path, mmap_mode="r")
            if self._settings.shard_use_stored_row_order and row_order_path.exists()
            else None
        )
        bundle = _ShardBundle(
            spec=spec,
            rows=load_shard_rows_file(spec.rows_path),
            row_order=row_order,
            arrays=arrays,
        )
        self._shard_cache[shard_index] = bundle
        self._enforce_cache_limit()
        return bundle

    def _locate(self, index: int) -> tuple[_ShardBundle, int, int]:
        index = int(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_index = bisect_right(self._cumulative_rows, index)
        previous_total = 0 if shard_index == 0 else self._cumulative_rows[shard_index - 1]
        local_index = index - previous_total
        bundle = self._load_shard(shard_index)
        physical_index = int(bundle.row_order[local_index]) if bundle.row_order is not None else int(local_index)
        return bundle, local_index, physical_index

    def __getitem__(self, index: int) -> dict[str, Any]:
        bundle, _local_index, physical_index = self._locate(index)
        arrays = bundle.arrays
        images = {
            image_key: np.asarray(arrays[f"{IMAGE_ARRAY_PREFIX}{image_key}"][physical_index], dtype=np.uint8)
            for image_key in self._settings.image_modalities
        }
        tactile = np.asarray(arrays["tactile"][physical_index], dtype=np.float32)
        output: dict[str, Any] = {
            "image": images,
            "image_mask": {image_key: np.asarray(True) for image_key in images},
            "state": np.asarray(arrays["state"][physical_index], dtype=np.float32),
            "tactile": tactile,
            "tactile_prompt": np.array(tactile, copy=True),
            "tactile_prompt_mask": np.ones((self._settings.tactile_dim,), dtype=bool),
            "state_mask": np.ones((self._settings.state_dim,), dtype=bool),
            "actions": np.array(arrays["actions"][physical_index], dtype=np.float32, copy=True),
            "action_mask": np.asarray(arrays["action_mask"][physical_index], dtype=bool),
            "sample_weight": np.asarray(arrays["sample_weight"][physical_index], dtype=np.float32),
            "prompt": np.asarray(self._settings.prompt),
            "frame_position": np.asarray(arrays["frame_position"][physical_index], dtype=np.int64),
            "frame_index": np.asarray(arrays["frame_index"][physical_index], dtype=np.int64),
            "timestamp": np.asarray(arrays["timestamp"][physical_index], dtype=np.float32),
            "source_row_index": np.asarray(
                int(bundle.rows.iloc[physical_index].get("source_row_index", -1)),
                dtype=np.int64,
            ),
        }
        if self._settings.include_planner_features:
            output.update(
                {
                    "planner_available": np.asarray(arrays["planner_available"][physical_index], dtype=bool),
                    "planner_state_belief": np.asarray(
                        arrays["planner_state_belief"][physical_index],
                        dtype=np.float32,
                    ),
                    "planner_progress_transition": np.asarray(
                        arrays["planner_progress_transition"][physical_index], dtype=np.float32
                    ),
                    "planner_uncertainty": np.asarray(arrays["planner_uncertainty"][physical_index], dtype=np.float32),
                    "planner_history_latent": np.asarray(
                        arrays["planner_history_latent"][physical_index],
                        dtype=np.float32,
                    ),
                }
            )
        if self._settings.load_tactile_images:
            output.update(
                {
                    "tactile_deform_images": np.asarray(
                        arrays["tactile_deform_images"][physical_index],
                        dtype=np.uint8,
                    ),
                    "tactile_raw_images": np.asarray(arrays["tactile_raw_images"][physical_index], dtype=np.uint8),
                    "tactile_deform_available": np.asarray(
                        arrays["tactile_deform_available"][physical_index], dtype=bool
                    ),
                    "tactile_raw_available": np.asarray(arrays["tactile_raw_available"][physical_index], dtype=bool),
                }
            )
        return output
