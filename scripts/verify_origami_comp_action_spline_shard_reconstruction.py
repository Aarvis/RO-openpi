from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
import dataclasses
import json
import os
from pathlib import Path
from queue import Empty
import shutil
import sys
from typing import Any
import uuid

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import openpi.training.config as _config
import openpi.training.origami_comp_action_spline_shards as _spline_shards
import openpi.training.origami_vla_dataset as _origami_vla_dataset


PERCENTILES = (0.0, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 75.0, 95.0, 99.0, 99.9, 99.99, 100.0)


@dataclass(frozen=True)
class VerificationConfig:
    settings: _origami_vla_dataset.OrigamiVlaSettings
    dataset_root: Path
    shard_root: Path
    split: str
    action_array_name: str
    state_array_name: str
    num_workers: int
    start_shard: int
    max_shards: int | None
    max_rows_per_shard: int | None
    output_dir: Path
    scratch_dir: Path
    progress_update_rows: int
    anchor_state_tolerance: float
    shard_state_tolerance: float
    keep_intermediate_error_arrays: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Exactly verify packed Origami comp action-spline shards by reconstructing absolute "
            "actions from shard control points/span widths and comparing against source action_65d."
        )
    )
    parser.add_argument("--config-name", type=str, default="pi05_origami_comp_action_spline")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--manifest-root", type=Path, default=None)
    parser.add_argument("--shard-root", type=Path, default=None)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--action-array-name", type=str, default="action_65d.npy")
    parser.add_argument("--state-array-name", type=str, default="state_65d.npy")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--start-shard", type=int, default=0)
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--max-rows-per-shard", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--progress-update-rows", type=int, default=256)
    parser.add_argument("--anchor-state-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--shard-state-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--keep-intermediate-error-arrays", action="store_true")
    return parser.parse_args()


def _resolve(args: argparse.Namespace) -> VerificationConfig:
    train_config = _config.get_config(args.config_name)
    if not isinstance(train_config.data, _config.OrigamiCompActionChunkDataConfig):
        raise TypeError(
            f"Config {args.config_name!r} uses {type(train_config.data).__name__}; "
            "expected OrigamiCompActionChunkDataConfig."
        )
    settings = _spline_shards.settings_from_data_factory(train_config.data, train_config.model)
    if settings.action_source != "spline":
        raise ValueError(
            f"Config {args.config_name!r} has action_source={settings.action_source!r}; "
            "expected action_source='spline'."
        )
    if int(settings.action_horizon) != int(settings.max_control_points) + 1:
        raise ValueError(
            f"Spline packed action_horizon must equal max_control_points + 1, got "
            f"{settings.action_horizon} and {settings.max_control_points}."
        )
    if args.dataset_root is not None:
        settings = dataclasses.replace(settings, dataset_root=str(args.dataset_root))
    if args.manifest_root is not None:
        settings = dataclasses.replace(settings, manifest_root=str(args.manifest_root))
    shard_root = args.shard_root or (Path(train_config.data.shard_root) if train_config.data.shard_root else None)
    if shard_root is None:
        raise ValueError("Set data.shard_root or pass --shard-root.")
    settings = dataclasses.replace(settings, dataset_backend="shard", shard_root=str(shard_root))

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path(settings.shard_root) / f"{args.split}_spline_reconstruction_verification"
    scratch_dir = output_dir / f".scratch_{uuid.uuid4().hex}"
    return VerificationConfig(
        settings=settings,
        dataset_root=Path(settings.dataset_root),
        shard_root=Path(settings.shard_root or shard_root),
        split=str(args.split),
        action_array_name=str(args.action_array_name),
        state_array_name=str(args.state_array_name),
        num_workers=max(1, int(args.num_workers)),
        start_shard=max(0, int(args.start_shard)),
        max_shards=(None if args.max_shards is None else max(0, int(args.max_shards))),
        max_rows_per_shard=(None if args.max_rows_per_shard is None else max(0, int(args.max_rows_per_shard))),
        output_dir=Path(output_dir),
        scratch_dir=scratch_dir,
        progress_update_rows=max(1, int(args.progress_update_rows)),
        anchor_state_tolerance=float(args.anchor_state_tolerance),
        shard_state_tolerance=float(args.shard_state_tolerance),
        keep_intermediate_error_arrays=bool(args.keep_intermediate_error_arrays),
    )


def _normalize_span_widths(span_widths: np.ndarray, expected_count: int) -> np.ndarray:
    widths = np.asarray(span_widths, dtype=np.float64).reshape(-1)
    if widths.shape[0] != expected_count:
        raise ValueError(f"Expected {expected_count} spline span widths, got {widths.shape[0]}.")
    if not np.all(np.isfinite(widths)):
        raise ValueError("Spline span widths contain non-finite values.")
    if np.any(widths <= 0.0):
        raise ValueError(f"Spline span widths must be positive, got min={float(widths.min())}.")
    total = float(np.sum(widths))
    if total <= 0.0:
        raise ValueError("Spline span widths sum to zero.")
    return widths / total


def _knot_vector_from_span_widths(span_widths: np.ndarray, degree: int) -> np.ndarray:
    widths = _normalize_span_widths(span_widths, int(span_widths.shape[0]))
    boundaries = np.concatenate([np.asarray([0.0], dtype=np.float64), np.cumsum(widths)])
    boundaries[-1] = 1.0
    return np.concatenate(
        [
            np.zeros((int(degree) + 1,), dtype=np.float64),
            boundaries[1:-1],
            np.ones((int(degree) + 1,), dtype=np.float64),
        ]
    )


def _decode_packed_spline(
    packed_actions: np.ndarray,
    packed_mask: np.ndarray,
    settings: _origami_vla_dataset.OrigamiVlaSettings,
) -> tuple[np.ndarray, np.ndarray]:
    actions = np.asarray(packed_actions, dtype=np.float64)
    mask = np.asarray(packed_mask, dtype=bool)
    expected_shape = (int(settings.max_control_points) + 1, int(settings.action_dim))
    if actions.shape != expected_shape or mask.shape != expected_shape:
        raise ValueError(f"Expected packed action/mask shape {expected_shape}, got {actions.shape} and {mask.shape}.")

    control_count = int(settings.max_control_points)
    span_count = int(settings.max_span_count)
    if not mask[:control_count, :].all():
        raise ValueError("Control-point mask is not fully enabled.")
    if mask[control_count + 1 :, :].any():
        raise ValueError("Unexpected mask entries after the span-width row.")
    if not mask[control_count, :span_count].all():
        raise ValueError("Span-width mask is not fully enabled.")
    if mask[control_count, span_count:].any():
        raise ValueError("Span-width row has enabled padding dimensions.")

    control_points = actions[:control_count, :]
    span_widths = actions[control_count, :span_count]
    knot_vector = _knot_vector_from_span_widths(span_widths, int(settings.degree))
    expected_knots = control_count + int(settings.degree) + 1
    if knot_vector.size != expected_knots:
        raise ValueError(f"Expected {expected_knots} knots, got {knot_vector.size}.")
    return control_points, knot_vector


def _validate_offsets(name: str, offsets: np.ndarray, total_size: int, sample_count: int) -> None:
    if offsets.ndim != 1 or offsets.size != sample_count + 1:
        raise ValueError(f"{name} offsets must have {sample_count + 1} entries, got {offsets.shape}.")
    if offsets[0] != 0 or offsets[-1] != total_size or np.any(np.diff(offsets) < 0):
        raise ValueError(f"{name} offsets are invalid for total size {total_size}.")


def _load_target_metadata(target_archive: Any, episode_uid: str) -> dict[str, np.ndarray | int]:
    archive_keys = set(getattr(target_archive, "files", ()))
    required = {
        "sample_ids",
        "local_degree",
        "anchor_state_65d",
        "frame_index_values",
        "frame_index_offsets",
        "v_values",
        "v_offsets",
    }
    missing = sorted(required.difference(archive_keys))
    if missing:
        raise ValueError(f"{episode_uid}: local target NPZ is missing keys: {missing}")

    sample_ids = np.asarray(target_archive["sample_ids"], dtype=np.int64)
    local_degree = int(np.asarray(target_archive["local_degree"]).reshape(-1)[0])
    anchor_state = np.asarray(target_archive["anchor_state_65d"], dtype=np.float64)
    frame_index_values = np.asarray(target_archive["frame_index_values"], dtype=np.int64)
    frame_index_offsets = np.asarray(target_archive["frame_index_offsets"], dtype=np.int64)
    v_values = np.asarray(target_archive["v_values"], dtype=np.float64)
    v_offsets = np.asarray(target_archive["v_offsets"], dtype=np.int64)

    sample_count = int(sample_ids.size)
    if local_degree != 3:
        raise ValueError(f"{episode_uid}: expected local degree 3, found {local_degree}.")
    if anchor_state.shape != (sample_count, 65):
        raise ValueError(f"{episode_uid}: invalid anchor_state_65d shape {anchor_state.shape}.")
    _validate_offsets("frame index", frame_index_offsets, int(frame_index_values.size), sample_count)
    _validate_offsets("v", v_offsets, int(v_values.size), sample_count)
    if frame_index_values.size != v_values.size:
        raise ValueError(f"{episode_uid}: frame-index and v-value counts differ.")
    return {
        "sample_ids": sample_ids,
        "anchor_state": anchor_state,
        "frame_index_values": frame_index_values,
        "frame_index_offsets": frame_index_offsets,
        "v_values": v_values,
        "v_offsets": v_offsets,
        "local_degree": local_degree,
    }


def _selected_shards(cfg: VerificationConfig) -> list[_spline_shards.ShardSpec]:
    shards = _spline_shards.load_shard_specs(
        cfg.shard_root,
        split=cfg.split,
        manifest_filename=cfg.settings.shard_manifest_name,
        complete_marker_name=cfg.settings.shard_complete_marker_name,
        require_complete=cfg.settings.shard_require_complete,
    )
    shards = [shard for shard in shards if int(shard.shard_id) >= int(cfg.start_shard)]
    if cfg.max_shards is not None:
        shards = shards[: int(cfg.max_shards)]
    if not shards:
        raise RuntimeError(f"No {cfg.split!r} shards selected under {cfg.shard_root}.")
    return shards


def _source_row_index(row: pd.Series, fallback: int) -> int:
    value = row.get("local_physical_row", fallback)
    if pd.isna(value):
        return int(fallback)
    return int(value)


def process_shard(
    spec: _spline_shards.ShardSpec,
    cfg: VerificationConfig,
    progress_queue: Any | None = None,
) -> dict[str, Any]:
    rows = pd.read_parquet(spec.rows_path).reset_index(drop=True)
    if cfg.max_rows_per_shard is not None:
        rows = rows.iloc[: int(cfg.max_rows_per_shard)].reset_index(drop=True)
    if rows.empty:
        raise ValueError(f"{spec.shard_name}: no rows selected.")

    arrays_dir = spec.shard_dir / "arrays"
    shard_actions = np.load(arrays_dir / "actions.npy", mmap_mode="r")
    shard_action_mask = np.load(arrays_dir / "action_mask.npy", mmap_mode="r")
    shard_state = np.load(arrays_dir / "state.npy", mmap_mode="r")
    shard_frame_position = np.load(arrays_dir / "frame_position.npy", mmap_mode="r")

    total_occurrences = 0
    target_metadata_cache: dict[str, dict[str, np.ndarray | int]] = {}
    for episode_uid, group in rows.groupby("episode_uid", sort=False):
        episode_uid = str(episode_uid)
        target_path = cfg.dataset_root / "episodes" / episode_uid / "arrays" / cfg.settings.local_target_npz_name
        with np.load(target_path, allow_pickle=False) as target_archive:
            metadata = _load_target_metadata(target_archive, episode_uid)
        target_metadata_cache[episode_uid] = metadata
        offsets = np.asarray(metadata["frame_index_offsets"], dtype=np.int64)
        sample_indices = group["local_target_npz_sample_index"].to_numpy(dtype=np.int64)
        total_occurrences += int(np.sum(offsets[sample_indices + 1] - offsets[sample_indices]))

    cfg.scratch_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{cfg.split}_{spec.shard_name}_{os.getpid()}"
    mae_path = cfg.scratch_dir / f"{stem}_mae.npy"
    max_abs_path = cfg.scratch_dir / f"{stem}_max_abs.npy"
    frame_count_path = cfg.scratch_dir / f"{stem}_frame_counts.npy"
    mae_values = np.lib.format.open_memmap(mae_path, mode="w+", dtype=np.float32, shape=(total_occurrences,))
    max_abs_values = np.lib.format.open_memmap(max_abs_path, mode="w+", dtype=np.float32, shape=(total_occurrences,))
    target_frame_counts = np.lib.format.open_memmap(frame_count_path, mode="w+", dtype=np.float32, shape=(len(rows),))

    def progress(event: str, **payload: Any) -> None:
        if progress_queue is None:
            return
        message = {
            "event": event,
            "worker_id": os.getpid(),
            "shard_id": int(spec.shard_id),
            "shard_name": spec.shard_name,
            **payload,
        }
        if callable(progress_queue):
            progress_queue(message)
        else:
            progress_queue.put(message)

    progress("shard_start", rows=len(rows), frame_occurrences=total_occurrences)

    cursor = 0
    target_cursor = 0
    processed = 0
    max_anchor_state_abs_error = 0.0
    max_shard_state_abs_error = 0.0
    max_frame_mae = 0.0
    max_frame_max_abs = 0.0
    action_cache: dict[str, np.ndarray] = {}
    state_cache: dict[str, np.ndarray] = {}

    for episode_uid, group in rows.groupby("episode_uid", sort=False):
        episode_uid = str(episode_uid)
        episode_arrays = cfg.dataset_root / "episodes" / episode_uid / "arrays"
        action = action_cache.get(episode_uid)
        if action is None:
            action = np.load(episode_arrays / cfg.action_array_name, mmap_mode="r")
            action_cache[episode_uid] = action
        state = state_cache.get(episode_uid)
        if state is None:
            state = np.load(episode_arrays / cfg.state_array_name, mmap_mode="r")
            state_cache[episode_uid] = state
        metadata = target_metadata_cache[episode_uid]
        sample_ids = np.asarray(metadata["sample_ids"], dtype=np.int64)
        anchor_state = np.asarray(metadata["anchor_state"], dtype=np.float64)
        frame_index_values = np.asarray(metadata["frame_index_values"], dtype=np.int64)
        frame_index_offsets = np.asarray(metadata["frame_index_offsets"], dtype=np.int64)
        v_values = np.asarray(metadata["v_values"], dtype=np.float64)
        v_offsets = np.asarray(metadata["v_offsets"], dtype=np.int64)

        for row in group.itertuples(index=True):
            row_series = rows.iloc[int(row.Index)]
            physical_row = _source_row_index(row_series, int(row.Index))
            sample_index = int(row_series["local_target_npz_sample_index"])
            start_frame = int(row_series["frame_position"])
            if sample_index < 0 or sample_index >= sample_ids.size:
                raise ValueError(f"{spec.shard_name}: sample index out of range: {sample_index}")
            if int(sample_ids[sample_index]) != start_frame:
                raise ValueError(
                    f"{spec.shard_name}: sample_id/frame_position mismatch for {episode_uid}: "
                    f"sample_id={int(sample_ids[sample_index])} frame_position={start_frame}"
                )
            if int(shard_frame_position[physical_row]) != start_frame:
                raise ValueError(
                    f"{spec.shard_name}: shard frame_position mismatch for physical_row={physical_row}: "
                    f"{int(shard_frame_position[physical_row])} != {start_frame}"
                )

            frame_start, frame_end = frame_index_offsets[sample_index : sample_index + 2]
            v_start, v_end = v_offsets[sample_index : sample_index + 2]
            frame_indices = frame_index_values[frame_start:frame_end]
            local_v = v_values[v_start:v_end]
            if frame_indices.size == 0 or frame_indices.size != local_v.size:
                raise ValueError(f"{spec.shard_name}: invalid frame/v pairing for sample_index={sample_index}.")
            if int(frame_indices[0]) != start_frame or int(frame_indices[-1]) >= action.shape[0]:
                raise ValueError(f"{spec.shard_name}: invalid source frame indices for sample_index={sample_index}.")
            if np.any(np.diff(frame_indices) != 1) or np.any(np.diff(local_v) < 0):
                raise ValueError(f"{spec.shard_name}: frame/v values must be ordered for sample_index={sample_index}.")
            if not np.isclose(local_v[0], 0.0, atol=1.0e-6) or not np.isclose(local_v[-1], 1.0, atol=1.0e-6):
                raise ValueError(f"{spec.shard_name}: local v endpoints must be 0 and 1 for sample_index={sample_index}.")

            control_points, knot_vector = _decode_packed_spline(
                shard_actions[physical_row],
                shard_action_mask[physical_row],
                cfg.settings,
            )
            current_state = np.asarray(shard_state[physical_row], dtype=np.float64)
            dataset_anchor_state = np.asarray(state[start_frame], dtype=np.float64)
            stored_anchor_state = anchor_state[sample_index]
            max_anchor_state_abs_error = max(
                max_anchor_state_abs_error,
                float(np.max(np.abs(stored_anchor_state - dataset_anchor_state))),
            )
            max_shard_state_abs_error = max(
                max_shard_state_abs_error,
                float(np.max(np.abs(current_state - dataset_anchor_state))),
            )
            reconstructed = BSpline(knot_vector, control_points, int(cfg.settings.degree))(local_v) + current_state
            difference = np.asarray(action[frame_indices], dtype=np.float64) - reconstructed
            frame_mae = np.mean(np.abs(difference), axis=1)
            frame_max_abs = np.max(np.abs(difference), axis=1)
            next_cursor = cursor + frame_mae.size
            mae_values[cursor:next_cursor] = frame_mae.astype(np.float32)
            max_abs_values[cursor:next_cursor] = frame_max_abs.astype(np.float32)
            cursor = next_cursor
            target_frame_counts[target_cursor] = float(frame_indices.size)
            target_cursor += 1
            max_frame_mae = max(max_frame_mae, float(np.max(frame_mae)))
            max_frame_max_abs = max(max_frame_max_abs, float(np.max(frame_max_abs)))

            processed += 1
            if processed % cfg.progress_update_rows == 0:
                progress(
                    "rows",
                    delta=cfg.progress_update_rows,
                    processed=processed,
                    max_mae_65d=max_frame_mae,
                    max_abs=max_frame_max_abs,
                )

    if processed % cfg.progress_update_rows:
        progress(
            "rows",
            delta=processed % cfg.progress_update_rows,
            processed=processed,
            max_mae_65d=max_frame_mae,
            max_abs=max_frame_max_abs,
        )
    if cursor != total_occurrences or target_cursor != len(rows):
        raise RuntimeError(
            f"{spec.shard_name}: metric cursor mismatch cursor={cursor}/{total_occurrences} "
            f"targets={target_cursor}/{len(rows)}"
        )
    mae_values.flush()
    max_abs_values.flush()
    target_frame_counts.flush()
    del mae_values, max_abs_values, target_frame_counts
    progress(
        "shard_done",
        rows=processed,
        frame_occurrences=total_occurrences,
        max_mae_65d=max_frame_mae,
        max_abs=max_frame_max_abs,
    )
    return {
        "split": cfg.split,
        "shard_id": int(spec.shard_id),
        "shard_name": spec.shard_name,
        "status": "processed",
        "num_rows": int(processed),
        "episodes": int(rows["episode_uid"].nunique()),
        "target_frame_occurrences": int(total_occurrences),
        "max_frame_mae_65d": float(max_frame_mae),
        "max_frame_max_abs": float(max_frame_max_abs),
        "max_anchor_state_abs_error": float(max_anchor_state_abs_error),
        "max_shard_state_abs_error": float(max_shard_state_abs_error),
        "anchor_state_tolerance_exceeded": bool(max_anchor_state_abs_error > cfg.anchor_state_tolerance),
        "shard_state_tolerance_exceeded": bool(max_shard_state_abs_error > cfg.shard_state_tolerance),
        "mae_values_path": str(mae_path),
        "max_abs_values_path": str(max_abs_path),
        "frame_count_values_path": str(frame_count_path),
    }


class ProgressRenderer:
    def __init__(self, total_rows: int, total_shards: int, worker_count: int) -> None:
        self.total_rows = tqdm(total=total_rows, desc="Verify shard spline rows", unit="row", position=0, dynamic_ncols=True)
        self.total_shards = tqdm(total=total_shards, desc="Verify shards", unit="shard", position=1, dynamic_ncols=True)
        self.worker_slots: dict[int, int] = {}
        self.bars = [
            tqdm(total=1, desc=f"Worker {slot + 1}: idle", unit="row", leave=False, dynamic_ncols=True, position=slot + 2)
            for slot in range(worker_count)
        ]
        self.completed_shards: set[int] = set()

    def close(self) -> None:
        self.total_rows.close()
        self.total_shards.close()
        for bar in self.bars:
            bar.close()

    def _bar(self, worker_id: int) -> tqdm:
        if worker_id not in self.worker_slots:
            self.worker_slots[worker_id] = min(len(self.worker_slots), max(0, len(self.bars) - 1))
        return self.bars[self.worker_slots[worker_id]]

    def handle_event(self, event: dict[str, Any]) -> None:
        event_type = str(event.get("event", ""))
        worker_id = int(event.get("worker_id", -1))
        shard_id = int(event.get("shard_id", -1))
        shard_name = str(event.get("shard_name", f"shard_{shard_id:05d}"))
        if event_type == "shard_start":
            bar = self._bar(worker_id)
            bar.reset(total=max(1, int(event.get("rows", 0))))
            bar.set_description(f"{shard_name}")
            bar.set_postfix(frame_occurrences=int(event.get("frame_occurrences", 0)))
            return
        if event_type == "rows":
            delta = int(event.get("delta", 0))
            if delta > 0:
                self.total_rows.update(delta)
                bar = self._bar(worker_id)
                bar.update(delta)
                bar.set_postfix(
                    max_mae_65d=f"{float(event.get('max_mae_65d', 0.0)):.6g}",
                    max_abs=f"{float(event.get('max_abs', 0.0)):.6g}",
                )
            return
        if event_type == "shard_done":
            if shard_id not in self.completed_shards:
                self.total_shards.update(1)
                self.completed_shards.add(shard_id)
            bar = self._bar(worker_id)
            remainder = int(bar.total or 0) - int(bar.n)
            if remainder > 0:
                bar.update(remainder)
            bar.set_postfix(
                max_mae_65d=f"{float(event.get('max_mae_65d', 0.0)):.6g}",
                max_abs=f"{float(event.get('max_abs', 0.0)):.6g}",
            )
            return


def _process_shard_with_progress(spec: _spline_shards.ShardSpec, cfg: VerificationConfig, progress_queue: Any) -> dict[str, Any]:
    return process_shard(spec, cfg, progress_queue)


def _drain_progress(progress_queue: Any, renderer: ProgressRenderer) -> None:
    while True:
        try:
            renderer.handle_event(progress_queue.get_nowait())
        except Empty:
            return


def process_all(cfg: VerificationConfig) -> list[dict[str, Any]]:
    shards = _selected_shards(cfg)
    total_rows = 0
    for spec in shards:
        rows = pd.read_parquet(spec.rows_path)
        total_rows += min(len(rows), cfg.max_rows_per_shard) if cfg.max_rows_per_shard is not None else len(rows)
    worker_count = min(max(1, cfg.num_workers), len(shards))
    renderer = ProgressRenderer(total_rows, len(shards), worker_count)
    results: list[dict[str, Any]] = []
    try:
        if worker_count == 1:
            for spec in shards:
                result = process_shard(
                    spec,
                    cfg,
                    renderer.handle_event,
                )
                results.append(result)
            return results
        from multiprocessing import Manager

        with Manager() as manager:
            progress_queue = manager.Queue()
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                pending = {
                    executor.submit(_process_shard_with_progress, spec, cfg, progress_queue): spec
                    for spec in shards
                }
                while pending:
                    completed, _ = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)
                    _drain_progress(progress_queue, renderer)
                    for future in completed:
                        spec = pending.pop(future)
                        try:
                            result = future.result()
                        except Exception:
                            tqdm.write(f"Failed shard: {spec.shard_name}")
                            raise
                        results.append(result)
                _drain_progress(progress_queue, renderer)
    finally:
        renderer.close()
    return sorted(results, key=lambda item: int(item["shard_id"]))


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temporary_path.replace(path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary_path, index=False)
    temporary_path.replace(path)


def _streaming_moments(values: np.memmap, chunk_size: int = 1_000_000) -> tuple[float, float, float, float]:
    count = int(values.size)
    if count == 0:
        return 0.0, 0.0, 0.0, 0.0
    total = 0.0
    total_squared = 0.0
    minimum = float("inf")
    maximum = float("-inf")
    for start in range(0, count, chunk_size):
        chunk = np.asarray(values[start : start + chunk_size], dtype=np.float64)
        total += float(np.sum(chunk))
        total_squared += float(np.dot(chunk, chunk))
        minimum = min(minimum, float(np.min(chunk)))
        maximum = max(maximum, float(np.max(chunk)))
    mean = total / count
    variance = max(0.0, total_squared / count - mean * mean)
    return mean, float(np.sqrt(variance)), minimum, maximum


def _percentile_key(percentile: float) -> str:
    if percentile == int(percentile):
        return f"p{int(percentile)}"
    return f"p{str(percentile).replace('.', '_')}"


def _exact_distribution(values_path: Path) -> dict[str, float | int]:
    values = np.load(values_path, mmap_mode="r+")
    count = int(values.size)
    if count == 0:
        return {"count": 0, "mean": 0.0, "std": 0.0, **{_percentile_key(p): 0.0 for p in PERCENTILES}}
    mean, std, minimum, maximum = _streaming_moments(values)
    distribution: dict[str, float | int] = {"count": count, "mean": mean, "std": std, "p0": minimum, "p100": maximum}
    for percentile in PERCENTILES[1:-1]:
        position = (count - 1) * percentile / 100.0
        lower = int(np.floor(position))
        upper = int(np.ceil(position))
        values.partition((lower, upper))
        lower_value = float(values[lower])
        upper_value = float(values[upper])
        distribution[_percentile_key(percentile)] = lower_value + (position - lower) * (upper_value - lower_value)
    return distribution


def _merge_metric_files(results: list[dict[str, Any]], key: str, output_path: Path) -> Path:
    paths = [Path(str(result[key])) for result in results]
    total = sum(int(np.load(path, mmap_mode="r").size) for path in paths)
    merged = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float32, shape=(total,))
    cursor = 0
    for path in paths:
        values = np.load(path, mmap_mode="r")
        next_cursor = cursor + int(values.size)
        merged[cursor:next_cursor] = values
        cursor = next_cursor
    merged.flush()
    del merged
    return output_path


def _format_distribution(distribution: dict[str, float | int]) -> str:
    keys = ("count", "mean", "std", "p0", "p1", "p2", "p5", "p10", "p25", "p50", "p75", "p95", "p99", "p99_9", "p99_99", "p100")
    return ", ".join(
        f"{key}={distribution[key]}" if key == "count" else f"{key}={float(distribution[key]):.9g}"
        for key in keys
    )


def _serialize_config(cfg: VerificationConfig) -> dict[str, Any]:
    output = asdict(cfg)
    output["settings"] = dataclasses.asdict(cfg.settings)
    output["dataset_root"] = str(cfg.dataset_root)
    output["shard_root"] = str(cfg.shard_root)
    output["output_dir"] = str(cfg.output_dir)
    output["scratch_dir"] = str(cfg.scratch_dir)
    return output


def write_summary(cfg: VerificationConfig, results: list[dict[str, Any]]) -> tuple[Path, dict[str, dict[str, float | int]]]:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    reported_results = [
        {key: value for key, value in result.items() if not key.endswith("_values_path")}
        for result in results
    ]
    _atomic_parquet(pd.DataFrame(reported_results), cfg.output_dir / f"{cfg.split}_spline_shard_reconstruction_per_shard.parquet")
    distributions: dict[str, dict[str, float | int]] = {}
    metric_specs = (
        ("shard_spline_frame_mae_65d", "mae_values_path"),
        ("shard_spline_frame_max_absolute_dimension_error", "max_abs_values_path"),
        ("target_frame_count_per_spline", "frame_count_values_path"),
    )
    for name, result_key in metric_specs:
        merged_path = cfg.scratch_dir / f"merged_{name}.npy"
        distributions[name] = _exact_distribution(_merge_metric_files(results, result_key, merged_path))
        if not cfg.keep_intermediate_error_arrays:
            merged_path.unlink()

    summary = {
        "config": _serialize_config(cfg),
        "shards_verified": len(results),
        "rows_verified": sum(int(result["num_rows"]) for result in results),
        "episodes_touched": sum(int(result["episodes"]) for result in results),
        "target_frame_occurrences_verified": sum(int(result["target_frame_occurrences"]) for result in results),
        "worst_anchor_state_abs_error": max((float(result["max_anchor_state_abs_error"]) for result in results), default=0.0),
        "worst_shard_state_abs_error": max((float(result["max_shard_state_abs_error"]) for result in results), default=0.0),
        "shards_exceeding_anchor_state_tolerance": sum(bool(result["anchor_state_tolerance_exceeded"]) for result in results),
        "shards_exceeding_shard_state_tolerance": sum(bool(result["shard_state_tolerance_exceeded"]) for result in results),
        "distributions": distributions,
        "per_shard_results": reported_results,
    }
    summary_path = cfg.output_dir / f"{cfg.split}_spline_shard_reconstruction_summary.json"
    _atomic_json(summary, summary_path)
    return summary_path, distributions


def cleanup_scratch(cfg: VerificationConfig) -> None:
    if cfg.keep_intermediate_error_arrays or not cfg.scratch_dir.exists():
        return
    resolved_scratch = cfg.scratch_dir.resolve()
    resolved_output = cfg.output_dir.resolve()
    if resolved_output not in resolved_scratch.parents or not resolved_scratch.name.startswith(".scratch_"):
        raise RuntimeError(f"Refusing to remove unexpected scratch path: {resolved_scratch}")
    shutil.rmtree(resolved_scratch)


def main() -> int:
    cfg = _resolve(parse_args())
    print("Origami comp action-spline shard reconstruction verification")
    print(f"  dataset_root       : {cfg.dataset_root}")
    print(f"  shard_root         : {cfg.shard_root}")
    print(f"  split              : {cfg.split}")
    print(f"  local target NPZ   : {cfg.settings.local_target_npz_name}")
    print(f"  packed target shape: ({cfg.settings.action_horizon}, {cfg.settings.action_dim})")
    print(f"  control points     : {cfg.settings.max_control_points}")
    print(f"  knot spans         : {cfg.settings.max_span_count}")
    print(f"  degree             : {cfg.settings.degree}")
    print(f"  num_workers        : {cfg.num_workers}")
    print(f"  output_dir         : {cfg.output_dir}")
    try:
        results = process_all(cfg)
        if not results:
            raise RuntimeError("No shards were verified.")
        summary_path, distributions = write_summary(cfg, results)
    finally:
        cleanup_scratch(cfg)

    print(f"Shards verified      : {len(results)}")
    print(f"Rows verified        : {sum(int(result['num_rows']) for result in results)}")
    print(f"Target frame samples : {sum(int(result['target_frame_occurrences']) for result in results)}")
    print(
        "Worst anchor error   : "
        f"{max((float(result['max_anchor_state_abs_error']) for result in results), default=0.0):.3e}"
    )
    print(
        "Worst shard-state err: "
        f"{max((float(result['max_shard_state_abs_error']) for result in results), default=0.0):.3e}"
    )
    print(f"Frame MAE-65D        : {_format_distribution(distributions['shard_spline_frame_mae_65d'])}")
    print(
        "Frame max abs dim err: "
        f"{_format_distribution(distributions['shard_spline_frame_max_absolute_dimension_error'])}"
    )
    print(f"Frames per spline    : {_format_distribution(distributions['target_frame_count_per_spline'])}")
    print(f"Summary              : {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
