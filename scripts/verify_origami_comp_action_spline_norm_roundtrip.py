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
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.origami_comp_action_spline_shards as _spline_shards
import openpi.training.origami_vla_dataset as _origami_vla_dataset
import verify_origami_comp_action_spline_shard_reconstruction as _base


PERCENTILES = _base.PERCENTILES


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
    roundtrip_tolerance: float
    keep_intermediate_error_arrays: bool
    norm_stats_dir: Path
    use_quantile_norm: bool
    span_logit_eps: float


@dataclass(frozen=True)
class PackedNormStats:
    state: _normalize.NormStats
    tactile_prompt: _normalize.NormStats | None
    control_points: _normalize.NormStats
    span_logits: _normalize.NormStats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the Origami comp action-spline training target round trip: physical shard spline "
            "target -> normalized model target -> unnormalized logits/control points -> softmax widths -> "
            "reconstructed spline."
        )
    )
    parser.add_argument("--config-name", type=str, default="pi05_origami_comp_action_spline")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--manifest-root", type=Path, default=None)
    parser.add_argument("--shard-root", type=Path, default=None)
    parser.add_argument("--norm-stats-dir", type=Path, default=None)
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
    parser.add_argument("--roundtrip-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--span-logit-eps", type=float, default=1.0e-6)
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
    if settings.spline_span_representation != "logits":
        raise ValueError(
            f"Config {args.config_name!r} uses spline_span_representation={settings.spline_span_representation!r}; "
            "expected 'logits'."
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

    norm_stats_dir = args.norm_stats_dir
    if norm_stats_dir is None:
        stats_dir = train_config.model.origami_vla.action_norm_stats_dir
        if not stats_dir:
            asset_id = train_config.data.assets.asset_id or train_config.data.repo_id
            if asset_id is None:
                raise ValueError("Could not infer norm stats dir; pass --norm-stats-dir.")
            stats_dir = str((Path(train_config.assets_base_dir) / train_config.name / str(asset_id)).resolve())
        norm_stats_dir = Path(stats_dir)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path(settings.shard_root) / f"{args.split}_spline_norm_roundtrip_verification"
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
        roundtrip_tolerance=float(args.roundtrip_tolerance),
        keep_intermediate_error_arrays=bool(args.keep_intermediate_error_arrays),
        norm_stats_dir=Path(norm_stats_dir),
        use_quantile_norm=bool(train_config.model.origami_vla.use_quantile_norm),
        span_logit_eps=float(args.span_logit_eps),
    )


def _load_norm_stats(cfg: VerificationConfig) -> PackedNormStats:
    loaded = _normalize.load(_download.maybe_download(str(cfg.norm_stats_dir)))
    required = ("state", "actions_control_points", "actions_span_logits")
    missing = [key for key in required if key not in loaded]
    if missing:
        raise KeyError(
            f"Missing norm stats {missing} under {cfg.norm_stats_dir}. "
            "Run scripts/compute_origami_comp_action_spline_norm_stats.py first."
        )
    if cfg.use_quantile_norm:
        for key in required:
            stats = loaded[key]
            if stats.q01 is None or stats.q99 is None:
                raise ValueError(f"Quantile norm is enabled, but {key} lacks q01/q99.")
    return PackedNormStats(
        state=loaded["state"],
        tactile_prompt=loaded.get("tactile_prompt"),
        control_points=loaded["actions_control_points"],
        span_logits=loaded["actions_span_logits"],
    )


def _apply_norm(x: np.ndarray, stats: _normalize.NormStats, *, use_quantiles: bool) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if use_quantiles:
        if stats.q01 is None or stats.q99 is None:
            raise ValueError("Quantile stats requested but q01/q99 are missing.")
        q01 = np.asarray(stats.q01, dtype=np.float32)[..., : x.shape[-1]]
        q99 = np.asarray(stats.q99, dtype=np.float32)[..., : x.shape[-1]]
        return (x - q01) / (q99 - q01 + 1.0e-6) * 2.0 - 1.0
    mean = np.asarray(stats.mean, dtype=np.float32)[..., : x.shape[-1]]
    std = np.asarray(stats.std, dtype=np.float32)[..., : x.shape[-1]]
    return (x - mean) / (std + 1.0e-6)


def _apply_unnorm(x: np.ndarray, stats: _normalize.NormStats, *, use_quantiles: bool) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if use_quantiles:
        if stats.q01 is None or stats.q99 is None:
            raise ValueError("Quantile stats requested but q01/q99 are missing.")
        q01 = np.asarray(stats.q01, dtype=np.float32)[..., : x.shape[-1]]
        q99 = np.asarray(stats.q99, dtype=np.float32)[..., : x.shape[-1]]
        return (x + 1.0) / 2.0 * (q99 - q01 + 1.0e-6) + q01
    mean = np.asarray(stats.mean, dtype=np.float32)[..., : x.shape[-1]]
    std = np.asarray(stats.std, dtype=np.float32)[..., : x.shape[-1]]
    return x * (std + 1.0e-6) + mean


def _span_widths_to_centered_logits(widths: np.ndarray, *, eps: float) -> np.ndarray:
    widths = np.asarray(widths, dtype=np.float32)
    logits = np.log(np.clip(widths, eps, None))
    return logits - np.mean(logits, axis=-1, keepdims=True)


def _softmax_widths(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=-1, keepdims=True)


def _normalize_packed_actions(
    packed_actions: np.ndarray,
    stats: PackedNormStats,
    cfg: VerificationConfig,
) -> np.ndarray:
    actions = np.asarray(packed_actions, dtype=np.float32)
    normalized = np.array(actions, copy=True)
    max_control_points = int(cfg.settings.max_control_points)
    max_span_count = int(cfg.settings.max_span_count)
    normalized[:max_control_points, :] = _apply_norm(
        actions[:max_control_points, :],
        stats.control_points,
        use_quantiles=cfg.use_quantile_norm,
    )
    span_logits = _span_widths_to_centered_logits(
        actions[max_control_points, :max_span_count],
        eps=cfg.span_logit_eps,
    )
    normalized[max_control_points, :max_span_count] = _apply_norm(
        span_logits,
        stats.span_logits,
        use_quantiles=cfg.use_quantile_norm,
    )
    return normalized


def _decode_normalized_actions(
    normalized_actions: np.ndarray,
    stats: PackedNormStats,
    cfg: VerificationConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normalized = np.asarray(normalized_actions, dtype=np.float32)
    max_control_points = int(cfg.settings.max_control_points)
    max_span_count = int(cfg.settings.max_span_count)
    control_points = _apply_unnorm(
        normalized[:max_control_points, :],
        stats.control_points,
        use_quantiles=cfg.use_quantile_norm,
    ).astype(np.float64)
    span_logits = _apply_unnorm(
        normalized[max_control_points, :max_span_count],
        stats.span_logits,
        use_quantiles=cfg.use_quantile_norm,
    )
    widths = _softmax_widths(span_logits)
    knots = _base._knot_vector_from_span_widths(widths, int(cfg.settings.degree))
    return control_points, widths, knots


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


def process_shard(
    spec: _spline_shards.ShardSpec,
    cfg: VerificationConfig,
    progress_queue: Any | None = None,
) -> dict[str, Any]:
    stats = _load_norm_stats(cfg)
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
            metadata = _base._load_target_metadata(target_archive, episode_uid)
        target_metadata_cache[episode_uid] = metadata
        offsets = np.asarray(metadata["frame_index_offsets"], dtype=np.int64)
        sample_indices = group["local_target_npz_sample_index"].to_numpy(dtype=np.int64)
        total_occurrences += int(np.sum(offsets[sample_indices + 1] - offsets[sample_indices]))

    cfg.scratch_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{cfg.split}_{spec.shard_name}_{os.getpid()}"
    mae_path = cfg.scratch_dir / f"{stem}_mae.npy"
    max_abs_path = cfg.scratch_dir / f"{stem}_max_abs.npy"
    baseline_delta_path = cfg.scratch_dir / f"{stem}_baseline_delta.npy"
    control_mae_path = cfg.scratch_dir / f"{stem}_control_mae.npy"
    control_max_abs_path = cfg.scratch_dir / f"{stem}_control_max_abs.npy"
    width_mae_path = cfg.scratch_dir / f"{stem}_width_mae.npy"
    width_max_abs_path = cfg.scratch_dir / f"{stem}_width_max_abs.npy"
    width_sum_error_path = cfg.scratch_dir / f"{stem}_width_sum_error.npy"
    frame_count_path = cfg.scratch_dir / f"{stem}_frame_counts.npy"

    mae_values = np.lib.format.open_memmap(mae_path, mode="w+", dtype=np.float32, shape=(total_occurrences,))
    max_abs_values = np.lib.format.open_memmap(max_abs_path, mode="w+", dtype=np.float32, shape=(total_occurrences,))
    baseline_delta_values = np.lib.format.open_memmap(
        baseline_delta_path,
        mode="w+",
        dtype=np.float32,
        shape=(total_occurrences,),
    )
    control_mae_values = np.lib.format.open_memmap(control_mae_path, mode="w+", dtype=np.float32, shape=(len(rows),))
    control_max_abs_values = np.lib.format.open_memmap(
        control_max_abs_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(rows),),
    )
    width_mae_values = np.lib.format.open_memmap(width_mae_path, mode="w+", dtype=np.float32, shape=(len(rows),))
    width_max_abs_values = np.lib.format.open_memmap(
        width_max_abs_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(rows),),
    )
    width_sum_error_values = np.lib.format.open_memmap(
        width_sum_error_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(rows),),
    )
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
    max_baseline_delta = 0.0
    max_control_error = 0.0
    max_width_error = 0.0
    max_width_sum_error = 0.0
    min_roundtrip_width = float("inf")
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
            physical_row = _base._source_row_index(row_series, int(row.Index))
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

            physical_control, physical_knot_vector = _base._decode_packed_spline(
                shard_actions[physical_row],
                shard_action_mask[physical_row],
                cfg.settings,
            )
            physical_widths = np.asarray(
                shard_actions[physical_row][int(cfg.settings.max_control_points), : int(cfg.settings.max_span_count)],
                dtype=np.float64,
            )
            normalized_actions = _normalize_packed_actions(shard_actions[physical_row], stats, cfg)
            decoded_control, decoded_widths, decoded_knot_vector = _decode_normalized_actions(
                normalized_actions,
                stats,
                cfg,
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

            baseline = BSpline(physical_knot_vector, physical_control, int(cfg.settings.degree))(local_v) + current_state
            reconstructed = BSpline(decoded_knot_vector, decoded_control, int(cfg.settings.degree))(local_v) + current_state
            difference = np.asarray(action[frame_indices], dtype=np.float64) - reconstructed
            frame_mae = np.mean(np.abs(difference), axis=1)
            frame_max_abs = np.max(np.abs(difference), axis=1)
            baseline_delta = np.max(np.abs(baseline - reconstructed), axis=1)

            control_diff = decoded_control - physical_control
            width_diff = decoded_widths.reshape(-1) - physical_widths.reshape(-1)
            control_mae = float(np.mean(np.abs(control_diff)))
            control_max_abs = float(np.max(np.abs(control_diff)))
            width_mae = float(np.mean(np.abs(width_diff)))
            width_max_abs = float(np.max(np.abs(width_diff)))
            width_sum_error = float(abs(np.sum(decoded_widths) - 1.0))

            next_cursor = cursor + frame_mae.size
            mae_values[cursor:next_cursor] = frame_mae.astype(np.float32)
            max_abs_values[cursor:next_cursor] = frame_max_abs.astype(np.float32)
            baseline_delta_values[cursor:next_cursor] = baseline_delta.astype(np.float32)
            cursor = next_cursor

            control_mae_values[target_cursor] = np.float32(control_mae)
            control_max_abs_values[target_cursor] = np.float32(control_max_abs)
            width_mae_values[target_cursor] = np.float32(width_mae)
            width_max_abs_values[target_cursor] = np.float32(width_max_abs)
            width_sum_error_values[target_cursor] = np.float32(width_sum_error)
            target_frame_counts[target_cursor] = float(frame_indices.size)
            target_cursor += 1

            max_frame_mae = max(max_frame_mae, float(np.max(frame_mae)))
            max_frame_max_abs = max(max_frame_max_abs, float(np.max(frame_max_abs)))
            max_baseline_delta = max(max_baseline_delta, float(np.max(baseline_delta)))
            max_control_error = max(max_control_error, control_max_abs)
            max_width_error = max(max_width_error, width_max_abs)
            max_width_sum_error = max(max_width_sum_error, width_sum_error)
            min_roundtrip_width = min(min_roundtrip_width, float(np.min(decoded_widths)))

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
    for array in (
        mae_values,
        max_abs_values,
        baseline_delta_values,
        control_mae_values,
        control_max_abs_values,
        width_mae_values,
        width_max_abs_values,
        width_sum_error_values,
        target_frame_counts,
    ):
        array.flush()
    del (
        mae_values,
        max_abs_values,
        baseline_delta_values,
        control_mae_values,
        control_max_abs_values,
        width_mae_values,
        width_max_abs_values,
        width_sum_error_values,
        target_frame_counts,
    )
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
        "max_baseline_roundtrip_abs_delta": float(max_baseline_delta),
        "max_control_point_abs_delta": float(max_control_error),
        "max_span_width_abs_delta": float(max_width_error),
        "max_span_sum_abs_error": float(max_width_sum_error),
        "min_roundtrip_span_width": float(min_roundtrip_width),
        "max_anchor_state_abs_error": float(max_anchor_state_abs_error),
        "max_shard_state_abs_error": float(max_shard_state_abs_error),
        "anchor_state_tolerance_exceeded": bool(max_anchor_state_abs_error > cfg.anchor_state_tolerance),
        "shard_state_tolerance_exceeded": bool(max_shard_state_abs_error > cfg.shard_state_tolerance),
        "roundtrip_tolerance_exceeded": bool(
            max(
                max_baseline_delta,
                max_control_error,
                max_width_error,
                max_width_sum_error,
            )
            > cfg.roundtrip_tolerance
        ),
        "mae_values_path": str(mae_path),
        "max_abs_values_path": str(max_abs_path),
        "baseline_delta_values_path": str(baseline_delta_path),
        "control_mae_values_path": str(control_mae_path),
        "control_max_abs_values_path": str(control_max_abs_path),
        "width_mae_values_path": str(width_mae_path),
        "width_max_abs_values_path": str(width_max_abs_path),
        "width_sum_error_values_path": str(width_sum_error_path),
        "frame_count_values_path": str(frame_count_path),
    }


def _process_shard_with_progress(spec: _spline_shards.ShardSpec, cfg: VerificationConfig, progress_queue: Any) -> dict[str, Any]:
    return process_shard(spec, cfg, progress_queue)


def _drain_progress(progress_queue: Any, renderer: _base.ProgressRenderer) -> None:
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
    renderer = _base.ProgressRenderer(total_rows, len(shards), worker_count)
    results: list[dict[str, Any]] = []
    try:
        if worker_count == 1:
            for spec in shards:
                results.append(process_shard(spec, cfg, renderer.handle_event))
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
                            results.append(future.result())
                        except Exception:
                            tqdm.write(f"Failed shard: {spec.shard_name}")
                            raise
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


def _serialize_config(cfg: VerificationConfig) -> dict[str, Any]:
    output = asdict(cfg)
    output["settings"] = dataclasses.asdict(cfg.settings)
    output["dataset_root"] = str(cfg.dataset_root)
    output["shard_root"] = str(cfg.shard_root)
    output["output_dir"] = str(cfg.output_dir)
    output["scratch_dir"] = str(cfg.scratch_dir)
    output["norm_stats_dir"] = str(cfg.norm_stats_dir)
    return output


def write_summary(cfg: VerificationConfig, results: list[dict[str, Any]]) -> tuple[Path, dict[str, dict[str, float | int]]]:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    reported_results = [
        {key: value for key, value in result.items() if not key.endswith("_values_path")}
        for result in results
    ]
    _atomic_parquet(pd.DataFrame(reported_results), cfg.output_dir / f"{cfg.split}_spline_norm_roundtrip_per_shard.parquet")
    distributions: dict[str, dict[str, float | int]] = {}
    metric_specs = (
        ("norm_roundtrip_frame_mae_65d", "mae_values_path"),
        ("norm_roundtrip_frame_max_absolute_dimension_error", "max_abs_values_path"),
        ("baseline_vs_norm_roundtrip_frame_max_abs", "baseline_delta_values_path"),
        ("control_point_roundtrip_mae", "control_mae_values_path"),
        ("control_point_roundtrip_max_abs", "control_max_abs_values_path"),
        ("span_width_roundtrip_mae", "width_mae_values_path"),
        ("span_width_roundtrip_max_abs", "width_max_abs_values_path"),
        ("span_width_sum_abs_error", "width_sum_error_values_path"),
        ("target_frame_count_per_spline", "frame_count_values_path"),
    )
    for name, result_key in metric_specs:
        merged_path = cfg.scratch_dir / f"merged_{name}.npy"
        distributions[name] = _base._exact_distribution(_merge_metric_files(results, result_key, merged_path))
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
        "worst_baseline_roundtrip_abs_delta": max((float(result["max_baseline_roundtrip_abs_delta"]) for result in results), default=0.0),
        "worst_control_point_abs_delta": max((float(result["max_control_point_abs_delta"]) for result in results), default=0.0),
        "worst_span_width_abs_delta": max((float(result["max_span_width_abs_delta"]) for result in results), default=0.0),
        "worst_span_sum_abs_error": max((float(result["max_span_sum_abs_error"]) for result in results), default=0.0),
        "minimum_roundtrip_span_width": min((float(result["min_roundtrip_span_width"]) for result in results), default=0.0),
        "shards_exceeding_anchor_state_tolerance": sum(bool(result["anchor_state_tolerance_exceeded"]) for result in results),
        "shards_exceeding_shard_state_tolerance": sum(bool(result["shard_state_tolerance_exceeded"]) for result in results),
        "shards_exceeding_roundtrip_tolerance": sum(bool(result["roundtrip_tolerance_exceeded"]) for result in results),
        "distributions": distributions,
        "per_shard_results": reported_results,
    }
    summary_path = cfg.output_dir / f"{cfg.split}_spline_norm_roundtrip_summary.json"
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
    print("Origami comp action-spline norm round-trip verification")
    print(f"  dataset_root       : {cfg.dataset_root}")
    print(f"  shard_root         : {cfg.shard_root}")
    print(f"  norm_stats_dir     : {cfg.norm_stats_dir}")
    print(f"  split              : {cfg.split}")
    print(f"  local target NPZ   : {cfg.settings.local_target_npz_name}")
    print(f"  packed target shape: ({cfg.settings.action_horizon}, {cfg.settings.action_dim})")
    print(f"  control points     : {cfg.settings.max_control_points}")
    print(f"  knot spans         : {cfg.settings.max_span_count}")
    print(f"  span representation: {cfg.settings.spline_span_representation}")
    print(f"  use_quantile_norm  : {cfg.use_quantile_norm}")
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
    print(
        "Worst roundtrip delta: "
        f"{max((float(result['max_baseline_roundtrip_abs_delta']) for result in results), default=0.0):.3e}"
    )
    print(
        "Worst ctrl-pt delta  : "
        f"{max((float(result['max_control_point_abs_delta']) for result in results), default=0.0):.3e}"
    )
    print(
        "Worst span-width err : "
        f"{max((float(result['max_span_width_abs_delta']) for result in results), default=0.0):.3e}"
    )
    print(
        "Worst span-sum err   : "
        f"{max((float(result['max_span_sum_abs_error']) for result in results), default=0.0):.3e}"
    )
    print(f"Frame MAE-65D        : {_base._format_distribution(distributions['norm_roundtrip_frame_mae_65d'])}")
    print(
        "Frame max abs dim err: "
        f"{_base._format_distribution(distributions['norm_roundtrip_frame_max_absolute_dimension_error'])}"
    )
    print(
        "Baseline roundtrip   : "
        f"{_base._format_distribution(distributions['baseline_vs_norm_roundtrip_frame_max_abs'])}"
    )
    print(f"Control point MAE    : {_base._format_distribution(distributions['control_point_roundtrip_mae'])}")
    print(f"Span width MAE       : {_base._format_distribution(distributions['span_width_roundtrip_mae'])}")
    print(f"Frames per spline    : {_base._format_distribution(distributions['target_frame_count_per_spline'])}")
    print(f"Summary              : {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
