from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import openpi.training.config as _config
import openpi.training.origami_comp_action_chunk_shards as _shards
import openpi.training.origami_vla_dataset as _origami_vla_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify Origami comp action-chunk shards against source videos, arrays, and planner exports."
    )
    parser.add_argument("--config-name", type=str, default="pi05_origami_comp_action_chunk")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--manifest-root", type=Path, default=None)
    parser.add_argument("--shard-root", type=Path, default=None)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--rtol", type=float, default=1.0e-5)
    parser.add_argument("--atol", type=float, default=1.0e-6)
    parser.add_argument("--max-failures", type=int, default=20)
    return parser.parse_args()


def _resolve(args: argparse.Namespace) -> tuple[_origami_vla_dataset.OrigamiVlaSettings, int]:
    config = _config.get_config(args.config_name)
    if not isinstance(config.data, _config.OrigamiCompActionChunkDataConfig):
        raise TypeError(
            f"Config {args.config_name!r} uses {type(config.data).__name__}; "
            "expected OrigamiCompActionChunkDataConfig."
        )
    settings = _shards.settings_from_data_factory(config.data, config.model)
    shard_root = args.shard_root or (Path(config.data.shard_root) if config.data.shard_root else None)
    if shard_root is None:
        raise ValueError("Set data.shard_root or pass --shard-root.")
    if args.dataset_root is not None:
        settings = dataclasses.replace(settings, dataset_root=str(args.dataset_root))
    if args.manifest_root is not None:
        settings = dataclasses.replace(settings, manifest_root=str(args.manifest_root))
    settings = dataclasses.replace(settings, dataset_backend="shard", shard_root=str(shard_root))
    image_size = int(args.image_size or config.data.shard_build.image_size)
    return settings, image_size


def _read_rgb_frame(video_path: Path, frame_position: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_position))
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"Could not read frame {frame_position} from {video_path}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()


def _read_live_sample(
    settings: _origami_vla_dataset.OrigamiVlaSettings,
    row: pd.Series,
    *,
    image_size: int,
    planner_cache: dict[tuple[str, str], Any],
) -> dict[str, Any]:
    episode_uid = str(row["episode_uid"])
    frame_position = int(row["frame_position"])
    episode_root = Path(settings.dataset_root) / "episodes" / episode_uid
    arrays_root = episode_root / "arrays"
    state = np.load(arrays_root / "state_65d.npy", mmap_mode="r")
    actions_source = np.load(arrays_root / settings.action_filename, mmap_mode="r")
    tactile = np.load(arrays_root / settings.tactile_filename, mmap_mode="r")
    timestamps = _shards._read_timestamps(arrays_root, state.shape[0])
    frame_index = _shards._read_frame_index(arrays_root, state.shape[0])

    images = {}
    for image_key, relpath in settings.image_modalities.items():
        frame = _read_rgb_frame(episode_root / relpath, frame_position)
        images[image_key] = _shards.resize_with_pad_uint8(frame, image_size, image_size)

    chunk, action_mask = _origami_vla_dataset.extract_action_chunk(
        actions_source,
        frame_position,
        action_horizon=settings.action_horizon,
        action_dim=settings.action_dim,
        action_chunk_stride=settings.action_chunk_stride,
    )
    sample_weight = row.get(settings.sample_weight_column, 1.0)
    if pd.isna(sample_weight):
        sample_weight = 1.0
    output: dict[str, Any] = {
        "image": images,
        "state": np.asarray(state[frame_position], dtype=np.float32),
        "tactile": np.asarray(tactile[frame_position], dtype=np.float32).reshape(-1),
        "actions": chunk,
        "action_mask": action_mask,
        "sample_weight": np.asarray(float(sample_weight), dtype=np.float32),
        "frame_position": np.asarray(frame_position, dtype=np.int64),
        "frame_index": np.asarray(int(frame_index[frame_position]), dtype=np.int64),
        "timestamp": np.asarray(float(timestamps[frame_position]), dtype=np.float32),
    }

    if settings.load_tactile_images:
        deform_frame = _read_rgb_frame(episode_root / settings.tactile_deform_video, frame_position)
        deform_images = _shards.split_tactile_grid(
            deform_frame,
            settings.tactile_deform_grid,
            episode_uid,
            int(settings.tactile_image_size),
        )
        raw_path = episode_root / settings.tactile_raw_video
        raw_available = raw_path.exists()
        if raw_available:
            try:
                raw_frame = _read_rgb_frame(raw_path, frame_position)
                raw_images = _shards.split_tactile_grid(
                    raw_frame,
                    settings.tactile_raw_grid,
                    episode_uid,
                    int(settings.tactile_image_size),
                )
            except RuntimeError:
                if settings.tactile_require_raw_video:
                    raise
                raw_available = False
                raw_images = np.zeros_like(deform_images)
        else:
            raw_images = np.zeros_like(deform_images)
        if raw_available and _shards._drop_tactile_raw_input(settings, episode_uid, frame_position):
            raw_available = False
            raw_images = np.zeros_like(deform_images)
        output["tactile_deform_images"] = deform_images
        output["tactile_raw_images"] = raw_images
        output["tactile_raw_available"] = np.asarray(raw_available, dtype=bool)

    if settings.include_planner_features:
        planner_enabled = _shards._row_bool(row.get("planner_enabled", True), default=True)
        output["planner_available"] = np.asarray(planner_enabled, dtype=bool)
        if not planner_enabled:
            output["planner_state_belief"] = np.zeros((settings.planner_belief_dim,), dtype=np.float32)
            output["planner_progress_transition"] = np.zeros((settings.planner_progress_dim,), dtype=np.float32)
            output["planner_uncertainty"] = np.zeros((settings.planner_uncertainty_dim,), dtype=np.float32)
            output["planner_history_latent"] = np.zeros((settings.planner_history_dim,), dtype=np.float32)
        else:
            view_mode = str(row["view_mode"])
            branch_value = row.get("planner_branch", settings.planner_branch)
            if pd.isna(branch_value) or not str(branch_value):
                branch_value = settings.planner_branch
            variant_value = row.get("planner_value_variant", settings.planner_value_variant)
            if pd.isna(variant_value) or not str(variant_value):
                variant_value = settings.planner_value_variant
            planner_output_dir = Path(str(row["planner_output_dir"]))
            cache_key = (view_mode, str(planner_output_dir))
            planner = planner_cache.get(cache_key)
            if planner is None:
                planner = np.load(planner_output_dir / settings.planner_arrays_filename, allow_pickle=False)
                planner_cache[cache_key] = planner
            planner_row_index = int(row["planner_row_index"])
            branch = str(branch_value)
            output["planner_state_belief"] = np.asarray(
                _shards._read_planner_feature(
                    planner,
                    _shards._planner_state_belief_key(str(variant_value)),
                    branch,
                    planner_row_index,
                ),
                dtype=np.float32,
            )
            output["planner_progress_transition"] = np.asarray(
                _shards._read_planner_feature(planner, "progress_transition", branch, planner_row_index),
                dtype=np.float32,
            )
            output["planner_uncertainty"] = np.asarray(
                _shards._read_planner_feature(planner, "uncertainty_features", branch, planner_row_index),
                dtype=np.float32,
            )
            output["planner_history_latent"] = np.asarray(
                _shards._read_planner_feature(planner, "temporal_latent", branch, planner_row_index),
                dtype=np.float32,
            )
    return output


def _compare_array(name: str, left: Any, right: Any, failures: list[str], *, rtol: float, atol: float) -> None:
    left_arr = np.asarray(left)
    right_arr = np.asarray(right)
    if left_arr.shape != right_arr.shape:
        failures.append(f"{name}: shape {left_arr.shape} != {right_arr.shape}")
        return
    exact_dtypes = (np.bool_, np.uint8)
    if left_arr.dtype in exact_dtypes or right_arr.dtype in exact_dtypes:
        if not np.array_equal(left_arr, right_arr):
            diff = int(np.count_nonzero(left_arr != right_arr))
            failures.append(f"{name}: exact mismatch count={diff}")
        return
    if not np.allclose(left_arr, right_arr, rtol=rtol, atol=atol):
        failures.append(
            f"{name}: max_abs={float(np.max(np.abs(left_arr - right_arr))):.6g} "
            f"mean_abs={float(np.mean(np.abs(left_arr - right_arr))):.6g}"
        )


def main() -> int:
    args = parse_args()
    settings, image_size = _resolve(args)
    shard_dataset = _shards.OrigamiCompActionChunkShardDataset(settings, split=args.split)
    manifest_settings = dataclasses.replace(settings, dataset_backend="video")
    manifest_rows = _origami_vla_dataset.load_manifest_rows(manifest_settings, args.split).reset_index(drop=True)
    rng = np.random.default_rng(int(args.seed))
    sample_count = min(int(args.num_samples), len(shard_dataset))
    indices = rng.choice(len(shard_dataset), size=sample_count, replace=False)
    failures: list[str] = []
    planner_cache: dict[tuple[str, str], Any] = {}

    print("Origami comp action-chunk shard verification")
    print(f"  config_name  : {args.config_name}")
    print(f"  dataset_root : {settings.dataset_root}")
    print(f"  manifest_root: {settings.manifest_root}")
    print(f"  shard_root   : {settings.shard_root}")
    print(f"  split        : {args.split}")
    print(f"  shard rows   : {len(shard_dataset):,}")
    print(f"  samples      : {sample_count:,}")

    scalar_keys = [
        "state",
        "tactile",
        "actions",
        "action_mask",
        "sample_weight",
        "frame_position",
        "frame_index",
        "timestamp",
    ]
    planner_keys = [
        "planner_available",
        "planner_state_belief",
        "planner_progress_transition",
        "planner_uncertainty",
        "planner_history_latent",
    ]
    tactile_keys = ["tactile_deform_images", "tactile_raw_images", "tactile_raw_available"]

    for sample_index in tqdm(indices, desc="Verify shard samples", unit="sample", dynamic_ncols=True):
        shard_sample = shard_dataset[int(sample_index)]
        source_row_index = int(shard_sample["source_row_index"])
        if source_row_index < 0 or source_row_index >= len(manifest_rows):
            failures.append(f"sample {sample_index}: source_row_index={source_row_index} out of range")
            if len(failures) >= args.max_failures:
                break
            continue
        source_row = manifest_rows.iloc[source_row_index]
        live_sample = _read_live_sample(settings, source_row, image_size=image_size, planner_cache=planner_cache)
        prefix = f"sample={sample_index} source_row={source_row_index}"
        for key in scalar_keys:
            _compare_array(
                f"{prefix}:{key}",
                shard_sample[key],
                live_sample[key],
                failures,
                rtol=args.rtol,
                atol=args.atol,
            )
        for image_key in settings.image_modalities:
            _compare_array(
                f"{prefix}:image/{image_key}",
                shard_sample["image"][image_key],
                live_sample["image"][image_key],
                failures,
                rtol=args.rtol,
                atol=args.atol,
            )
        if settings.load_tactile_images:
            for key in tactile_keys:
                _compare_array(
                    f"{prefix}:{key}",
                    shard_sample[key],
                    live_sample[key],
                    failures,
                    rtol=args.rtol,
                    atol=args.atol,
                )
        if settings.include_planner_features:
            for key in planner_keys:
                _compare_array(
                    f"{prefix}:{key}",
                    shard_sample[key],
                    live_sample[key],
                    failures,
                    rtol=args.rtol,
                    atol=args.atol,
                )
        if len(failures) >= args.max_failures:
            break

    for archive in planner_cache.values():
        close = getattr(archive, "close", None)
        if close is not None:
            close()

    if failures:
        print("\nFAILED")
        for failure in failures[: args.max_failures]:
            print(f"- {failure}")
        return 1
    print("\nOK: sampled shard rows match source videos, arrays, planner features, masks, and weights.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
