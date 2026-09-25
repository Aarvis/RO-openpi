"""Sampled raw-source reconstruction verification for Phase-3 shards.

Run only on the remote build machine, where both the source dataset and the
completed shard package are available.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import openpi.training.origami_comp_action_chunk_phase3_shards as _phase3
import openpi.training.origami_comp_action_chunk_shards as _chunk
import openpi.training.origami_vla_dataset as _dataset


def _frame(path: Path, position: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open {path}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, position)
        ok, value = capture.read()
        if not ok or value is None:
            raise RuntimeError(f"Could not read frame {position} from {path}")
        return cv2.cvtColor(value, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()


def main() -> int:
    parser = argparse.ArgumentParser(description="Sample Phase-3 shard rows against raw source data.")
    parser.add_argument("--config-name", default="pi05_origami_comp_action_chunk_phase3_build")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--shard-root", type=Path, default=None)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--include-images", action="store_true")
    parser.add_argument("--atol", type=float, default=1.0e-6)
    parser.add_argument("--rtol", type=float, default=1.0e-5)
    args = parser.parse_args()
    config, data = _phase3.phase3_config(args.config_name)
    build = data.shard_build
    settings = _chunk.settings_from_data_factory(data, config.model)
    source_root = args.dataset_root or Path(settings.dataset_root)
    shard_root = args.shard_root or Path(data.shard_root or "")
    specs = _phase3.load_phase3_specs(shard_root, split=args.split)
    rng = np.random.default_rng(args.seed)
    failures: list[str] = []
    candidates: list[tuple[Any, int]] = []
    for spec in specs:
        rows = pd.read_parquet(spec.shard_dir / _phase3.ROWS_FILENAME)
        candidates.extend((spec, int(row)) for row in rows["local_physical_row"].to_numpy(dtype=np.int64))
    chosen = rng.choice(len(candidates), size=min(args.num_samples, len(candidates)), replace=False)
    for choice in tqdm(chosen, desc="Reconstruct Phase-3 source samples", unit="sample"):
        spec, row_id = candidates[int(choice)]
        rows = pd.read_parquet(spec.shard_dir / _phase3.ROWS_FILENAME)
        row = rows.iloc[row_id]
        episode_uid, position = str(row["episode_uid"]), int(row["frame_position"])
        arrays_dir = spec.shard_dir / "arrays"
        episode_arrays = source_root / "episodes" / episode_uid / "arrays"
        source_state = np.load(episode_arrays / "state_65d.npy", mmap_mode="r")
        source_tactile = np.load(episode_arrays / settings.tactile_filename, mmap_mode="r")
        source_actions = np.load(episode_arrays / settings.action_filename, mmap_mode="r")
        for name, source in (("state", source_state[position]), ("tactile", source_tactile[position])):
            shard = np.load(arrays_dir / f"{name}.npy", mmap_mode="r")[row_id]
            if not np.allclose(shard, source, atol=args.atol, rtol=args.rtol):
                failures.append(f"{spec.shard_name}:{row_id}: {name} mismatch")
        source_timestamps = _chunk._read_timestamps(episode_arrays, source_state.shape[0])
        source_frame_index = _chunk._read_frame_index(episode_arrays, source_state.shape[0])
        for name, source in (
            ("frame_position", position),
            ("frame_index", int(source_frame_index[position])),
            ("timestamp", float(source_timestamps[position])),
            ("sample_weight", 1.0),
        ):
            shard = np.load(arrays_dir / f"{name}.npy", mmap_mode="r")[row_id]
            if not np.isclose(shard, source, atol=args.atol, rtol=args.rtol):
                failures.append(f"{spec.shard_name}:{row_id}: {name} mismatch")
        for stride in build.mixed_speed.ordered_strides:
            target, mask = _dataset.extract_action_chunk(
                source_actions,
                position,
                action_horizon=build.mixed_speed.action_horizon,
                action_dim=settings.action_dim,
                action_chunk_stride=stride,
            )
            shard = np.load(arrays_dir / f"actions_stride_{stride}.npy", mmap_mode="r")[row_id]
            if not np.all(mask) or not np.allclose(shard, target, atol=args.atol, rtol=args.rtol):
                failures.append(f"{spec.shard_name}:{row_id}: stride-{stride} action mismatch")
        if args.include_images:
            episode_root = source_root / "episodes" / episode_uid
            for image_key, relpath in settings.image_modalities.items():
                source = _chunk.resize_with_pad_uint8(_frame(episode_root / relpath, position), build.image_size, build.image_size)
                shard = np.load(arrays_dir / f"{_chunk.IMAGE_ARRAY_PREFIX}{image_key}.npy", mmap_mode="r")[row_id]
                if not np.array_equal(shard, source):
                    failures.append(f"{spec.shard_name}:{row_id}: image {image_key} mismatch")
            if settings.load_tactile_images:
                deform = _chunk.split_tactile_grid(
                    _frame(episode_root / settings.tactile_deform_video, position),
                    settings.tactile_deform_grid,
                    episode_uid,
                    settings.tactile_image_size,
                )
                shard_deform = np.load(arrays_dir / "tactile_deform_images.npy", mmap_mode="r")[row_id]
                if not np.array_equal(shard_deform, deform):
                    failures.append(f"{spec.shard_name}:{row_id}: tactile deform image mismatch")
                raw_path = episode_root / settings.tactile_raw_video
                raw_available = raw_path.exists()
                raw = np.zeros_like(deform)
                if raw_available:
                    raw = _chunk.split_tactile_grid(
                        _frame(raw_path, position), settings.tactile_raw_grid, episode_uid, settings.tactile_image_size
                    )
                shard_raw = np.load(arrays_dir / "tactile_raw_images.npy", mmap_mode="r")[row_id]
                shard_available = bool(np.load(arrays_dir / "tactile_raw_available.npy", mmap_mode="r")[row_id])
                if shard_available != raw_available or not np.array_equal(shard_raw, raw):
                    failures.append(f"{spec.shard_name}:{row_id}: tactile raw image/availability mismatch")
        if failures:
            break
    if failures:
        print("FAILED")
        print("\n".join(f"- {failure}" for failure in failures))
        return 1
    print("OK: sampled Phase-3 state, tactile, and all stride targets reconstruct from raw source.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
