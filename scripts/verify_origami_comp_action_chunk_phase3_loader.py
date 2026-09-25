"""Verify Phase-3 logical loading, dropout accounting, and batch planning."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import openpi.training.origami_comp_action_chunk_phase3_dataset as _phase3_dataset
import openpi.training.origami_comp_action_chunk_phase3_shards as _phase3


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify portable Phase-3 logical loader behavior.")
    parser.add_argument("--config-name", default="pi05_origami_comp_action_chunk_phase3")
    parser.add_argument("--shard-root", type=Path, default=None)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--global-batch-size", type=int, default=None)
    parser.add_argument("--logical-chunk-rows", type=int, default=8192)
    parser.add_argument("--loader-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> int:
    args = _args()
    config, data = _phase3.phase3_config(args.config_name)
    runtime = data.create(config.assets_dirs, config.model)
    if runtime.origami_vla is None or not runtime.origami_vla.phase3_mixed_speed_enabled:
        raise RuntimeError(f"{args.config_name!r} is not a Phase-3 runtime config.")
    settings = runtime.origami_vla
    root = args.shard_root or Path(settings.shard_root or "")
    # The direct loader checks below must use the command-line root too, not
    # merely the root used for the preceding portable scan.
    settings = dataclasses.replace(settings, shard_root=str(root))
    batch_size = int(args.global_batch_size or config.batch_size)
    if batch_size <= 0:
        raise ValueError("--global-batch-size must be positive.")
    chunk_rows = max(1, int(args.logical_chunk_rows))
    failures: list[str] = []
    physical_raw_available = physical_raw_missing = 0
    logical_raw_present = logical_raw_dropped = logical_raw_missing = 0
    logical_by_speed: dict[str, int] = {str(s): 0 for s in data.shard_build.mixed_speed.ordered_strides}
    raw_by_speed: dict[str, dict[str, int]] = {
        str(s): {"present": 0, "dropped": 0, "missing": 0} for s in data.shard_build.mixed_speed.ordered_strides
    }
    total_logical = 0
    specs = _phase3.load_phase3_specs(root, split=args.split, require_complete=True)
    for spec in tqdm(specs, desc="Verify Phase-3 logical plan", unit="shard"):
        metadata = json.loads((spec.shard_dir / _phase3.METADATA_FILENAME).read_text(encoding="utf-8"))
        if metadata.get("format") != settings.phase3_shard_format:
            failures.append(f"{spec.shard_name}: wrong format")
            continue
        arrays_dir = spec.shard_dir / "arrays"
        rows_frame = pd.read_parquet(spec.shard_dir / _phase3.ROWS_FILENAME)
        episode_uids = rows_frame["episode_uid"].astype(str).to_numpy()
        frame_positions = rows_frame["frame_position"].to_numpy(dtype=np.int64)
        plan = np.load(spec.shard_dir / str(metadata["virtual_plan"]["filename"]), mmap_mode="r")
        raw_available = np.load(arrays_dir / "tactile_raw_available.npy", mmap_mode="r")
        planner_available = np.load(arrays_dir / "planner_available.npy", mmap_mode="r")
        if np.any(planner_available):
            failures.append(f"{spec.shard_name}: planner_available is not all false")
        physical_raw_available += int(np.count_nonzero(raw_available))
        physical_raw_missing += int(len(raw_available) - np.count_nonzero(raw_available))
        for image_key in settings.image_modalities:
            image = np.load(arrays_dir / f"image_{image_key}.npy", mmap_mode="r")
            if image.shape != (spec.num_rows, 224, 224, 3) or image.dtype != np.uint8:
                failures.append(f"{spec.shard_name}: image_{image_key} is not OpenPI 224x224 uint8 input")
        for start in range(0, len(plan), chunk_rows):
            block = np.asarray(plan[start : start + chunk_rows])
            for physical, stride, occurrence in block.tolist():
                physical = int(physical)
                stride = int(stride)
                occurrence = int(occurrence)
                key = str(stride)
                if physical < 0 or physical >= len(rows_frame) or key not in logical_by_speed:
                    failures.append(f"{spec.shard_name}: invalid virtual occurrence")
                    continue
                logical_by_speed[key] += 1
                total_logical += 1
                if not bool(raw_available[physical]):
                    logical_raw_missing += 1
                    raw_by_speed[key]["missing"] += 1
                    continue
                dropped = _phase3.drop_tactile_raw_for_virtual_sample(
                    settings,
                    episode_uid=episode_uids[physical],
                    frame_position=int(frame_positions[physical]),
                    stride=stride,
                    occurrence_id=occurrence,
                )
                if dropped:
                    logical_raw_dropped += 1
                    raw_by_speed[key]["dropped"] += 1
                else:
                    logical_raw_present += 1
                    raw_by_speed[key]["present"] += 1
    dataset = _phase3_dataset.OrigamiCompActionChunkPhase3Dataset(settings, split=args.split)
    rng = np.random.default_rng(args.seed)
    for index in rng.choice(len(dataset), size=min(len(dataset), max(0, int(args.loader_samples))), replace=False):
        bundle, local_index = dataset._locate(int(index))
        physical, expected_speed, occurrence = (int(value) for value in bundle.plan[local_index])
        item = dataset[int(index)]
        speed = int(item["phase3_speed_id"])
        if speed != expected_speed:
            failures.append(f"logical sample {index}: speed differs from virtual plan")
        expected_actions = np.asarray(bundle.arrays[f"actions_stride_{expected_speed}"][physical], dtype=np.float32)
        if not np.array_equal(item["actions"], expected_actions):
            failures.append(f"logical sample {index}: selected action target differs from stride array")
        if not np.all(item["action_mask"]):
            failures.append(f"logical sample {index}: action mask is not all true")
        expected_prompt = settings.phase3_prompt_template.format(speed=speed)
        if str(item["prompt"]) != expected_prompt:
            failures.append(f"logical sample {index}: prompt/speed mismatch")
        if bool(item.get("planner_available", False)):
            failures.append(f"logical sample {index}: planner unexpectedly available")
        for name in (
            "planner_state_belief",
            "planner_progress_transition",
            "planner_uncertainty",
            "planner_history_latent",
        ):
            if name in item and np.any(item[name] != 0.0):
                failures.append(f"logical sample {index}: {name} is not zero")
        if item["actions"].shape != (settings.action_horizon, settings.action_dim):
            failures.append(f"logical sample {index}: incorrect selected action shape")
        if settings.load_tactile_images:
            row = bundle.rows.iloc[physical]
            expected_raw = bool(bundle.arrays["tactile_raw_available"][physical]) and not _phase3.drop_tactile_raw_for_virtual_sample(
                settings,
                episode_uid=str(row["episode_uid"]),
                frame_position=int(row["frame_position"]),
                stride=expected_speed,
                occurrence_id=occurrence,
            )
            if bool(item["tactile_raw_available"]) != expected_raw:
                failures.append(f"logical sample {index}: raw tactile availability differs from occurrence rule")
            if not expected_raw and np.any(item["tactile_raw_images"]):
                failures.append(f"logical sample {index}: dropped raw tactile image is not zero")
    remainder = total_logical % batch_size
    eligible_raw = logical_raw_present + logical_raw_dropped
    raw_drop_rate = None if not eligible_raw else logical_raw_dropped / eligible_raw
    if raw_drop_rate is not None:
        expected = float(settings.tactile_raw_input_dropout_prob)
        tolerance = max(0.01, 6.0 * np.sqrt(expected * (1.0 - expected) / eligible_raw))
        if abs(raw_drop_rate - expected) > tolerance:
            failures.append(
                f"logical raw-drop rate {raw_drop_rate:.6f} differs from configured {expected:.6f} by more than {tolerance:.6f}"
            )
    print("Phase-3 logical loader report")
    print(f"  split                       : {args.split}")
    print(f"  physical raw available/missing: {physical_raw_available:,}/{physical_raw_missing:,}")
    print(f"  logical rows by speed       : {logical_by_speed}")
    print(f"  logical raw present         : {logical_raw_present:,}")
    print(f"  logical raw policy-dropped  : {logical_raw_dropped:,}")
    print(f"  logical raw source-missing  : {logical_raw_missing:,}")
    print(f"  logical raw-drop rate       : {raw_drop_rate if raw_drop_rate is not None else 'n/a'}")
    print(f"  raw accounting by speed     : {raw_by_speed}")
    print(f"  batch size/full batches/remainder: {batch_size}/{total_logical // batch_size:,}/{remainder}")
    if failures:
        print("FAILED")
        print("\n".join(f"- {failure}" for failure in failures[:50]))
        return 1
    print("OK: Phase-3 logical loader contract is valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
