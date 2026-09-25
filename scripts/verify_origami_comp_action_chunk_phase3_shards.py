"""Portable structural/full verification for completed Phase-3 shards."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import openpi.training.origami_comp_action_chunk_phase3_shards as _phase3


def _check_numeric(array: np.ndarray, *, name: str, chunk_rows: int, failures: list[str]) -> None:
    if array.dtype.kind not in "fc":
        return
    for start in range(0, len(array), chunk_rows):
        values = np.asarray(array[start : start + chunk_rows])
        if not np.isfinite(values).all():
            failures.append(f"{name}: non-finite values in rows {start}:{min(len(array), start + chunk_rows)}")
            return


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify completed Phase-3 mixed-speed shards without raw data.")
    parser.add_argument("--config-name", default="pi05_origami_comp_action_chunk_phase3")
    parser.add_argument("--shard-root", type=Path, default=None)
    parser.add_argument("--split", choices=("train", "val", "all"), default="all")
    parser.add_argument("--mode", choices=("structure", "full"), default="full")
    parser.add_argument("--chunk-rows", type=int, default=4096)
    parser.add_argument("--max-failures", type=int, default=50)
    args = parser.parse_args()
    config, data = _phase3.phase3_config(args.config_name)
    build = data.shard_build
    root = args.shard_root or Path(data.shard_root or "")
    manifest = json.loads((root / _phase3.SHARD_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    failures: list[str] = []
    if manifest.get("format") != _phase3.FORMAT_VERSION:
        failures.append(f"unexpected format: {manifest.get('format')!r}")
    if _phase3.canonical_json(manifest.get("mixed_speed")) != _phase3.canonical_json(
        dataclasses.asdict(build.mixed_speed)
    ):
        failures.append("top-level mixed-speed contract differs from config")
    splits = ("train", "val") if args.split == "all" else (args.split,)
    seen_uids: dict[str, set[str]] = {split: set() for split in splits}
    expected_strides = set(build.mixed_speed.ordered_strides)
    for entry in tqdm(manifest.get("shards", []), desc="Verify Phase-3 shards", unit="shard"):
        split = str(entry.get("split"))
        if split not in splits:
            continue
        directory = root / str(entry["relative_dir"])
        if not (directory / _phase3.COMPLETE_MARKER).is_file():
            failures.append(f"{directory}: missing complete marker")
            continue
        metadata_path = directory / _phase3.METADATA_FILENAME
        rows_path = directory / _phase3.ROWS_FILENAME
        if not metadata_path.is_file() or not rows_path.is_file():
            failures.append(f"{directory}: missing metadata or rows")
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("format") != _phase3.FORMAT_VERSION:
            failures.append(f"{directory}: unexpected metadata format")
        if _phase3.canonical_json(metadata.get("mixed_speed")) != _phase3.canonical_json(
            dataclasses.asdict(build.mixed_speed)
        ):
            failures.append(f"{directory}: mixed-speed contract differs from config")
        rows = pd.read_parquet(rows_path)
        if len(rows) != int(entry["num_rows"]) or len(rows) != int(metadata["num_rows"]):
            failures.append(f"{directory}: metadata/manifest/rows count mismatch")
        if rows.duplicated(subset=["episode_uid", "frame_position"]).any():
            failures.append(f"{directory}: duplicate episode/frame rows")
        if "local_physical_row" not in rows or not np.array_equal(
            rows["local_physical_row"].to_numpy(dtype=np.int64), np.arange(len(rows), dtype=np.int64)
        ):
            failures.append(f"{directory}: local_physical_row is not contiguous from zero")
        expected_episodes = tuple(str(uid) for uid in entry["episodes"])
        actual_episodes = tuple(str(uid) for uid in rows["episode_uid"].drop_duplicates())
        if expected_episodes != actual_episodes:
            failures.append(f"{directory}: episode order differs from manifest")
        if not bool(entry.get("complete")):
            failures.append(f"{directory}: top-level manifest does not mark the shard complete")
        seen_uids[split].update(actual_episodes)
        specs = metadata.get("arrays", {})
        arrays: dict[str, np.ndarray] = {}
        for name, descriptor in specs.items():
            path = directory / descriptor["filename"]
            if not path.is_file():
                failures.append(f"{directory}: missing {name}")
                continue
            array = np.load(path, mmap_mode="r")
            arrays[name] = array
            if list(array.shape) != list(descriptor["shape"]) or str(array.dtype) != descriptor["dtype"]:
                failures.append(f"{directory}: {name} descriptor mismatch")
            if args.mode == "full":
                _check_numeric(array, name=f"{directory}:{name}", chunk_rows=max(1, args.chunk_rows), failures=failures)
        for stride in expected_strides:
            action_name = f"actions_stride_{stride}"
            shape = (len(rows), build.mixed_speed.action_horizon, config.model.action_dim)
            if action_name not in arrays or arrays[action_name].shape != shape:
                failures.append(f"{directory}: invalid {action_name} shape")
        if "planner_available" in arrays and np.any(arrays["planner_available"]):
            failures.append(f"{directory}: planner_available contains true values")
        for name in ("planner_state_belief", "planner_progress_transition", "planner_uncertainty", "planner_history_latent"):
            if name in arrays and np.any(arrays[name] != 0.0):
                failures.append(f"{directory}: {name} is not all zero")
        plan_info: dict[str, Any] = metadata.get("virtual_plan", {})
        plan_path = directory / str(plan_info.get("filename", ""))
        if not plan_path.is_file():
            failures.append(f"{directory}: missing virtual plan")
        else:
            plan = np.load(plan_path, mmap_mode="r")
            if plan.dtype != np.dtype(np.uint32) or plan.ndim != 2 or plan.shape[1] != 3:
                failures.append(f"{directory}: invalid virtual-plan dtype or shape")
            else:
                if len(plan) != int(plan_info.get("logical_samples", -1)):
                    failures.append(f"{directory}: virtual-plan count mismatch")
                if len(plan) and (np.any(plan[:, 0] >= len(rows)) or not set(np.unique(plan[:, 1])).issubset(expected_strides)):
                    failures.append(f"{directory}: invalid virtual-plan references")
                if len(plan):
                    ordered = plan[np.lexsort((plan[:, 2], plan[:, 1], plan[:, 0]))]
                    pair_starts = np.r_[True, np.any(ordered[1:, :2] != ordered[:-1, :2], axis=1)]
                    boundaries = np.r_[np.flatnonzero(pair_starts), len(ordered)]
                    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
                        if not np.array_equal(ordered[start:end, 2], np.arange(end - start, dtype=np.uint32)):
                            failures.append(f"{directory}: invalid virtual-plan occurrence IDs")
                            break
                for stride, detail in plan_info.get("by_speed", {}).items():
                    actual = int(np.count_nonzero(plan[:, 1] == int(stride)))
                    if actual != int(detail["logical_samples"]):
                        failures.append(f"{directory}: speed-{stride} virtual-plan count mismatch")
                # The plan is deterministic from physical rows, shard id, and
                # the configured episode-coverage contract. Rebuilding it
                # catches a plan that is self-consistent in metadata but has
                # the wrong speed mixture or episode selection.
                expected_plan, expected_info = _phase3.build_virtual_plan(
                    rows, build.mixed_speed, shard_id=int(metadata.get("shard_id", entry["shard_id"]))
                )
                expected_plan_filename = f"arrays/{build.virtual_sample_plan_name}"
                if plan_info.get("filename") != expected_plan_filename:
                    failures.append(f"{directory}: unexpected virtual-plan filename")
                # ``filename`` is storage location, not coverage metadata;
                # compare it above and compare only the deterministic plan
                # fields here. Comparing the full object would make every
                # valid shard fail because build_virtual_plan has no filename.
                actual_coverage_info = {key: plan_info.get(key) for key in expected_info}
                if _phase3.canonical_json(actual_coverage_info) != _phase3.canonical_json(expected_info):
                    failures.append(f"{directory}: virtual-plan coverage metadata differs from config")
                if args.mode == "full" and not np.array_equal(plan, expected_plan):
                    failures.append(f"{directory}: virtual plan differs from deterministic expected plan")
        if len(failures) >= args.max_failures:
            break
    if args.split == "all" and seen_uids["train"] & seen_uids["val"]:
        failures.append(f"train/val shard episode overlap: {sorted(seen_uids['train'] & seen_uids['val'])[:5]}")
    if failures:
        print("FAILED")
        print("\n".join(f"- {failure}" for failure in failures[: args.max_failures]))
        return 1
    print(f"OK: completed Phase-3 {args.split} shards pass {args.mode} verification.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
