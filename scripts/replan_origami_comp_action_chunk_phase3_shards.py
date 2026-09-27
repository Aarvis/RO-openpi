"""Rebuild only the logical virtual plans of completed Phase-3 shards.

This intentionally never reads raw source episodes or rewrites physical
observation, tactile, state, or action arrays. It is for changing the
episode-balanced speed mixture after a completed physical Phase-3 shard build.
"""

from __future__ import annotations

import argparse
from concurrent import futures
import dataclasses
import json
import multiprocessing
import os
from pathlib import Path
import sys
import traceback
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import openpi.training.origami_comp_action_chunk_phase3_shards as _phase3


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Atomically replan completed Origami Phase-3 virtual sample plans."
    )
    parser.add_argument("--config-name", default="pi05_origami_comp_action_chunk_phase3_build")
    parser.add_argument("--shard-root", type=Path, default=None)
    parser.add_argument("--split", choices=("train", "val", "all"), default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Report target shards and coverage without writing anything.",
    )
    return parser.parse_args()


def _resolve(args: argparse.Namespace) -> tuple[Any, Any, Path, tuple[str, ...], int]:
    config, data = _phase3.phase3_config(args.config_name)
    build = data.shard_build
    shard_root = args.shard_root or (Path(build.shard_root) if build.shard_root else None)
    if shard_root is None:
        raise ValueError("Set shard_root in config or pass --shard-root.")
    split = args.split or build.split
    splits = ("train", "val") if split == "all" else (str(split),)
    workers = int(args.num_workers if args.num_workers is not None else build.num_workers)
    if workers <= 0:
        raise ValueError("--num-workers must be positive.")
    return config, build, shard_root, splits, workers


def _atomic_save_plan(path: Path, plan: np.ndarray) -> None:
    """Write one complete NPY file, then atomically replace the old plan."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.replan.tmp")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, plan, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_required_action_arrays(
    metadata: dict[str, Any], directory: Path, build: Any, *, action_dim: int
) -> None:
    arrays = metadata.get("arrays", {})
    for stride in build.mixed_speed.ordered_strides:
        name = f"actions_stride_{stride}"
        descriptor = arrays.get(name)
        if not isinstance(descriptor, dict):
            raise ValueError(f"{directory}: missing {name} in physical shard metadata.")
        path = directory / str(descriptor.get("filename", ""))
        if not path.is_file():
            raise FileNotFoundError(f"{directory}: missing {name} array: {path}")
        array = np.load(path, mmap_mode="r")
        try:
            expected_shape = (int(metadata["num_rows"]), int(build.mixed_speed.action_horizon), int(action_dim))
            if array.dtype != np.dtype(np.float32) or array.shape != expected_shape:
                raise ValueError(
                    f"{directory}: invalid {name}: expected float32/{expected_shape}, "
                    f"got {array.dtype}/{array.shape}."
                )
        finally:
            _phase3.close_memmap(array)


def _is_current_plan(
    metadata: dict[str, Any],
    *,
    plan_path: Path,
    expected_plan: np.ndarray,
    expected_info: dict[str, Any],
    build: Any,
) -> bool:
    plan_info = metadata.get("virtual_plan", {})
    expected_filename = f"arrays/{build.virtual_sample_plan_name}"
    if (
        _phase3.canonical_json(metadata.get("mixed_speed"))
        != _phase3.canonical_json(dataclasses.asdict(build.mixed_speed))
        or plan_info.get("filename") != expected_filename
        or {key: plan_info.get(key) for key in expected_info} != expected_info
        or not plan_path.is_file()
    ):
        return False
    current = np.load(plan_path, mmap_mode="r")
    try:
        return current.dtype == np.dtype(np.uint32) and current.shape == expected_plan.shape and np.array_equal(
            current, expected_plan
        )
    finally:
        _phase3.close_memmap(current)


def _replan_one(task: dict[str, Any]) -> dict[str, Any]:
    """Replan one completed shard; safe to rerun after an interrupted pass."""
    try:
        config, build, _root, _splits, _workers = _resolve(
            argparse.Namespace(
                config_name=task["config_name"],
                shard_root=Path(task["shard_root"]),
                split=task["split"],
                num_workers=1,
                plan_only=False,
            )
        )
        directory = Path(task["directory"])
        if not (directory / _phase3.COMPLETE_MARKER).is_file():
            raise RuntimeError(f"{directory}: refusing to replan an incomplete shard.")
        metadata_path = directory / _phase3.METADATA_FILENAME
        rows_path = directory / _phase3.ROWS_FILENAME
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("format") != _phase3.FORMAT_VERSION:
            raise ValueError(f"{directory}: unsupported shard format {metadata.get('format')!r}.")
        rows = pd.read_parquet(rows_path)
        if len(rows) != int(metadata.get("num_rows", -1)):
            raise ValueError(f"{directory}: rows/metadata count mismatch.")
        if rows.duplicated(subset=["episode_uid", "frame_position"]).any():
            raise ValueError(f"{directory}: duplicate physical episode/frame rows.")
        _validate_required_action_arrays(metadata, directory, build, action_dim=int(config.model.action_dim))
        shard_id = int(metadata.get("shard_id", task["shard_id"]))
        plan, plan_info = _phase3.build_virtual_plan(rows, build.mixed_speed, shard_id=shard_id)
        plan_path = directory / "arrays" / build.virtual_sample_plan_name
        if _is_current_plan(
            metadata,
            plan_path=plan_path,
            expected_plan=plan,
            expected_info=plan_info,
            build=build,
        ):
            return {"status": "skipped", "shard_name": task["shard_name"]}
        _atomic_save_plan(plan_path, plan)
        revised = dict(metadata)
        revised["mixed_speed"] = dataclasses.asdict(build.mixed_speed)
        revised["virtual_plan"] = {"filename": f"arrays/{build.virtual_sample_plan_name}", **plan_info}
        # The original fingerprint describes immutable physical build inputs.
        # Preserve it and add separate provenance for this logical replan.
        revised["virtual_plan_fingerprint"] = _phase3.fingerprint(
            {
                "physical_build_fingerprint": metadata.get("fingerprint"),
                "mixed_speed": dataclasses.asdict(build.mixed_speed),
                "shard_id": shard_id,
                "rows": int(len(rows)),
            }
        )
        _phase3.atomic_write_json(metadata_path, revised)
        return {"status": "replanned", "shard_name": task["shard_name"], "logical_samples": int(len(plan))}
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "failed",
            "shard_name": task.get("shard_name", str(task.get("directory"))),
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def main() -> int:
    args = _args()
    _config, build, shard_root, splits, workers = _resolve(args)
    manifest_path = shard_root / _phase3.SHARD_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != _phase3.FORMAT_VERSION:
        raise ValueError(f"Unsupported Phase-3 shard format: {manifest.get('format')!r}")
    entries = [entry for entry in manifest.get("shards", []) if str(entry.get("split")) in splits]
    available_splits = {str(entry.get("split")) for entry in manifest.get("shards", [])}
    if set(splits) != available_splits:
        raise ValueError(
            "The root mixed-speed contract applies to every split. Replan all available "
            f"splits {sorted(available_splits)} together (use --split all)."
        )
    if not entries:
        raise RuntimeError(f"No shards found for split(s) {splits} in {shard_root}.")
    incomplete = [
        entry["relative_dir"]
        for entry in entries
        if not (shard_root / str(entry["relative_dir"]) / _phase3.COMPLETE_MARKER).is_file()
    ]
    if incomplete:
        raise RuntimeError(
            "Refusing to replan while selected shards are incomplete: " + ", ".join(map(str, incomplete[:10]))
        )
    tasks = [
        {
            "config_name": args.config_name,
            "shard_root": str(shard_root),
            "split": str(entry["split"]),
            "directory": str(shard_root / str(entry["relative_dir"])),
            "shard_id": int(entry["shard_id"]),
            "shard_name": str(entry["shard_name"]),
        }
        for entry in entries
    ]
    print(
        f"Phase-3 replan: {len(tasks)} completed shard(s), split={','.join(splits)}, "
        f"coverage={build.mixed_speed.stride_episode_coverage}"
    )
    if args.plan_only:
        return 0
    failures: list[dict[str, Any]] = []
    replanned = skipped = 0
    context = multiprocessing.get_context("spawn")
    with tqdm(total=len(tasks), desc="Replan Phase-3 virtual plans", unit="shard") as progress:
        with futures.ProcessPoolExecutor(
            max_workers=min(workers, len(tasks)), mp_context=context
        ) as executor:
            for result in executor.map(_replan_one, tasks):
                progress.update(1)
                status = result["status"]
                if status == "replanned":
                    replanned += 1
                elif status == "skipped":
                    skipped += 1
                else:
                    failures.append(result)
                progress.set_postfix_str(f"replanned={replanned}, skipped={skipped}, failed={len(failures)}")
    if failures:
        for failure in failures:
            print(f"FAILED {failure['shard_name']}: {failure['error']}\n{failure['traceback']}")
        return 1
    # Publish the new root-level contract only after every selected shard has
    # its matching plan and metadata. If interrupted, rerun before verifying.
    revised_manifest = dict(manifest)
    revised_manifest["config_name"] = args.config_name
    revised_manifest["mixed_speed"] = dataclasses.asdict(build.mixed_speed)
    revised_manifest["virtual_plan_replan_fingerprint"] = _phase3.fingerprint(
        {
            "mixed_speed": dataclasses.asdict(build.mixed_speed),
            "selected_splits": splits,
            "shards": [str(entry["relative_dir"]) for entry in entries],
        }
    )
    _phase3.atomic_write_json(manifest_path, revised_manifest)
    print(f"OK: replanned={replanned}, already-current={skipped}, total={len(tasks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
