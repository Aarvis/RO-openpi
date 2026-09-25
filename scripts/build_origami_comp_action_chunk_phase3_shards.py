"""Build resumable, mixed-speed, mask-free Phase-3 Origami shards.

Run this script only on the remote build machine. It intentionally does not
depend on a training loader or a training batch size.
"""

from __future__ import annotations

import argparse
from concurrent import futures
import dataclasses
import json
import multiprocessing
import os
from pathlib import Path
import queue
import shutil
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
import openpi.training.origami_comp_action_chunk_shards as _chunk
import openpi.training.origami_vla_dataset as _dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build resumable Origami Phase-3 mixed-speed shards.")
    parser.add_argument("--config-name", default="pi05_origami_comp_action_chunk_phase3_build")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--manifest-root", type=Path, default=None)
    parser.add_argument("--shard-root", type=Path, default=None)
    parser.add_argument("--split", choices=("train", "val", "all"), default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--target-shard-bytes", type=str, default=None)
    parser.add_argument("--max-episodes-per-shard", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--progress-update-frames",
        type=int,
        default=None,
        help="Send a live worker-progress update after this many decoded frames (default: config value, 256).",
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--start-shard", type=int, default=0)
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args()


def _resolve(args: argparse.Namespace):
    config, data = _phase3.phase3_config(args.config_name)
    build = data.shard_build
    assert hasattr(build, "mixed_speed")
    settings = _chunk.settings_from_data_factory(data, config.model)
    settings = dataclasses.replace(settings, dataset_backend="video")
    if args.dataset_root is not None:
        settings = dataclasses.replace(settings, dataset_root=str(args.dataset_root))
    if args.manifest_root is not None:
        settings = dataclasses.replace(settings, manifest_root=str(args.manifest_root))
    shard_root = args.shard_root or (Path(build.shard_root) if build.shard_root else None)
    if shard_root is None:
        raise ValueError("Set shard_root in config or pass --shard-root.")
    changes: dict[str, Any] = {"shard_root": str(shard_root)}
    for attr in (
        "split",
        "num_workers",
        "target_shard_bytes",
        "max_episodes_per_shard",
        "seed",
        "progress_update_frames",
        "overwrite",
    ):
        value = getattr(args, attr)
        if value is not None:
            changes[attr] = value
    build = dataclasses.replace(build, **changes)
    return config, data, settings, build, shard_root


def _season(uid: str) -> str:
    for marker in ("__episode_", "_episode_"):
        if marker in uid:
            return uid.split(marker, 1)[0]
    return uid


def _episode_infos(rows: pd.DataFrame, dataset_root: Path, season_column: str, row_bytes: int) -> list[dict[str, Any]]:
    metadata_path = dataset_root / "metadata" / "episodes.parquet"
    metadata = pd.read_parquet(metadata_path) if metadata_path.exists() else pd.DataFrame()
    season_by_uid: dict[str, str] = {}
    if not metadata.empty and "episode_uid" in metadata and season_column in metadata:
        season_by_uid = {
            str(row["episode_uid"]): str(row[season_column])
            for _index, row in metadata[["episode_uid", season_column]].iterrows()
        }
    return [
        {
            "episode_uid": str(uid),
            "season": season_by_uid.get(str(uid), _season(str(uid))),
            "num_rows": int(len(group)),
            "estimated_bytes": int(len(group)) * row_bytes,
        }
        for uid, group in rows.groupby("episode_uid", sort=False)
    ]


def _plan_shards(infos: list[dict[str, Any]], *, target_bytes: int, max_episodes: int | None, seed: int):
    by_season: dict[str, list[dict[str, Any]]] = {}
    for info in infos:
        by_season.setdefault(info["season"], []).append(info)
    rng = np.random.default_rng(seed)
    seasons = sorted(by_season)
    rng.shuffle(seasons)
    for values in by_season.values():
        rng.shuffle(values)
    result: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    current_bytes = 0

    def close() -> None:
        nonlocal current, current_bytes
        if not current:
            return
        shard_id = len(result)
        result.append(
            {
                "shard_id": shard_id,
                "shard_name": f"shard_{shard_id:05d}",
                "episodes": tuple(str(item["episode_uid"]) for item in current),
                "seasons": tuple(str(item["season"]) for item in current),
                "num_rows": int(sum(item["num_rows"] for item in current)),
                "estimated_bytes": int(current_bytes),
            }
        )
        current, current_bytes = [], 0

    while any(by_season[season] for season in seasons):
        for season in seasons:
            if not by_season[season]:
                continue
            item = by_season[season].pop(0)
            if current and (
                current_bytes + item["estimated_bytes"] > target_bytes
                or (max_episodes is not None and len(current) >= max_episodes)
            ):
                close()
            current.append(item)
            current_bytes += item["estimated_bytes"]
            if current_bytes >= target_bytes or (max_episodes is not None and len(current) >= max_episodes):
                close()
    close()
    return result


def _source_rows_files(shard_root: Path, split: str, rows: pd.DataFrame, plan: list[dict[str, Any]]):
    source_root = shard_root / "_source_rows" / split
    source_root.mkdir(parents=True, exist_ok=True)
    by_episode = {str(uid): group.copy() for uid, group in rows.groupby("episode_uid", sort=False)}
    tasks = []
    for spec in plan:
        source = pd.concat([by_episode[uid] for uid in spec["episodes"]], ignore_index=True)
        path = source_root / f"{spec['shard_name']}.parquet"
        source.to_parquet(path, index=False)
        tasks.append({**spec, "source_rows_path": str(path)})
    return tasks


def _remove_temporary_source_rows(shard_root: Path) -> None:
    """Remove build-only worker inputs; completed shards are self-contained."""
    temporary_root = shard_root / "_source_rows"
    if temporary_root.exists():
        shutil.rmtree(temporary_root)


def _build_fingerprint(task: dict[str, Any], build: Any, settings: Any) -> str:
    return _phase3.fingerprint(
        {
            "format": _phase3.FORMAT_VERSION,
            "shard_id": task["shard_id"],
            "episodes": task["episodes"],
            "mixed_speed": dataclasses.asdict(build.mixed_speed),
            "image_size": build.image_size,
            "image_modalities": settings.image_modalities,
            "tactile": settings.load_tactile_images,
            "dims": [settings.state_dim, settings.action_dim, settings.tactile_dim],
        }
    )


def _phase3_row_bytes(settings: Any, build: Any) -> int:
    specs = _phase3.phase3_array_specs(settings, build.mixed_speed, num_rows=1, image_size=build.image_size)
    return int(
        sum(np.dtype(dtype).itemsize * int(np.prod(shape[1:], dtype=np.int64)) for dtype, shape in specs.values())
    )


def _marker(path: Path, payload: dict[str, Any]) -> None:
    _phase3.atomic_write_json(path, payload)


def _emit_progress(progress_queue: Any | None, **event: Any) -> None:
    """Best-effort worker-to-parent progress reporting.

    Shard writing must not fail merely because the parent renderer exits or a
    terminal cannot consume a cosmetic progress update.
    """
    if progress_queue is None:
        return
    try:
        progress_queue.put_nowait(event)
    except Exception:  # noqa: BLE001
        pass


def _write_episode(
    *,
    episode_rows: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    settings: Any,
    build: Any,
    progress_queue: Any | None = None,
    shard_id: int | None = None,
) -> None:
    episode_uid = str(episode_rows["episode_uid"].iloc[0])
    episode_root = Path(settings.dataset_root) / "episodes" / episode_uid
    arrays_root = episode_root / "arrays"
    state = np.load(arrays_root / "state_65d.npy", mmap_mode="r")
    actions = np.load(arrays_root / settings.action_filename, mmap_mode="r")
    tactile = np.load(arrays_root / settings.tactile_filename, mmap_mode="r")
    timestamps = _chunk._read_timestamps(arrays_root, state.shape[0])
    frame_index = _chunk._read_frame_index(arrays_root, state.shape[0])
    readers: dict[str, _chunk.SequentialVideoReader] = {}
    deform_reader = raw_reader = None
    update_every = max(1, int(build.progress_update_frames))
    pending_progress_frames = 0

    def report_frames(*, force: bool = False) -> None:
        nonlocal pending_progress_frames
        if pending_progress_frames and (force or pending_progress_frames >= update_every):
            _emit_progress(
                progress_queue,
                kind="frames",
                pid=os.getpid(),
                shard_id=shard_id,
                frames=int(pending_progress_frames),
            )
            pending_progress_frames = 0

    try:
        for image_key, relpath in settings.image_modalities.items():
            readers[image_key] = _chunk.SequentialVideoReader(episode_root / relpath)
        if settings.load_tactile_images:
            deform_reader = _chunk.SequentialVideoReader(episode_root / settings.tactile_deform_video)
            raw_path = episode_root / settings.tactile_raw_video
            raw_reader = _chunk.SequentialVideoReader(raw_path) if raw_path.exists() else None
        for row in episode_rows.itertuples(index=False):
            physical = int(row.local_physical_row)
            position = int(row.frame_position)
            arrays["state"][physical] = np.asarray(state[position], dtype=np.float32)
            arrays["tactile"][physical] = np.asarray(tactile[position], dtype=np.float32).reshape(-1)
            arrays["frame_position"][physical] = position
            arrays["frame_index"][physical] = int(frame_index[position])
            arrays["timestamp"][physical] = float(timestamps[position])
            arrays["sample_weight"][physical] = 1.0
            arrays["planner_available"][physical] = False
            for stride in build.mixed_speed.ordered_strides:
                chunk, mask = _dataset.extract_action_chunk(
                    actions,
                    position,
                    action_horizon=build.mixed_speed.action_horizon,
                    action_dim=settings.action_dim,
                    action_chunk_stride=int(stride),
                )
                if not bool(np.all(mask)):
                    raise ValueError(f"{episode_uid} frame {position} has an incomplete stride-{stride} horizon.")
                arrays[f"actions_stride_{stride}"][physical] = chunk
            for image_key, reader in readers.items():
                arrays[f"{_chunk.IMAGE_ARRAY_PREFIX}{image_key}"][physical] = _chunk.resize_with_pad_uint8(
                    reader.read(position), int(build.image_size), int(build.image_size)
                )
            if deform_reader is not None:
                deform = _chunk.split_tactile_grid(
                    deform_reader.read(position), settings.tactile_deform_grid, episode_uid, settings.tactile_image_size
                )
                arrays["tactile_deform_images"][physical] = deform
                arrays["tactile_deform_available"][physical] = True
                raw_available = raw_reader is not None
                raw = np.zeros_like(deform)
                if raw_reader is not None:
                    raw = _chunk.split_tactile_grid(
                        raw_reader.read(position), settings.tactile_raw_grid, episode_uid, settings.tactile_image_size
                    )
                # Phase 3 keeps raw tactile intact in physical storage. The
                # future shard loader applies deterministic dropout per
                # (episode, frame, stride, occurrence), not per physical row.
                arrays["tactile_raw_images"][physical] = raw
                arrays["tactile_raw_available"][physical] = raw_available
            pending_progress_frames += 1
            report_frames()
    finally:
        report_frames(force=True)
        for reader in readers.values():
            reader.close()
        if deform_reader is not None:
            deform_reader.close()
        if raw_reader is not None:
            raw_reader.close()


def _build_one(task: dict[str, Any]) -> dict[str, Any]:
    progress_queue = task.get("progress_queue")
    shard_id = int(task["shard_id"])
    total_frames = int(task["num_rows"])
    total_episodes = len(task["episodes"])
    _emit_progress(
        progress_queue,
        kind="start",
        pid=os.getpid(),
        shard_id=shard_id,
        shard_name=str(task["shard_name"]),
        total_frames=total_frames,
        total_episodes=total_episodes,
    )
    try:
        config, data = _phase3.phase3_config(task["config_name"])
        build = dataclasses.replace(data.shard_build, progress_update_frames=int(task["progress_update_frames"]))
        settings = _chunk.settings_from_data_factory(data, config.model)
        settings = dataclasses.replace(settings, dataset_backend="video", dataset_root=task["dataset_root"])
        shard_root = Path(task["shard_root"])
        split, name = str(task["split"]), str(task["shard_name"])
        final_dir = shard_root / split / name
        incomplete = shard_root / split / f"{name}.incomplete"
        final_marker = final_dir / _phase3.COMPLETE_MARKER
        if final_marker.exists() and not bool(task["overwrite"]):
            _emit_progress(
                progress_queue,
                kind="frames",
                pid=os.getpid(),
                shard_id=shard_id,
                frames=total_frames,
            )
            _emit_progress(
                progress_queue,
                kind="episodes",
                pid=os.getpid(),
                shard_id=shard_id,
                episodes=total_episodes,
            )
            _emit_progress(progress_queue, kind="complete", pid=os.getpid(), shard_id=shard_id)
            return {"status": "skipped", "shard_id": task["shard_id"], "shard_name": name}
        if bool(task["overwrite"]):
            # --overwrite is an explicit request to discard this exact shard,
            # including an old resumable attempt. Normal resumes never take
            # this branch.
            if final_dir.exists():
                shutil.rmtree(final_dir)
            if incomplete.exists():
                shutil.rmtree(incomplete)
        fingerprint = _build_fingerprint(task, build, settings)
        rows = _phase3.physical_rows(pd.read_parquet(task["source_rows_path"]), tuple(task["episodes"]))
        arrays_dir = incomplete / "arrays"
        state_path = incomplete / build.build_state_name
        if incomplete.exists():
            if not state_path.exists():
                raise RuntimeError(f"Incomplete shard has no build state: {incomplete}")
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("fingerprint") != fingerprint:
                raise RuntimeError(f"Refusing to resume incompatible incomplete shard: {incomplete}")
            existing = pd.read_parquet(incomplete / _phase3.ROWS_FILENAME)
            if len(existing) != len(rows) or not existing[["episode_uid", "frame_position"]].equals(rows[["episode_uid", "frame_position"]]):
                raise RuntimeError(f"Refusing to resume incompatible row layout: {incomplete}")
            arrays = _phase3.create_arrays(
                arrays_dir,
                _phase3.phase3_array_specs(settings, build.mixed_speed, num_rows=len(rows), image_size=build.image_size),
                mode="r+",
            )
        else:
            arrays_dir.mkdir(parents=True, exist_ok=True)
            rows.to_parquet(incomplete / _phase3.ROWS_FILENAME, index=False)
            arrays = _phase3.create_arrays(
                arrays_dir,
                _phase3.phase3_array_specs(settings, build.mixed_speed, num_rows=len(rows), image_size=build.image_size),
                mode="w+",
            )
            # These fields are not sourced from planner artifacts in Phase 3.
            # Explicit initialization makes the all-zero verifier meaningful.
            arrays["planner_available"][:] = False
            for key in (
                "planner_state_belief",
                "planner_progress_transition",
                "planner_uncertainty",
                "planner_history_latent",
            ):
                arrays[key][:] = 0.0
            _marker(state_path, {"fingerprint": fingerprint, "episodes": list(task["episodes"]), "complete_episodes": []})
        progress = incomplete / build.episode_progress_dir_name
        progress.mkdir(exist_ok=True)
        for episode_uid, group in rows.groupby("episode_uid", sort=False):
            done = progress / f"{episode_uid}{build.episode_complete_suffix}"
            if done.exists():
                marker = json.loads(done.read_text(encoding="utf-8"))
                if (
                    marker.get("fingerprint") != fingerprint
                    or str(marker.get("episode_uid")) != str(episode_uid)
                    or int(marker.get("rows", -1)) != int(len(group))
                ):
                    raise RuntimeError(f"Refusing an invalid episode completion marker: {done}")
                _emit_progress(
                    progress_queue,
                    kind="frames",
                    pid=os.getpid(),
                    shard_id=shard_id,
                    frames=int(len(group)),
                )
                _emit_progress(
                    progress_queue,
                    kind="episodes",
                    pid=os.getpid(),
                    shard_id=shard_id,
                    episodes=1,
                )
                continue
            _write_episode(
                episode_rows=group,
                arrays=arrays,
                settings=settings,
                build=build,
                progress_queue=progress_queue,
                shard_id=shard_id,
            )
            _phase3.flush_memmaps(arrays)
            _marker(done, {"episode_uid": str(episode_uid), "rows": int(len(group)), "fingerprint": fingerprint})
            state = json.loads(state_path.read_text(encoding="utf-8"))
            completed = list(state.get("complete_episodes", []))
            completed.append(str(episode_uid))
            state["complete_episodes"] = completed
            _marker(state_path, state)
            _emit_progress(
                progress_queue,
                kind="episodes",
                pid=os.getpid(),
                shard_id=shard_id,
                episodes=1,
            )
        plan, plan_metadata = _phase3.build_virtual_plan(rows, build.mixed_speed, shard_id=int(task["shard_id"]))
        plan_path = arrays_dir / build.virtual_sample_plan_name
        plan_array = _chunk.create_memmap(plan_path, dtype=np.uint32, shape=plan.shape)
        plan_array[:] = plan
        plan_array.flush()
        _phase3.close_memmap(plan_array)
        _phase3.flush_memmaps(arrays)
        for array in arrays.values():
            _phase3.close_memmap(array)
        metadata = {
            "format": _phase3.FORMAT_VERSION,
            "fingerprint": fingerprint,
            "split": split,
            "shard_id": int(task["shard_id"]),
            "shard_name": name,
            "num_rows": int(len(rows)),
            "episodes": list(task["episodes"]),
            "mixed_speed": dataclasses.asdict(build.mixed_speed),
            "virtual_plan": {"filename": f"arrays/{build.virtual_sample_plan_name}", **plan_metadata},
            "arrays": {
                key: {"filename": f"arrays/{key}.npy", "dtype": str(np.dtype(dtype)), "shape": list(shape)}
                for key, (dtype, shape) in _phase3.phase3_array_specs(
                    settings, build.mixed_speed, num_rows=len(rows), image_size=build.image_size
                ).items()
            },
        }
        _phase3.atomic_write_json(incomplete / _phase3.METADATA_FILENAME, metadata)
        (incomplete / _phase3.COMPLETE_MARKER).write_text("complete\n", encoding="utf-8")
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        incomplete.replace(final_dir)
        _emit_progress(progress_queue, kind="complete", pid=os.getpid(), shard_id=shard_id)
        return {"status": "built", "shard_id": task["shard_id"], "shard_name": name, "rows": len(rows)}
    except Exception as exc:  # noqa: BLE001
        _emit_progress(progress_queue, kind="failed", pid=os.getpid(), shard_id=shard_id)
        return {"status": "failed", "shard_id": task["shard_id"], "shard_name": task["shard_name"], "error": str(exc), "traceback": traceback.format_exc()}


def main() -> int:
    args = parse_args()
    _config, _data, settings, build, shard_root = _resolve(args)
    split_names = ("train", "val") if build.split == "all" else (build.split,)
    manifest_entries: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    for split in split_names:
        rows = _dataset.load_manifest_rows(settings, split).reset_index(drop=True)
        if rows.empty:
            continue
        row_bytes = _phase3_row_bytes(settings, build)
        infos = _episode_infos(rows, Path(settings.dataset_root), build.season_column, row_bytes)
        target_bytes = _chunk.parse_size_bytes(build.target_shard_bytes)
        assert target_bytes is not None
        plan = _plan_shards(infos, target_bytes=target_bytes, max_episodes=build.max_episodes_per_shard, seed=build.seed)
        split_tasks = _source_rows_files(shard_root, split, rows, plan)
        for task in split_tasks:
            manifest_entries.append(
                {
                    "split": split, "shard_id": task["shard_id"], "shard_name": task["shard_name"],
                    "relative_dir": f"{split}/{task['shard_name']}", "num_rows": task["num_rows"],
                    "episodes": list(task["episodes"]), "seasons": list(task["seasons"]), "complete": False,
                }
            )
            if task["shard_id"] >= args.start_shard and (args.max_shards is None or len(tasks) < args.max_shards):
                tasks.append(
                    {
                        **task,
                        "config_name": args.config_name,
                        "dataset_root": str(settings.dataset_root),
                        "shard_root": str(shard_root),
                        "split": split,
                        "overwrite": bool(build.overwrite),
                        "progress_update_frames": int(build.progress_update_frames),
                    }
                )
    root_metadata = {
        "format": _phase3.FORMAT_VERSION,
        "config_name": args.config_name,
        "mixed_speed": dataclasses.asdict(build.mixed_speed),
        "source_manifest_root": str(settings.manifest_root),
        "shards": manifest_entries,
    }
    shard_root.mkdir(parents=True, exist_ok=True)
    _phase3.atomic_write_json(shard_root / _phase3.SHARD_MANIFEST_FILENAME, root_metadata)
    print(f"Phase-3 plan: {len(manifest_entries)} shards, {len(tasks)} selected, split={build.split}")
    if args.plan_only:
        _remove_temporary_source_rows(shard_root)
        return 0
    workers = min(max(1, int(build.num_workers)), max(1, len(tasks)))
    context = multiprocessing.get_context("spawn")
    failures: list[dict[str, Any]] = []
    # The parent owns all terminal rendering. Each active process reports
    # small frame deltas through a manager queue, giving one stable live bar
    # per worker plus one overall completed-shards bar.
    with context.Manager() as manager:
        progress_queue = manager.Queue()
        total_episodes = sum(len(task["episodes"]) for task in tasks)
        overall_bar = tqdm(total=len(tasks), desc="Build Phase-3 shards", unit="shard", position=0, leave=True)
        episode_bar = tqdm(total=total_episodes, desc="Build Phase-3 episodes", unit="episode", position=1, leave=True)
        worker_bars = [
            tqdm(total=1, desc=f"Worker {slot + 1}: idle", unit="frame", position=slot + 2, leave=True)
            for slot in range(workers)
        ]
        slot_by_pid: dict[int, int] = {}
        idle_slots: list[int] = list(range(workers))
        worker_state: dict[int, dict[str, int]] = {}

        def render(event: dict[str, Any]) -> None:
            pid = int(event.get("pid", -1))
            kind = str(event.get("kind", ""))
            if kind == "start":
                slot = slot_by_pid.get(pid)
                if slot is None:
                    slot = idle_slots.pop(0) if idle_slots else len(slot_by_pid) % workers
                    slot_by_pid[pid] = slot
                total_frames = int(event["total_frames"])
                shard_total_episodes = int(event["total_episodes"])
                shard_id = int(event["shard_id"])
                bar = worker_bars[slot]
                bar.reset(total=total_frames)
                bar.set_description_str(f"Worker {slot + 1}: shard {shard_id:05d}")
                worker_state[pid] = {"episodes": 0, "total_episodes": shard_total_episodes}
                bar.set_postfix_str(f"episodes 0/{shard_total_episodes}", refresh=True)
                return
            slot = slot_by_pid.get(pid)
            if slot is None:
                return
            bar = worker_bars[slot]
            state = worker_state.get(pid)
            if kind == "frames":
                bar.update(int(event.get("frames", 0)))
            elif kind == "episodes" and state is not None:
                completed_episodes = int(event.get("episodes", 0))
                state["episodes"] += completed_episodes
                episode_bar.update(completed_episodes)
                episode_bar.set_postfix_str(f"completed {episode_bar.n}/{total_episodes}", refresh=True)
                bar.set_postfix_str(f"episodes {state['episodes']}/{state['total_episodes']}", refresh=True)
            elif kind == "complete":
                if state is not None:
                    bar.n = bar.total or bar.n
                    bar.set_postfix_str(
                        f"episodes {state['episodes']}/{state['total_episodes']} (complete)",
                        refresh=True,
                    )
            elif kind == "failed":
                if state is not None:
                    bar.set_postfix_str(
                        f"episodes {state['episodes']}/{state['total_episodes']} (failed)",
                        refresh=True,
                    )

        def drain_progress() -> None:
            while True:
                try:
                    event = progress_queue.get_nowait()
                except queue.Empty:
                    return
                render(event)

        try:
            with futures.ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
                futures_by_task = {
                    executor.submit(_build_one, {**task, "progress_queue": progress_queue}): task for task in tasks
                }
                pending = set(futures_by_task)
                completed = 0
                while pending:
                    drain_progress()
                    done, pending = futures.wait(pending, timeout=0.25, return_when=futures.FIRST_COMPLETED)
                    for future in done:
                        result = future.result()
                        completed += 1
                        overall_bar.update(1)
                        overall_bar.set_postfix_str(f"completed {completed}/{len(tasks)}", refresh=True)
                        tqdm.write(f"{result['status']}: {result['shard_name']}")
                        if result["status"] == "failed":
                            failures.append(result)
                drain_progress()
        finally:
            for bar in worker_bars:
                bar.close()
            episode_bar.close()
            overall_bar.close()
    manifest = json.loads((shard_root / _phase3.SHARD_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    for entry in manifest["shards"]:
        entry["complete"] = (shard_root / entry["relative_dir"] / _phase3.COMPLETE_MARKER).is_file()
    _phase3.atomic_write_json(shard_root / _phase3.SHARD_MANIFEST_FILENAME, manifest)
    # The incomplete-shard directories contain all resume state. These
    # temporary source-row copies are only process-pool inputs and must not
    # become part of the portable training package.
    _remove_temporary_source_rows(shard_root)
    if failures:
        for failure in failures:
            print(f"FAILED {failure['shard_name']}: {failure['error']}\n{failure['traceback']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
