from __future__ import annotations

import argparse
from concurrent import futures
import dataclasses
import json
import multiprocessing
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

import openpi.training.config as _config
import openpi.training.origami_comp_action_chunk_shards as _shards
import openpi.training.origami_vla_dataset as _origami_vla_dataset


_WORKER_PROGRESS_QUEUE: Any | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build uncompressed, memory-mappable Origami pi0.5 comp action-chunk shards. "
            "Each shard owns complete episodes only; episodes are never split across shards."
        )
    )
    parser.add_argument("--config-name", type=str, default="pi05_origami_comp_action_chunk")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--manifest-root", type=Path, default=None)
    parser.add_argument("--shard-root", type=Path, default=None)
    parser.add_argument("--split", choices=("train", "val", "all"), default=None)
    parser.add_argument("--target-shard-bytes", type=str, default=None)
    parser.add_argument("--target-num-shards", type=int, default=None)
    parser.add_argument("--max-episodes-per-shard", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--row-order", choices=("episode_sequential", "shuffled_index"), default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--start-shard", type=int, default=0)
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--progress-update-frames", type=int, default=None)
    parser.add_argument("--progress-max-active-bars", type=int, default=None)
    parser.add_argument("--progress-poll-seconds", type=float, default=None)
    parser.add_argument("--progress-leave-active-bars", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args()


def _resolve(args: argparse.Namespace) -> tuple[
    _config.TrainConfig,
    _origami_vla_dataset.OrigamiVlaSettings,
    _config.OrigamiCompActionChunkShardBuildConfig,
]:
    config = _config.get_config(args.config_name)
    if not isinstance(config.data, _config.OrigamiCompActionChunkDataConfig):
        raise TypeError(
            f"Config {args.config_name!r} uses {type(config.data).__name__}; "
            "expected OrigamiCompActionChunkDataConfig."
        )
    settings = _shards.settings_from_data_factory(config.data, config.model)
    build = config.data.shard_build

    shard_root = args.shard_root or (Path(build.shard_root) if build.shard_root else None) or (
        Path(config.data.shard_root) if config.data.shard_root else None
    )
    if shard_root is None:
        raise ValueError("Set data.shard_root/data.shard_build.shard_root or pass --shard-root.")
    if args.dataset_root is not None:
        settings = dataclasses.replace(settings, dataset_root=str(args.dataset_root))
    if args.manifest_root is not None:
        settings = dataclasses.replace(settings, manifest_root=str(args.manifest_root))
    settings = dataclasses.replace(
        settings,
        dataset_backend="video",
        shard_root=str(shard_root),
        shard_manifest_name=build.shard_manifest_name,
        shard_rows_name=build.rows_name,
        shard_complete_marker_name=build.complete_marker_name,
    )
    if args.split is not None:
        build = dataclasses.replace(build, split=args.split)
    if args.target_shard_bytes is not None:
        build = dataclasses.replace(build, target_shard_bytes=args.target_shard_bytes)
    if args.target_num_shards is not None:
        build = dataclasses.replace(build, target_num_shards=args.target_num_shards)
    if args.max_episodes_per_shard is not None:
        build = dataclasses.replace(build, max_episodes_per_shard=args.max_episodes_per_shard)
    if args.num_workers is not None:
        build = dataclasses.replace(build, num_workers=args.num_workers)
    if args.seed is not None:
        build = dataclasses.replace(build, seed=args.seed)
    if args.row_order is not None:
        build = dataclasses.replace(build, row_order=args.row_order)
    if args.image_size is not None:
        build = dataclasses.replace(build, image_size=args.image_size)
    if args.overwrite is not None:
        build = dataclasses.replace(build, overwrite=bool(args.overwrite))
    if args.skip_existing is not None:
        build = dataclasses.replace(build, skip_existing=bool(args.skip_existing))
    if args.max_shards is not None:
        build = dataclasses.replace(build, max_shards_per_run=args.max_shards)
    if args.progress_update_frames is not None:
        build = dataclasses.replace(build, progress_update_frames=args.progress_update_frames)
    if args.progress_max_active_bars is not None:
        build = dataclasses.replace(build, progress_max_active_bars=args.progress_max_active_bars)
    if args.progress_poll_seconds is not None:
        build = dataclasses.replace(build, progress_poll_seconds=args.progress_poll_seconds)
    if args.progress_leave_active_bars is not None:
        build = dataclasses.replace(build, progress_leave_active_bars=bool(args.progress_leave_active_bars))
    return config, settings, build


def _load_episode_table(dataset_root: Path, rows: pd.DataFrame, season_column: str) -> pd.DataFrame:
    metadata_path = dataset_root / "metadata" / "episodes.parquet"
    if metadata_path.exists():
        episodes = pd.read_parquet(metadata_path)
    else:
        episodes = pd.DataFrame({"episode_uid": sorted(rows["episode_uid"].astype(str).unique())})
    if "episode_uid" not in episodes.columns:
        raise KeyError(f"{metadata_path} is missing episode_uid")
    if season_column not in episodes.columns:
        episodes = episodes.copy()
        episodes[season_column] = episodes["episode_uid"].map(_infer_season)
    return episodes


def _check_manifest_metadata(
    settings: _origami_vla_dataset.OrigamiVlaSettings,
    *,
    split: str,
    rows: pd.DataFrame,
) -> None:
    manifest_path = Path(settings.manifest_root) / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest metadata was required but not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    counts = manifest.get("counts", {})
    expected_key = f"{split}_rows"
    expected_rows = counts.get(expected_key)
    if expected_rows is not None and int(expected_rows) != int(len(rows)):
        raise ValueError(
            f"Manifest metadata {expected_key}={expected_rows} does not match "
            f"filtered {split} parquet rows={len(rows)}."
        )
    split_info = manifest.get("split", {})
    episode_key = f"{split}_episode_uids"
    expected_episode_uids = split_info.get(episode_key)
    if expected_episode_uids is not None:
        actual_episode_uids = sorted(str(uid) for uid in rows["episode_uid"].drop_duplicates().tolist())
        if sorted(str(uid) for uid in expected_episode_uids) != actual_episode_uids:
            raise ValueError(f"Manifest metadata {episode_key} does not match {split} parquet episode_uids.")


def _infer_season(episode_uid: str) -> str:
    marker = "__episode_"
    if marker in episode_uid:
        return episode_uid.split(marker, 1)[0]
    marker = "_episode_"
    if marker in episode_uid:
        return episode_uid.split(marker, 1)[0]
    return episode_uid


def _make_episode_infos(
    *,
    rows: pd.DataFrame,
    episode_table: pd.DataFrame,
    season_column: str,
    row_bytes: int,
) -> list[dict[str, Any]]:
    row_counts = rows.groupby("episode_uid", sort=False).size().astype("int64")
    metadata = episode_table.set_index("episode_uid", drop=False)
    infos: list[dict[str, Any]] = []
    for episode_uid, row_count in row_counts.items():
        episode_uid = str(episode_uid)
        if episode_uid in metadata.index:
            season = str(metadata.loc[episode_uid].get(season_column, _infer_season(episode_uid)))
        else:
            season = _infer_season(episode_uid)
        infos.append(
            {
                "episode_uid": episode_uid,
                "source_season": season,
                "num_rows": int(row_count),
                "estimated_bytes": int(row_count) * int(row_bytes),
            }
        )
    return infos


def _target_bytes(build: _config.OrigamiCompActionChunkShardBuildConfig, episode_infos: list[dict[str, Any]]) -> int:
    total_bytes = int(sum(int(info["estimated_bytes"]) for info in episode_infos))
    if build.target_num_shards is not None:
        if build.target_num_shards <= 0:
            raise ValueError("target_num_shards must be positive when set.")
        return max(1, int(np.ceil(total_bytes / int(build.target_num_shards))))
    parsed = _shards.parse_size_bytes(build.target_shard_bytes)
    if parsed is None or parsed <= 0:
        raise ValueError(f"Invalid target_shard_bytes: {build.target_shard_bytes!r}")
    return int(parsed)


def _build_plan(
    *,
    episode_infos: list[dict[str, Any]],
    target_bytes: int,
    max_episodes_per_shard: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    by_season: dict[str, list[dict[str, Any]]] = {}
    for info in episode_infos:
        by_season.setdefault(str(info["source_season"]), []).append(info)
    season_names = sorted(by_season)
    rng.shuffle(season_names)
    for episodes in by_season.values():
        rng.shuffle(episodes)

    shards: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    current_bytes = 0

    def close_current() -> None:
        nonlocal current, current_bytes
        if not current:
            return
        shard_id = len(shards)
        shard_name = f"shard_{shard_id:05d}"
        shards.append(
            {
                "shard_id": shard_id,
                "shard_name": shard_name,
                "episodes": tuple(str(item["episode_uid"]) for item in current),
                "seasons": tuple(str(item["source_season"]) for item in current),
                "num_rows": int(sum(int(item["num_rows"]) for item in current)),
                "estimated_bytes": int(current_bytes),
            }
        )
        current = []
        current_bytes = 0

    while any(by_season[name] for name in season_names):
        progressed = False
        for season in season_names:
            if not by_season[season]:
                continue
            episode = by_season[season].pop(0)
            episode_bytes = int(episode["estimated_bytes"])
            if current:
                too_many_episodes = max_episodes_per_shard is not None and len(current) >= int(max_episodes_per_shard)
                too_many_bytes = current_bytes + episode_bytes > int(target_bytes)
                if too_many_episodes or too_many_bytes:
                    close_current()
            current.append(episode)
            current_bytes += episode_bytes
            progressed = True
            if max_episodes_per_shard is not None and len(current) >= int(max_episodes_per_shard):
                close_current()
            elif current_bytes >= int(target_bytes):
                close_current()
        if not progressed:
            break
    close_current()
    return shards


def _write_plan_files(
    *,
    shard_root: Path,
    split: str,
    rows: pd.DataFrame,
    shards: list[dict[str, Any]],
    settings: _origami_vla_dataset.OrigamiVlaSettings,
    build: _config.OrigamiCompActionChunkShardBuildConfig,
    row_bytes: int,
    target_bytes: int,
) -> list[dict[str, Any]]:
    shard_root.mkdir(parents=True, exist_ok=True)
    plan_rows: list[dict[str, Any]] = []
    per_shard_rows_root = shard_root / "_source_rows" / split
    per_shard_rows_root.mkdir(parents=True, exist_ok=True)
    rows_by_episode = {str(uid): frame.copy() for uid, frame in rows.groupby("episode_uid", sort=False)}
    for shard in shards:
        shard_name = str(shard["shard_name"])
        episode_parts = [rows_by_episode[episode_uid] for episode_uid in shard["episodes"]]
        shard_rows = pd.concat(episode_parts, axis=0, ignore_index=True)
        source_rows_path = per_shard_rows_root / f"{shard_name}.parquet"
        shard_rows.to_parquet(source_rows_path, index=False)
        relative_dir = f"{split}/{shard_name}"
        for episode_order, episode_uid in enumerate(shard["episodes"]):
            episode_rows = int(len(rows_by_episode[episode_uid]))
            plan_rows.append(
                {
                    "split": split,
                    "shard_id": int(shard["shard_id"]),
                    "shard_name": shard_name,
                    "relative_dir": relative_dir,
                    "episode_order": int(episode_order),
                    "episode_uid": episode_uid,
                    "source_season": shard["seasons"][episode_order],
                    "episode_rows": episode_rows,
                    "episode_estimated_bytes": episode_rows * row_bytes,
                    "shard_num_rows": int(shard["num_rows"]),
                    "shard_estimated_bytes": int(shard["estimated_bytes"]),
                    "source_rows_path": str(source_rows_path),
                }
            )
    plan_frame = pd.DataFrame(plan_rows)
    plan_path = shard_root / build.shard_plan_name
    if plan_path.exists():
        existing_plan = pd.read_parquet(plan_path)
        if "split" in existing_plan.columns:
            existing_plan = existing_plan[existing_plan["split"].astype(str) != split]
        plan_frame = pd.concat([existing_plan, plan_frame], axis=0, ignore_index=True)
        plan_frame = plan_frame.sort_values(["split", "shard_id", "episode_order"]).reset_index(drop=True)
    plan_frame.to_parquet(plan_path, index=False)
    manifest_path = shard_root / build.shard_manifest_name
    existing_shards: list[dict[str, Any]] = []
    if manifest_path.exists():
        try:
            existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            existing_shards = [
                item for item in existing_manifest.get("shards", []) if str(item.get("split")) != split
            ]
        except json.JSONDecodeError:
            existing_shards = []
    current_shards = [
        {
            "split": split,
            "shard_id": int(shard["shard_id"]),
            "shard_name": str(shard["shard_name"]),
            "relative_dir": f"{split}/{shard['shard_name']}",
            "num_rows": int(shard["num_rows"]),
            "estimated_bytes": int(shard["estimated_bytes"]),
            "episodes": list(shard["episodes"]),
            "rows_name": build.rows_name,
            "metadata_name": build.metadata_name,
            "complete_marker_name": build.complete_marker_name,
            "complete": bool((shard_root / split / str(shard["shard_name"]) / build.complete_marker_name).exists()),
        }
        for shard in shards
    ]
    manifest = {
        "format": "origami_comp_action_chunk_uncompressed_shards_v1",
        "dataset_root": settings.dataset_root,
        "manifest_root": settings.manifest_root,
        "split": split,
        "row_bytes_estimate": int(row_bytes),
        "target_shard_bytes": int(target_bytes),
        "settings": {
            "action_horizon": int(settings.action_horizon),
            "action_chunk_stride": int(settings.action_chunk_stride),
            "state_dim": int(settings.state_dim),
            "action_dim": int(settings.action_dim),
            "tactile_dim": int(settings.tactile_dim),
            "image_size": int(build.image_size),
            "tactile_image_size": int(settings.tactile_image_size),
            "load_tactile_images": bool(settings.load_tactile_images),
            "include_planner_features": bool(settings.include_planner_features),
            "planner_belief_dim": int(settings.planner_belief_dim),
            "planner_progress_dim": int(settings.planner_progress_dim),
            "planner_uncertainty_dim": int(settings.planner_uncertainty_dim),
            "planner_history_dim": int(settings.planner_history_dim),
            "row_order": build.row_order,
            "rows_name": build.rows_name,
            "metadata_name": build.metadata_name,
            "complete_marker_name": build.complete_marker_name,
        },
        "shards": sorted(
            existing_shards + current_shards,
            key=lambda item: (str(item.get("split")), int(item.get("shard_id", 0))),
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return plan_rows


def _assert_child_path(path: Path, root: Path) -> None:
    path = path.resolve()
    root = root.resolve()
    if path == root or root not in path.parents:
        raise ValueError(f"Refusing to remove path outside shard root: {path}")


def _load_npz(path: Path) -> Any:
    return np.load(path, allow_pickle=False)


def _array_specs(
    settings: _origami_vla_dataset.OrigamiVlaSettings,
    *,
    num_rows: int,
    image_size: int,
) -> dict[str, tuple[np.dtype | type, tuple[int, ...]]]:
    specs: dict[str, tuple[np.dtype | type, tuple[int, ...]]] = {
        "state": (np.float32, (num_rows, settings.state_dim)),
        "tactile": (np.float32, (num_rows, settings.tactile_dim)),
        "actions": (np.float32, (num_rows, settings.action_horizon, settings.action_dim)),
        "action_mask": (np.bool_, (num_rows, settings.action_horizon, settings.action_dim)),
        "sample_weight": (np.float32, (num_rows,)),
        "frame_position": (np.int64, (num_rows,)),
        "frame_index": (np.int64, (num_rows,)),
        "timestamp": (np.float32, (num_rows,)),
    }
    for image_key in settings.image_modalities:
        specs[f"{_shards.IMAGE_ARRAY_PREFIX}{image_key}"] = (np.uint8, (num_rows, image_size, image_size, 3))
    if settings.load_tactile_images:
        fingers = int(settings.tactile_deform_grid["rows"]) * int(settings.tactile_deform_grid["cols"])
        specs["tactile_deform_images"] = (
            np.uint8,
            (num_rows, fingers, 3, settings.tactile_image_size, settings.tactile_image_size),
        )
        specs["tactile_raw_images"] = (
            np.uint8,
            (num_rows, fingers, 3, settings.tactile_image_size, settings.tactile_image_size),
        )
        specs["tactile_raw_available"] = (np.bool_, (num_rows,))
    if settings.include_planner_features:
        specs["planner_available"] = (np.bool_, (num_rows,))
        specs["planner_state_belief"] = (np.float32, (num_rows, settings.planner_belief_dim))
        specs["planner_progress_transition"] = (np.float32, (num_rows, settings.planner_progress_dim))
        specs["planner_uncertainty"] = (np.float32, (num_rows, settings.planner_uncertainty_dim))
        specs["planner_history_latent"] = (np.float32, (num_rows, settings.planner_history_dim))
    return specs


def _planner_zero_values(settings: _origami_vla_dataset.OrigamiVlaSettings) -> dict[str, np.ndarray]:
    return {
        "planner_state_belief": np.zeros((settings.planner_belief_dim,), dtype=np.float32),
        "planner_progress_transition": np.zeros((settings.planner_progress_dim,), dtype=np.float32),
        "planner_uncertainty": np.zeros((settings.planner_uncertainty_dim,), dtype=np.float32),
        "planner_history_latent": np.zeros((settings.planner_history_dim,), dtype=np.float32),
    }


def _progress_put(progress_queue: Any | None, event: str, **payload: Any) -> None:
    if progress_queue is None:
        return
    payload["event"] = event
    try:
        progress_queue.put(payload)
    except Exception:
        pass


def _init_worker(progress_queue: Any | None) -> None:
    global _WORKER_PROGRESS_QUEUE
    _WORKER_PROGRESS_QUEUE = progress_queue


def _build_one_shard(task: dict[str, Any]) -> dict[str, Any]:
    settings: _origami_vla_dataset.OrigamiVlaSettings = task["settings"]
    build: _config.OrigamiCompActionChunkShardBuildConfig = task["build"]
    progress_queue = task.get("progress_queue", _WORKER_PROGRESS_QUEUE)
    shard_root = Path(task["shard_root"])
    source_rows_path = Path(task["source_rows_path"])
    shard_id = int(task["shard_id"])
    shard_name = str(task["shard_name"])
    split = str(task["split"])
    final_dir = shard_root / split / shard_name
    tmp_dir = shard_root / split / f"{shard_name}.incomplete"
    marker_path = final_dir / build.complete_marker_name
    if build.skip_existing and marker_path.exists() and not build.overwrite:
        metadata_path = final_dir / build.metadata_name
        num_rows = None
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                num_rows = metadata.get("num_rows")
                episode_count = len(metadata.get("episodes", []))
            except Exception:
                num_rows = None
                episode_count = int(task.get("episode_count", 0))
        else:
            episode_count = int(task.get("episode_count", 0))
        _progress_put(
            progress_queue,
            "shard_skip",
            shard_id=shard_id,
            shard_name=shard_name,
            split=split,
            rows=int(num_rows or 0),
            episodes=int(episode_count),
        )
        return {
            "shard_id": shard_id,
            "shard_name": shard_name,
            "status": "skipped",
            "num_rows": num_rows,
            "episodes": episode_count,
        }
    try:
        if tmp_dir.exists():
            _assert_child_path(tmp_dir, shard_root)
            shutil.rmtree(tmp_dir)
        if final_dir.exists():
            if not build.overwrite:
                raise FileExistsError(f"Shard already exists and overwrite is false: {final_dir}")
            _assert_child_path(final_dir, shard_root)
            shutil.rmtree(final_dir)
        arrays_dir = tmp_dir / "arrays"
        arrays_dir.mkdir(parents=True, exist_ok=True)

        rows = pd.read_parquet(source_rows_path)
        episode_order = tuple(str(uid) for uid in rows["episode_uid"].drop_duplicates().tolist())
        rows = _shards._episode_rows_for_physical_write(rows, episode_order)
        rows = rows.reset_index(drop=True)
        rows["local_physical_row"] = np.arange(len(rows), dtype=np.int64)
        num_rows = int(len(rows))
        _progress_put(
            progress_queue,
            "shard_start",
            shard_id=shard_id,
            shard_name=shard_name,
            split=split,
            rows=num_rows,
            episodes=len(episode_order),
        )
        arrays = {
            name: _shards.create_memmap(
                arrays_dir / f"{name}.npy",
                dtype=dtype,
                shape=shape,
            )
            for name, (dtype, shape) in _array_specs(
                settings,
                num_rows=num_rows,
                image_size=int(build.image_size),
            ).items()
        }

        dataset_root = Path(settings.dataset_root)
        planner_archives: dict[tuple[str, str], Any] = {}
        frame_counter = 0
        planner_zeros = _planner_zero_values(settings)
        progress_update_frames = max(1, int(build.progress_update_frames))
        for episode_uid, episode_rows in rows.groupby("episode_uid", sort=False):
            episode_uid = str(episode_uid)
            episode_total = int(len(episode_rows))
            episode_counter = 0
            pending_progress = 0
            _progress_put(
                progress_queue,
                "episode_start",
                shard_id=shard_id,
                shard_name=shard_name,
                split=split,
                episode_uid=episode_uid,
                rows=episode_total,
            )
            episode_root = dataset_root / "episodes" / episode_uid
            arrays_root = episode_root / "arrays"
            state = np.load(arrays_root / "state_65d.npy", mmap_mode="r")
            actions_source = np.load(arrays_root / settings.action_filename, mmap_mode="r")
            tactile = np.load(arrays_root / settings.tactile_filename, mmap_mode="r")
            timestamps = _shards._read_timestamps(arrays_root, state.shape[0])
            frame_index = _shards._read_frame_index(arrays_root, state.shape[0])
            video_readers: dict[str, _shards.SequentialVideoReader] = {}
            tactile_deform_reader: _shards.SequentialVideoReader | None = None
            tactile_raw_reader: _shards.SequentialVideoReader | None = None
            try:
                for image_key, relpath in settings.image_modalities.items():
                    video_path = episode_root / relpath
                    if not video_path.exists():
                        raise FileNotFoundError(f"Missing video for {episode_uid}:{image_key}: {video_path}")
                    video_readers[image_key] = _shards.SequentialVideoReader(video_path)
                if settings.load_tactile_images:
                    deform_path = episode_root / settings.tactile_deform_video
                    raw_path = episode_root / settings.tactile_raw_video
                    if not deform_path.exists():
                        raise FileNotFoundError(f"Missing tactile deform video for {episode_uid}: {deform_path}")
                    if settings.tactile_require_raw_video and not raw_path.exists():
                        raise FileNotFoundError(f"Missing required tactile raw video for {episode_uid}: {raw_path}")
                    tactile_deform_reader = _shards.SequentialVideoReader(deform_path)
                    tactile_raw_reader = _shards.SequentialVideoReader(raw_path) if raw_path.exists() else None

                for row in episode_rows.to_dict(orient="records"):
                    local_row = int(row["local_physical_row"])
                    frame_position = int(row["frame_position"])
                    arrays["state"][local_row] = np.asarray(state[frame_position], dtype=np.float32)
                    arrays["tactile"][local_row] = np.asarray(tactile[frame_position], dtype=np.float32).reshape(-1)
                    chunk, mask = _origami_vla_dataset.extract_action_chunk(
                        actions_source,
                        frame_position,
                        action_horizon=settings.action_horizon,
                        action_dim=settings.action_dim,
                        action_chunk_stride=settings.action_chunk_stride,
                    )
                    arrays["actions"][local_row] = chunk
                    arrays["action_mask"][local_row] = mask
                    sample_weight = row.get(settings.sample_weight_column, 1.0)
                    if pd.isna(sample_weight):
                        sample_weight = 1.0
                    arrays["sample_weight"][local_row] = np.asarray(float(sample_weight), dtype=np.float32)
                    arrays["frame_position"][local_row] = np.asarray(frame_position, dtype=np.int64)
                    arrays["frame_index"][local_row] = np.asarray(int(frame_index[frame_position]), dtype=np.int64)
                    arrays["timestamp"][local_row] = np.asarray(float(timestamps[frame_position]), dtype=np.float32)

                    for image_key, reader in video_readers.items():
                        frame = reader.read(frame_position)
                        arrays[f"{_shards.IMAGE_ARRAY_PREFIX}{image_key}"][local_row] = _shards.resize_with_pad_uint8(
                            frame,
                            int(build.image_size),
                            int(build.image_size),
                        )

                    if settings.load_tactile_images:
                        assert tactile_deform_reader is not None
                        deform_frame = tactile_deform_reader.read(frame_position)
                        deform_images = _shards.split_tactile_grid(
                            deform_frame,
                            settings.tactile_deform_grid,
                            episode_uid,
                            int(settings.tactile_image_size),
                        )
                        arrays["tactile_deform_images"][local_row] = deform_images
                        raw_available = tactile_raw_reader is not None
                        if raw_available:
                            try:
                                raw_frame = tactile_raw_reader.read(frame_position)
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
                        arrays["tactile_raw_images"][local_row] = raw_images
                        arrays["tactile_raw_available"][local_row] = np.asarray(raw_available, dtype=bool)

                    if settings.include_planner_features:
                        planner_enabled = _shards._row_bool(row.get("planner_enabled", True), default=True)
                        arrays["planner_available"][local_row] = np.asarray(planner_enabled, dtype=bool)
                        if not planner_enabled:
                            for key, value in planner_zeros.items():
                                arrays[key][local_row] = value
                        else:
                            view_mode = str(row["view_mode"])
                            branch_value = row.get("planner_branch", settings.planner_branch)
                            if pd.isna(branch_value) or not str(branch_value):
                                branch_value = settings.planner_branch
                            branch = str(branch_value)
                            variant_value = row.get("planner_value_variant", settings.planner_value_variant)
                            if pd.isna(variant_value) or not str(variant_value):
                                variant_value = settings.planner_value_variant
                            value_variant = str(variant_value)
                            planner_output_dir = Path(str(row["planner_output_dir"]))
                            planner_key = (view_mode, str(planner_output_dir))
                            planner = planner_archives.get(planner_key)
                            if planner is None:
                                planner = _load_npz(planner_output_dir / settings.planner_arrays_filename)
                                planner_archives[planner_key] = planner
                            planner_row_index = int(row["planner_row_index"])
                            arrays["planner_state_belief"][local_row] = np.asarray(
                                _shards._read_planner_feature(
                                    planner,
                                    _shards._planner_state_belief_key(value_variant),
                                    branch,
                                    planner_row_index,
                                ),
                                dtype=np.float32,
                            )
                            arrays["planner_progress_transition"][local_row] = np.asarray(
                                _shards._read_planner_feature(
                                    planner, "progress_transition", branch, planner_row_index
                                ),
                                dtype=np.float32,
                            )
                            arrays["planner_uncertainty"][local_row] = np.asarray(
                                _shards._read_planner_feature(
                                    planner, "uncertainty_features", branch, planner_row_index
                                ),
                                dtype=np.float32,
                            )
                            arrays["planner_history_latent"][local_row] = np.asarray(
                                _shards._read_planner_feature(
                                    planner, "temporal_latent", branch, planner_row_index
                                ),
                                dtype=np.float32,
                            )
                    frame_counter += 1
                    episode_counter += 1
                    pending_progress += 1
                    if pending_progress >= progress_update_frames:
                        _progress_put(
                            progress_queue,
                            "frames",
                            shard_id=shard_id,
                            shard_name=shard_name,
                            split=split,
                            episode_uid=episode_uid,
                            delta=pending_progress,
                            episode_delta=pending_progress,
                            episode_done=episode_counter,
                            episode_total=episode_total,
                            frame_position=frame_position,
                        )
                        pending_progress = 0
            finally:
                for reader in video_readers.values():
                    reader.close()
                if tactile_deform_reader is not None:
                    tactile_deform_reader.close()
                if tactile_raw_reader is not None:
                    tactile_raw_reader.close()
            if pending_progress:
                _progress_put(
                    progress_queue,
                    "frames",
                    shard_id=shard_id,
                    shard_name=shard_name,
                    split=split,
                    episode_uid=episode_uid,
                    delta=pending_progress,
                    episode_delta=pending_progress,
                    episode_done=episode_counter,
                    episode_total=episode_total,
                    frame_position=int(episode_rows["frame_position"].iloc[-1]),
                )
            _progress_put(
                progress_queue,
                "episode_done",
                shard_id=shard_id,
                shard_name=shard_name,
                split=split,
                episode_uid=episode_uid,
                rows=episode_total,
            )

        for archive in planner_archives.values():
            close = getattr(archive, "close", None)
            if close is not None:
                close()

        row_order = _shards.make_shard_row_order(
            num_rows,
            seed=int(build.seed),
            shard_id=shard_id,
            mode=build.row_order,
        )
        if row_order is not None:
            row_order_array = _shards.create_memmap(
                arrays_dir / "row_order.npy",
                dtype=np.int64,
                shape=row_order.shape,
            )
            row_order_array[:] = row_order
            row_order_array.flush()
            _shards._close_memmap(row_order_array)

        for array in arrays.values():
            flush = getattr(array, "flush", None)
            if flush is not None:
                flush()
            _shards._close_memmap(array)

        rows.to_parquet(tmp_dir / build.rows_name, index=False)
        metadata = {
            "format": "origami_comp_action_chunk_uncompressed_shard_v1",
            "split": split,
            "shard_id": shard_id,
            "shard_name": shard_name,
            "num_rows": num_rows,
            "episodes": episode_order,
            "row_order": build.row_order,
            "arrays": {
                name: {
                    "filename": f"arrays/{name}.npy",
                    "dtype": str(np.dtype(dtype)),
                    "shape": list(shape),
                }
                for name, (dtype, shape) in _array_specs(
                    settings,
                    num_rows=num_rows,
                    image_size=int(build.image_size),
                ).items()
            },
        }
        (tmp_dir / build.metadata_name).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        (tmp_dir / build.complete_marker_name).write_text("complete\n", encoding="utf-8")
        tmp_dir.rename(final_dir)
        _progress_put(
            progress_queue,
            "shard_done",
            shard_id=shard_id,
            shard_name=shard_name,
            split=split,
            rows=num_rows,
            frames=frame_counter,
            episodes=len(episode_order),
        )
        return {
            "shard_id": shard_id,
            "shard_name": shard_name,
            "status": "built",
            "num_rows": num_rows,
            "frames": frame_counter,
        }
    except Exception as exc:  # noqa: BLE001
        _progress_put(
            progress_queue,
            "shard_failed",
            shard_id=shard_id,
            shard_name=shard_name,
            split=split,
            error=str(exc),
        )
        return {
            "shard_id": shard_id,
            "shard_name": shard_name,
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def _update_manifest_completion(shard_root: Path, build: _config.OrigamiCompActionChunkShardBuildConfig) -> None:
    manifest_path = shard_root / build.shard_manifest_name
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for shard in manifest.get("shards", []):
        shard_dir = shard_root / str(shard["relative_dir"])
        shard["complete"] = bool((shard_dir / build.complete_marker_name).exists())
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _short_episode_uid(episode_uid: str, *, max_len: int = 42) -> str:
    episode_uid = str(episode_uid)
    if len(episode_uid) <= max_len:
        return episode_uid
    keep = max_len - 3
    left = keep // 2
    right = keep - left
    return f"{episode_uid[:left]}...{episode_uid[-right:]}"


def _run_shard_tasks_with_progress(
    *,
    tasks: list[dict[str, Any]],
    worker_count: int,
    build: _config.OrigamiCompActionChunkShardBuildConfig,
    split: str,
) -> list[dict[str, Any]]:
    if not tasks:
        return []

    total_rows = int(sum(int(task.get("shard_num_rows", 0)) for task in tasks))
    total_episodes = int(sum(int(task.get("episode_count", 0)) for task in tasks))
    total_shards = len(tasks)
    max_active_bars = max(0, min(int(build.progress_max_active_bars), int(worker_count)))
    poll_seconds = max(0.05, float(build.progress_poll_seconds))
    leave_active = bool(build.progress_leave_active_bars)

    context = multiprocessing.get_context("spawn")
    use_process_pool = True
    try:
        progress_queue = context.Queue()
    except (OSError, PermissionError):
        use_process_pool = False
        progress_queue = queue.Queue()
    active: dict[int, dict[str, Any]] = {}
    completed_shards: set[int] = set()
    completed_episodes: set[tuple[int, str]] = set()
    free_slots = list(range(max_active_bars))

    def close_active(shard_id: int) -> None:
        state = active.pop(shard_id, None)
        if state is None:
            return
        episode_bar = state.get("episode_bar")
        if episode_bar is not None:
            episode_bar.close()
        shard_bar = state.get("shard_bar")
        if shard_bar is not None:
            shard_bar.close()
        slot = state.get("slot")
        if slot is not None:
            free_slots.append(int(slot))
            free_slots.sort()

    def handle_event(
        event: dict[str, Any],
        *,
        dataset_bar: tqdm,
        episode_bar_total: tqdm,
        shard_bar_total: tqdm,
    ) -> None:
        event_type = str(event.get("event", ""))
        shard_id = int(event.get("shard_id", -1))
        shard_name = str(event.get("shard_name", f"shard_{shard_id:05d}"))

        if event_type == "shard_start":
            if shard_id in active:
                return
            slot = free_slots.pop(0) if free_slots else None
            shard_bar = None
            if slot is not None:
                shard_bar = tqdm(
                    total=int(event.get("rows", 0)),
                    desc=f"{split} {shard_name}",
                    unit="frame",
                    position=3 + int(slot) * 2,
                    leave=leave_active,
                    dynamic_ncols=True,
                )
            active[shard_id] = {
                "slot": slot,
                "shard_bar": shard_bar,
                "episode_bar": None,
                "rows": int(event.get("rows", 0)),
                "seen": 0,
                "episode_seen": 0,
                "episode_total": 0,
            }
            return

        if event_type == "episode_start":
            state = active.get(shard_id)
            if state is None:
                return
            old_episode_bar = state.get("episode_bar")
            if old_episode_bar is not None:
                old_episode_bar.close()
            state["episode_seen"] = 0
            state["episode_total"] = int(event.get("rows", 0))
            episode_uid = str(event.get("episode_uid", ""))
            state["episode_uid"] = episode_uid
            state["episode_bar"] = None
            slot = state.get("slot")
            if slot is not None:
                state["episode_bar"] = tqdm(
                    total=int(event.get("rows", 0)),
                    desc=f"{split} ep {_short_episode_uid(episode_uid)}",
                    unit="frame",
                    position=4 + int(slot) * 2,
                    leave=leave_active,
                    dynamic_ncols=True,
                )
            return

        if event_type == "frames":
            delta = int(event.get("delta", 0))
            episode_delta = int(event.get("episode_delta", delta))
            if delta > 0:
                dataset_bar.update(delta)
            state = active.get(shard_id)
            if state is not None:
                state["seen"] = int(state.get("seen", 0)) + delta
                state["episode_seen"] = int(event.get("episode_done", state.get("episode_seen", 0)))
                state["episode_total"] = int(event.get("episode_total", state.get("episode_total", 0)))
                shard_bar = state.get("shard_bar")
                if shard_bar is not None and delta > 0:
                    shard_bar.update(delta)
                    shard_bar.set_postfix_str(
                        f"episode={_short_episode_uid(str(event.get('episode_uid', '')))} "
                        f"ep={state['episode_seen']}/{state['episode_total']}"
                    )
                current_episode_bar = state.get("episode_bar")
                if current_episode_bar is not None and episode_delta > 0:
                    current_episode_bar.update(episode_delta)
            return

        if event_type == "episode_done":
            episode_uid = str(event.get("episode_uid", ""))
            key = (shard_id, episode_uid)
            if key not in completed_episodes:
                episode_bar_total.update(1)
                completed_episodes.add(key)
            state = active.get(shard_id)
            if state is not None:
                current_episode_bar = state.get("episode_bar")
                if current_episode_bar is not None:
                    remainder = int(current_episode_bar.total or 0) - int(current_episode_bar.n)
                    if remainder > 0:
                        current_episode_bar.update(remainder)
                    current_episode_bar.close()
                    state["episode_bar"] = None
            return

        if event_type == "shard_skip":
            if shard_id not in completed_shards:
                rows = int(event.get("rows", 0))
                episodes = int(event.get("episodes", 0))
                if rows > 0:
                    dataset_bar.update(rows)
                if episodes > 0:
                    episode_bar_total.update(episodes)
                shard_bar_total.update(1)
                completed_shards.add(shard_id)
            tqdm.write(f"skipped {shard_name} rows={event.get('rows')}")
            return

        if event_type in {"shard_done", "shard_failed"}:
            if shard_id not in completed_shards:
                state = active.get(shard_id)
                if event_type == "shard_done":
                    rows = int(event.get("rows", 0))
                    seen = int(state.get("seen", 0)) if state is not None else 0
                    remainder = max(0, rows - seen)
                    if remainder:
                        dataset_bar.update(remainder)
                        shard_bar = state.get("shard_bar") if state is not None else None
                        if shard_bar is not None:
                            shard_bar.update(remainder)
                shard_bar_total.update(1)
                completed_shards.add(shard_id)
            close_active(shard_id)
            if event_type == "shard_done":
                tqdm.write(
                    f"built {shard_name} rows={event.get('rows')} "
                    f"frames={event.get('frames')} episodes={event.get('episodes')}"
                )
            else:
                tqdm.write(f"FAILED {shard_name}: {event.get('error')}")

    def drain_events(
        *,
        dataset_bar: tqdm,
        episode_bar_total: tqdm,
        shard_bar_total: tqdm,
        wait: bool,
    ) -> None:
        timeout = poll_seconds if wait else 0.0
        while True:
            try:
                event = progress_queue.get(timeout=timeout)
            except queue.Empty:
                return
            handle_event(event, dataset_bar=dataset_bar, episode_bar_total=episode_bar_total, shard_bar_total=shard_bar_total)
            timeout = 0.0

    results: list[dict[str, Any]] = []
    try:
        with (
            tqdm(
                total=total_rows,
                desc=f"{split} dataset",
                unit="frame",
                position=0,
                leave=True,
                dynamic_ncols=True,
            ) as dataset_bar,
            tqdm(
                total=total_episodes,
                desc=f"{split} episodes",
                unit="episode",
                position=1,
                leave=True,
                dynamic_ncols=True,
            ) as episode_bar_total,
            tqdm(
                total=total_shards,
                desc=f"{split} shards",
                unit="shard",
                position=2,
                leave=True,
                dynamic_ncols=True,
            ) as shard_bar_total,
        ):
            if use_process_pool:
                executor_cm = futures.ProcessPoolExecutor(
                    max_workers=worker_count,
                    mp_context=context,
                    initializer=_init_worker,
                    initargs=(progress_queue,),
                )
            else:
                tqdm.write(
                    "multiprocessing progress queue is unavailable; "
                    "using thread workers for this run"
                )
                executor_cm = futures.ThreadPoolExecutor(
                    max_workers=worker_count,
                    initializer=_init_worker,
                    initargs=(progress_queue,),
                )
            with executor_cm as executor:
                future_map = {executor.submit(_build_one_shard, task): task for task in tasks}
                pending = set(future_map)
                while pending:
                    drain_events(
                        dataset_bar=dataset_bar,
                        episode_bar_total=episode_bar_total,
                        shard_bar_total=shard_bar_total,
                        wait=True,
                    )
                    done = {future for future in pending if future.done()}
                    for future in done:
                        task = future_map[future]
                        shard_id = int(task["shard_id"])
                        try:
                            result = future.result()
                        except Exception as exc:  # noqa: BLE001
                            result = {
                                "shard_id": shard_id,
                                "shard_name": str(task["shard_name"]),
                                "status": "failed",
                                "error": repr(exc),
                                "traceback": traceback.format_exc(),
                            }
                        results.append(result)
                        if shard_id not in completed_shards:
                            event_name = {
                                "built": "shard_done",
                                "skipped": "shard_skip",
                                "failed": "shard_failed",
                            }.get(str(result.get("status")), "shard_failed")
                            synthetic_event = {
                                "event": event_name,
                                "shard_id": shard_id,
                                "shard_name": str(result.get("shard_name", task["shard_name"])),
                                "rows": int(result.get("num_rows") or task.get("shard_num_rows") or 0),
                                "frames": int(result.get("frames") or 0),
                                "episodes": int(result.get("episodes") or task.get("episode_count") or 0),
                                "error": result.get("error"),
                            }
                            handle_event(
                                synthetic_event,
                                dataset_bar=dataset_bar,
                                episode_bar_total=episode_bar_total,
                                shard_bar_total=shard_bar_total,
                            )
                    pending -= done
                drain_events(
                    dataset_bar=dataset_bar,
                    episode_bar_total=episode_bar_total,
                    shard_bar_total=shard_bar_total,
                    wait=False,
                )
    finally:
        for shard_id in list(active):
            close_active(shard_id)
    return results


def main() -> int:
    args = parse_args()
    config, settings, build = _resolve(args)
    del config
    shard_root = Path(settings.shard_root or build.shard_root or "")
    split_names = ("train", "val") if build.split == "all" else (build.split,)
    all_failures: list[dict[str, Any]] = []

    print("Origami comp action-chunk shard build")
    print(f"  config_name   : {args.config_name}")
    print(f"  dataset_root  : {settings.dataset_root}")
    print(f"  manifest_root : {settings.manifest_root}")
    print(f"  shard_root    : {shard_root}")
    print(f"  split         : {build.split}")
    print(f"  row_order     : {build.row_order}")
    print(f"  num_workers   : {build.num_workers}")
    print("  episode policy: whole episodes only; no episode is split across shards")

    for split in split_names:
        rows = _origami_vla_dataset.load_manifest_rows(settings, split).reset_index(drop=True)
        if rows.empty:
            print(f"Skipping {split}: no rows")
            continue
        if build.require_manifest_verified:
            _check_manifest_metadata(settings, split=split, rows=rows)
        if "source_row_index" not in rows.columns:
            rows = rows.reset_index().rename(columns={"index": "source_row_index"})
        row_bytes = _shards.estimate_row_bytes(settings, image_size=int(build.image_size))
        episode_table = _load_episode_table(Path(settings.dataset_root), rows, build.season_column)
        episode_infos = _make_episode_infos(
            rows=rows,
            episode_table=episode_table,
            season_column=build.season_column,
            row_bytes=row_bytes,
        )
        target_bytes = _target_bytes(build, episode_infos)
        plan = _build_plan(
            episode_infos=episode_infos,
            target_bytes=target_bytes,
            max_episodes_per_shard=build.max_episodes_per_shard,
            seed=int(build.seed),
        )
        plan_rows = _write_plan_files(
            shard_root=shard_root,
            split=split,
            rows=rows,
            shards=plan,
            settings=settings,
            build=build,
            row_bytes=row_bytes,
            target_bytes=target_bytes,
        )
        plan_frame = pd.DataFrame(plan_rows)
        print(f"{split} rows          : {len(rows):,}")
        print(f"{split} episodes      : {len(episode_infos):,}")
        print(f"{split} shards planned: {len(plan):,}")
        print(f"{split} row bytes est : {row_bytes:,}")
        print(f"{split} target bytes  : {target_bytes:,}")
        if args.plan_only:
            continue

        shard_groups = (
            plan_frame.groupby(["shard_id", "shard_name", "relative_dir", "source_rows_path"], as_index=False)
            .agg(
                shard_num_rows=("shard_num_rows", "first"),
                episode_count=("episode_uid", "nunique"),
            )
            .sort_values("shard_id")
            .to_dict(orient="records")
        )
        shard_groups = [item for item in shard_groups if int(item["shard_id"]) >= int(args.start_shard)]
        if build.max_shards_per_run is not None:
            shard_groups = shard_groups[: int(build.max_shards_per_run)]
        tasks = [
            {
                "settings": settings,
                "build": build,
                "shard_root": str(shard_root),
                "split": split,
                "shard_id": int(item["shard_id"]),
                "shard_name": str(item["shard_name"]),
                "source_rows_path": str(item["source_rows_path"]),
                "shard_num_rows": int(item["shard_num_rows"]),
                "episode_count": int(item["episode_count"]),
            }
            for item in shard_groups
        ]
        if not tasks:
            continue
        worker_count = max(1, min(int(build.num_workers), len(tasks)))
        results = _run_shard_tasks_with_progress(
            tasks=tasks,
            worker_count=worker_count,
            build=build,
            split=split,
        )
        for result in sorted(results, key=lambda item: int(item["shard_id"])):
            status = result["status"]
            if status == "failed":
                all_failures.append(result)
        _update_manifest_completion(shard_root, build)

    if all_failures:
        failure_path = shard_root / "failed_shards.json"
        failure_path.write_text(json.dumps(all_failures, indent=2), encoding="utf-8")
        print(f"Shard build failed for {len(all_failures)} shards. Details: {failure_path}")
        return 1
    print("OK: shard build complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
