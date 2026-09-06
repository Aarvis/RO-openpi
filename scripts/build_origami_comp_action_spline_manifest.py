from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_origami_comp_action_chunk_manifest as _chunk_manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description=(
            "Build OpenPI train/val manifests for Origami pi0.5 comp action-spline training. "
            "Rows point at local delta cubic spline targets generated per source frame."
        )
    )
    parser.add_argument("--config-name", type=str, default=None)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--checkpoint-planner-manifest-root", type=Path, default=None)
    parser.add_argument("--ignore-checkpoint-planner-split", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--num-val-episodes", type=int, default=12)
    parser.add_argument("--val-seed", type=int, default=1234)
    parser.add_argument("--val-episode-uids", type=str, default="")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--local-target-npz-name", type=str, default="local_delta_action_cubic_knotspans10.npz")
    parser.add_argument(
        "--local-target-index-name",
        type=str,
        default="local_delta_action_cubic_knotspans10_index.parquet",
    )
    parser.add_argument("--target-knot-spans", type=int, default=10)
    parser.add_argument("--max-control-points", type=int, default=13)
    parser.add_argument("--max-span-count", type=int, default=10)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--planner-export-root", type=Path, default=None)
    parser.add_argument(
        "--planner-assignment-mode",
        choices=("expand_view_modes", "episode_sampled"),
        default="expand_view_modes",
    )
    parser.add_argument("--train-planner-view-modes", nargs="*", default=None)
    parser.add_argument("--val-planner-view-modes", nargs="*", default=None)
    parser.add_argument("--planner-branch", type=str, default="posterior")
    parser.add_argument("--planner-value-variant", choices=("final", "raw"), default="final")
    parser.add_argument("--planner-assignment-seed", type=int, default=1234)
    parser.add_argument("--planner-dropout-episode-prob", type=float, default=0.16)
    parser.add_argument("--planner-view-mode-probs", nargs="*", default=None)
    parser.add_argument("--planner-branch-probs", nargs="*", default=None)
    parser.add_argument("--planner-index-name", type=str, default="planner_vla_rollout_index.parquet")
    parser.add_argument("--planner-arrays-name", type=str, default="planner_vla_rollout_features.npz")
    parser.add_argument("--planner-complete-marker-name", type=str, default="export_complete.marker")
    parser.add_argument("--require-planner-complete-marker", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-missing-planner-rows", action="store_true")
    parser.add_argument("--speed-weighting", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--speed-label-relpaths",
        nargs="*",
        default=["labels/checkpoints.json", "labels/transfer_checkpoints.json"],
    )
    parser.add_argument("--speed-stats-split", choices=("train", "val", "all"), default="train")
    parser.add_argument("--speed-semantic-group-size", type=int, default=2)
    parser.add_argument("--speed-final-unpaired-policy", choices=("keep", "drop", "error"), default="keep")
    parser.add_argument("--speed-done-policy", choices=("neutral", "weighted"), default="neutral")
    parser.add_argument("--speed-alpha", type=float, default=1.5)
    parser.add_argument("--speed-min-weight", type=float, default=0.5)
    parser.add_argument("--speed-max-weight", type=float, default=2.0)
    parser.add_argument("--speed-epsilon-frames", type=float, default=1.0e-6)
    parser.add_argument("--speed-weight-val", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--train-index-name", type=str, default="train_index.parquet")
    parser.add_argument("--val-index-name", type=str, default="val_index.parquet")
    args = parser.parse_args(argv)
    args._explicit_flags = {arg.split("=", 1)[0] for arg in raw_args if arg.startswith("--")}
    return args


def _cli_flag_supplied(args: argparse.Namespace, *flags: str) -> bool:
    return _chunk_manifest._cli_flag_supplied(args, *flags)


def _resolve_args_from_config(args: argparse.Namespace) -> argparse.Namespace:
    if args.config_name is None:
        return args
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    import openpi.training.config as train_config

    config = train_config.get_config(str(args.config_name))
    data_config = config.data
    if not isinstance(data_config, train_config.OrigamiCompActionChunkDataConfig):
        raise TypeError(
            f"Config {args.config_name!r} uses {type(data_config).__name__}; "
            "expected OrigamiCompActionChunkDataConfig."
        )
    if data_config.action_source != "spline":
        raise ValueError(
            f"Config {args.config_name!r} has action_source={data_config.action_source!r}; "
            "expected action_source='spline'."
        )
    build_config = data_config.manifest_build

    def set_if_config(attr: str, value: Any, *flags: str) -> None:
        if not _cli_flag_supplied(args, *flags):
            setattr(args, attr, value)

    set_if_config("dataset_root", Path(data_config.dataset_root), "--dataset-root")
    set_if_config("output_root", Path(data_config.manifest_root), "--output-root")
    set_if_config(
        "checkpoint_planner_manifest_root",
        _chunk_manifest._path_or_none(build_config.checkpoint_planner_manifest_root),
        "--checkpoint-planner-manifest-root",
    )
    set_if_config(
        "ignore_checkpoint_planner_split",
        bool(build_config.ignore_checkpoint_planner_split),
        "--ignore-checkpoint-planner-split",
        "--no-ignore-checkpoint-planner-split",
    )
    set_if_config("num_val_episodes", int(build_config.num_val_episodes), "--num-val-episodes")
    set_if_config("val_seed", int(build_config.val_seed), "--val-seed")
    set_if_config("val_episode_uids", _chunk_manifest._csv(build_config.val_episode_uids), "--val-episode-uids")
    set_if_config("frame_stride", int(build_config.frame_stride), "--frame-stride")
    set_if_config("local_target_npz_name", str(data_config.local_target_npz_name), "--local-target-npz-name")
    set_if_config("local_target_index_name", str(data_config.local_target_index_name), "--local-target-index-name")
    set_if_config("target_knot_spans", int(config.model.origami_vla.max_span_count), "--target-knot-spans")
    set_if_config("max_control_points", int(config.model.origami_vla.max_control_points), "--max-control-points")
    set_if_config("max_span_count", int(config.model.origami_vla.max_span_count), "--max-span-count")
    set_if_config("degree", int(config.model.origami_vla.degree), "--degree")
    set_if_config("planner_export_root", _chunk_manifest._path_or_none(build_config.planner_export_root), "--planner-export-root")
    set_if_config("planner_assignment_mode", str(build_config.planner_assignment_mode), "--planner-assignment-mode")
    set_if_config("train_planner_view_modes", list(build_config.train_planner_view_modes) or None, "--train-planner-view-modes")
    set_if_config("val_planner_view_modes", list(build_config.val_planner_view_modes) or None, "--val-planner-view-modes")
    set_if_config("planner_branch", str(data_config.planner_branch), "--planner-branch")
    set_if_config("planner_value_variant", str(build_config.planner_value_variant), "--planner-value-variant")
    set_if_config("planner_assignment_seed", int(build_config.planner_assignment_seed), "--planner-assignment-seed")
    set_if_config("planner_dropout_episode_prob", float(build_config.planner_dropout_episode_prob), "--planner-dropout-episode-prob")
    set_if_config(
        "planner_view_mode_probs",
        _chunk_manifest._probability_map_cli_values(build_config.planner_view_mode_probs),
        "--planner-view-mode-probs",
    )
    set_if_config(
        "planner_branch_probs",
        _chunk_manifest._probability_map_cli_values(build_config.planner_branch_probs),
        "--planner-branch-probs",
    )
    set_if_config("planner_index_name", str(build_config.planner_index_name), "--planner-index-name")
    set_if_config("planner_arrays_name", str(build_config.planner_arrays_name), "--planner-arrays-name")
    set_if_config("planner_complete_marker_name", str(build_config.planner_complete_marker_name), "--planner-complete-marker-name")
    set_if_config(
        "require_planner_complete_marker",
        bool(build_config.require_planner_complete_marker),
        "--require-planner-complete-marker",
        "--no-require-planner-complete-marker",
    )
    set_if_config("allow_missing_planner_rows", bool(build_config.allow_missing_planner_rows), "--allow-missing-planner-rows")
    set_if_config("speed_weighting", bool(build_config.speed_weighting), "--speed-weighting", "--no-speed-weighting")
    set_if_config("speed_label_relpaths", list(build_config.speed_label_relpaths), "--speed-label-relpaths")
    set_if_config("speed_stats_split", str(build_config.speed_stats_split), "--speed-stats-split")
    set_if_config("speed_semantic_group_size", int(build_config.speed_semantic_group_size), "--speed-semantic-group-size")
    set_if_config("speed_final_unpaired_policy", str(build_config.speed_final_unpaired_policy), "--speed-final-unpaired-policy")
    set_if_config("speed_done_policy", str(build_config.speed_done_policy), "--speed-done-policy")
    set_if_config("speed_alpha", float(build_config.speed_alpha), "--speed-alpha")
    set_if_config("speed_min_weight", float(build_config.speed_min_weight), "--speed-min-weight")
    set_if_config("speed_max_weight", float(build_config.speed_max_weight), "--speed-max-weight")
    set_if_config("speed_epsilon_frames", float(build_config.speed_epsilon_frames), "--speed-epsilon-frames")
    set_if_config("speed_weight_val", bool(build_config.speed_weight_val), "--speed-weight-val", "--no-speed-weight-val")
    set_if_config("train_index_name", str(build_config.train_index_name), "--train-index-name")
    set_if_config("val_index_name", str(build_config.val_index_name), "--val-index-name")
    return args


def _normalize_target_index(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    frame = frame.copy()
    rename_map = {
        "sample_id": "frame_position",
        "current_frame_position": "frame_position",
        "current_frame_index": "frame_index",
        "npz_sample_index": "local_target_npz_sample_index",
        "num_control_points": "local_num_control_points",
        "num_knots_total": "local_num_knots_total",
    }
    for old, new in rename_map.items():
        if old in frame.columns and new not in frame.columns:
            frame = frame.rename(columns={old: new})
    if "target_valid" in frame.columns:
        frame = frame[frame["target_valid"].astype(bool)].copy()
    required = {
        "episode_uid",
        "frame_position",
        "local_target_npz_sample_index",
        "target_knot_spans_actual",
        "horizon_was_truncated",
        "local_num_control_points",
        "local_num_knots_total",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"{path} is missing required columns: {sorted(missing)}")
    if "frame_index" not in frame.columns:
        frame["frame_index"] = frame["frame_position"]
    for column in (
        "frame_position",
        "frame_index",
        "local_target_npz_sample_index",
        "target_knot_spans_actual",
        "local_num_control_points",
        "local_num_knots_total",
    ):
        frame[column] = frame[column].astype("int64")
    frame["horizon_was_truncated"] = frame["horizon_was_truncated"].astype(bool)
    return frame


def _spline_sample_weights(
    *,
    target_archive: Any,
    sample_indices: np.ndarray,
    fallback_frame_positions: np.ndarray,
    frame_weights: np.ndarray,
) -> np.ndarray:
    archive_keys = set(getattr(target_archive, "files", ()))
    if "frame_index_values" not in archive_keys or "frame_index_offsets" not in archive_keys:
        safe = np.clip(fallback_frame_positions.astype(np.int64), 0, max(int(frame_weights.shape[0]) - 1, 0))
        return frame_weights[safe].astype(np.float32)
    values = np.asarray(target_archive["frame_index_values"], dtype=np.int64)
    offsets = np.asarray(target_archive["frame_index_offsets"], dtype=np.int64)
    weights = np.ones((sample_indices.shape[0],), dtype=np.float32)
    for out_index, sample_index in enumerate(sample_indices.astype(np.int64).tolist()):
        start, end = int(offsets[sample_index]), int(offsets[sample_index + 1])
        frame_indices = values[start:end]
        if frame_indices.size == 0:
            continue
        safe = np.clip(frame_indices, 0, max(int(frame_weights.shape[0]) - 1, 0))
        weights[out_index] = np.asarray(frame_weights[safe], dtype=np.float32).mean()
    return weights


def _build_split_frame(
    *,
    dataset_root: Path,
    split_name: str,
    episode_table: pd.DataFrame,
    episode_uids: list[str],
    frame_stride: int,
    local_target_npz_name: str,
    local_target_index_name: str,
    target_knot_spans: int,
    max_control_points: int,
    max_span_count: int,
    degree: int,
    planner_export_root: Path | None,
    planner_view_modes: list[str],
    planner_branch: str,
    planner_value_variant: str,
    planner_assignments: dict[str, Any] | None,
    planner_index_name: str,
    planner_arrays_name: str,
    planner_complete_marker_name: str,
    require_planner_complete_marker: bool,
    allow_missing_planner_rows: bool,
    speed_context: Any | None,
    use_speed_weights: bool,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    episode_table = episode_table.set_index("episode_uid", drop=False)
    for episode_uid in tqdm(episode_uids, desc=f"Build {split_name} spline manifest", unit="episode", dynamic_ncols=True):
        episode_meta = episode_table.loc[episode_uid]
        episode_root = dataset_root / "episodes" / episode_uid
        arrays_root = episode_root / "arrays"
        state_path = arrays_root / "state_65d.npy"
        tactile_path = arrays_root / "tactile_60d.npy"
        target_npz_path = arrays_root / local_target_npz_name
        target_index_path = arrays_root / local_target_index_name
        required = [state_path, tactile_path, target_npz_path, target_index_path]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing required arrays/targets for {episode_uid}: {missing}")

        metadata_num_frames = int(episode_meta["num_frames"])
        state_len = int(np.load(state_path, mmap_mode="r").shape[0])
        tactile_len = int(np.load(tactile_path, mmap_mode="r").shape[0])
        num_frames = min(metadata_num_frames, state_len, tactile_len)
        target_frame = _normalize_target_index(pd.read_parquet(target_index_path), target_index_path)
        target_frame = target_frame[target_frame["episode_uid"].astype(str) == episode_uid].copy()
        target_frame = target_frame[target_frame["target_knot_spans_actual"] == int(target_knot_spans)].copy()
        target_frame = target_frame[~target_frame["horizon_was_truncated"]].copy()
        target_frame = target_frame[target_frame["local_num_control_points"] == int(max_control_points)].copy()
        target_frame = target_frame[target_frame["local_num_knots_total"] == int(max_control_points + degree + 1)].copy()
        target_frame = target_frame[target_frame["local_target_npz_sample_index"] >= 0].copy()
        target_frame = target_frame[target_frame["frame_position"] < int(num_frames)].copy()
        target_frame = target_frame[target_frame["frame_position"] % max(1, int(frame_stride)) == 0].copy()
        if target_frame.empty:
            continue

        frame_index_array = _chunk_manifest._read_frame_index(arrays_root, num_frames)
        timestamps_array = _chunk_manifest._read_timestamps(arrays_root, num_frames)
        target_frame["frame_index"] = frame_index_array[target_frame["frame_position"].to_numpy(dtype=np.int64)]
        target_frame["timestamp"] = timestamps_array[target_frame["frame_position"].to_numpy(dtype=np.int64)]

        if speed_context is not None and use_speed_weights:
            frame_speed_weights, _frame_semantic_ids = _chunk_manifest._frame_speed_weights(
                episode_uid=episode_uid,
                num_frames=num_frames,
                speed_context=speed_context,
            )
            with np.load(target_npz_path, allow_pickle=False) as target_archive:
                sample_weights = _spline_sample_weights(
                    target_archive=target_archive,
                    sample_indices=target_frame["local_target_npz_sample_index"].to_numpy(dtype=np.int64),
                    fallback_frame_positions=target_frame["frame_position"].to_numpy(dtype=np.int64),
                    frame_weights=frame_speed_weights,
                )
        else:
            sample_weights = np.ones((len(target_frame),), dtype=np.float32)

        keep_columns = [
            "episode_uid",
            "frame_position",
            "frame_index",
            "timestamp",
            "local_target_npz_sample_index",
            "local_num_control_points",
            "local_num_knots_total",
            "target_knot_spans_requested",
            "target_knot_spans_actual",
            "horizon_was_truncated",
            "u_start",
            "u_end",
            "horizon_boundary_u",
            "start_knot_span_index",
            "end_knot_span_index",
        ]
        available_columns = [column for column in keep_columns if column in target_frame.columns]
        base_frame = target_frame[available_columns].copy()
        base_frame["split"] = split_name
        base_frame["sample_weight"] = sample_weights.astype("float32")
        base_frame["action_source"] = "spline"
        base_frame["action_horizon"] = int(max_control_points + 1)
        base_frame["action_dim"] = 65
        base_frame["max_control_points"] = int(max_control_points)
        base_frame["max_span_count"] = int(max_span_count)
        base_frame["degree"] = int(degree)

        if planner_export_root is None:
            rows.append(base_frame)
            continue
        if planner_assignments is not None:
            assignment = planner_assignments.get(episode_uid)
            if assignment is None:
                raise RuntimeError(f"No planner assignment was generated for {episode_uid}")
            if not assignment.enabled:
                rows.append(_chunk_manifest._attach_disabled_planner_rows(base_frame))
                continue
            rows.extend(
                _chunk_manifest._attach_planner_rows(
                    base_frame=base_frame,
                    planner_export_root=planner_export_root,
                    episode_uid=episode_uid,
                    view_modes=[assignment.view_mode],
                    planner_branch=assignment.branch,
                    planner_value_variant=assignment.value_variant,
                    planner_index_name=planner_index_name,
                    planner_arrays_name=planner_arrays_name,
                    planner_complete_marker_name=planner_complete_marker_name,
                    require_complete_marker=require_planner_complete_marker,
                    allow_missing_rows=allow_missing_planner_rows,
                )
            )
            continue
        rows.extend(
            _chunk_manifest._attach_planner_rows(
                base_frame=base_frame,
                planner_export_root=planner_export_root,
                episode_uid=episode_uid,
                view_modes=planner_view_modes,
                planner_branch=planner_branch,
                planner_value_variant=planner_value_variant,
                planner_index_name=planner_index_name,
                planner_arrays_name=planner_arrays_name,
                planner_complete_marker_name=planner_complete_marker_name,
                require_complete_marker=require_planner_complete_marker,
                allow_missing_rows=allow_missing_planner_rows,
            )
        )
    if not rows:
        return pd.DataFrame()
    combined = pd.concat(rows, axis=0, ignore_index=True)
    for column in (
        "frame_position",
        "frame_index",
        "local_target_npz_sample_index",
        "local_num_control_points",
        "local_num_knots_total",
    ):
        if column in combined.columns:
            combined[column] = combined[column].astype("int64")
    return combined


def main() -> int:
    args = _resolve_args_from_config(parse_args())
    if args.dataset_root is None:
        raise ValueError("--dataset-root is required unless supplied by --config-name.")
    if args.output_root is None:
        raise ValueError("--output-root is required unless supplied by --config-name.")
    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be positive.")
    if args.target_knot_spans <= 0:
        raise ValueError("--target-knot-spans must be positive.")
    if args.max_control_points <= 0:
        raise ValueError("--max-control-points must be positive.")
    if args.max_span_count != args.target_knot_spans:
        raise ValueError("--max-span-count must match --target-knot-spans for this fixed-width shard format.")

    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)
    planner_export_root = None if args.planner_export_root is None else Path(args.planner_export_root)
    planner_view_mode_probs = _chunk_manifest._parse_probability_map(
        args.planner_view_mode_probs,
        default={"frame_stride_10": 0.5, "random_mix": 0.25, "fixed_7": 0.25, "fixed_15": 0.0},
        option_name="--planner-view-mode-probs",
    )
    planner_branch_probs = _chunk_manifest._parse_probability_map(
        args.planner_branch_probs,
        default={"posterior": 0.5, "prior": 0.5},
        option_name="--planner-branch-probs",
    )
    if planner_export_root is not None and args.planner_assignment_mode == "episode_sampled":
        train_planner_view_modes = [key for key, value in planner_view_mode_probs.items() if value > 0.0]
        val_planner_view_modes = [key for key, value in planner_view_mode_probs.items() if value > 0.0]
    else:
        train_planner_view_modes = (
            _chunk_manifest._resolve_planner_view_modes(args.train_planner_view_modes, split_name="train")
            if planner_export_root is not None
            else []
        )
        val_planner_view_modes = (
            _chunk_manifest._resolve_planner_view_modes(args.val_planner_view_modes, split_name="val")
            if planner_export_root is not None
            else []
        )
    episode_table = _chunk_manifest._load_episode_table(dataset_root)
    episode_uids = [str(uid) for uid in episode_table["episode_uid"].tolist()]
    train_episodes, val_episodes = _chunk_manifest._make_split(args, episode_uids)
    speed_settings = _chunk_manifest._load_speed_weight_settings(args)
    speed_context = _chunk_manifest._build_speed_weight_context(
        dataset_root=dataset_root,
        train_episodes=train_episodes,
        val_episodes=val_episodes,
        settings=speed_settings,
    )
    train_planner_assignments = _chunk_manifest._build_planner_assignments(
        train_episodes,
        enabled=planner_export_root is not None,
        assignment_mode=str(args.planner_assignment_mode),
        seed=int(args.planner_assignment_seed),
        dropout_episode_prob=float(args.planner_dropout_episode_prob),
        view_mode_probs=planner_view_mode_probs,
        branch_probs=planner_branch_probs,
        value_variant=str(args.planner_value_variant),
    )
    val_planner_assignments = _chunk_manifest._build_planner_assignments(
        val_episodes,
        enabled=planner_export_root is not None,
        assignment_mode=str(args.planner_assignment_mode),
        seed=int(args.planner_assignment_seed) + 3001,
        dropout_episode_prob=float(args.planner_dropout_episode_prob),
        view_mode_probs=planner_view_mode_probs,
        branch_probs=planner_branch_probs,
        value_variant=str(args.planner_value_variant),
    )

    common_kwargs = {
        "dataset_root": dataset_root,
        "episode_table": episode_table,
        "frame_stride": int(args.frame_stride),
        "local_target_npz_name": str(args.local_target_npz_name),
        "local_target_index_name": str(args.local_target_index_name),
        "target_knot_spans": int(args.target_knot_spans),
        "max_control_points": int(args.max_control_points),
        "max_span_count": int(args.max_span_count),
        "degree": int(args.degree),
        "planner_export_root": planner_export_root,
        "planner_index_name": str(args.planner_index_name),
        "planner_arrays_name": str(args.planner_arrays_name),
        "planner_complete_marker_name": str(args.planner_complete_marker_name),
        "require_planner_complete_marker": bool(args.require_planner_complete_marker),
        "allow_missing_planner_rows": bool(args.allow_missing_planner_rows),
        "speed_context": speed_context,
    }
    train_frame = _build_split_frame(
        split_name="train",
        episode_uids=train_episodes,
        planner_view_modes=train_planner_view_modes,
        planner_branch=str(args.planner_branch),
        planner_value_variant=str(args.planner_value_variant),
        planner_assignments=train_planner_assignments,
        use_speed_weights=speed_context is not None,
        **common_kwargs,
    )
    val_frame = _build_split_frame(
        split_name="val",
        episode_uids=val_episodes,
        planner_view_modes=val_planner_view_modes,
        planner_branch=str(args.planner_branch),
        planner_value_variant=str(args.planner_value_variant),
        planner_assignments=val_planner_assignments,
        use_speed_weights=speed_context is not None and speed_settings.weight_val,
        **common_kwargs,
    )
    if train_frame.empty and not val_frame.empty:
        train_frame = train_frame.reindex(columns=val_frame.columns)
    if val_frame.empty and not train_frame.empty:
        val_frame = val_frame.reindex(columns=train_frame.columns)

    output_root.mkdir(parents=True, exist_ok=True)
    train_path = output_root / args.train_index_name
    val_path = output_root / args.val_index_name
    train_frame.to_parquet(train_path, index=False)
    val_frame.to_parquet(val_path, index=False)
    manifest: dict[str, Any] = {
        "format": "origami_comp_action_spline_manifest_v1",
        "config_name": args.config_name,
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "train_index_name": train_path.name,
        "val_index_name": val_path.name,
        "frame_stride": int(args.frame_stride),
        "target": {
            "action_source": "spline",
            "local_target_npz_name": str(args.local_target_npz_name),
            "local_target_index_name": str(args.local_target_index_name),
            "target_knot_spans": int(args.target_knot_spans),
            "max_control_points": int(args.max_control_points),
            "max_span_count": int(args.max_span_count),
            "degree": int(args.degree),
            "packed_action_horizon": int(args.max_control_points + 1),
            "packed_action_dim": 65,
            "packed_layout": "rows[0:max_control_points]=control_points; row[max_control_points,0:max_span_count]=span_widths",
        },
        "planner": {
            "enabled": planner_export_root is not None,
            "export_root": None if planner_export_root is None else str(planner_export_root),
            "assignment_mode": str(args.planner_assignment_mode),
            "assignment_seed": int(args.planner_assignment_seed),
            "dropout_episode_prob": float(args.planner_dropout_episode_prob),
            "view_mode_probs": planner_view_mode_probs,
            "branch_probs": planner_branch_probs,
            "value_variant": str(args.planner_value_variant),
            "train_view_modes": train_planner_view_modes,
            "val_view_modes": val_planner_view_modes,
            "branch": str(args.planner_branch),
            "index_name": str(args.planner_index_name),
            "arrays_name": str(args.planner_arrays_name),
            "complete_marker_name": str(args.planner_complete_marker_name),
            "require_complete_marker": bool(args.require_planner_complete_marker),
            "allow_missing_rows": bool(args.allow_missing_planner_rows),
            "train_assignment_summary": _chunk_manifest._planner_assignment_summary(train_planner_assignments),
            "val_assignment_summary": _chunk_manifest._planner_assignment_summary(val_planner_assignments),
        },
        "speed_weighting": {
            "enabled": bool(speed_settings.enabled),
            "label_priority_relpaths": list(speed_settings.label_priority_relpaths),
            "stats_split": speed_settings.stats_split,
            "semantic_group_size": int(speed_settings.semantic_group_size),
            "final_unpaired_policy": speed_settings.final_unpaired_policy,
            "done_policy": speed_settings.done_policy,
            "alpha": float(speed_settings.alpha),
            "w_min": float(speed_settings.w_min),
            "w_max": float(speed_settings.w_max),
            "epsilon_frames": float(speed_settings.epsilon_frames),
            "train_weighted": bool(speed_context is not None),
            "val_weighted": bool(speed_context is not None and speed_settings.weight_val),
            "semantic_checkpoint_duration_medians_frames": (
                {}
                if speed_context is None
                else {str(key): float(value) for key, value in sorted(speed_context.median_durations.items())}
            ),
        },
        "split": {
            "train_episode_uids": train_episodes,
            "val_episode_uids": val_episodes,
        },
        "counts": {
            "train_rows": int(len(train_frame)),
            "val_rows": int(len(val_frame)),
            "train_episodes": int(len(train_episodes)),
            "val_episodes": int(len(val_episodes)),
        },
        "sample_weight_stats": {
            "train": _chunk_manifest._weight_stats(train_frame),
            "val": _chunk_manifest._weight_stats(val_frame),
        },
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Origami comp action-spline manifest")
    print(f"  dataset_root  : {dataset_root}")
    print(f"  output_root   : {output_root}")
    print(f"  train_rows    : {len(train_frame)}")
    print(f"  val_rows      : {len(val_frame)}")
    print(f"  train_episodes: {len(train_episodes)}")
    print(f"  val_episodes  : {len(val_episodes)}")
    print(f"  target        : spans={args.target_knot_spans} cps={args.max_control_points} degree={args.degree}")
    if speed_context is not None:
        train_stats = _chunk_manifest._weight_stats(train_frame)
        val_stats = _chunk_manifest._weight_stats(val_frame)
        print(
            "  speed weights : "
            f"enabled train_mean={train_stats['mean']} train_min={train_stats['min']} "
            f"train_max={train_stats['max']} val_mean={val_stats['mean']}"
        )
    if planner_export_root is not None:
        print(f"  planner_root  : {planner_export_root}")
        print(f"  planner_mode  : {args.planner_assignment_mode}")
        if train_planner_assignments is None:
            print(f"  planner_branch: {args.planner_branch}")
        else:
            print(f"  train_planner : {_chunk_manifest._planner_assignment_summary(train_planner_assignments)}")
            print(f"  val_planner   : {_chunk_manifest._planner_assignment_summary(val_planner_assignments)}")
    print(f"  train_file    : {train_path}")
    print(f"  val_file      : {val_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
