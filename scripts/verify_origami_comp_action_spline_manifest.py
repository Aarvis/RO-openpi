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
import verify_origami_comp_action_chunk_manifest as _chunk_verify


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description=(
            "Verify an Origami pi0.5 comp action-spline manifest before shard building. "
            "This checks dataset files, local spline target coverage, planner pointers, and sample weights."
        )
    )
    parser.add_argument("--config-name", type=str, default=None)
    parser.add_argument("--manifest-root", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--train-index-name", type=str, default="train_index.parquet")
    parser.add_argument("--val-index-name", type=str, default="val_index.parquet")
    parser.add_argument("--local-target-npz-name", type=str, default="local_delta_action_cubic_knotspans10.npz")
    parser.add_argument("--local-target-index-name", type=str, default="local_delta_action_cubic_knotspans10_index.parquet")
    parser.add_argument("--target-knot-spans", type=int, default=10)
    parser.add_argument("--max-control-points", type=int, default=13)
    parser.add_argument("--max-span-count", type=int, default=10)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--planner-arrays-name", type=str, default="planner_vla_rollout_features.npz")
    parser.add_argument("--planner-index-name", type=str, default="planner_vla_rollout_index.parquet")
    parser.add_argument("--planner-complete-marker-name", type=str, default="export_complete.marker")
    parser.add_argument("--require-planner-complete-marker", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--check-dataset-files", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--expect-train-episodes", type=int, default=None)
    parser.add_argument("--expect-val-episodes", type=int, default=None)
    parser.add_argument("--expect-planner-enabled-ratio", type=float, default=None)
    parser.add_argument("--planner-enabled-ratio-tolerance", type=float, default=0.02)
    parser.add_argument("--expect-view-mode-probs", nargs="*", default=None)
    parser.add_argument("--expect-branch-probs", nargs="*", default=None)
    parser.add_argument("--ratio-tolerance", type=float, default=0.03)
    parser.add_argument("--expect-planner-value-variant", choices=("final", "raw"), default=None)
    parser.add_argument("--planner-belief-dim", type=int, default=29)
    parser.add_argument("--planner-progress-dim", type=int, default=2)
    parser.add_argument("--planner-uncertainty-dim", type=int, default=3)
    parser.add_argument("--planner-history-dim", type=int, default=512)
    args = parser.parse_args(argv)
    args._explicit_flags = {arg.split("=", 1)[0] for arg in raw_args if arg.startswith("--")}
    return args


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
    origami_vla = config.model.origami_vla

    def set_if_config(attr: str, value: Any, *flags: str) -> None:
        if not _chunk_manifest._cli_flag_supplied(args, *flags):
            setattr(args, attr, value)

    set_if_config("manifest_root", Path(data_config.manifest_root), "--manifest-root")
    set_if_config("dataset_root", Path(data_config.dataset_root), "--dataset-root")
    set_if_config("train_index_name", str(build_config.train_index_name), "--train-index-name")
    set_if_config("val_index_name", str(build_config.val_index_name), "--val-index-name")
    set_if_config("local_target_npz_name", str(data_config.local_target_npz_name), "--local-target-npz-name")
    set_if_config("local_target_index_name", str(data_config.local_target_index_name), "--local-target-index-name")
    set_if_config("target_knot_spans", int(origami_vla.max_span_count), "--target-knot-spans")
    set_if_config("max_control_points", int(origami_vla.max_control_points), "--max-control-points")
    set_if_config("max_span_count", int(origami_vla.max_span_count), "--max-span-count")
    set_if_config("degree", int(origami_vla.degree), "--degree")
    set_if_config("planner_arrays_name", str(build_config.planner_arrays_name), "--planner-arrays-name")
    set_if_config("planner_index_name", str(build_config.planner_index_name), "--planner-index-name")
    set_if_config("planner_complete_marker_name", str(build_config.planner_complete_marker_name), "--planner-complete-marker-name")
    set_if_config(
        "require_planner_complete_marker",
        bool(build_config.require_planner_complete_marker),
        "--require-planner-complete-marker",
        "--no-require-planner-complete-marker",
    )
    if build_config.planner_export_root and build_config.planner_assignment_mode == "episode_sampled":
        set_if_config("expect_planner_enabled_ratio", 1.0 - float(build_config.planner_dropout_episode_prob), "--expect-planner-enabled-ratio")
        set_if_config(
            "expect_view_mode_probs",
            _chunk_manifest._probability_map_cli_values(build_config.planner_view_mode_probs),
            "--expect-view-mode-probs",
        )
        set_if_config(
            "expect_branch_probs",
            _chunk_manifest._probability_map_cli_values(build_config.planner_branch_probs),
            "--expect-branch-probs",
        )
    set_if_config("expect_planner_value_variant", str(build_config.planner_value_variant), "--expect-planner-value-variant")
    set_if_config("expect_val_episodes", int(build_config.num_val_episodes), "--expect-val-episodes")
    set_if_config("planner_belief_dim", int(origami_vla.belief_dim), "--planner-belief-dim")
    set_if_config("planner_history_dim", int(origami_vla.history_dim), "--planner-history-dim")
    return args


def _check_dataset_and_targets(
    dataset_root: Path,
    frame: pd.DataFrame,
    *,
    local_target_npz_name: str,
    local_target_index_name: str,
    failures: list[str],
) -> None:
    required_arrays = ["state_65d.npy", "tactile_60d.npy", local_target_npz_name, local_target_index_name]
    required_videos = ["head_left.mp4", "wrist_left.mp4", "wrist_right.mp4", "tactile_deform.mp4"]
    episode_uids = _chunk_verify._split_episode_uids(frame)
    for episode_uid in tqdm(episode_uids, desc="Verify dataset and target files", unit="episode", dynamic_ncols=True):
        episode_root = dataset_root / "episodes" / episode_uid
        arrays_root = episode_root / "arrays"
        videos_root = episode_root / "videos"
        for name in required_arrays:
            path = arrays_root / name
            if not path.exists():
                failures.append(f"Missing array/target for {episode_uid}: {path}")
        for name in required_videos:
            path = videos_root / name
            if not path.exists():
                failures.append(f"Missing video for {episode_uid}: {path}")


def _check_spline_rows(
    frame: pd.DataFrame,
    *,
    split_name: str,
    target_knot_spans: int,
    max_control_points: int,
    max_span_count: int,
    degree: int,
    failures: list[str],
) -> None:
    if frame.empty:
        return
    required = {
        "episode_uid",
        "frame_position",
        "local_target_npz_sample_index",
        "local_num_control_points",
        "local_num_knots_total",
        "target_knot_spans_actual",
        "horizon_was_truncated",
        "sample_weight",
    }
    missing = required.difference(frame.columns)
    if missing:
        failures.append(f"{split_name} is missing spline manifest columns: {sorted(missing)}")
        return
    _chunk_verify._check(
        bool((frame["target_knot_spans_actual"].astype("int64") == int(target_knot_spans)).all()),
        f"{split_name} has rows whose target_knot_spans_actual != {target_knot_spans}",
        failures,
    )
    _chunk_verify._check(
        bool((frame["local_num_control_points"].astype("int64") == int(max_control_points)).all()),
        f"{split_name} has rows whose local_num_control_points != {max_control_points}",
        failures,
    )
    expected_knots = int(max_control_points + degree + 1)
    _chunk_verify._check(
        bool((frame["local_num_knots_total"].astype("int64") == expected_knots).all()),
        f"{split_name} has rows whose local_num_knots_total != {expected_knots}",
        failures,
    )
    _chunk_verify._check(
        bool((frame["horizon_was_truncated"].astype(bool) == False).all()),
        f"{split_name} contains truncated local spline horizons.",
        failures,
    )
    if "max_span_count" in frame.columns:
        _chunk_verify._check(
            bool((frame["max_span_count"].astype("int64") == int(max_span_count)).all()),
            f"{split_name} has rows whose max_span_count != {max_span_count}",
            failures,
        )


def main() -> int:
    args = _resolve_args_from_config(parse_args())
    if args.manifest_root is None:
        raise ValueError("--manifest-root is required unless supplied by --config-name.")
    manifest_root = Path(args.manifest_root)
    manifest = _chunk_verify._load_manifest(manifest_root)
    dataset_root = Path(args.dataset_root) if args.dataset_root is not None else Path(manifest["dataset_root"])

    train_frame = _chunk_verify._load_split_frame(manifest_root, args.train_index_name)
    val_frame = _chunk_verify._load_split_frame(manifest_root, args.val_index_name)
    failures: list[str] = []

    print("Origami comp action-spline manifest verification")
    print(f"config_name  : {args.config_name}")
    print(f"manifest_root: {manifest_root}")
    print(f"dataset_root : {dataset_root}")
    _chunk_verify._summarize_assignments("train", train_frame, failures)
    _chunk_verify._summarize_assignments("val", val_frame, failures)
    _check_spline_rows(
        train_frame,
        split_name="train",
        target_knot_spans=int(args.target_knot_spans),
        max_control_points=int(args.max_control_points),
        max_span_count=int(args.max_span_count),
        degree=int(args.degree),
        failures=failures,
    )
    _check_spline_rows(
        val_frame,
        split_name="val",
        target_knot_spans=int(args.target_knot_spans),
        max_control_points=int(args.max_control_points),
        max_span_count=int(args.max_span_count),
        degree=int(args.degree),
        failures=failures,
    )

    train_episodes = _chunk_verify._split_episode_uids(train_frame)
    val_episodes = _chunk_verify._split_episode_uids(val_frame)
    if args.expect_train_episodes is not None:
        _chunk_verify._check(
            len(train_episodes) == args.expect_train_episodes,
            f"train episodes={len(train_episodes)}, expected {args.expect_train_episodes}",
            failures,
        )
    if args.expect_val_episodes is not None:
        _chunk_verify._check(
            len(val_episodes) == args.expect_val_episodes,
            f"val episodes={len(val_episodes)}, expected {args.expect_val_episodes}",
            failures,
        )

    train_planner = _chunk_verify._planner_episode_table(train_frame)
    if not train_planner.empty:
        enabled = train_planner[train_planner["planner_enabled"]]
        if args.expect_planner_enabled_ratio is not None:
            actual = len(enabled) / float(len(train_planner))
            if abs(actual - args.expect_planner_enabled_ratio) > args.planner_enabled_ratio_tolerance:
                failures.append(
                    f"train planner enabled ratio is {actual:.4f}, expected {args.expect_planner_enabled_ratio:.4f} "
                    f"+/- {args.planner_enabled_ratio_tolerance:.4f}"
                )
        _chunk_verify._check_ratio(
            enabled["view_mode"].value_counts().to_dict(),
            _chunk_verify._parse_probability_map(args.expect_view_mode_probs),
            total=len(enabled),
            name="train planner view-mode",
            tolerance=float(args.ratio_tolerance),
            failures=failures,
        )
        _chunk_verify._check_ratio(
            enabled["planner_branch"].value_counts().to_dict(),
            _chunk_verify._parse_probability_map(args.expect_branch_probs),
            total=len(enabled),
            name="train planner branch",
            tolerance=float(args.ratio_tolerance),
            failures=failures,
        )
        if args.expect_planner_value_variant is not None and not enabled.empty:
            variants = set(str(value) for value in enabled["planner_value_variant"].drop_duplicates().tolist())
            _chunk_verify._check(
                variants == {args.expect_planner_value_variant},
                f"train planner value variants are {sorted(variants)}, expected only {args.expect_planner_value_variant!r}",
                failures,
            )

    if args.check_dataset_files:
        all_frames = pd.concat([train_frame, val_frame], axis=0, ignore_index=True)
        _check_dataset_and_targets(
            dataset_root,
            all_frames,
            local_target_npz_name=str(args.local_target_npz_name),
            local_target_index_name=str(args.local_target_index_name),
            failures=failures,
        )
    if "planner_enabled" in train_frame.columns:
        _chunk_verify._check_planner_exports(
            train_frame,
            planner_arrays_name=args.planner_arrays_name,
            planner_index_name=args.planner_index_name,
            planner_complete_marker_name=args.planner_complete_marker_name,
            require_complete_marker=bool(args.require_planner_complete_marker),
            belief_dim=int(args.planner_belief_dim),
            progress_dim=int(args.planner_progress_dim),
            uncertainty_dim=int(args.planner_uncertainty_dim),
            history_dim=int(args.planner_history_dim),
            failures=failures,
        )
    if "planner_enabled" in val_frame.columns:
        _chunk_verify._check_planner_exports(
            val_frame,
            planner_arrays_name=args.planner_arrays_name,
            planner_index_name=args.planner_index_name,
            planner_complete_marker_name=args.planner_complete_marker_name,
            require_complete_marker=bool(args.require_planner_complete_marker),
            belief_dim=int(args.planner_belief_dim),
            progress_dim=int(args.planner_progress_dim),
            uncertainty_dim=int(args.planner_uncertainty_dim),
            history_dim=int(args.planner_history_dim),
            failures=failures,
        )

    if failures:
        print("\nFAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("\nOK: manifest is ready for comp action-spline shard building.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
