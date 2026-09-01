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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description=(
            "Verify an Origami pi0.5 comp action-chunk manifest before VLA training. "
            "This is CPU-only and checks dataset files, horizon rows, planner assignment "
            "coverage, and planner rollout export pointers."
        )
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default=None,
        help="Optional OpenPI training config name to use as the manifest-verifier source of truth.",
    )
    parser.add_argument("--manifest-root", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--train-index-name", type=str, default="train_index.parquet")
    parser.add_argument("--val-index-name", type=str, default="val_index.parquet")
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


def _cli_flag_supplied(args: argparse.Namespace, *flags: str) -> bool:
    explicit_flags = getattr(args, "_explicit_flags", set())
    return any(flag in explicit_flags for flag in flags)


def _probability_map_cli_values(values: dict[str, float]) -> list[str] | None:
    if not values:
        return None
    return [f"{key}={float(weight)}" for key, weight in values.items()]


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
    build_config = data_config.manifest_build
    origami_vla = config.model.origami_vla

    def set_if_config(attr: str, value: Any, *flags: str) -> None:
        if not _cli_flag_supplied(args, *flags):
            setattr(args, attr, value)

    set_if_config("manifest_root", Path(data_config.manifest_root), "--manifest-root")
    set_if_config("dataset_root", Path(data_config.dataset_root), "--dataset-root")
    set_if_config("train_index_name", str(build_config.train_index_name), "--train-index-name")
    set_if_config("val_index_name", str(build_config.val_index_name), "--val-index-name")
    set_if_config("planner_arrays_name", str(build_config.planner_arrays_name), "--planner-arrays-name")
    set_if_config("planner_index_name", str(build_config.planner_index_name), "--planner-index-name")
    set_if_config(
        "planner_complete_marker_name",
        str(build_config.planner_complete_marker_name),
        "--planner-complete-marker-name",
    )
    set_if_config(
        "require_planner_complete_marker",
        bool(build_config.require_planner_complete_marker),
        "--require-planner-complete-marker",
        "--no-require-planner-complete-marker",
    )
    if build_config.planner_export_root and build_config.planner_assignment_mode == "episode_sampled":
        set_if_config(
            "expect_planner_enabled_ratio",
            1.0 - float(build_config.planner_dropout_episode_prob),
            "--expect-planner-enabled-ratio",
        )
        set_if_config(
            "expect_view_mode_probs",
            _probability_map_cli_values(build_config.planner_view_mode_probs),
            "--expect-view-mode-probs",
        )
        set_if_config(
            "expect_branch_probs",
            _probability_map_cli_values(build_config.planner_branch_probs),
            "--expect-branch-probs",
        )
    set_if_config(
        "expect_planner_value_variant",
        str(build_config.planner_value_variant),
        "--expect-planner-value-variant",
    )
    set_if_config("expect_val_episodes", int(build_config.num_val_episodes), "--expect-val-episodes")
    set_if_config("planner_belief_dim", int(origami_vla.belief_dim), "--planner-belief-dim")
    set_if_config("planner_history_dim", int(origami_vla.history_dim), "--planner-history-dim")
    return args


def _load_manifest(manifest_root: Path) -> dict[str, Any]:
    path = manifest_root / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_split_frame(manifest_root: Path, filename: str) -> pd.DataFrame:
    path = manifest_root / filename
    if not path.exists():
        raise FileNotFoundError(f"Manifest split parquet not found: {path}")
    return pd.read_parquet(path)


def _parse_probability_map(values: list[str] | None) -> dict[str, float] | None:
    if not values:
        return None
    result: dict[str, float] = {}
    for value in values:
        for item in str(value).split(","):
            item = item.strip()
            if not item:
                continue
            key, sep, weight = item.partition("=")
            if not sep:
                raise ValueError(f"Expected probability entry name=value, got {item!r}")
            result[key.strip()] = float(weight)
    total = float(sum(result.values()))
    if total <= 0.0:
        raise ValueError(f"Expected probabilities to sum to > 0, got {result}")
    return {key: value / total for key, value in result.items()}


def _row_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "none"}
    return bool(value)


def _planner_key(base_key: str, branch: str) -> str:
    branch = str(branch or "alias")
    if branch in {"alias", "compatibility", "default"}:
        return base_key
    return f"{branch}_{base_key}"


def _planner_state_key(value_variant: str) -> str:
    if value_variant == "final":
        return "final_state_belief"
    if value_variant == "raw":
        return "raw_state_belief"
    raise ValueError(f"Unsupported planner value variant: {value_variant!r}")


def _check(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def _split_episode_uids(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return []
    return sorted(str(uid) for uid in frame["episode_uid"].drop_duplicates().tolist())


def _planner_episode_table(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "planner_enabled" not in frame.columns:
        return pd.DataFrame()
    work = frame.copy()
    work["planner_enabled"] = work["planner_enabled"].map(lambda value: _row_bool(value, default=True))
    cols = ["episode_uid", "planner_enabled", "view_mode", "planner_branch", "planner_value_variant"]
    for col in cols:
        if col not in work.columns:
            work[col] = ""
    return work[cols].drop_duplicates().reset_index(drop=True)


def _check_ratio(
    actual_counts: dict[str, int],
    expected_probs: dict[str, float] | None,
    *,
    total: int,
    name: str,
    tolerance: float,
    failures: list[str],
) -> None:
    if expected_probs is None or total <= 0:
        return
    for key, expected_prob in expected_probs.items():
        actual_prob = actual_counts.get(key, 0) / float(total)
        if abs(actual_prob - expected_prob) > tolerance:
            failures.append(
                f"{name} ratio for {key!r} is {actual_prob:.4f}, expected {expected_prob:.4f} +/- {tolerance:.4f}"
            )


def _check_dataset_files(dataset_root: Path, episode_uids: list[str], failures: list[str]) -> None:
    required_arrays = ["state_65d.npy", "action_65d.npy", "tactile_60d.npy", "timestamps.npy", "frame_index.npy"]
    required_videos = [
        "head_left.mp4",
        "wrist_left.mp4",
        "wrist_right.mp4",
        "tactile_deform.mp4",
        "tactile_raw.mp4",
    ]
    for episode_uid in tqdm(episode_uids, desc="Verify dataset files", unit="episode", dynamic_ncols=True):
        episode_root = dataset_root / "episodes" / episode_uid
        arrays_root = episode_root / "arrays"
        videos_root = episode_root / "videos"
        lengths: dict[str, int] = {}
        for name in required_arrays:
            path = arrays_root / name
            if not path.exists():
                failures.append(f"Missing array for {episode_uid}: {path}")
                continue
            try:
                lengths[name] = int(np.load(path, mmap_mode="r").shape[0])
            except Exception as exc:  # noqa: BLE001
                failures.append(f"Failed to read array for {episode_uid}: {path}: {exc}")
        if lengths:
            min_len = min(lengths.values())
            bad = {key: value for key, value in lengths.items() if value != min_len}
            if bad:
                failures.append(f"Array length mismatch for {episode_uid}: {lengths}")
        for name in required_videos:
            path = videos_root / name
            if not path.exists():
                failures.append(f"Missing video for {episode_uid}: {path}")


def _check_planner_exports(
    frame: pd.DataFrame,
    *,
    planner_arrays_name: str,
    planner_index_name: str,
    planner_complete_marker_name: str,
    require_complete_marker: bool,
    belief_dim: int,
    progress_dim: int,
    uncertainty_dim: int,
    history_dim: int,
    failures: list[str],
) -> None:
    if frame.empty or "planner_enabled" not in frame.columns:
        return
    work = frame.copy()
    work["planner_enabled"] = work["planner_enabled"].map(lambda value: _row_bool(value, default=True))
    enabled = work[work["planner_enabled"]]
    if enabled.empty:
        return

    required_cols = {"episode_uid", "frame_position", "view_mode", "planner_branch", "planner_value_variant", "planner_output_dir", "planner_row_index"}
    missing_cols = required_cols.difference(enabled.columns)
    if missing_cols:
        failures.append(f"Planner-enabled rows are missing columns: {sorted(missing_cols)}")
        return

    group_cols = ["episode_uid", "view_mode", "planner_branch", "planner_value_variant", "planner_output_dir"]
    groups = enabled.groupby(group_cols, sort=True, dropna=False)
    for group_key, group in tqdm(groups, desc="Verify planner exports", unit="group", dynamic_ncols=True):
        episode_uid, view_mode, branch, value_variant, planner_output_dir = [str(value) for value in group_key]
        output_dir = Path(planner_output_dir)
        index_path = output_dir / planner_index_name
        arrays_path = output_dir / planner_arrays_name
        marker_path = output_dir / planner_complete_marker_name
        if not output_dir.exists():
            failures.append(f"Missing planner output dir for {episode_uid}:{view_mode}: {output_dir}")
            continue
        if not index_path.exists():
            failures.append(f"Missing planner index for {episode_uid}:{view_mode}: {index_path}")
            continue
        if not arrays_path.exists():
            failures.append(f"Missing planner arrays for {episode_uid}:{view_mode}: {arrays_path}")
            continue
        if require_complete_marker and not marker_path.exists():
            failures.append(f"Missing planner complete marker for {episode_uid}:{view_mode}: {marker_path}")
            continue

        try:
            planner_index = pd.read_parquet(index_path)
            planner = np.load(arrays_path, allow_pickle=False)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"Failed to read planner export for {episode_uid}:{view_mode}: {exc}")
            continue

        row_indices = group["planner_row_index"].to_numpy(dtype=np.int64)
        if np.any(row_indices < 0):
            failures.append(f"Negative planner_row_index in enabled rows for {episode_uid}:{view_mode}")
            continue
        if len(planner_index) == 0 or int(row_indices.max(initial=-1)) >= len(planner_index):
            failures.append(
                f"planner_row_index out of range for {episode_uid}:{view_mode}: max={row_indices.max(initial=-1)} len={len(planner_index)}"
            )
            continue
        if "frame_position" in planner_index.columns:
            expected = group["frame_position"].to_numpy(dtype=np.int64)
            actual = planner_index.iloc[row_indices]["frame_position"].to_numpy(dtype=np.int64)
            if not np.array_equal(expected, actual):
                failures.append(f"Planner row/frame_position mismatch for {episode_uid}:{view_mode}")

        shape_checks = {
            _planner_key(_planner_state_key(value_variant), branch): belief_dim,
            _planner_key("progress_transition", branch): progress_dim,
            _planner_key("uncertainty_features", branch): uncertainty_dim,
            _planner_key("temporal_latent", branch): history_dim,
        }
        for key, dim in shape_checks.items():
            if key not in planner:
                failures.append(f"Planner key {key!r} missing for {episode_uid}:{view_mode}. Available={sorted(planner.files)}")
                continue
            array = planner[key]
            if array.shape[0] < len(planner_index):
                failures.append(f"Planner key {key!r} has too few rows for {episode_uid}:{view_mode}: {array.shape}")
            if array.shape[-1] != dim:
                failures.append(f"Planner key {key!r} has dim {array.shape[-1]}, expected {dim} for {episode_uid}:{view_mode}")


def _summarize_assignments(split_name: str, frame: pd.DataFrame, failures: list[str]) -> None:
    episodes = _split_episode_uids(frame)
    print(f"{split_name} rows: {len(frame):,}")
    print(f"{split_name} episodes: {len(episodes):,}")
    if "horizon_clipped_to_episode_end" in frame.columns:
        clipped = int(frame["horizon_clipped_to_episode_end"].astype(bool).sum())
        print(f"{split_name} clipped horizon rows: {clipped:,}")
        _check(clipped == 0, f"{split_name} contains {clipped} clipped horizon rows.", failures)
    if "sample_weight" in frame.columns and len(frame):
        values = frame["sample_weight"].to_numpy(dtype=np.float64)
        finite = np.isfinite(values)
        _check(bool(finite.all()), f"{split_name} has non-finite sample weights.", failures)
        _check(bool((values[finite] > 0.0).all()), f"{split_name} has non-positive sample weights.", failures)
        print(
            f"{split_name} sample weights: min={values[finite].min():.4f} "
            f"mean={values[finite].mean():.4f} max={values[finite].max():.4f}"
        )

    planner_table = _planner_episode_table(frame)
    if planner_table.empty:
        print(f"{split_name} planner: not present")
        return
    per_episode_counts = planner_table.groupby("episode_uid").size()
    multi = per_episode_counts[per_episode_counts > 1]
    if not multi.empty:
        failures.append(f"{split_name} has episodes with multiple planner assignments: {multi.head(10).to_dict()}")
    enabled = planner_table[planner_table["planner_enabled"]]
    disabled = planner_table[~planner_table["planner_enabled"]]
    print(f"{split_name} planner enabled episodes: {len(enabled):,}")
    print(f"{split_name} planner disabled episodes: {len(disabled):,}")
    if len(planner_table):
        print(f"{split_name} planner enabled ratio: {len(enabled) / float(len(planner_table)):.4f}")
    if not enabled.empty:
        print(f"{split_name} planner view modes: {enabled['view_mode'].value_counts().sort_index().to_dict()}")
        print(f"{split_name} planner branches: {enabled['planner_branch'].value_counts().sort_index().to_dict()}")
        print(f"{split_name} planner variants: {enabled['planner_value_variant'].value_counts().sort_index().to_dict()}")


def main() -> int:
    args = _resolve_args_from_config(parse_args())
    if args.manifest_root is None:
        raise ValueError("--manifest-root is required unless it is supplied by --config-name.")

    manifest_root = Path(args.manifest_root)
    manifest = _load_manifest(manifest_root)
    dataset_root = args.dataset_root
    if dataset_root is None:
        dataset_root = Path(manifest["dataset_root"])
    else:
        dataset_root = Path(dataset_root)

    train_frame = _load_split_frame(manifest_root, args.train_index_name)
    val_frame = _load_split_frame(manifest_root, args.val_index_name)
    failures: list[str] = []

    train_episodes = _split_episode_uids(train_frame)
    val_episodes = _split_episode_uids(val_frame)
    if args.expect_train_episodes is not None:
        _check(len(train_episodes) == args.expect_train_episodes, f"train episodes={len(train_episodes)}, expected {args.expect_train_episodes}", failures)
    if args.expect_val_episodes is not None:
        _check(len(val_episodes) == args.expect_val_episodes, f"val episodes={len(val_episodes)}, expected {args.expect_val_episodes}", failures)

    print("Origami comp action-chunk manifest verification")
    print(f"config_name  : {args.config_name}")
    print(f"manifest_root: {manifest_root}")
    print(f"dataset_root : {dataset_root}")
    _summarize_assignments("train", train_frame, failures)
    _summarize_assignments("val", val_frame, failures)

    train_planner = _planner_episode_table(train_frame)
    if not train_planner.empty:
        enabled = train_planner[train_planner["planner_enabled"]]
        if args.expect_planner_enabled_ratio is not None:
            actual = len(enabled) / float(len(train_planner))
            if abs(actual - args.expect_planner_enabled_ratio) > args.planner_enabled_ratio_tolerance:
                failures.append(
                    f"train planner enabled ratio is {actual:.4f}, expected {args.expect_planner_enabled_ratio:.4f} "
                    f"+/- {args.planner_enabled_ratio_tolerance:.4f}"
                )
        _check_ratio(
            enabled["view_mode"].value_counts().to_dict(),
            _parse_probability_map(args.expect_view_mode_probs),
            total=len(enabled),
            name="train planner view-mode",
            tolerance=float(args.ratio_tolerance),
            failures=failures,
        )
        _check_ratio(
            enabled["planner_branch"].value_counts().to_dict(),
            _parse_probability_map(args.expect_branch_probs),
            total=len(enabled),
            name="train planner branch",
            tolerance=float(args.ratio_tolerance),
            failures=failures,
        )
        if args.expect_planner_value_variant is not None and not enabled.empty:
            variants = set(str(value) for value in enabled["planner_value_variant"].drop_duplicates().tolist())
            _check(
                variants == {args.expect_planner_value_variant},
                f"train planner value variants are {sorted(variants)}, expected only {args.expect_planner_value_variant!r}",
                failures,
            )

    if args.check_dataset_files:
        all_episodes = sorted(set(train_episodes).union(val_episodes))
        _check_dataset_files(dataset_root, all_episodes, failures)

    if "planner_enabled" in train_frame.columns:
        _check_planner_exports(
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
        _check_planner_exports(
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
    print("\nOK: manifest is ready for comp action-chunk VLA training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
