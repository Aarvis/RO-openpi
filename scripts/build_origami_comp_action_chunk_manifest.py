from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build OpenPI train/val manifests for Origami pi0.5 direct action-chunk training. "
            "Rows point at current frame positions; the dataset loader builds action_65d chunks on demand."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-planner-manifest-root",
        type=Path,
        default=None,
        help="Optional root containing manifest.json with checkpoint-planner train/val episode split.",
    )
    parser.add_argument(
        "--ignore-checkpoint-planner-split",
        action="store_true",
        help=(
            "Do not reuse the train/val split from --checkpoint-planner-manifest-root. "
            "Use this when checkpoint-planner exports are used as features but VLA training "
            "should build its own split."
        ),
    )
    parser.add_argument("--num-val-episodes", type=int, default=12)
    parser.add_argument("--val-seed", type=int, default=1234)
    parser.add_argument(
        "--val-episode-uids",
        type=str,
        default="",
        help="Comma-separated explicit validation episode UIDs. Overrides --num-val-episodes.",
    )
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--action-horizon", type=int, default=10)
    parser.add_argument("--action-chunk-stride", type=int, default=1)
    parser.add_argument(
        "--keep-horizon-clipped",
        action="store_true",
        help="Keep tail frames that do not have a complete future action chunk. By default they are dropped.",
    )
    parser.add_argument(
        "--planner-export-root",
        type=Path,
        default=None,
        help="Optional checkpoint-planner VLA rollout export root.",
    )
    parser.add_argument(
        "--planner-assignment-mode",
        choices=("expand_view_modes", "episode_sampled"),
        default="expand_view_modes",
        help=(
            "How planner exports are attached. expand_view_modes keeps the historical behavior "
            "of one row per requested view mode. episode_sampled assigns one planner policy, or "
            "planner dropout, per episode."
        ),
    )
    parser.add_argument(
        "--train-planner-view-modes",
        nargs="*",
        default=None,
        help="Planner history/view modes to include for train. Defaults to all four exported modes.",
    )
    parser.add_argument(
        "--val-planner-view-modes",
        nargs="*",
        default=None,
        help="Planner history/view modes to include for val. Defaults to frame_stride_10 and random_mix.",
    )
    parser.add_argument("--planner-branch", type=str, default="posterior")
    parser.add_argument(
        "--planner-value-variant",
        choices=("final", "raw"),
        default="final",
        help="Planner belief/checkpoint variant to consume. final uses prior-gamma adjusted exports; raw uses raw logits/belief.",
    )
    parser.add_argument("--planner-assignment-seed", type=int, default=1234)
    parser.add_argument(
        "--planner-dropout-episode-prob",
        type=float,
        default=0.16,
        help="Episode-level probability of disabling planner prefixes in episode_sampled mode.",
    )
    parser.add_argument(
        "--planner-view-mode-probs",
        nargs="*",
        default=None,
        help=(
            "View-mode probability map for episode_sampled mode, e.g. "
            "frame_stride_10=0.5 random_mix=0.25 fixed_7=0.25 fixed_15=0.0. "
            "Defaults to that distribution."
        ),
    )
    parser.add_argument(
        "--planner-branch-probs",
        nargs="*",
        default=None,
        help=(
            "Planner branch probability map for episode_sampled mode, e.g. posterior=0.5 prior=0.5. "
            "Defaults to posterior=0.5 prior=0.5."
        ),
    )
    parser.add_argument("--planner-index-name", type=str, default="planner_vla_rollout_index.parquet")
    parser.add_argument("--planner-arrays-name", type=str, default="planner_vla_rollout_features.npz")
    parser.add_argument("--planner-complete-marker-name", type=str, default="export_complete.marker")
    parser.add_argument(
        "--require-planner-complete-marker",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require export_complete.marker for each requested planner export.",
    )
    parser.add_argument(
        "--allow-missing-planner-rows",
        action="store_true",
        help="Drop action-chunk rows without matching planner export rows instead of raising.",
    )
    parser.add_argument(
        "--speed-weighting",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Compute checkpoint-speed sample weights. Each action-chunk row receives the mean frame weight over "
            "its future action horizon."
        ),
    )
    parser.add_argument(
        "--speed-label-relpaths",
        nargs="*",
        default=["labels/checkpoints.json", "labels/transfer_checkpoints.json"],
        help="Episode-relative label files to try, in priority order, when --speed-weighting is enabled.",
    )
    parser.add_argument(
        "--speed-stats-split",
        choices=("train", "val", "all"),
        default="train",
        help="Episode split used to compute per-checkpoint median durations.",
    )
    parser.add_argument("--speed-semantic-group-size", type=int, default=2)
    parser.add_argument(
        "--speed-final-unpaired-policy",
        choices=("keep", "drop", "error"),
        default="keep",
        help="How to handle a trailing label group smaller than --speed-semantic-group-size.",
    )
    parser.add_argument(
        "--speed-done-policy",
        choices=("neutral", "weighted"),
        default="neutral",
        help="Use neutral weight 1.0 for done/reset spans, or include them in speed weighting.",
    )
    parser.add_argument("--speed-alpha", type=float, default=1.5)
    parser.add_argument("--speed-min-weight", type=float, default=0.5)
    parser.add_argument("--speed-max-weight", type=float, default=2.0)
    parser.add_argument("--speed-epsilon-frames", type=float, default=1.0e-6)
    parser.add_argument(
        "--speed-weight-val",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also write speed-efficiency weights into validation rows. By default validation stays unweighted.",
    )
    parser.add_argument("--train-index-name", type=str, default="train_index.parquet")
    parser.add_argument("--val-index-name", type=str, default="val_index.parquet")
    return parser.parse_args()


@dataclass(frozen=True)
class SemanticCheckpointSpan:
    semantic_checkpoint_id: int
    label: str
    start_frame: int
    end_frame_exclusive: int
    num_frames: int
    is_done: bool


@dataclass(frozen=True)
class EpisodeLabelInfo:
    episode_uid: str
    label_path: Path
    semantic_spans: tuple[SemanticCheckpointSpan, ...]


@dataclass(frozen=True)
class SpeedWeightSettings:
    enabled: bool
    label_priority_relpaths: tuple[str, ...]
    stats_split: str
    semantic_group_size: int
    final_unpaired_policy: str
    done_policy: str
    alpha: float
    w_min: float
    w_max: float
    epsilon_frames: float
    weight_val: bool


@dataclass(frozen=True)
class SpeedWeightContext:
    settings: SpeedWeightSettings
    label_infos: dict[str, EpisodeLabelInfo]
    median_durations: dict[int, float]


@dataclass(frozen=True)
class PlannerAssignment:
    enabled: bool
    view_mode: str = ""
    branch: str = ""
    value_variant: str = "final"


def _load_episode_table(dataset_root: Path) -> pd.DataFrame:
    path = dataset_root / "metadata" / "episodes.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Episode table not found: {path}")
    frame = pd.read_parquet(path)
    required = {"episode_uid", "num_frames"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"{path} is missing required columns: {sorted(missing)}")
    return frame.sort_values("episode_uid").reset_index(drop=True)


def _load_split_from_checkpoint_planner(
    checkpoint_planner_manifest_root: Path | None,
    episode_uids: list[str],
) -> tuple[list[str], list[str]] | None:
    if checkpoint_planner_manifest_root is None:
        return None
    path = checkpoint_planner_manifest_root / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint-planner split manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    split = manifest.get("split", {})
    train = [str(uid) for uid in split.get("train_episode_uids", [])]
    val = [str(uid) for uid in split.get("val_episode_uids", [])]
    available = set(episode_uids)
    train = [uid for uid in train if uid in available]
    val = [uid for uid in val if uid in available]
    if not train:
        raise RuntimeError(f"No checkpoint-planner train episodes overlap with {checkpoint_planner_manifest_root}")
    return train, val


def _make_split(args: argparse.Namespace, episode_uids: list[str]) -> tuple[list[str], list[str]]:
    reused = None
    if not bool(args.ignore_checkpoint_planner_split):
        reused = _load_split_from_checkpoint_planner(args.checkpoint_planner_manifest_root, episode_uids)
    if reused is not None:
        return reused

    explicit_val = [uid.strip() for uid in args.val_episode_uids.split(",") if uid.strip()]
    if explicit_val:
        missing = sorted(set(explicit_val).difference(episode_uids))
        if missing:
            raise RuntimeError(f"Explicit validation episodes are not present in dataset: {missing}")
        val_set = set(explicit_val)
    else:
        rng = np.random.default_rng(int(args.val_seed))
        shuffled = np.asarray(episode_uids, dtype=object)
        rng.shuffle(shuffled)
        val_set = set(str(uid) for uid in shuffled[: max(0, int(args.num_val_episodes))])

    train = [uid for uid in episode_uids if uid not in val_set]
    val = [uid for uid in episode_uids if uid in val_set]
    if not train:
        raise RuntimeError("Train split is empty.")
    return train, val


def _parse_probability_map(
    values: list[str] | None,
    *,
    default: dict[str, float],
    option_name: str,
) -> dict[str, float]:
    raw_values = list(values or [])
    if not raw_values:
        raw = dict(default)
    else:
        raw: dict[str, float] = {}
        for value in raw_values:
            for item in str(value).split(","):
                item = item.strip()
                if not item:
                    continue
                key, sep, weight = item.partition("=")
                if not sep:
                    raise ValueError(f"{option_name} entries must be name=value, got {item!r}")
                key = key.strip()
                if not key:
                    raise ValueError(f"{option_name} contains an empty key in {item!r}")
                if key in raw:
                    raise ValueError(f"{option_name} contains duplicate key {key!r}")
                raw[key] = float(weight)
    if not raw:
        raise ValueError(f"{option_name} must contain at least one entry.")
    if any(value < 0.0 for value in raw.values()):
        raise ValueError(f"{option_name} probabilities must be non-negative: {raw}")
    total = float(sum(raw.values()))
    if total <= 0.0:
        raise ValueError(f"{option_name} probabilities must sum to > 0: {raw}")
    return {key: float(value) / total for key, value in raw.items()}


def _exact_category_counts(probabilities: dict[str, float], total: int) -> dict[str, int]:
    if total <= 0:
        return {key: 0 for key in probabilities}
    keys = list(probabilities)
    expected = {key: float(probabilities[key]) * total for key in keys}
    counts = {key: int(np.floor(expected[key])) for key in keys}
    remainder = total - sum(counts.values())
    order = sorted(keys, key=lambda key: (expected[key] - counts[key], probabilities[key]), reverse=True)
    for key in order[:remainder]:
        counts[key] += 1
    return counts


def _assign_categories(
    episode_uids: list[str],
    probabilities: dict[str, float],
    *,
    seed: int,
) -> dict[str, str]:
    if not episode_uids:
        return {}
    shuffled = np.asarray(episode_uids, dtype=object)
    np.random.default_rng(int(seed)).shuffle(shuffled)
    counts = _exact_category_counts(probabilities, len(episode_uids))
    assignments: dict[str, str] = {}
    offset = 0
    for key in probabilities:
        count = counts[key]
        for uid in shuffled[offset : offset + count].tolist():
            assignments[str(uid)] = key
        offset += count
    if len(assignments) != len(episode_uids):
        raise RuntimeError("Planner assignment failed to cover every episode.")
    return assignments


def _build_planner_assignments(
    episode_uids: list[str],
    *,
    enabled: bool,
    assignment_mode: str,
    seed: int,
    dropout_episode_prob: float,
    view_mode_probs: dict[str, float],
    branch_probs: dict[str, float],
    value_variant: str,
) -> dict[str, PlannerAssignment] | None:
    if not enabled or assignment_mode == "expand_view_modes":
        return None
    if assignment_mode != "episode_sampled":
        raise ValueError(f"Unsupported planner assignment mode: {assignment_mode!r}")
    if not 0.0 <= float(dropout_episode_prob) <= 1.0:
        raise ValueError(f"--planner-dropout-episode-prob must satisfy 0 <= p <= 1, got {dropout_episode_prob}")

    shuffled = np.asarray(episode_uids, dtype=object)
    np.random.default_rng(int(seed)).shuffle(shuffled)
    disabled_count = int(round(len(episode_uids) * float(dropout_episode_prob)))
    disabled = {str(uid) for uid in shuffled[:disabled_count].tolist()}
    enabled_uids = [uid for uid in episode_uids if uid not in disabled]

    view_assignments = _assign_categories(enabled_uids, view_mode_probs, seed=int(seed) + 1009)
    branch_assignments = _assign_categories(enabled_uids, branch_probs, seed=int(seed) + 2003)

    assignments: dict[str, PlannerAssignment] = {}
    for uid in episode_uids:
        if uid in disabled:
            assignments[uid] = PlannerAssignment(enabled=False, value_variant=value_variant)
        else:
            assignments[uid] = PlannerAssignment(
                enabled=True,
                view_mode=view_assignments[uid],
                branch=branch_assignments[uid],
                value_variant=value_variant,
            )
    return assignments


def _planner_assignment_summary(assignments: dict[str, PlannerAssignment] | None) -> dict[str, Any]:
    if assignments is None:
        return {"mode": "expand_view_modes"}
    enabled = [assignment for assignment in assignments.values() if assignment.enabled]
    disabled_count = len(assignments) - len(enabled)

    def counts(values: list[str]) -> dict[str, int]:
        result: dict[str, int] = {}
        for value in values:
            result[value] = result.get(value, 0) + 1
        return dict(sorted(result.items()))

    return {
        "mode": "episode_sampled",
        "episodes": len(assignments),
        "enabled_episodes": len(enabled),
        "disabled_episodes": disabled_count,
        "view_modes": counts([assignment.view_mode for assignment in enabled]),
        "branches": counts([assignment.branch for assignment in enabled]),
        "value_variants": counts([assignment.value_variant for assignment in enabled]),
    }


def _load_speed_weight_settings(args: argparse.Namespace) -> SpeedWeightSettings:
    label_relpaths = tuple(str(value) for value in args.speed_label_relpaths if str(value).strip())
    if not label_relpaths:
        raise ValueError("--speed-label-relpaths must contain at least one path when --speed-weighting is enabled.")
    if args.speed_semantic_group_size < 1:
        raise ValueError("--speed-semantic-group-size must be >= 1.")
    if args.speed_alpha < 0.0:
        raise ValueError("--speed-alpha must be >= 0.")
    if args.speed_min_weight <= 0.0 or args.speed_max_weight <= 0.0:
        raise ValueError("--speed-min-weight and --speed-max-weight must be > 0.")
    if args.speed_max_weight < args.speed_min_weight:
        raise ValueError("--speed-max-weight must be >= --speed-min-weight.")
    if args.speed_epsilon_frames <= 0.0:
        raise ValueError("--speed-epsilon-frames must be > 0.")
    return SpeedWeightSettings(
        enabled=bool(args.speed_weighting),
        label_priority_relpaths=label_relpaths,
        stats_split=str(args.speed_stats_split),
        semantic_group_size=int(args.speed_semantic_group_size),
        final_unpaired_policy=str(args.speed_final_unpaired_policy),
        done_policy=str(args.speed_done_policy),
        alpha=float(args.speed_alpha),
        w_min=float(args.speed_min_weight),
        w_max=float(args.speed_max_weight),
        epsilon_frames=float(args.speed_epsilon_frames),
        weight_val=bool(args.speed_weight_val),
    )


def _resolve_label_path(episode_root: Path, relpaths: tuple[str, ...]) -> Path | None:
    for relpath in relpaths:
        path = episode_root / relpath
        if path.exists():
            return path
    return None


def _load_label_info(episode_root: Path, settings: SpeedWeightSettings) -> EpisodeLabelInfo:
    label_path = _resolve_label_path(episode_root, settings.label_priority_relpaths)
    if label_path is None:
        raise FileNotFoundError(
            f"No checkpoint label file found for {episode_root.name} under {settings.label_priority_relpaths}"
        )

    payload = json.loads(label_path.read_text(encoding="utf-8"))
    segments = payload.get("segments", [])
    if not isinstance(segments, list) or not segments:
        raise ValueError(f"{label_path}: expected a non-empty 'segments' list")
    segments = sorted(segments, key=lambda row: int(row["start_frame"]))

    spans: list[SemanticCheckpointSpan] = []
    semantic_id = 0
    group_size = settings.semantic_group_size
    for start_idx in range(0, len(segments), group_size):
        chunk = segments[start_idx : start_idx + group_size]
        if len(chunk) < group_size:
            if settings.final_unpaired_policy == "drop":
                break
            if settings.final_unpaired_policy == "error":
                raise ValueError(
                    f"{label_path}: trailing segment group has size {len(chunk)} but "
                    f"--speed-semantic-group-size={group_size}"
                )

        start_frame = int(chunk[0]["start_frame"])
        end_frame_exclusive = int(chunk[-1]["end_frame_exclusive"])
        if end_frame_exclusive <= start_frame:
            raise ValueError(
                f"{label_path}: invalid semantic span [{start_frame}, {end_frame_exclusive}) "
                f"for semantic checkpoint {semantic_id}"
            )

        label = str(chunk[0].get("label", ""))
        is_done = len(chunk) < group_size or label.strip().lower() == "reset to default position"
        spans.append(
            SemanticCheckpointSpan(
                semantic_checkpoint_id=semantic_id,
                label=label,
                start_frame=start_frame,
                end_frame_exclusive=end_frame_exclusive,
                num_frames=end_frame_exclusive - start_frame,
                is_done=is_done,
            )
        )
        semantic_id += 1

    if not spans:
        raise ValueError(f"{label_path}: no semantic checkpoint spans were produced")
    return EpisodeLabelInfo(
        episode_uid=episode_root.name,
        label_path=label_path,
        semantic_spans=tuple(spans),
    )


def _build_duration_medians(
    label_infos: dict[str, EpisodeLabelInfo],
    stats_episode_uids: list[str],
    settings: SpeedWeightSettings,
) -> dict[int, float]:
    values_by_semantic_id: dict[int, list[float]] = {}
    for episode_uid in stats_episode_uids:
        info = label_infos[episode_uid]
        for span in info.semantic_spans:
            if span.is_done and settings.done_policy == "neutral":
                continue
            values_by_semantic_id.setdefault(span.semantic_checkpoint_id, []).append(float(span.num_frames))

    return {
        semantic_checkpoint_id: float(np.median(np.asarray(values, dtype=np.float64)))
        for semantic_checkpoint_id, values in values_by_semantic_id.items()
        if values
    }


def _semantic_weight(
    span: SemanticCheckpointSpan,
    median_durations: dict[int, float],
    settings: SpeedWeightSettings,
) -> tuple[float, float]:
    if span.is_done and settings.done_policy == "neutral":
        return 1.0, 1.0
    median_duration = median_durations.get(span.semantic_checkpoint_id)
    if median_duration is None:
        return 1.0, 1.0
    ratio = float(median_duration / max(float(span.num_frames), settings.epsilon_frames))
    weight = float(np.clip(ratio**settings.alpha, settings.w_min, settings.w_max))
    return weight, ratio


def _build_speed_weight_context(
    *,
    dataset_root: Path,
    train_episodes: list[str],
    val_episodes: list[str],
    settings: SpeedWeightSettings,
) -> SpeedWeightContext | None:
    if not settings.enabled:
        return None

    episode_uid_set = set(train_episodes)
    if settings.weight_val or settings.stats_split in {"val", "all"}:
        episode_uid_set.update(val_episodes)
    episode_uids = sorted(episode_uid_set)
    label_infos: dict[str, EpisodeLabelInfo] = {}
    for episode_uid in tqdm(
        episode_uids,
        desc="Load checkpoint labels for speed weights",
        unit="episode",
        dynamic_ncols=True,
    ):
        label_infos[episode_uid] = _load_label_info(dataset_root / "episodes" / episode_uid, settings)

    if settings.stats_split == "train":
        stats_episode_uids = train_episodes
    elif settings.stats_split == "val":
        stats_episode_uids = val_episodes
    else:
        stats_episode_uids = [*train_episodes, *val_episodes]
    stats_episode_uids = [uid for uid in stats_episode_uids if uid in label_infos]
    if not stats_episode_uids:
        raise RuntimeError(f"No episodes are available for speed-weight stats split {settings.stats_split!r}.")

    median_durations = _build_duration_medians(label_infos, stats_episode_uids, settings)
    if not median_durations:
        raise RuntimeError("No semantic checkpoint duration medians were computed for speed weighting.")

    return SpeedWeightContext(
        settings=settings,
        label_infos=label_infos,
        median_durations=median_durations,
    )


def _frame_speed_weights(
    *,
    episode_uid: str,
    num_frames: int,
    speed_context: SpeedWeightContext,
) -> tuple[np.ndarray, np.ndarray]:
    info = speed_context.label_infos[episode_uid]
    frame_weights = np.ones((num_frames,), dtype=np.float32)
    frame_semantic_ids = np.full((num_frames,), -1, dtype=np.int32)
    for span in info.semantic_spans:
        if span.end_frame_exclusive > num_frames:
            raise ValueError(
                f"{info.label_path}: semantic span end {span.end_frame_exclusive} exceeds available frame count "
                f"{num_frames} for {episode_uid}"
            )
        weight, _ratio = _semantic_weight(span, speed_context.median_durations, speed_context.settings)
        frame_weights[span.start_frame : span.end_frame_exclusive] = weight
        frame_semantic_ids[span.start_frame : span.end_frame_exclusive] = span.semantic_checkpoint_id
    return frame_weights, frame_semantic_ids


def _action_chunk_sample_weights(
    *,
    frame_positions: np.ndarray,
    action_horizon: int,
    action_chunk_stride: int,
    action_len: int,
    frame_weights: np.ndarray,
) -> np.ndarray:
    offsets = np.arange(action_horizon, dtype=np.int64) * int(action_chunk_stride)
    horizon_positions = frame_positions[:, None].astype(np.int64) + offsets[None, :]
    valid_limit = min(int(action_len), int(frame_weights.shape[0]))
    valid = horizon_positions < valid_limit
    safe_positions = np.clip(horizon_positions, 0, max(valid_limit - 1, 0))
    values = frame_weights[safe_positions] * valid.astype(np.float32)
    denom = np.maximum(valid.sum(axis=1).astype(np.float32), 1.0)
    return (values.sum(axis=1) / denom).astype(np.float32)


def _weight_stats(frame: pd.DataFrame) -> dict[str, float | int | None]:
    if frame.empty or "sample_weight" not in frame.columns:
        return {"count": 0, "min": None, "max": None, "mean": None}
    values = frame["sample_weight"].to_numpy(dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def _read_frame_index(arrays_root: Path, num_frames: int) -> np.ndarray:
    path = arrays_root / "frame_index.npy"
    if path.exists():
        values = np.asarray(np.load(path, mmap_mode="r"), dtype=np.int64)
        if values.shape[0] < num_frames:
            raise ValueError(f"{path} has {values.shape[0]} rows, expected at least {num_frames}")
        return values
    return np.arange(num_frames, dtype=np.int64)


def _read_timestamps(arrays_root: Path, num_frames: int) -> np.ndarray:
    path = arrays_root / "timestamps.npy"
    if path.exists():
        values = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32)
        if values.shape[0] < num_frames:
            raise ValueError(f"{path} has {values.shape[0]} rows, expected at least {num_frames}")
        return values
    return np.arange(num_frames, dtype=np.float32)


def _default_train_planner_view_modes() -> list[str]:
    return ["frame_stride_10", "fixed_7", "fixed_15", "random_mix"]


def _default_val_planner_view_modes() -> list[str]:
    return ["frame_stride_10", "random_mix"]


def _resolve_planner_view_modes(values: list[str] | None, *, split_name: str) -> list[str]:
    if values:
        return [str(value) for value in values]
    if split_name == "val":
        return _default_val_planner_view_modes()
    return _default_train_planner_view_modes()


def _attach_planner_rows(
    *,
    base_frame: pd.DataFrame,
    planner_export_root: Path,
    episode_uid: str,
    view_modes: list[str],
    planner_branch: str,
    planner_value_variant: str,
    planner_index_name: str,
    planner_arrays_name: str,
    planner_complete_marker_name: str,
    require_complete_marker: bool,
    allow_missing_rows: bool,
) -> list[pd.DataFrame]:
    expanded: list[pd.DataFrame] = []
    for view_mode in view_modes:
        planner_output_dir = planner_export_root / episode_uid / view_mode
        planner_index_path = planner_output_dir / planner_index_name
        planner_arrays_path = planner_output_dir / planner_arrays_name
        marker_path = planner_output_dir / planner_complete_marker_name

        missing = []
        if not planner_index_path.exists():
            missing.append(str(planner_index_path))
        if not planner_arrays_path.exists():
            missing.append(str(planner_arrays_path))
        if require_complete_marker and not marker_path.exists():
            missing.append(str(marker_path))
        if missing:
            raise FileNotFoundError(f"Missing planner export files for {episode_uid}:{view_mode}: {missing}")

        planner_index = pd.read_parquet(planner_index_path)
        if "planner_row_index" not in planner_index.columns:
            planner_index = planner_index.copy()
            planner_index["planner_row_index"] = planner_index.index.astype("int64")
        required = {"frame_position", "planner_row_index"}
        missing_columns = required.difference(planner_index.columns)
        if missing_columns:
            raise KeyError(f"{planner_index_path} is missing required columns: {sorted(missing_columns)}")

        planner_index = planner_index[["frame_position", "planner_row_index"]].copy()
        planner_index["frame_position"] = planner_index["frame_position"].astype("int64")
        planner_index["planner_row_index"] = planner_index["planner_row_index"].astype("int64")
        merged = base_frame.merge(planner_index, on="frame_position", how="left", validate="one_to_one")
        missing_count = int(merged["planner_row_index"].isna().sum())
        if missing_count and not allow_missing_rows:
            raise RuntimeError(
                f"{episode_uid}:{view_mode} is missing planner rows for {missing_count} action-chunk frames."
            )
        if missing_count:
            merged = merged[merged["planner_row_index"].notna()].reset_index(drop=True)
        if merged.empty:
            continue
        merged["planner_row_index"] = merged["planner_row_index"].astype("int64")
        merged["planner_enabled"] = True
        merged["view_mode"] = view_mode
        merged["planner_branch"] = planner_branch
        merged["planner_value_variant"] = planner_value_variant
        merged["planner_output_dir"] = str(planner_output_dir)
        expanded.append(merged)
    return expanded


def _attach_disabled_planner_rows(base_frame: pd.DataFrame) -> pd.DataFrame:
    frame = base_frame.copy()
    frame["planner_enabled"] = False
    frame["view_mode"] = "planner_disabled"
    frame["planner_branch"] = ""
    frame["planner_value_variant"] = "none"
    frame["planner_output_dir"] = ""
    frame["planner_row_index"] = np.full(len(frame), -1, dtype=np.int64)
    return frame


def _build_split_frame(
    *,
    dataset_root: Path,
    split_name: str,
    episode_table: pd.DataFrame,
    episode_uids: list[str],
    frame_stride: int,
    action_horizon: int,
    action_chunk_stride: int,
    drop_horizon_clipped: bool,
    planner_export_root: Path | None,
    planner_view_modes: list[str],
    planner_branch: str,
    planner_value_variant: str,
    planner_assignments: dict[str, PlannerAssignment] | None,
    planner_index_name: str,
    planner_arrays_name: str,
    planner_complete_marker_name: str,
    require_planner_complete_marker: bool,
    allow_missing_planner_rows: bool,
    speed_context: SpeedWeightContext | None,
    use_speed_weights: bool,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    columns = [
        "episode_uid",
        "split",
        "frame_position",
        "frame_index",
        "timestamp",
        "sample_weight",
        "action_horizon",
        "action_chunk_stride",
        "horizon_clipped_to_episode_end",
    ]
    episode_table = episode_table.set_index("episode_uid", drop=False)
    for episode_uid in tqdm(episode_uids, desc=f"Build {split_name} manifest", unit="episode", dynamic_ncols=True):
        episode_meta = episode_table.loc[episode_uid]
        episode_root = dataset_root / "episodes" / episode_uid
        arrays_root = episode_root / "arrays"
        state_path = arrays_root / "state_65d.npy"
        action_path = arrays_root / "action_65d.npy"
        tactile_path = arrays_root / "tactile_60d.npy"
        required = [state_path, action_path, tactile_path]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing required arrays for {episode_uid}: {missing}")

        metadata_num_frames = int(episode_meta["num_frames"])
        state_len = int(np.load(state_path, mmap_mode="r").shape[0])
        action_len = int(np.load(action_path, mmap_mode="r").shape[0])
        tactile_len = int(np.load(tactile_path, mmap_mode="r").shape[0])
        num_frames = min(metadata_num_frames, state_len, tactile_len)
        if drop_horizon_clipped:
            max_start = min(num_frames, action_len - (action_horizon - 1) * action_chunk_stride)
            frame_positions = np.arange(max(0, max_start), dtype=np.int64)
        else:
            frame_positions = np.arange(num_frames, dtype=np.int64)
        frame_positions = frame_positions[:: max(1, int(frame_stride))]
        if frame_positions.size == 0:
            continue

        if speed_context is not None and use_speed_weights:
            frame_speed_weights, _frame_semantic_ids = _frame_speed_weights(
                episode_uid=episode_uid,
                num_frames=num_frames,
                speed_context=speed_context,
            )
            sample_weights = _action_chunk_sample_weights(
                frame_positions=frame_positions,
                action_horizon=action_horizon,
                action_chunk_stride=action_chunk_stride,
                action_len=action_len,
                frame_weights=frame_speed_weights,
            )
        else:
            sample_weights = np.ones(frame_positions.shape[0], dtype=np.float32)

        frame_index = _read_frame_index(arrays_root, num_frames)[frame_positions]
        timestamps = _read_timestamps(arrays_root, num_frames)[frame_positions]
        base_frame = pd.DataFrame(
            {
                "episode_uid": episode_uid,
                "split": split_name,
                "frame_position": frame_positions.astype("int64"),
                "frame_index": frame_index.astype("int64"),
                "timestamp": timestamps.astype("float32"),
                "sample_weight": sample_weights.astype("float32"),
                "action_horizon": np.full(frame_positions.shape[0], action_horizon, dtype=np.int64),
                "action_chunk_stride": np.full(frame_positions.shape[0], action_chunk_stride, dtype=np.int64),
                "horizon_clipped_to_episode_end": (
                    frame_positions + (action_horizon - 1) * action_chunk_stride >= action_len
                ),
            }
        )
        if planner_export_root is None:
            rows.append(base_frame)
            continue
        if planner_assignments is not None:
            assignment = planner_assignments.get(episode_uid)
            if assignment is None:
                raise RuntimeError(f"No planner assignment was generated for {episode_uid}")
            if not assignment.enabled:
                rows.append(_attach_disabled_planner_rows(base_frame))
                continue
            rows.extend(
                _attach_planner_rows(
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
            _attach_planner_rows(
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
        return pd.DataFrame(columns=columns)
    return pd.concat(rows, axis=0, ignore_index=True)


def main() -> int:
    args = parse_args()
    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be positive.")
    if args.action_horizon <= 0:
        raise ValueError("--action-horizon must be positive.")
    if args.action_chunk_stride <= 0:
        raise ValueError("--action-chunk-stride must be positive.")

    dataset_root = args.dataset_root
    output_root = args.output_root
    planner_export_root = args.planner_export_root
    planner_view_mode_probs = _parse_probability_map(
        args.planner_view_mode_probs,
        default={"frame_stride_10": 0.5, "random_mix": 0.25, "fixed_7": 0.25, "fixed_15": 0.0},
        option_name="--planner-view-mode-probs",
    )
    planner_branch_probs = _parse_probability_map(
        args.planner_branch_probs,
        default={"posterior": 0.5, "prior": 0.5},
        option_name="--planner-branch-probs",
    )
    if planner_export_root is not None and args.planner_assignment_mode == "episode_sampled":
        train_planner_view_modes = [key for key, value in planner_view_mode_probs.items() if value > 0.0]
        val_planner_view_modes = [key for key, value in planner_view_mode_probs.items() if value > 0.0]
    else:
        train_planner_view_modes = (
            _resolve_planner_view_modes(args.train_planner_view_modes, split_name="train")
            if planner_export_root is not None
            else []
        )
        val_planner_view_modes = (
            _resolve_planner_view_modes(args.val_planner_view_modes, split_name="val")
            if planner_export_root is not None
            else []
        )
    episode_table = _load_episode_table(dataset_root)
    episode_uids = [str(uid) for uid in episode_table["episode_uid"].tolist()]
    train_episodes, val_episodes = _make_split(args, episode_uids)
    speed_settings = _load_speed_weight_settings(args)
    speed_context = _build_speed_weight_context(
        dataset_root=dataset_root,
        train_episodes=train_episodes,
        val_episodes=val_episodes,
        settings=speed_settings,
    )

    train_planner_assignments = _build_planner_assignments(
        train_episodes,
        enabled=planner_export_root is not None,
        assignment_mode=str(args.planner_assignment_mode),
        seed=int(args.planner_assignment_seed),
        dropout_episode_prob=float(args.planner_dropout_episode_prob),
        view_mode_probs=planner_view_mode_probs,
        branch_probs=planner_branch_probs,
        value_variant=str(args.planner_value_variant),
    )
    val_planner_assignments = _build_planner_assignments(
        val_episodes,
        enabled=planner_export_root is not None,
        assignment_mode=str(args.planner_assignment_mode),
        seed=int(args.planner_assignment_seed) + 3001,
        dropout_episode_prob=float(args.planner_dropout_episode_prob),
        view_mode_probs=planner_view_mode_probs,
        branch_probs=planner_branch_probs,
        value_variant=str(args.planner_value_variant),
    )

    drop_horizon_clipped = not bool(args.keep_horizon_clipped)
    train_frame = _build_split_frame(
        dataset_root=dataset_root,
        split_name="train",
        episode_table=episode_table,
        episode_uids=train_episodes,
        frame_stride=int(args.frame_stride),
        action_horizon=int(args.action_horizon),
        action_chunk_stride=int(args.action_chunk_stride),
        drop_horizon_clipped=drop_horizon_clipped,
        planner_export_root=planner_export_root,
        planner_view_modes=train_planner_view_modes,
        planner_branch=str(args.planner_branch),
        planner_value_variant=str(args.planner_value_variant),
        planner_assignments=train_planner_assignments,
        planner_index_name=str(args.planner_index_name),
        planner_arrays_name=str(args.planner_arrays_name),
        planner_complete_marker_name=str(args.planner_complete_marker_name),
        require_planner_complete_marker=bool(args.require_planner_complete_marker),
        allow_missing_planner_rows=bool(args.allow_missing_planner_rows),
        speed_context=speed_context,
        use_speed_weights=speed_context is not None,
    )
    val_frame = _build_split_frame(
        dataset_root=dataset_root,
        split_name="val",
        episode_table=episode_table,
        episode_uids=val_episodes,
        frame_stride=int(args.frame_stride),
        action_horizon=int(args.action_horizon),
        action_chunk_stride=int(args.action_chunk_stride),
        drop_horizon_clipped=drop_horizon_clipped,
        planner_export_root=planner_export_root,
        planner_view_modes=val_planner_view_modes,
        planner_branch=str(args.planner_branch),
        planner_value_variant=str(args.planner_value_variant),
        planner_assignments=val_planner_assignments,
        planner_index_name=str(args.planner_index_name),
        planner_arrays_name=str(args.planner_arrays_name),
        planner_complete_marker_name=str(args.planner_complete_marker_name),
        require_planner_complete_marker=bool(args.require_planner_complete_marker),
        allow_missing_planner_rows=bool(args.allow_missing_planner_rows),
        speed_context=speed_context,
        use_speed_weights=speed_context is not None and speed_settings.weight_val,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    train_path = output_root / args.train_index_name
    val_path = output_root / args.val_index_name
    train_frame.to_parquet(train_path, index=False)
    val_frame.to_parquet(val_path, index=False)

    manifest: dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "train_index_name": train_path.name,
        "val_index_name": val_path.name,
        "frame_stride": int(args.frame_stride),
        "action_horizon": int(args.action_horizon),
        "action_chunk_stride": int(args.action_chunk_stride),
        "drop_horizon_clipped": drop_horizon_clipped,
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
            "train_assignment_summary": _planner_assignment_summary(train_planner_assignments),
            "val_assignment_summary": _planner_assignment_summary(val_planner_assignments),
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
            "train": _weight_stats(train_frame),
            "val": _weight_stats(val_frame),
        },
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Origami action-chunk manifest")
    print(f"  dataset_root  : {dataset_root}")
    print(f"  output_root   : {output_root}")
    print(f"  train_rows    : {len(train_frame)}")
    print(f"  val_rows      : {len(val_frame)}")
    print(f"  train_episodes: {len(train_episodes)}")
    print(f"  val_episodes  : {len(val_episodes)}")
    if speed_context is not None:
        print("  speed weights : enabled")
        print(
            "  speed params  : "
            f"alpha={speed_settings.alpha} w_min={speed_settings.w_min} w_max={speed_settings.w_max} "
            f"stats_split={speed_settings.stats_split} val_weighted={speed_settings.weight_val}"
        )
        train_stats = _weight_stats(train_frame)
        val_stats = _weight_stats(val_frame)
        print(
            "  train weights : "
            f"mean={train_stats['mean']} min={train_stats['min']} max={train_stats['max']}"
        )
        print(
            "  val weights   : "
            f"mean={val_stats['mean']} min={val_stats['min']} max={val_stats['max']}"
        )
    if planner_export_root is not None:
        print(f"  planner_root  : {planner_export_root}")
        print(f"  planner_mode  : {args.planner_assignment_mode}")
        print(f"  train_modes   : {train_planner_view_modes}")
        print(f"  val_modes     : {val_planner_view_modes}")
        print(f"  value_variant : {args.planner_value_variant}")
        if train_planner_assignments is None:
            print(f"  planner_branch: {args.planner_branch}")
        else:
            print(f"  train_planner : {_planner_assignment_summary(train_planner_assignments)}")
            print(f"  val_planner   : {_planner_assignment_summary(val_planner_assignments)}")
    print(f"  train_file    : {train_path}")
    print(f"  val_file      : {val_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
