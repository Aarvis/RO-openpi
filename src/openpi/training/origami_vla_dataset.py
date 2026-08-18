from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


@dataclasses.dataclass(frozen=True)
class OrigamiVlaSettings:
    dataset_root: str
    manifest_root: str
    manifest_train_name: str = "train_index.parquet"
    manifest_val_name: str = "val_index.parquet"
    local_target_npz_name: str = "local_delta_reached_state_targets_K15_include_current_restrict_true_state.npz"
    planner_arrays_filename: str = "planner_vla_rollout_features.npz"
    planner_index_filename: str = "planner_vla_rollout_index.parquet"
    max_control_points: int = 18
    max_span_count: int = 15
    degree: int = 3
    state_dim: int = 65
    action_dim: int = 65
    prompt: str = "Fold paper into airplane"
    image_modalities: dict[str, str] = dataclasses.field(
        default_factory=lambda: {
            "ooi_rgb": "videos/ooi.mp4",
            "base_0_rgb": "videos/head_left.mp4",
            "left_wrist_0_rgb": "videos/wrist_left.mp4",
            "right_wrist_0_rgb": "videos/wrist_right.mp4",
        }
    )
    fail_on_missing_modalities: bool = True
    max_rows: int | None = None


def _ensure_path(value: str | Path) -> Path:
    return value if isinstance(value, Path) else Path(value)


def load_manifest_rows(settings: OrigamiVlaSettings, split: str) -> pd.DataFrame:
    manifest_root = _ensure_path(settings.manifest_root)
    split_name = split.strip().lower()
    if split_name == "train":
        path = manifest_root / settings.manifest_train_name
    elif split_name == "val":
        path = manifest_root / settings.manifest_val_name
    elif split_name == "all":
        train_frame = pd.read_parquet(manifest_root / settings.manifest_train_name)
        val_frame = pd.read_parquet(manifest_root / settings.manifest_val_name)
        frame = pd.concat([train_frame, val_frame], axis=0, ignore_index=True)
        return frame if settings.max_rows is None else frame.iloc[: settings.max_rows].reset_index(drop=True)
    else:
        raise ValueError(f"Unsupported split {split!r}. Expected train, val, or all.")
    frame = pd.read_parquet(path)
    if settings.max_rows is not None:
        frame = frame.iloc[: settings.max_rows].reset_index(drop=True)
    return frame


def local_knots_to_span_widths(local_knots: np.ndarray, degree: int) -> np.ndarray:
    del degree
    boundaries = np.unique(np.asarray(local_knots, dtype=np.float32))
    widths = np.diff(boundaries)
    if widths.ndim != 1:
        raise ValueError(f"Expected 1D widths, got shape {widths.shape}")
    positive = widths[widths > 1e-8]
    if positive.size == 0:
        raise ValueError("Local spline produced no positive knot spans.")
    total = float(np.sum(positive))
    return (positive / max(total, 1e-8)).astype(np.float32, copy=False)


def pack_spline_actions(
    control_points: np.ndarray,
    span_widths: np.ndarray,
    *,
    max_control_points: int,
    max_span_count: int,
    action_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    control_points = np.asarray(control_points, dtype=np.float32)
    span_widths = np.asarray(span_widths, dtype=np.float32).reshape(-1)
    if control_points.ndim != 2:
        raise ValueError(f"Expected control points to be 2D, got shape {control_points.shape}")
    if control_points.shape[1] != action_dim:
        raise ValueError(f"Expected control-point dim {action_dim}, got {control_points.shape[1]}")
    if control_points.shape[0] > max_control_points:
        raise ValueError(f"Expected <= {max_control_points} control points, got {control_points.shape[0]}")
    if span_widths.shape[0] > max_span_count:
        raise ValueError(f"Expected <= {max_span_count} span widths, got {span_widths.shape[0]}")

    actions = np.zeros((max_control_points + 1, action_dim), dtype=np.float32)
    action_mask = np.zeros((max_control_points + 1, action_dim), dtype=bool)

    actions[: control_points.shape[0], :] = control_points
    action_mask[: control_points.shape[0], :] = True
    actions[max_control_points, : span_widths.shape[0]] = span_widths
    action_mask[max_control_points, : span_widths.shape[0]] = True
    return actions, action_mask


def extract_target_sample(
    target_archive: Any,
    sample_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    sample_offsets = np.asarray(target_archive["control_point_offsets"], dtype=np.int64)
    knot_offsets = np.asarray(target_archive["local_knot_offsets"], dtype=np.int64)
    control_points = np.asarray(target_archive["local_delta_control_points"], dtype=np.float32)
    local_knots = np.asarray(target_archive["local_knots"], dtype=np.float32)

    cp_start, cp_end = int(sample_offsets[sample_index]), int(sample_offsets[sample_index + 1])
    knot_start, knot_end = int(knot_offsets[sample_index]), int(knot_offsets[sample_index + 1])
    if cp_end <= cp_start:
        raise ValueError(f"Empty control-point slice for sample_index={sample_index}")
    if knot_end <= knot_start:
        raise ValueError(f"Empty knot slice for sample_index={sample_index}")
    return control_points[cp_start:cp_end], local_knots[knot_start:knot_end]


@dataclasses.dataclass
class _EpisodeBundle:
    episode_root: Path
    state: np.ndarray
    timestamps: np.ndarray
    target_archive: Any
    planner_archives: dict[str, Any]


class OrigamiVlaDataset:
    def __init__(self, settings: OrigamiVlaSettings, *, split: str):
        self._settings = settings
        self._dataset_root = _ensure_path(settings.dataset_root)
        self._rows = load_manifest_rows(settings, split).to_dict(orient="records")
        self._episode_cache: dict[str, _EpisodeBundle] = {}
        self._video_cache: dict[str, cv2.VideoCapture] = {}

    def __len__(self) -> int:
        return len(self._rows)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_episode_cache"] = {}
        state["_video_cache"] = {}
        return state

    def __del__(self) -> None:
        for capture in self._video_cache.values():
            capture.release()

    def _episode_bundle(self, episode_uid: str) -> _EpisodeBundle:
        cached = self._episode_cache.get(episode_uid)
        if cached is not None:
            return cached

        episode_root = self._dataset_root / "episodes" / episode_uid
        arrays_root = episode_root / "arrays"
        if not episode_root.is_dir():
            raise FileNotFoundError(f"Episode directory not found: {episode_root}")

        bundle = _EpisodeBundle(
            episode_root=episode_root,
            state=np.load(arrays_root / "state_65d.npy", mmap_mode="r"),
            timestamps=np.load(arrays_root / "timestamps.npy", mmap_mode="r"),
            target_archive=np.load(arrays_root / self._settings.local_target_npz_name, allow_pickle=False),
            planner_archives={},
        )
        for row in self._rows:
            if row["episode_uid"] != episode_uid:
                continue
            view_mode = str(row["view_mode"])
            if view_mode in bundle.planner_archives:
                continue
            planner_npz = (
                _ensure_path(row["planner_output_dir"])
                / self._settings.planner_arrays_filename
            )
            bundle.planner_archives[view_mode] = np.load(planner_npz, allow_pickle=False)
        self._episode_cache[episode_uid] = bundle
        return bundle

    def _video_capture(self, video_path: Path) -> cv2.VideoCapture:
        key = str(video_path)
        capture = self._video_cache.get(key)
        if capture is not None:
            return capture
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        self._video_cache[key] = capture
        return capture

    def _read_video_frame(self, video_path: Path, frame_position: int) -> np.ndarray:
        capture = self._video_capture(video_path)
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_position))
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"Could not read frame {frame_position} from {video_path}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self._rows[int(index)]
        episode_uid = str(row["episode_uid"])
        frame_position = int(row["frame_position"])
        target_sample_index = int(row["local_target_npz_sample_index"])
        planner_row_index = int(row["planner_row_index"])
        view_mode = str(row["view_mode"])

        bundle = self._episode_bundle(episode_uid)
        control_points, local_knots = extract_target_sample(bundle.target_archive, target_sample_index)
        span_widths = local_knots_to_span_widths(local_knots, self._settings.degree)
        actions, action_mask = pack_spline_actions(
            control_points,
            span_widths,
            max_control_points=self._settings.max_control_points,
            max_span_count=self._settings.max_span_count,
            action_dim=self._settings.action_dim,
        )

        planner = bundle.planner_archives[view_mode]
        images: dict[str, np.ndarray] = {}
        image_masks: dict[str, np.ndarray] = {}
        missing_modalities: list[str] = []
        for image_key, relpath in self._settings.image_modalities.items():
            video_path = bundle.episode_root / relpath
            if not video_path.exists():
                missing_modalities.append(str(video_path))
                continue
            images[image_key] = self._read_video_frame(video_path, frame_position)
            image_masks[image_key] = np.asarray(True)
        if missing_modalities and self._settings.fail_on_missing_modalities:
            raise FileNotFoundError(f"Missing required image modalities for {episode_uid}: {missing_modalities}")

        state = np.asarray(bundle.state[frame_position], dtype=np.float32)
        if state.shape[-1] != self._settings.state_dim:
            raise ValueError(f"Expected state dim {self._settings.state_dim}, got {state.shape[-1]}")

        return {
            "image": images,
            "image_mask": image_masks,
            "state": state,
            "state_mask": np.ones((self._settings.state_dim,), dtype=bool),
            "actions": actions,
            "action_mask": action_mask,
            "prompt": np.asarray(self._settings.prompt),
            "planner_state_belief": np.asarray(planner["final_state_belief"][planner_row_index], dtype=np.float32),
            "planner_progress_transition": np.asarray(
                planner["progress_transition"][planner_row_index], dtype=np.float32
            ),
            "planner_uncertainty": np.asarray(planner["uncertainty_features"][planner_row_index], dtype=np.float32),
            "planner_history_latent": np.asarray(planner["temporal_latent"][planner_row_index], dtype=np.float32),
            "frame_position": np.asarray(frame_position, dtype=np.int64),
            "frame_index": np.asarray(int(row["frame_index"]), dtype=np.int64),
            "timestamp": np.asarray(bundle.timestamps[frame_position], dtype=np.float32),
        }
