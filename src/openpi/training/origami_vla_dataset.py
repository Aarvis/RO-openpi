from __future__ import annotations

from collections import OrderedDict
import dataclasses
import hashlib
from pathlib import Path
from typing import Any, Literal

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
    planner_branch: str = "alias"
    planner_value_variant: Literal["final", "raw"] = "final"
    planner_belief_dim: int = 29
    planner_progress_dim: int = 2
    planner_uncertainty_dim: int = 3
    planner_history_dim: int = 512
    action_source: Literal["spline", "action_chunk"] = "spline"
    action_filename: str = "action_65d.npy"
    action_chunk_stride: int = 1
    drop_horizon_clipped: bool = False
    max_control_points: int = 18
    action_horizon: int = 19
    max_span_count: int = 15
    degree: int = 3
    state_dim: int = 65
    action_dim: int = 65
    tactile_dim: int = 60
    prompt: str = "Fold paper into airplane"
    sample_weight_column: str = "sample_weight"
    require_sample_weight: bool = False
    tactile_filename: str = "tactile_60d.npy"
    dataset_backend: Literal["video", "shard"] = "video"
    shard_root: str | None = None
    shard_manifest_name: str = "shard_manifest.json"
    shard_rows_name: str = "rows.parquet"
    shard_complete_marker_name: str = "complete.marker"
    shard_require_complete: bool = True
    shard_max_cached_shards: int = 2
    shard_use_stored_row_order: bool = True
    image_source_type: Literal["video", "frame_cache"] = "video"
    image_modalities: dict[str, str] = dataclasses.field(
        default_factory=lambda: {
            "ooi_rgb": "videos/ooi.mp4",
            "base_0_rgb": "videos/head_left.mp4",
            "left_wrist_0_rgb": "videos/wrist_left.mp4",
            "right_wrist_0_rgb": "videos/wrist_right.mp4",
        }
    )
    frame_cache_root_relpath: str = "arrays/vla_frame_cache_224_uint8"
    frame_cache_modalities: dict[str, str] = dataclasses.field(
        default_factory=lambda: {
            "ooi_rgb": "ooi_rgb_224x224_uint8.npy",
            "base_0_rgb": "base_0_rgb_224x224_uint8.npy",
            "left_wrist_0_rgb": "left_wrist_0_rgb_224x224_uint8.npy",
            "right_wrist_0_rgb": "right_wrist_0_rgb_224x224_uint8.npy",
        }
    )
    load_tactile_images: bool = False
    tactile_deform_video: str = "videos/tactile_deform.mp4"
    tactile_raw_video: str = "videos/tactile_raw.mp4"
    tactile_require_raw_video: bool = False
    tactile_image_size: int = 224
    tactile_raw_input_dropout_prob: float = 0.0
    tactile_raw_dropout_seed: int = 1234
    tactile_raw_grid: dict[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "rows": 2,
            "cols": 5,
            "expected_width": 1600,
            "expected_height": 480,
            "resize_mode": "resize",
        }
    )
    tactile_deform_grid: dict[str, Any] = dataclasses.field(
        default_factory=lambda: {
            "rows": 2,
            "cols": 5,
            "expected_width": 1200,
            "expected_height": 480,
            "resize_mode": "resize",
        }
    )
    include_planner_features: bool = True
    fail_on_missing_modalities: bool = True
    limit_loader_caches: bool = False
    max_cached_episodes: int = 8
    max_cached_videos: int = 32
    max_rows: int | None = None

    def __post_init__(self) -> None:
        if self.action_chunk_stride <= 0:
            raise ValueError(f"action_chunk_stride must be positive, got {self.action_chunk_stride}")
        if self.action_horizon <= 0:
            raise ValueError(f"action_horizon must be positive, got {self.action_horizon}")
        if "/" in self.planner_branch or "\\" in self.planner_branch:
            raise ValueError(f"planner_branch must be a simple branch name, got {self.planner_branch!r}")
        if self.planner_value_variant not in {"final", "raw"}:
            raise ValueError(f"planner_value_variant must be 'final' or 'raw', got {self.planner_value_variant!r}")
        if self.planner_belief_dim <= 0:
            raise ValueError(f"planner_belief_dim must be positive, got {self.planner_belief_dim}")
        if self.planner_progress_dim <= 0:
            raise ValueError(f"planner_progress_dim must be positive, got {self.planner_progress_dim}")
        if self.planner_uncertainty_dim <= 0:
            raise ValueError(f"planner_uncertainty_dim must be positive, got {self.planner_uncertainty_dim}")
        if self.planner_history_dim <= 0:
            raise ValueError(f"planner_history_dim must be positive, got {self.planner_history_dim}")
        if self.tactile_image_size <= 0:
            raise ValueError(f"tactile_image_size must be positive, got {self.tactile_image_size}")
        if not 0.0 <= self.tactile_raw_input_dropout_prob <= 1.0:
            raise ValueError(
                "tactile_raw_input_dropout_prob must satisfy 0 <= p <= 1, "
                f"got {self.tactile_raw_input_dropout_prob}"
            )
        if self.max_cached_episodes <= 0:
            raise ValueError(f"max_cached_episodes must be positive, got {self.max_cached_episodes}")
        if self.max_cached_videos <= 0:
            raise ValueError(f"max_cached_videos must be positive, got {self.max_cached_videos}")
        if self.shard_max_cached_shards <= 0:
            raise ValueError(f"shard_max_cached_shards must be positive, got {self.shard_max_cached_shards}")
        if self.dataset_backend == "shard" and not self.shard_root:
            raise ValueError("dataset_backend='shard' requires shard_root to be set.")


def _ensure_path(value: str | Path) -> Path:
    return value if isinstance(value, Path) else Path(value)


def _planner_feature_key(base_key: str, branch: str) -> str:
    branch = str(branch or "alias")
    if branch in {"alias", "compatibility", "default"}:
        return base_key
    return f"{branch}_{base_key}"


def _read_planner_feature(planner: Any, base_key: str, branch: str, row_index: int) -> np.ndarray:
    key = _planner_feature_key(base_key, branch)
    if key not in planner:
        available = sorted(str(name) for name in planner.files)
        raise KeyError(
            f"Planner rollout feature {key!r} is missing. "
            f"Set planner_branch to one of the exported branches or use 'alias'. Available keys: {available}"
        )
    return planner[key][row_index]


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


def _planner_state_belief_key(value_variant: str) -> str:
    if value_variant == "final":
        return "final_state_belief"
    if value_variant == "raw":
        return "raw_state_belief"
    raise ValueError(f"Unsupported planner value variant: {value_variant!r}")


def _filter_horizon_clipped_rows(settings: OrigamiVlaSettings, frame: pd.DataFrame) -> pd.DataFrame:
    if not settings.drop_horizon_clipped or frame.empty:
        return frame
    if "horizon_clipped_to_episode_end" in frame.columns:
        return frame[~frame["horizon_clipped_to_episode_end"].astype(bool)].reset_index(drop=True)
    if settings.action_source != "action_chunk":
        return frame

    dataset_root = _ensure_path(settings.dataset_root)
    keep = np.ones(len(frame), dtype=bool)
    action_lengths: dict[str, int] = {}
    for episode_uid, group in frame.groupby("episode_uid", sort=False):
        episode_uid = str(episode_uid)
        action_len = action_lengths.get(episode_uid)
        if action_len is None:
            action_path = dataset_root / "episodes" / episode_uid / "arrays" / settings.action_filename
            action_len = int(np.load(action_path, mmap_mode="r").shape[0])
            action_lengths[episode_uid] = action_len
        horizon_end = (
            group["frame_position"].to_numpy(dtype=np.int64)
            + (int(settings.action_horizon) - 1) * int(settings.action_chunk_stride)
        )
        keep[group.index.to_numpy(dtype=np.int64)] = horizon_end < action_len
    return frame[keep].reset_index(drop=True)


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
        frame = _filter_horizon_clipped_rows(settings, frame)
        return frame if settings.max_rows is None else frame.iloc[: settings.max_rows].reset_index(drop=True)
    else:
        raise ValueError(f"Unsupported split {split!r}. Expected train, val, or all.")
    frame = pd.read_parquet(path)
    frame = _filter_horizon_clipped_rows(settings, frame)
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


def extract_action_chunk(
    action_array: np.ndarray,
    frame_position: int,
    *,
    action_horizon: int,
    action_dim: int,
    action_chunk_stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    actions = np.zeros((action_horizon, action_dim), dtype=np.float32)
    action_mask = np.zeros((action_horizon, action_dim), dtype=bool)
    positions = int(frame_position) + np.arange(action_horizon, dtype=np.int64) * int(action_chunk_stride)
    valid = (positions >= 0) & (positions < int(action_array.shape[0]))
    if np.any(valid):
        valid_positions = positions[valid]
        valid_actions = np.asarray(action_array[valid_positions], dtype=np.float32)
        if valid_actions.ndim != 2 or valid_actions.shape[-1] != action_dim:
            raise ValueError(f"Expected action chunk dim {action_dim}, got {valid_actions.shape}")
        actions[valid] = valid_actions
        action_mask[valid, :] = True
    return actions, action_mask


@dataclasses.dataclass
class _EpisodeBundle:
    episode_root: Path
    state: np.ndarray
    actions: np.ndarray | None
    tactile: np.ndarray
    timestamps: np.ndarray
    target_archive: Any | None
    planner_archives: dict[str, Any]
    frame_cache_archives: dict[str, np.ndarray]


class OrigamiVlaDataset:
    def __init__(self, settings: OrigamiVlaSettings, *, split: str):
        self._settings = settings
        self._dataset_root = _ensure_path(settings.dataset_root)
        self._episode_cache: OrderedDict[str, _EpisodeBundle] = OrderedDict()
        self._video_cache: OrderedDict[str, cv2.VideoCapture] = OrderedDict()
        self._rows = load_manifest_rows(settings, split).to_dict(orient="records")
        self._planner_output_dirs_by_episode = self._index_planner_output_dirs()

    def __len__(self) -> int:
        return len(self._rows)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_episode_cache"] = OrderedDict()
        state["_video_cache"] = OrderedDict()
        return state

    def __del__(self) -> None:
        for capture in getattr(self, "_video_cache", {}).values():
            capture.release()
        for bundle in getattr(self, "_episode_cache", {}).values():
            self._close_episode_bundle(bundle)

    def _index_planner_output_dirs(self) -> dict[str, dict[str, Path]]:
        if not self._settings.include_planner_features:
            return {}
        by_episode: dict[str, dict[str, Path]] = {}
        for row in self._rows:
            if not _row_bool(row.get("planner_enabled", True), default=True):
                continue
            episode_uid = str(row["episode_uid"])
            view_mode = str(row["view_mode"])
            episode_dirs = by_episode.setdefault(episode_uid, {})
            if view_mode in episode_dirs:
                continue
            planner_output_dir = row.get("planner_output_dir")
            if planner_output_dir is None or pd.isna(planner_output_dir) or not str(planner_output_dir):
                raise ValueError(f"Planner is enabled for {episode_uid}:{view_mode}, but planner_output_dir is empty.")
            episode_dirs[view_mode] = _ensure_path(planner_output_dir)
        return by_episode

    def _close_np_archive(self, archive: Any | None) -> None:
        close = getattr(archive, "close", None)
        if close is not None:
            close()

    def _close_episode_bundle(self, bundle: _EpisodeBundle) -> None:
        self._close_np_archive(bundle.target_archive)
        for archive in bundle.planner_archives.values():
            self._close_np_archive(archive)

    def _enforce_episode_cache_limit(self) -> None:
        if not self._settings.limit_loader_caches:
            return
        while len(self._episode_cache) > int(self._settings.max_cached_episodes):
            _episode_uid, bundle = self._episode_cache.popitem(last=False)
            self._close_episode_bundle(bundle)

    def _enforce_video_cache_limit(self) -> None:
        if not self._settings.limit_loader_caches:
            return
        while len(self._video_cache) > int(self._settings.max_cached_videos):
            _video_path, capture = self._video_cache.popitem(last=False)
            capture.release()

    def _episode_bundle(self, episode_uid: str) -> _EpisodeBundle:
        cached = self._episode_cache.get(episode_uid)
        if cached is not None:
            self._episode_cache.move_to_end(episode_uid)
            return cached

        episode_root = self._dataset_root / "episodes" / episode_uid
        arrays_root = episode_root / "arrays"
        if not episode_root.is_dir():
            raise FileNotFoundError(f"Episode directory not found: {episode_root}")

        bundle = _EpisodeBundle(
            episode_root=episode_root,
            state=np.load(arrays_root / "state_65d.npy", mmap_mode="r"),
            actions=(
                np.load(arrays_root / self._settings.action_filename, mmap_mode="r")
                if self._settings.action_source == "action_chunk"
                else None
            ),
            tactile=np.load(arrays_root / self._settings.tactile_filename, mmap_mode="r"),
            timestamps=np.load(arrays_root / "timestamps.npy", mmap_mode="r"),
            target_archive=(
                np.load(arrays_root / self._settings.local_target_npz_name, allow_pickle=False)
                if self._settings.action_source == "spline"
                else None
            ),
            planner_archives={},
            frame_cache_archives={},
        )
        if self._settings.include_planner_features:
            for view_mode, planner_output_dir in self._planner_output_dirs_by_episode.get(episode_uid, {}).items():
                planner_npz = planner_output_dir / self._settings.planner_arrays_filename
                bundle.planner_archives[view_mode] = np.load(planner_npz, allow_pickle=False)

        if self._settings.image_source_type == "frame_cache":
            frame_cache_root = episode_root / self._settings.frame_cache_root_relpath
            missing_modalities: list[str] = []
            for image_key, filename in self._settings.frame_cache_modalities.items():
                cache_path = frame_cache_root / filename
                if not cache_path.exists():
                    missing_modalities.append(str(cache_path))
                    continue
                bundle.frame_cache_archives[image_key] = np.load(cache_path, mmap_mode="r")
            if missing_modalities and self._settings.fail_on_missing_modalities:
                raise FileNotFoundError(
                    f"Missing required frame-cache modalities for {episode_uid}: {missing_modalities}"
                )
        self._episode_cache[episode_uid] = bundle
        self._enforce_episode_cache_limit()
        return bundle

    def _video_capture(self, video_path: Path) -> cv2.VideoCapture:
        key = str(video_path)
        capture = self._video_cache.get(key)
        if capture is not None:
            self._video_cache.move_to_end(key)
            return capture
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        self._video_cache[key] = capture
        self._enforce_video_cache_limit()
        return capture

    def _read_video_frame(self, video_path: Path, frame_position: int) -> np.ndarray:
        capture = self._video_capture(video_path)
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_position))
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"Could not read frame {frame_position} from {video_path}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _read_cached_frame(self, cache: np.ndarray, frame_position: int, image_key: str, episode_uid: str) -> np.ndarray:
        if frame_position < 0 or frame_position >= int(cache.shape[0]):
            raise IndexError(
                f"Frame position {frame_position} out of range for cached modality {image_key!r} "
                f"in {episode_uid}; cache has {int(cache.shape[0])} frames."
            )
        frame = np.asarray(cache[frame_position], dtype=np.uint8)
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError(
                f"Cached frame for {episode_uid}:{image_key} must have shape [H, W, 3], got {frame.shape}"
            )
        return frame

    def _resize_tactile_cell(self, cell: np.ndarray, mode: str) -> np.ndarray:
        image_size = int(self._settings.tactile_image_size)
        if mode == "resize":
            return cv2.resize(cell, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        if mode == "pad":
            height, width = cell.shape[:2]
            scale = min(image_size / height, image_size / width)
            new_h = max(1, int(round(height * scale)))
            new_w = max(1, int(round(width * scale)))
            resized = cv2.resize(cell, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            canvas = np.zeros((image_size, image_size, 3), dtype=np.uint8)
            top = (image_size - new_h) // 2
            left = (image_size - new_w) // 2
            canvas[top : top + new_h, left : left + new_w] = resized
            return canvas
        raise ValueError(f"Unknown tactile resize_mode: {mode!r}")

    def _split_tactile_grid(self, frame: np.ndarray, grid_cfg: dict[str, Any], episode_uid: str) -> np.ndarray:
        rows = int(grid_cfg["rows"])
        cols = int(grid_cfg["cols"])
        height, width, channels = frame.shape
        expected_height = int(grid_cfg.get("expected_height", height))
        expected_width = int(grid_cfg.get("expected_width", width))
        if (width, height) != (expected_width, expected_height):
            raise ValueError(
                f"Tactile grid shape {(width, height)} != expected {(expected_width, expected_height)} "
                f"for {episode_uid}"
            )
        if channels != 3 or height % rows or width % cols:
            raise ValueError(f"Invalid tactile grid shape {frame.shape} for {rows}x{cols} in {episode_uid}")
        cell_h = height // rows
        cell_w = width // cols
        mode = str(grid_cfg.get("resize_mode", "resize"))
        cells: list[np.ndarray] = []
        for row in range(rows):
            for col in range(cols):
                cell = frame[row * cell_h : (row + 1) * cell_h, col * cell_w : (col + 1) * cell_w]
                cells.append(self._resize_tactile_cell(cell, mode))
        return np.ascontiguousarray(np.stack(cells, axis=0).transpose(0, 3, 1, 2))

    def _drop_tactile_raw_input(self, episode_uid: str, frame_position: int) -> bool:
        probability = float(self._settings.tactile_raw_input_dropout_prob)
        if probability <= 0.0:
            return False
        if probability >= 1.0:
            return True
        payload = f"{self._settings.tactile_raw_dropout_seed}:{episode_uid}:{int(frame_position)}".encode("utf-8")
        value = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") / float(1 << 64)
        return value < probability

    def _read_tactile_images(self, bundle: _EpisodeBundle, frame_position: int, episode_uid: str) -> dict[str, np.ndarray]:
        deform_path = bundle.episode_root / self._settings.tactile_deform_video
        raw_path = bundle.episode_root / self._settings.tactile_raw_video
        if not deform_path.exists():
            raise FileNotFoundError(f"Missing tactile deform video for {episode_uid}: {deform_path}")
        if self._settings.tactile_require_raw_video and not raw_path.exists():
            raise FileNotFoundError(f"Missing required tactile raw video for {episode_uid}: {raw_path}")

        deform_frame = self._read_video_frame(deform_path, frame_position)
        deform_images = self._split_tactile_grid(deform_frame, self._settings.tactile_deform_grid, episode_uid)
        raw_available = raw_path.exists()
        if raw_available:
            try:
                raw_frame = self._read_video_frame(raw_path, frame_position)
            except RuntimeError:
                if self._settings.tactile_require_raw_video:
                    raise
                raw_available = False
                raw_images = np.zeros_like(deform_images)
            else:
                raw_images = self._split_tactile_grid(raw_frame, self._settings.tactile_raw_grid, episode_uid)
        else:
            raw_images = np.zeros_like(deform_images)

        if raw_available and self._drop_tactile_raw_input(episode_uid, frame_position):
            raw_images = np.zeros_like(deform_images)
            raw_available = False

        return {
            "tactile_deform_images": deform_images,
            "tactile_raw_images": raw_images,
            "tactile_raw_available": np.asarray(raw_available, dtype=bool),
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self._rows[int(index)]
        episode_uid = str(row["episode_uid"])
        frame_position = int(row["frame_position"])
        view_mode = str(row.get("view_mode", "action_chunk"))

        bundle = self._episode_bundle(episode_uid)
        if self._settings.action_source == "spline":
            if bundle.target_archive is None:
                raise RuntimeError(f"Spline action source requested, but target archive is missing for {episode_uid}.")
            target_sample_index = int(row["local_target_npz_sample_index"])
            control_points, local_knots = extract_target_sample(bundle.target_archive, target_sample_index)
            span_widths = local_knots_to_span_widths(local_knots, self._settings.degree)
            actions, action_mask = pack_spline_actions(
                control_points,
                span_widths,
                max_control_points=self._settings.max_control_points,
                max_span_count=self._settings.max_span_count,
                action_dim=self._settings.action_dim,
            )
        elif self._settings.action_source == "action_chunk":
            if bundle.actions is None:
                raise RuntimeError(f"Action-chunk source requested, but action array is missing for {episode_uid}.")
            actions, action_mask = extract_action_chunk(
                bundle.actions,
                frame_position,
                action_horizon=self._settings.action_horizon,
                action_dim=self._settings.action_dim,
                action_chunk_stride=self._settings.action_chunk_stride,
            )
        else:
            raise ValueError(f"Unsupported action_source: {self._settings.action_source!r}")
        images: dict[str, np.ndarray] = {}
        image_masks: dict[str, np.ndarray] = {}
        missing_modalities: list[str] = []
        if self._settings.image_source_type == "frame_cache":
            for image_key, _filename in self._settings.frame_cache_modalities.items():
                cache = bundle.frame_cache_archives.get(image_key)
                if cache is None:
                    missing_modalities.append(str(bundle.episode_root / self._settings.frame_cache_root_relpath / _filename))
                    continue
                images[image_key] = self._read_cached_frame(cache, frame_position, image_key, episode_uid)
                image_masks[image_key] = np.asarray(True)
        else:
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
        tactile = np.asarray(bundle.tactile[frame_position], dtype=np.float32).reshape(-1)
        if tactile.shape[-1] != self._settings.tactile_dim:
            raise ValueError(
                f"Expected tactile dim {self._settings.tactile_dim}, got {tactile.shape[-1]} for {episode_uid}"
            )

        if self._settings.require_sample_weight and self._settings.sample_weight_column not in row:
            raise KeyError(
                f"Manifest row for {episode_uid} is missing required sample-weight column "
                f"{self._settings.sample_weight_column!r}"
            )
        sample_weight_value = row.get(self._settings.sample_weight_column, 1.0)
        if pd.isna(sample_weight_value):
            if self._settings.require_sample_weight:
                raise ValueError(
                    f"Manifest row for {episode_uid} frame_position={frame_position} has NaN sample weight."
                )
            sample_weight_value = 1.0

        output = {
            "image": images,
            "image_mask": image_masks,
            "state": state,
            "tactile": tactile,
            "tactile_prompt": np.array(tactile, copy=True),
            "tactile_prompt_mask": np.ones((self._settings.tactile_dim,), dtype=bool),
            "state_mask": np.ones((self._settings.state_dim,), dtype=bool),
            "actions": actions,
            "action_mask": action_mask,
            "sample_weight": np.asarray(float(sample_weight_value), dtype=np.float32),
            "prompt": np.asarray(self._settings.prompt),
            "frame_position": np.asarray(frame_position, dtype=np.int64),
            "frame_index": np.asarray(int(row["frame_index"]), dtype=np.int64),
            "timestamp": np.asarray(bundle.timestamps[frame_position], dtype=np.float32),
        }
        if self._settings.include_planner_features:
            planner_enabled = _row_bool(row.get("planner_enabled", True), default=True)
            output["planner_available"] = np.asarray(planner_enabled, dtype=bool)
            if not planner_enabled:
                output.update(
                    {
                        "planner_state_belief": np.zeros((self._settings.planner_belief_dim,), dtype=np.float32),
                        "planner_progress_transition": np.zeros(
                            (self._settings.planner_progress_dim,), dtype=np.float32
                        ),
                        "planner_uncertainty": np.zeros((self._settings.planner_uncertainty_dim,), dtype=np.float32),
                        "planner_history_latent": np.zeros((self._settings.planner_history_dim,), dtype=np.float32),
                    }
                )
            else:
                planner_row_index = int(row["planner_row_index"])
                planner_branch_value = row.get("planner_branch", self._settings.planner_branch)
                if pd.isna(planner_branch_value):
                    planner_branch_value = self._settings.planner_branch
                planner_branch = str(planner_branch_value or self._settings.planner_branch)
                planner_value_variant = row.get("planner_value_variant", self._settings.planner_value_variant)
                if pd.isna(planner_value_variant):
                    planner_value_variant = self._settings.planner_value_variant
                planner_value_variant = str(planner_value_variant or self._settings.planner_value_variant)
                planner = bundle.planner_archives[view_mode]
                output.update(
                    {
                        "planner_state_belief": np.asarray(
                            _read_planner_feature(
                                planner,
                                _planner_state_belief_key(planner_value_variant),
                                planner_branch,
                                planner_row_index,
                            ),
                            dtype=np.float32,
                        ),
                        "planner_progress_transition": np.asarray(
                            _read_planner_feature(planner, "progress_transition", planner_branch, planner_row_index),
                            dtype=np.float32,
                        ),
                        "planner_uncertainty": np.asarray(
                            _read_planner_feature(planner, "uncertainty_features", planner_branch, planner_row_index),
                            dtype=np.float32,
                        ),
                        "planner_history_latent": np.asarray(
                            _read_planner_feature(planner, "temporal_latent", planner_branch, planner_row_index),
                            dtype=np.float32,
                        ),
                    }
                )
        if self._settings.load_tactile_images:
            output.update(self._read_tactile_images(bundle, frame_position, episode_uid))
        return output
