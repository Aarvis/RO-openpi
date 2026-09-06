from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import pandas as pd

import openpi.training.origami_vla_dataset as _origami_vla_dataset


SHARD_MANIFEST_FILENAME = "shard_manifest.json"
SHARD_PLAN_FILENAME = "shard_plan.parquet"
SHARD_ROWS_FILENAME = "rows.parquet"
SHARD_METADATA_FILENAME = "metadata.json"
SHARD_COMPLETE_MARKER = "complete.marker"

IMAGE_ARRAY_PREFIX = "image__"


@dataclasses.dataclass(frozen=True)
class ShardSpec:
    split: str
    shard_id: int
    shard_name: str
    shard_dir: Path
    rows_path: Path
    num_rows: int
    episodes: tuple[str, ...]
    estimated_bytes: int


@dataclasses.dataclass
class _ShardBundle:
    spec: ShardSpec
    rows: pd.DataFrame
    row_order: np.ndarray | None
    arrays: dict[str, np.ndarray]


def estimate_row_bytes(settings: _origami_vla_dataset.OrigamiVlaSettings, *, image_size: int) -> int:
    if settings.action_source != "action_chunk":
        raise ValueError(
            "Origami comp action-chunk shards require "
            f"action_source='action_chunk', got {settings.action_source!r}."
        )
    image_size = int(image_size)
    image_bytes = len(settings.image_modalities) * image_size * image_size * 3
    tactile_fingers = int(settings.tactile_deform_grid["rows"]) * int(settings.tactile_deform_grid["cols"])
    tactile_bytes = 0
    if settings.load_tactile_images:
        tactile_bytes += 2 * tactile_fingers * 3 * int(settings.tactile_image_size) * int(settings.tactile_image_size)
        tactile_bytes += 1
    numeric_bytes = 0
    numeric_bytes += settings.state_dim * np.dtype(np.float32).itemsize
    numeric_bytes += settings.tactile_dim * np.dtype(np.float32).itemsize
    numeric_bytes += settings.action_horizon * settings.action_dim * np.dtype(np.float32).itemsize
    numeric_bytes += settings.action_horizon * settings.action_dim * np.dtype(np.bool_).itemsize
    numeric_bytes += np.dtype(np.float32).itemsize  # sample_weight
    numeric_bytes += np.dtype(np.int64).itemsize * 2  # frame_position, frame_index
    numeric_bytes += np.dtype(np.float32).itemsize  # timestamp
    if settings.include_planner_features:
        numeric_bytes += np.dtype(np.bool_).itemsize
        numeric_bytes += settings.planner_belief_dim * np.dtype(np.float32).itemsize
        numeric_bytes += settings.planner_progress_dim * np.dtype(np.float32).itemsize
        numeric_bytes += settings.planner_uncertainty_dim * np.dtype(np.float32).itemsize
        numeric_bytes += settings.planner_history_dim * np.dtype(np.float32).itemsize
    return int(image_bytes + tactile_bytes + numeric_bytes)


def parse_size_bytes(value: str | int | float | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    text = str(value).strip().lower()
    if not text:
        return None
    units = {
        "b": 1,
        "kb": 1_000,
        "kib": 1024,
        "mb": 1_000_000,
        "mib": 1024**2,
        "gb": 1_000_000_000,
        "gib": 1024**3,
        "tb": 1_000_000_000_000,
        "tib": 1024**4,
    }
    for suffix, multiplier in sorted(units.items(), key=lambda item: len(item[0]), reverse=True):
        if text.endswith(suffix):
            return int(float(text[: -len(suffix)].strip()) * multiplier)
    return int(float(text))


def resolve_shard_root(settings: _origami_vla_dataset.OrigamiVlaSettings) -> Path:
    if not settings.shard_root:
        raise ValueError("dataset_backend='shard' requires data.shard_root to be set.")
    return Path(settings.shard_root)


def settings_from_data_factory(data_factory: Any, model_config: Any) -> _origami_vla_dataset.OrigamiVlaSettings:
    if not model_config.origami_vla.enabled and data_factory.action_source != "action_chunk":
        raise ValueError("Spline Origami VLA data config requires model.origami_vla.enabled=True.")
    return _origami_vla_dataset.OrigamiVlaSettings(
        dataset_root=data_factory.dataset_root,
        manifest_root=data_factory.manifest_root,
        local_target_npz_name=data_factory.local_target_npz_name,
        local_target_index_name=data_factory.local_target_index_name,
        action_source=data_factory.action_source,
        action_filename=data_factory.action_filename,
        action_chunk_stride=data_factory.action_chunk_stride,
        drop_horizon_clipped=data_factory.drop_horizon_clipped,
        planner_arrays_filename=data_factory.planner_arrays_filename,
        planner_index_filename=data_factory.planner_index_filename,
        planner_branch=data_factory.planner_branch,
        planner_value_variant=data_factory.planner_value_variant,
        planner_belief_dim=model_config.origami_vla.belief_dim,
        planner_history_dim=model_config.origami_vla.history_dim,
        tactile_filename=data_factory.tactile_filename,
        dataset_backend=data_factory.dataset_backend,
        shard_root=data_factory.shard_root,
        shard_manifest_name=data_factory.shard_manifest_name,
        shard_rows_name=data_factory.shard_rows_name,
        shard_complete_marker_name=data_factory.shard_complete_marker_name,
        shard_require_complete=data_factory.shard_require_complete,
        shard_max_cached_shards=data_factory.shard_max_cached_shards,
        shard_use_stored_row_order=data_factory.shard_use_stored_row_order,
        max_control_points=model_config.origami_vla.max_control_points,
        action_horizon=model_config.action_horizon,
        max_span_count=model_config.origami_vla.max_span_count,
        degree=model_config.origami_vla.degree,
        spline_span_representation=model_config.origami_vla.spline_span_representation,
        state_dim=int(model_config.state_dim or model_config.action_dim),
        action_dim=model_config.action_dim,
        tactile_dim=model_config.origami_vla.tactile_dim,
        prompt=data_factory.prompt,
        sample_weight_column=data_factory.sample_weight_column,
        require_sample_weight=data_factory.require_sample_weight,
        image_source_type=data_factory.image_source_type,
        image_modalities=dict(data_factory.image_modalities),
        frame_cache_root_relpath=data_factory.frame_cache_root_relpath,
        frame_cache_modalities=dict(data_factory.frame_cache_modalities),
        load_tactile_images=data_factory.load_tactile_images or model_config.origami_vla.ftp_tactile_enabled,
        tactile_deform_video=data_factory.tactile_deform_video,
        tactile_raw_video=data_factory.tactile_raw_video,
        tactile_require_raw_video=data_factory.tactile_require_raw_video,
        tactile_image_size=data_factory.tactile_image_size,
        tactile_raw_input_dropout_prob=data_factory.tactile_raw_input_dropout_prob,
        tactile_raw_dropout_seed=data_factory.tactile_raw_dropout_seed,
        tactile_image_input_dropout_prob=data_factory.tactile_image_input_dropout_prob,
        tactile_image_dropout_seed=data_factory.tactile_image_dropout_seed,
        tactile_raw_grid=dict(data_factory.tactile_raw_grid),
        tactile_deform_grid=dict(data_factory.tactile_deform_grid),
        include_planner_features=data_factory.include_planner_features,
        fail_on_missing_modalities=data_factory.fail_on_missing_modalities,
        limit_loader_caches=data_factory.limit_loader_caches,
        max_cached_episodes=data_factory.max_cached_episodes,
        max_cached_videos=data_factory.max_cached_videos,
        max_rows=data_factory.max_rows,
    )


def load_shard_manifest(shard_root: Path, *, manifest_filename: str = SHARD_MANIFEST_FILENAME) -> dict[str, Any]:
    path = shard_root / manifest_filename
    if not path.exists():
        raise FileNotFoundError(f"Shard manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_shard_specs(
    shard_root: Path,
    *,
    split: str,
    manifest_filename: str = SHARD_MANIFEST_FILENAME,
    complete_marker_name: str = SHARD_COMPLETE_MARKER,
    require_complete: bool = True,
) -> list[ShardSpec]:
    manifest = load_shard_manifest(shard_root, manifest_filename=manifest_filename)
    shards: list[ShardSpec] = []
    for item in manifest.get("shards", []):
        if str(item.get("split")) != split:
            continue
        shard_dir = shard_root / str(item["relative_dir"])
        marker = shard_dir / str(item.get("complete_marker_name", complete_marker_name))
        if require_complete and not marker.exists():
            continue
        shards.append(
            ShardSpec(
                split=str(item["split"]),
                shard_id=int(item["shard_id"]),
                shard_name=str(item["shard_name"]),
                shard_dir=shard_dir,
                rows_path=shard_dir / str(item.get("rows_name", SHARD_ROWS_FILENAME)),
                num_rows=int(item["num_rows"]),
                episodes=tuple(str(uid) for uid in item.get("episodes", [])),
                estimated_bytes=int(item.get("estimated_bytes", 0)),
            )
        )
    shards.sort(key=lambda spec: spec.shard_id)
    if not shards:
        raise RuntimeError(f"No complete {split!r} shards found under {shard_root}.")
    return shards


def _row_bool(value: Any, *, default: bool) -> bool:
    return _origami_vla_dataset._row_bool(value, default=default)


def _planner_state_belief_key(value_variant: str) -> str:
    return _origami_vla_dataset._planner_state_belief_key(value_variant)


def _planner_feature_key(base_key: str, branch: str) -> str:
    return _origami_vla_dataset._planner_feature_key(base_key, branch)


def _drop_tactile_raw_input(
    settings: _origami_vla_dataset.OrigamiVlaSettings,
    episode_uid: str,
    frame_position: int,
) -> bool:
    probability = float(settings.tactile_raw_input_dropout_prob)
    if probability <= 0.0:
        return False
    if probability >= 1.0:
        return True
    payload = f"{settings.tactile_raw_dropout_seed}:{episode_uid}:{int(frame_position)}".encode("utf-8")
    value = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") / float(1 << 64)
    return value < probability


def resize_with_pad_uint8(image: np.ndarray, height: int, width: int) -> np.ndarray:
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected RGB image [H, W, 3], got shape {image.shape}")
    cur_height, cur_width = image.shape[:2]
    ratio = max(cur_width / float(width), cur_height / float(height))
    resized_height = max(1, int(cur_height / ratio))
    resized_width = max(1, int(cur_width / ratio))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    canvas[pad_h0 : pad_h0 + resized_height, pad_w0 : pad_w0 + resized_width] = resized
    if remainder_h or remainder_w:
        # The divmod values above already place the extra pixel on the bottom/right,
        # matching the existing OpenPI resize-with-pad convention.
        pass
    return canvas


def resize_tactile_cell(cell: np.ndarray, image_size: int, mode: str) -> np.ndarray:
    if mode == "resize":
        return cv2.resize(cell, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    if mode == "pad":
        return resize_with_pad_uint8(cell, image_size, image_size)
    raise ValueError(f"Unknown tactile resize_mode: {mode!r}")


def split_tactile_grid(frame: np.ndarray, grid_cfg: dict[str, Any], episode_uid: str, image_size: int) -> np.ndarray:
    rows = int(grid_cfg["rows"])
    cols = int(grid_cfg["cols"])
    height, width, channels = frame.shape
    expected_height = int(grid_cfg.get("expected_height", height))
    expected_width = int(grid_cfg.get("expected_width", width))
    if (width, height) != (expected_width, expected_height):
        raise ValueError(
            f"Tactile grid shape {(width, height)} != expected {(expected_width, expected_height)} for {episode_uid}"
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
            cells.append(resize_tactile_cell(cell, image_size, mode))
    return np.ascontiguousarray(np.stack(cells, axis=0).transpose(0, 3, 1, 2))


class SequentialVideoReader:
    def __init__(self, path: Path):
        self.path = path
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open video: {path}")
        self.next_frame_position = 0

    def close(self) -> None:
        self.capture.release()

    def read(self, frame_position: int) -> np.ndarray:
        frame_position = int(frame_position)
        if frame_position != self.next_frame_position:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
            self.next_frame_position = frame_position
        ok, frame = self.capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"Could not read frame {frame_position} from {self.path}")
        self.next_frame_position = frame_position + 1
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _safe_makedirs(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def create_memmap(path: Path, *, dtype: np.dtype | type, shape: tuple[int, ...]) -> np.ndarray:
    _safe_makedirs(path.parent)
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def _close_memmap(array: np.ndarray) -> None:
    mmap = getattr(array, "_mmap", None)
    if mmap is not None:
        mmap.close()


def _read_planner_feature(planner: Any, base_key: str, branch: str, row_index: int) -> np.ndarray:
    return _origami_vla_dataset._read_planner_feature(planner, base_key, branch, row_index)


def _read_frame_index(arrays_root: Path, num_frames: int) -> np.ndarray:
    path = arrays_root / "frame_index.npy"
    if path.exists():
        return np.load(path, mmap_mode="r")
    return np.arange(num_frames, dtype=np.int64)


def _read_timestamps(arrays_root: Path, num_frames: int) -> np.ndarray:
    path = arrays_root / "timestamps.npy"
    if path.exists():
        return np.load(path, mmap_mode="r")
    return np.arange(num_frames, dtype=np.float32)


def _episode_rows_for_physical_write(rows: pd.DataFrame, episode_order: tuple[str, ...]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for episode_uid in episode_order:
        episode_rows = rows[rows["episode_uid"].astype(str) == episode_uid].sort_values("frame_position")
        if not episode_rows.empty:
            parts.append(episode_rows)
    if not parts:
        return rows.iloc[0:0].copy()
    return pd.concat(parts, axis=0, ignore_index=True)


def make_shard_row_order(
    num_rows: int,
    *,
    seed: int,
    shard_id: int,
    mode: Literal["episode_sequential", "shuffled_index"],
) -> np.ndarray | None:
    if mode == "episode_sequential":
        return None
    if mode != "shuffled_index":
        raise ValueError(f"Unsupported shard row order mode: {mode!r}")
    rng = np.random.default_rng(int(seed) + int(shard_id) * 1_000_003)
    return rng.permutation(num_rows).astype(np.int64, copy=False)


def load_rows_for_shard(shard_dir: Path, *, rows_name: str = SHARD_ROWS_FILENAME) -> pd.DataFrame:
    rows_path = shard_dir / rows_name
    if not rows_path.exists():
        raise FileNotFoundError(f"Shard rows parquet not found: {rows_path}")
    return pd.read_parquet(rows_path)


def load_shard_rows_file(rows_path: Path) -> pd.DataFrame:
    if not rows_path.exists():
        raise FileNotFoundError(f"Shard rows parquet not found: {rows_path}")
    return pd.read_parquet(rows_path)


class OrigamiCompActionChunkShardDataset:
    def __init__(self, settings: _origami_vla_dataset.OrigamiVlaSettings, *, split: str):
        if settings.action_source != "action_chunk":
            raise ValueError(
                "Origami comp action-chunk shards require "
                f"action_source='action_chunk', got {settings.action_source!r}."
            )
        self._settings = settings
        self._split = split
        self._shard_root = resolve_shard_root(settings)
        self._shards = load_shard_specs(
            self._shard_root,
            split=split,
            manifest_filename=settings.shard_manifest_name,
            complete_marker_name=settings.shard_complete_marker_name,
            require_complete=settings.shard_require_complete,
        )
        self._cumulative_rows: list[int] = []
        total = 0
        for spec in self._shards:
            total += int(spec.num_rows)
            self._cumulative_rows.append(total)
        self._shard_cache: OrderedDict[int, _ShardBundle] = OrderedDict()

    def __len__(self) -> int:
        return self._cumulative_rows[-1]

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_shard_cache"] = OrderedDict()
        return state

    def close(self) -> None:
        cache = getattr(self, "_shard_cache", None)
        if cache is None:
            return
        while cache:
            _shard_id, bundle = cache.popitem(last=False)
            self._close_shard_bundle(bundle)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _close_shard_bundle(self, bundle: _ShardBundle) -> None:
        close_memmap = _close_memmap
        if not callable(close_memmap):
            return
        for array in bundle.arrays.values():
            close_memmap(array)
        if bundle.row_order is not None:
            close_memmap(bundle.row_order)

    def _enforce_cache_limit(self) -> None:
        while len(self._shard_cache) > int(self._settings.shard_max_cached_shards):
            _shard_id, bundle = self._shard_cache.popitem(last=False)
            self._close_shard_bundle(bundle)

    def _load_shard(self, shard_index: int) -> _ShardBundle:
        cached = self._shard_cache.get(shard_index)
        if cached is not None:
            self._shard_cache.move_to_end(shard_index)
            return cached
        spec = self._shards[shard_index]
        arrays_dir = spec.shard_dir / "arrays"
        arrays: dict[str, np.ndarray] = {
            "state": np.load(arrays_dir / "state.npy", mmap_mode="r"),
            "tactile": np.load(arrays_dir / "tactile.npy", mmap_mode="r"),
            "actions": np.load(arrays_dir / "actions.npy", mmap_mode="r"),
            "action_mask": np.load(arrays_dir / "action_mask.npy", mmap_mode="r"),
            "sample_weight": np.load(arrays_dir / "sample_weight.npy", mmap_mode="r"),
            "frame_position": np.load(arrays_dir / "frame_position.npy", mmap_mode="r"),
            "frame_index": np.load(arrays_dir / "frame_index.npy", mmap_mode="r"),
            "timestamp": np.load(arrays_dir / "timestamp.npy", mmap_mode="r"),
        }
        for image_key in self._settings.image_modalities:
            arrays[f"{IMAGE_ARRAY_PREFIX}{image_key}"] = np.load(
                arrays_dir / f"{IMAGE_ARRAY_PREFIX}{image_key}.npy",
                mmap_mode="r",
            )
        if self._settings.load_tactile_images:
            arrays["tactile_deform_images"] = np.load(arrays_dir / "tactile_deform_images.npy", mmap_mode="r")
            arrays["tactile_raw_images"] = np.load(arrays_dir / "tactile_raw_images.npy", mmap_mode="r")
            deform_path = arrays_dir / "tactile_deform_available.npy"
            arrays["tactile_deform_available"] = (
                np.load(deform_path, mmap_mode="r")
                if deform_path.exists()
                else np.ones((spec.num_rows,), dtype=bool)
            )
            arrays["tactile_raw_available"] = np.load(arrays_dir / "tactile_raw_available.npy", mmap_mode="r")
        if self._settings.include_planner_features:
            arrays["planner_available"] = np.load(arrays_dir / "planner_available.npy", mmap_mode="r")
            arrays["planner_state_belief"] = np.load(arrays_dir / "planner_state_belief.npy", mmap_mode="r")
            arrays["planner_progress_transition"] = np.load(
                arrays_dir / "planner_progress_transition.npy",
                mmap_mode="r",
            )
            arrays["planner_uncertainty"] = np.load(arrays_dir / "planner_uncertainty.npy", mmap_mode="r")
            arrays["planner_history_latent"] = np.load(arrays_dir / "planner_history_latent.npy", mmap_mode="r")
        row_order_path = arrays_dir / "row_order.npy"
        row_order = (
            np.load(row_order_path, mmap_mode="r")
            if self._settings.shard_use_stored_row_order and row_order_path.exists()
            else None
        )
        bundle = _ShardBundle(
            spec=spec,
            rows=load_shard_rows_file(spec.rows_path),
            row_order=row_order,
            arrays=arrays,
        )
        self._shard_cache[shard_index] = bundle
        self._enforce_cache_limit()
        return bundle

    def _locate(self, index: int) -> tuple[_ShardBundle, int, int]:
        index = int(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_index = bisect_right(self._cumulative_rows, index)
        previous_total = 0 if shard_index == 0 else self._cumulative_rows[shard_index - 1]
        local_index = index - previous_total
        bundle = self._load_shard(shard_index)
        physical_index = int(bundle.row_order[local_index]) if bundle.row_order is not None else int(local_index)
        return bundle, local_index, physical_index

    def __getitem__(self, index: int) -> dict[str, Any]:
        bundle, _local_index, physical_index = self._locate(index)
        arrays = bundle.arrays
        images = {
            image_key: np.asarray(arrays[f"{IMAGE_ARRAY_PREFIX}{image_key}"][physical_index], dtype=np.uint8)
            for image_key in self._settings.image_modalities
        }
        image_masks = {image_key: np.asarray(True) for image_key in images}
        tactile = np.asarray(arrays["tactile"][physical_index], dtype=np.float32)
        output: dict[str, Any] = {
            "image": images,
            "image_mask": image_masks,
            "state": np.asarray(arrays["state"][physical_index], dtype=np.float32),
            "tactile": tactile,
            "tactile_prompt": np.array(tactile, copy=True),
            "tactile_prompt_mask": np.ones((self._settings.tactile_dim,), dtype=bool),
            "state_mask": np.ones((self._settings.state_dim,), dtype=bool),
            # DeltaActions mutates action targets in place, while shard arrays are read-only memmap views.
            "actions": np.array(arrays["actions"][physical_index], dtype=np.float32, copy=True),
            "action_mask": np.asarray(arrays["action_mask"][physical_index], dtype=bool),
            "sample_weight": np.asarray(arrays["sample_weight"][physical_index], dtype=np.float32),
            "prompt": np.asarray(self._settings.prompt),
            "frame_position": np.asarray(arrays["frame_position"][physical_index], dtype=np.int64),
            "frame_index": np.asarray(arrays["frame_index"][physical_index], dtype=np.int64),
            "timestamp": np.asarray(arrays["timestamp"][physical_index], dtype=np.float32),
            "source_row_index": np.asarray(
                int(bundle.rows.iloc[physical_index].get("source_row_index", -1)),
                dtype=np.int64,
            ),
        }
        if self._settings.include_planner_features:
            output.update(
                {
                    "planner_available": np.asarray(arrays["planner_available"][physical_index], dtype=bool),
                    "planner_state_belief": np.asarray(
                        arrays["planner_state_belief"][physical_index],
                        dtype=np.float32,
                    ),
                    "planner_progress_transition": np.asarray(
                        arrays["planner_progress_transition"][physical_index], dtype=np.float32
                    ),
                    "planner_uncertainty": np.asarray(arrays["planner_uncertainty"][physical_index], dtype=np.float32),
                    "planner_history_latent": np.asarray(
                        arrays["planner_history_latent"][physical_index],
                        dtype=np.float32,
                    ),
                }
            )
        if self._settings.load_tactile_images:
            output.update(
                {
                    "tactile_deform_images": np.asarray(
                        arrays["tactile_deform_images"][physical_index],
                        dtype=np.uint8,
                    ),
                    "tactile_raw_images": np.asarray(arrays["tactile_raw_images"][physical_index], dtype=np.uint8),
                    "tactile_deform_available": np.asarray(
                        arrays["tactile_deform_available"][physical_index], dtype=bool
                    ),
                    "tactile_raw_available": np.asarray(arrays["tactile_raw_available"][physical_index], dtype=bool),
                }
            )
        return output
