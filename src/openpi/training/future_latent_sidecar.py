from __future__ import annotations

from collections import OrderedDict
import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

import openpi.shared.future_latent_order as _future_latent_order
import openpi.transforms as _transforms


CAMERAS = _future_latent_order.SIDECAR_CAMERA_COLUMNS
SAFE_CAMERA_TO_MODEL_ORDER = _future_latent_order.POLICY_FUTURE_LATENT_CAMERA_ORDER
MODEL_CAMERA_TO_SIDECAR_CAMERA = {
    "top": "top",
    "left_wrist": "left_wrist",
    "right_wrist": "right_wrist",
}


def safe_dataset_name(repo_id: str) -> str:
    return repo_id.replace("/", "__").replace("\\", "__")


def _scalar(value: Any) -> int | float | str | None:
    if value is None:
        return None
    array = np.asarray(value)
    if array.shape == ():
        return array.item()
    if array.size == 1:
        return array.reshape(()).item()
    return None


def _decode_latent(value: Any, *, dtype: np.dtype, shape: tuple[int, int]) -> np.ndarray:
    if value is None:
        return np.zeros(shape, dtype=np.float32)
    return np.frombuffer(value, dtype=dtype).reshape(shape).astype(np.float32, copy=False)


@dataclasses.dataclass(frozen=True)
class _SidecarEntry:
    path: Path
    row_index: int


@dataclasses.dataclass
class FutureLatentSidecarTransform(_transforms.DataTransformFn):
    sidecar_root: str
    source_repo_id: str
    required: bool = True
    cache_size: int = 2

    def __post_init__(self) -> None:
        self._dataset_root = Path(self.sidecar_root) / safe_dataset_name(self.source_repo_id)
        self._index: dict[tuple[str, Any], _SidecarEntry] | None = None
        self._cache: OrderedDict[Path, pl.DataFrame] = OrderedDict()

    def __call__(self, data: _transforms.DataDict) -> _transforms.DataDict:
        entry = self._lookup_entry(data)
        if entry is None:
            if self.required:
                keys = {
                    "index": _scalar(data.get("index")),
                    "episode_index": _scalar(data.get("episode_index")),
                    "frame_index": _scalar(data.get("frame_index")),
                }
                raise KeyError(
                    f"No future latent sidecar row found for {self.source_repo_id} with keys {keys} "
                    f"under {self._dataset_root}"
                )
            data["future_latent"] = self._empty_future_latent()
            return data

        frame = self._load_frame(entry.path)
        row = frame.row(entry.row_index, named=True)
        data["future_latent"] = self._future_latent_from_row(row)
        return data

    def _ensure_index(self) -> dict[tuple[str, Any], _SidecarEntry]:
        if self._index is not None:
            return self._index
        files = sorted(self._dataset_root.glob("*.parquet"))
        if not files:
            if self.required:
                raise FileNotFoundError(f"No future latent sidecar parquet files found under {self._dataset_root}")
            self._index = {}
            return self._index

        index: dict[tuple[str, Any], _SidecarEntry] = {}
        for path in files:
            schema = pl.read_parquet_schema(path)
            columns = [column for column in ("index", "source_index", "episode_index", "frame_index", "task_index") if column in schema]
            if not columns:
                continue
            metadata = pl.scan_parquet(path).select(columns).with_row_index("__row_index").collect()
            for row in metadata.iter_rows(named=True):
                entry = _SidecarEntry(path=path, row_index=int(row["__row_index"]))
                for key in ("index", "source_index"):
                    value = _scalar(row.get(key))
                    if value is not None:
                        index.setdefault((key, int(value)), entry)
                episode = _scalar(row.get("episode_index"))
                frame = _scalar(row.get("frame_index"))
                if episode is not None and frame is not None:
                    index.setdefault(("episode_frame", (int(episode), int(frame))), entry)
        self._index = index
        return self._index

    def _lookup_entry(self, data: _transforms.DataDict) -> _SidecarEntry | None:
        index = self._ensure_index()
        for key in ("index", "source_index"):
            value = _scalar(data.get(key))
            if value is not None:
                entry = index.get((key, int(value)))
                if entry is not None:
                    return entry
        episode = _scalar(data.get("episode_index"))
        frame = _scalar(data.get("frame_index"))
        if episode is not None and frame is not None:
            return index.get(("episode_frame", (int(episode), int(frame))))
        return None

    def _load_frame(self, path: Path) -> pl.DataFrame:
        cached = self._cache.get(path)
        if cached is not None:
            self._cache.move_to_end(path)
            return cached

        schema = pl.read_parquet_schema(path)
        columns = [
            column
            for column in self._sidecar_columns()
            if column in schema
        ]
        frame = pl.read_parquet(path, columns=columns)
        self._cache[path] = frame
        self._cache.move_to_end(path)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return frame

    def _sidecar_columns(self) -> list[str]:
        columns = ["latent_dtype", "latent_shape"]
        for camera in CAMERAS:
            columns.extend(
                [
                    f"{camera}_future_latent_pred",
                    f"{camera}_future_latent_true",
                    f"{camera}_future_latent_valid",
                ]
            )
        return columns

    def _future_latent_from_row(self, row: dict[str, Any]) -> dict[str, np.ndarray]:
        dtype = np.dtype(str(row.get("latent_dtype", "float16")))
        shape_value = row.get("latent_shape", [24, 512])
        shape = tuple(int(value) for value in shape_value)
        pred = []
        true = []
        valid = []
        for model_camera in SAFE_CAMERA_TO_MODEL_ORDER:
            camera = MODEL_CAMERA_TO_SIDECAR_CAMERA[model_camera]
            camera_valid = bool(row.get(f"{camera}_future_latent_valid", False))
            valid.append(camera_valid)
            pred.append(
                _decode_latent(row.get(f"{camera}_future_latent_pred"), dtype=dtype, shape=shape)
                if camera_valid
                else np.zeros(shape, dtype=np.float32)
            )
            true.append(
                _decode_latent(row.get(f"{camera}_future_latent_true"), dtype=dtype, shape=shape)
                if camera_valid
                else np.zeros(shape, dtype=np.float32)
            )
        return {
            "pred": np.stack(pred, axis=0).astype(np.float32, copy=False),
            "true": np.stack(true, axis=0).astype(np.float32, copy=False),
            "valid_mask": np.asarray(valid, dtype=np.bool_),
        }

    def _empty_future_latent(self) -> dict[str, np.ndarray]:
        shape = (len(SAFE_CAMERA_TO_MODEL_ORDER), 24, 512)
        return {
            "pred": np.zeros(shape, dtype=np.float32),
            "true": np.zeros(shape, dtype=np.float32),
            "valid_mask": np.zeros((len(SAFE_CAMERA_TO_MODEL_ORDER),), dtype=np.bool_),
        }
