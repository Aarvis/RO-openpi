from __future__ import annotations

from collections import OrderedDict
import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

import openpi.transforms as _transforms


def _scalar(value: Any) -> int | float | str | None:
    if value is None:
        return None
    array = np.asarray(value)
    if array.shape == ():
        return array.item()
    if array.size == 1:
        return array.reshape(()).item()
    return None


def _episode_sidecar_path(sidecar_root: Path, episode_index: int) -> Path:
    return sidecar_root / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}" / "predicted_robot_local_splines.npz"


def _episode_parquet_path(dataset_root: Path, episode_index: int) -> Path:
    return dataset_root / "data" / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}.parquet"


def _parse_episode_index(path: Path) -> int:
    return int(path.parent.name.split("_")[-1])


def _resolve_dataset_root(dataset: Any, source_repo_id: str) -> Path:
    candidates = [source_repo_id]
    for attr in ("root", "repo_path", "dataset_path", "data_dir", "repo_id"):
        value = getattr(dataset, attr, None)
        if value is not None:
            candidates.append(str(value))
    for candidate in candidates:
        path = Path(candidate)
        if path.exists() and (path / "data").exists() and (path / "meta").exists():
            return path
    raise FileNotFoundError(
        "Could not resolve a LeRobot dataset root for robot spline sidecar expansion. "
        f"Tried source_repo_id={source_repo_id!r} and dataset attributes."
    )


@dataclasses.dataclass(frozen=True)
class _ExpandedItem:
    dataset_index: int
    pairing_slot: int


@dataclasses.dataclass
class RobotSplineExpandedDataset:
    dataset: Any
    sidecar_root: str
    source_repo_id: str
    required: bool = True

    def __post_init__(self) -> None:
        dataset_root = _resolve_dataset_root(self.dataset, self.source_repo_id)
        self._dataset_root = dataset_root
        self._sidecar_root = Path(self.sidecar_root)
        self._items = self._build_items()

    def _build_items(self) -> list[_ExpandedItem]:
        files = sorted(self._sidecar_root.glob("chunk-*/episode_*/predicted_robot_local_splines.npz"))
        if not files:
            if self.required:
                raise FileNotFoundError(f"No predicted robot spline sidecar files found under {self._sidecar_root}")
            return []

        items: list[_ExpandedItem] = []
        for sidecar_path in files:
            episode_index = _parse_episode_index(sidecar_path)
            parquet_path = _episode_parquet_path(self._dataset_root, episode_index)
            if not parquet_path.exists():
                if self.required:
                    raise FileNotFoundError(parquet_path)
                continue

            table = pq.read_table(parquet_path, columns=["index", "frame_index"])
            frame_to_dataset_index = {
                int(frame_index): int(dataset_index)
                for dataset_index, frame_index in zip(
                    table["index"].to_numpy(),
                    table["frame_index"].to_numpy(),
                    strict=True,
                )
            }

            with np.load(sidecar_path, allow_pickle=False) as archive:
                frame_indices = np.asarray(archive["frame_indices"], dtype=np.int64)
                prediction_valid_mask = np.asarray(archive["prediction_valid_mask"], dtype=bool)

            if prediction_valid_mask.ndim != 2:
                raise ValueError(
                    f"Expected prediction_valid_mask to have shape [frames, slots], got {prediction_valid_mask.shape} "
                    f"for {sidecar_path}"
                )

            for row_index, frame_index in enumerate(frame_indices.tolist()):
                dataset_index = frame_to_dataset_index.get(int(frame_index))
                if dataset_index is None:
                    if self.required:
                        raise KeyError(
                            f"Frame index {frame_index} from {sidecar_path} not found in {parquet_path}"
                        )
                    continue
                valid_slots = np.flatnonzero(prediction_valid_mask[row_index])
                for slot_index in valid_slots.tolist():
                    items.append(_ExpandedItem(dataset_index=dataset_index, pairing_slot=int(slot_index)))
        return items

    def __getitem__(self, index) -> dict:
        item = self._items[index]
        sample = dict(self.dataset[item.dataset_index])
        sample["robot_spline_slot"] = np.asarray(item.pairing_slot, dtype=np.int32)
        sample.setdefault("source_index", np.asarray(item.dataset_index, dtype=np.int64))
        return sample

    def __len__(self) -> int:
        return len(self._items)


@dataclasses.dataclass
class _EpisodeArchive:
    frame_to_row: dict[int, int]
    prediction_valid_mask: np.ndarray
    coefficients: np.ndarray
    knots: np.ndarray


@dataclasses.dataclass
class RobotSplineSidecarTransform(_transforms.DataTransformFn):
    sidecar_root: str
    required: bool = True
    cache_size: int = 4

    def __post_init__(self) -> None:
        self._sidecar_root = Path(self.sidecar_root)
        self._cache: OrderedDict[int, _EpisodeArchive] = OrderedDict()
        self._empty_shapes = self._discover_shapes()

    def __call__(self, data: _transforms.DataDict) -> _transforms.DataDict:
        episode_index = _scalar(data.get("episode_index"))
        frame_index = _scalar(data.get("frame_index"))
        slot = _scalar(data.get("robot_spline_slot"))
        if episode_index is None or frame_index is None or slot is None:
            if self.required:
                raise KeyError(
                    "robot spline sidecar lookup requires episode_index, frame_index, and robot_spline_slot in the sample"
                )
            data["robot_spline"] = self._empty_robot_spline()
            return data

        archive = self._load_episode_archive(int(episode_index))
        row_index = archive.frame_to_row.get(int(frame_index))
        slot_index = int(slot)
        if row_index is None or slot_index < 0 or slot_index >= archive.prediction_valid_mask.shape[1]:
            if self.required:
                raise KeyError(
                    f"No robot spline sidecar entry for episode={episode_index}, frame={frame_index}, slot={slot_index}"
                )
            data["robot_spline"] = self._empty_robot_spline()
            return data
        if not bool(archive.prediction_valid_mask[row_index, slot_index]):
            if self.required:
                raise ValueError(
                    f"Robot spline sidecar entry is invalid for episode={episode_index}, frame={frame_index}, slot={slot_index}"
                )
            data["robot_spline"] = self._empty_robot_spline()
            return data

        data["robot_spline"] = {
            "coefficients": archive.coefficients[row_index, slot_index].astype(np.float32, copy=False),
            "knots": archive.knots[row_index, slot_index].astype(np.float32, copy=False),
        }
        return data

    def _discover_shapes(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        first_file = next(iter(sorted(self._sidecar_root.glob("chunk-*/episode_*/predicted_robot_local_splines.npz"))), None)
        if first_file is None:
            return (0, 0), (0,)
        with np.load(first_file, allow_pickle=False) as archive:
            coeff_shape = tuple(np.asarray(archive["predicted_robot_coefficients"]).shape[-2:])
            knot_shape = tuple(np.asarray(archive["predicted_robot_knots"]).shape[-1:])
        return coeff_shape, knot_shape

    def _empty_robot_spline(self) -> dict[str, np.ndarray]:
        coeff_shape, knot_shape = self._empty_shapes
        return {
            "coefficients": np.zeros(coeff_shape, dtype=np.float32),
            "knots": np.zeros(knot_shape, dtype=np.float32),
        }

    def _load_episode_archive(self, episode_index: int) -> _EpisodeArchive:
        cached = self._cache.get(episode_index)
        if cached is not None:
            self._cache.move_to_end(episode_index)
            return cached

        path = _episode_sidecar_path(self._sidecar_root, episode_index)
        if not path.exists():
            if self.required:
                raise FileNotFoundError(path)
            coeff_shape, knot_shape = self._empty_shapes
            archive = _EpisodeArchive(
                frame_to_row={},
                prediction_valid_mask=np.zeros((0, 0), dtype=bool),
                coefficients=np.zeros((0, 0, *coeff_shape), dtype=np.float32),
                knots=np.zeros((0, 0, *knot_shape), dtype=np.float32),
            )
            return archive

        with np.load(path, allow_pickle=False) as loaded:
            frame_indices = np.asarray(loaded["frame_indices"], dtype=np.int64)
            prediction_valid_mask = np.asarray(loaded["prediction_valid_mask"], dtype=bool)
            coefficients = np.asarray(loaded["predicted_robot_coefficients"], dtype=np.float32)
            knots = np.asarray(loaded["predicted_robot_knots"], dtype=np.float32)

        archive = _EpisodeArchive(
            frame_to_row={int(frame_index): row_index for row_index, frame_index in enumerate(frame_indices.tolist())},
            prediction_valid_mask=prediction_valid_mask,
            coefficients=coefficients,
            knots=knots,
        )
        self._cache[episode_index] = archive
        self._cache.move_to_end(episode_index)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return archive
