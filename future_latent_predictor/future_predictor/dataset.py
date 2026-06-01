from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Iterator

import numpy as np
import polars as pl
import torch
from torch.utils.data import IterableDataset
from torch.utils.data import get_worker_info

from future_latent_predictor.resampler_autoencoder.model import CAMERAS


@dataclass(frozen=True)
class FuturePredictionDataConfig:
    data_root: Path
    include_datasets: tuple[str, ...] = ()
    cameras: tuple[str, ...] = CAMERAS
    embedding_shape: tuple[int, int] = (256, 2048)
    shuffle_files: bool = True
    shuffle_rows: bool = True
    seed: int = 42
    split: str = "train"
    val_fraction: float = 0.0
    dataset_weights: dict[str, float] | None = None


class FuturePredictionDataset(IterableDataset):
    def __init__(
        self,
        config: FuturePredictionDataConfig,
        *,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        super().__init__()
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0
        all_files = find_parquet_files(config.data_root, config.include_datasets)
        self.files = split_files(
            all_files,
            data_root=config.data_root,
            split=config.split,
            val_fraction=config.val_fraction,
            seed=config.seed,
        )
        self.weighted_file_groups = weighted_file_groups(
            self.files,
            data_root=config.data_root,
            dataset_weights=config.dataset_weights or {},
        )
        if not self.files:
            raise FileNotFoundError(
                f"No parquet shards found for split {config.split!r} under {config.data_root}. "
                f"Check data.val_fraction and include_datasets."
            )
        if config.dataset_weights and not self.weighted_file_groups:
            raise ValueError("dataset_weights disabled every dataset; at least one dataset must have weight > 0.")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_data_workers = worker_info.num_workers if worker_info is not None else 1
        global_worker_id = self.rank * num_data_workers + worker_id
        num_global_workers = self.world_size * num_data_workers

        if self.config.dataset_weights:
            yield from self._iter_weighted_files(global_worker_id=global_worker_id)
            return

        files = list(self.files)
        if self.config.shuffle_files:
            rng = random.Random(self.config.seed + self.epoch)
            rng.shuffle(files)
        files = [
            path
            for index, path in enumerate(files)
            if index % num_global_workers == global_worker_id
        ]
        if not files:
            return

        for file_index, path in enumerate(files):
            yield from self._iter_file(path, file_seed=self.config.seed + self.epoch * 1_000_003 + file_index)

    def _iter_weighted_files(self, *, global_worker_id: int) -> Iterator[dict[str, torch.Tensor]]:
        dataset_names = list(self.weighted_file_groups)
        if not dataset_names:
            return
        weights = [self.weighted_file_groups[name]["weight"] for name in dataset_names]
        rng = random.Random(self.config.seed + self.epoch * 1_000_003 + global_worker_id * 97_531)
        file_counter = 0
        while True:
            dataset_name = rng.choices(dataset_names, weights=weights, k=1)[0]
            files = self.weighted_file_groups[dataset_name]["files"]
            path = rng.choice(files)
            yield from self._iter_file(path, file_seed=rng.randint(0, 2**31 - 1) + file_counter)
            file_counter += 1

    def _iter_file(self, path: Path, *, file_seed: int) -> Iterator[dict[str, torch.Tensor]]:
        frame = pl.read_parquet(path, columns=required_columns(self.config.cameras))
        row_indices = list(range(frame.height))
        if self.config.shuffle_rows:
            rng = random.Random(file_seed)
            rng.shuffle(row_indices)

        for row_index in row_indices:
            row = frame.row(row_index, named=True)
            sample = sample_from_row(
                row,
                cameras=self.config.cameras,
                embedding_shape=self.config.embedding_shape,
            )
            if sample is not None:
                yield sample


def find_parquet_files(data_root: Path, include_datasets: tuple[str, ...]) -> list[Path]:
    if include_datasets:
        roots = [data_root / dataset_name for dataset_name in include_datasets]
    else:
        roots = [data_root]
    files: list[Path] = []
    for root in roots:
        files.extend(sorted(root.rglob("*.parquet")))
    return sorted(files)


def parquet_file_summary(
    *,
    data_root: Path,
    include_datasets: tuple[str, ...] = (),
    val_fraction: float = 0.0,
    seed: int = 42,
) -> dict[str, object]:
    if include_datasets:
        roots = [(dataset_name, data_root / dataset_name) for dataset_name in include_datasets]
    else:
        roots = [(data_root.name, data_root)]

    root_summaries: list[dict[str, object]] = []
    all_files: list[Path] = []
    for dataset_name, root in roots:
        files = sorted(root.rglob("*.parquet"))
        all_files.extend(files)
        root_summaries.append(
            {
                "dataset": dataset_name,
                "path": str(root),
                "exists": root.exists(),
                "is_symlink": root.is_symlink(),
                "resolved_path": str(root.resolve()) if root.exists() or root.is_symlink() else None,
                "parquet_files": len(files),
            }
        )

    all_files = sorted(all_files)
    train_files = split_files(
        all_files,
        data_root=data_root,
        split="train",
        val_fraction=val_fraction,
        seed=seed,
    )
    val_files = split_files(
        all_files,
        data_root=data_root,
        split="val",
        val_fraction=val_fraction,
        seed=seed,
    )

    def split_summary(files: list[Path]) -> dict[str, object]:
        groups = group_files_by_dataset(files, data_root=data_root)
        return {
            "total_parquet_files": len(files),
            "datasets": {dataset_name: len(dataset_files) for dataset_name, dataset_files in groups.items()},
        }

    return {
        "data_root": str(data_root),
        "include_datasets": list(include_datasets),
        "configured_roots": root_summaries,
        "total_parquet_files": len(all_files),
        "missing_or_empty_datasets": [
            summary["dataset"] for summary in root_summaries if int(summary["parquet_files"]) == 0
        ],
        "train": split_summary(train_files),
        "val": split_summary(val_files),
    }


def split_files(files: list[Path], *, data_root: Path, split: str, val_fraction: float, seed: int) -> list[Path]:
    if split not in {"train", "val"}:
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")
    if val_fraction == 0.0 or len(files) < 2:
        return files if split == "train" else []

    val_files: set[Path] = set()
    for dataset_name, dataset_files in group_files_by_dataset(files, data_root=data_root).items():
        if len(dataset_files) < 2:
            continue
        shuffled = list(dataset_files)
        random.Random(seed + stable_int(dataset_name)).shuffle(shuffled)
        val_count = max(1, round(len(shuffled) * val_fraction))
        val_count = min(val_count, len(shuffled) - 1)
        val_files.update(shuffled[:val_count])
    if split == "val":
        return sorted(val_files)
    return sorted(path for path in files if path not in val_files)


def dataset_name_for_file(path: Path, *, data_root: Path) -> str:
    try:
        relative = path.relative_to(data_root)
    except ValueError:
        return path.parent.name
    if len(relative.parts) > 1:
        return relative.parts[0]
    return path.parent.name


def group_files_by_dataset(files: list[Path], *, data_root: Path) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in files:
        groups[dataset_name_for_file(path, data_root=data_root)].append(path)
    return {name: sorted(paths) for name, paths in groups.items()}


def stable_int(value: str) -> int:
    total = 0
    for index, character in enumerate(value):
        total += (index + 1) * ord(character)
    return total


def weighted_file_groups(
    files: list[Path],
    *,
    data_root: Path,
    dataset_weights: dict[str, float],
) -> dict[str, dict[str, object]]:
    groups = group_files_by_dataset(files, data_root=data_root)
    weighted_groups: dict[str, dict[str, object]] = {}
    for dataset_name, dataset_files in groups.items():
        weight = float(dataset_weights.get(dataset_name, dataset_weights.get(dataset_name.replace("__", "/"), 1.0)))
        if weight <= 0.0:
            continue
        weighted_groups[dataset_name] = {"files": dataset_files, "weight": weight}
    return weighted_groups


def estimate_num_prediction_samples(
    *,
    data_root: Path,
    include_datasets: tuple[str, ...] = (),
    split: str = "train",
    val_fraction: float = 0.0,
    seed: int = 42,
) -> int:
    total_rows = 0
    files = split_files(
        find_parquet_files(data_root, include_datasets),
        data_root=data_root,
        split=split,
        val_fraction=val_fraction,
        seed=seed,
    )
    for path in files:
        total_rows += pl.scan_parquet(path).select(pl.len()).collect().item()
    return total_rows


def infer_state_dim(*, data_root: Path, include_datasets: tuple[str, ...] = ()) -> int:
    files = find_parquet_files(data_root, include_datasets)
    if not files:
        raise FileNotFoundError(f"No parquet shards found under {data_root}")
    frame = pl.scan_parquet(files[0]).select("state").limit(1).collect()
    if frame.height == 0:
        raise ValueError(f"Cannot infer state_dim from empty parquet shard: {files[0]}")
    return len(frame.row(0, named=True)["state"])


def required_columns(cameras: tuple[str, ...]) -> list[str]:
    columns = ["embedding_dtype", "state"]
    for camera in cameras:
        columns.extend(
            [
                f"{camera}_embedding_valid",
                f"{camera}_embedding_t_5_valid",
                f"{camera}_embedding_t",
                f"{camera}_embedding_t_5",
            ]
        )
    return columns


def sample_from_row(
    row: dict[str, object],
    *,
    cameras: tuple[str, ...],
    embedding_shape: tuple[int, int],
) -> dict[str, torch.Tensor] | None:
    dtype = np.dtype(str(row["embedding_dtype"]))
    current = np.zeros((len(cameras), *embedding_shape), dtype=np.float16)
    future = np.zeros((len(cameras), *embedding_shape), dtype=np.float16)
    valid = np.zeros((len(cameras),), dtype=np.bool_)

    for camera_index, camera in enumerate(cameras):
        current_valid = bool(row[f"{camera}_embedding_valid"])
        future_valid = bool(row[f"{camera}_embedding_t_5_valid"])
        is_valid = current_valid and future_valid
        valid[camera_index] = is_valid
        if not is_valid:
            continue

        current_value = row[f"{camera}_embedding_t"]
        future_value = row[f"{camera}_embedding_t_5"]
        if current_value is None or future_value is None:
            valid[camera_index] = False
            continue
        current[camera_index] = np.frombuffer(current_value, dtype=dtype).reshape(embedding_shape).astype(
            np.float16,
            copy=False,
        )
        future[camera_index] = np.frombuffer(future_value, dtype=dtype).reshape(embedding_shape).astype(
            np.float16,
            copy=False,
        )

    if not valid.any():
        return None

    return {
        "current_embeddings": torch.from_numpy(current),
        "future_embeddings": torch.from_numpy(future),
        "valid": torch.from_numpy(valid),
        "state": torch.tensor(row["state"], dtype=torch.float32),
    }
