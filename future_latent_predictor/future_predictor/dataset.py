from __future__ import annotations

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
        self.files = split_files(
            find_parquet_files(config.data_root, config.include_datasets),
            split=config.split,
            val_fraction=config.val_fraction,
            seed=config.seed,
        )
        if not self.files:
            raise FileNotFoundError(
                f"No parquet shards found for split {config.split!r} under {config.data_root}. "
                f"Check data.val_fraction and include_datasets."
            )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_data_workers = worker_info.num_workers if worker_info is not None else 1
        global_worker_id = self.rank * num_data_workers + worker_id
        num_global_workers = self.world_size * num_data_workers

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


def split_files(files: list[Path], *, split: str, val_fraction: float, seed: int) -> list[Path]:
    if split not in {"train", "val"}:
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")
    if val_fraction == 0.0 or len(files) < 2:
        return files if split == "train" else []

    shuffled = list(files)
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, round(len(shuffled) * val_fraction))
    val_count = min(val_count, len(shuffled) - 1)
    val_files = set(shuffled[:val_count])
    if split == "val":
        return sorted(val_files)
    return sorted(path for path in shuffled if path not in val_files)


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
