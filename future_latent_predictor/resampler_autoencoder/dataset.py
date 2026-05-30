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
class GeneratedEmbeddingDataConfig:
    data_root: Path
    include_datasets: tuple[str, ...] = ()
    cameras: tuple[str, ...] = CAMERAS
    embedding_shape: tuple[int, int] = (256, 2048)
    use_t: bool = True
    use_t_5: bool = True
    shuffle_files: bool = True
    shuffle_rows: bool = True
    seed: int = 42
    split: str = "train"
    val_fraction: float = 0.0


class GeneratedEmbeddingDataset(IterableDataset):
    def __init__(
        self,
        config: GeneratedEmbeddingDataConfig,
        *,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        super().__init__()
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0
        self.files = _split_files(
            _find_parquet_files(config.data_root, config.include_datasets),
            split=config.split,
            val_fraction=config.val_fraction,
            seed=config.seed,
        )
        if not self.files:
            raise FileNotFoundError(
                f"No parquet shards found for split {config.split!r} under {config.data_root}. "
                f"Check data.val_fraction and include_datasets."
            )
        if not config.use_t and not config.use_t_5:
            raise ValueError("At least one of use_t or use_t_5 must be true.")

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
        columns = _required_columns(self.config.cameras)
        frame = pl.read_parquet(path, columns=columns)
        row_indices = list(range(frame.height))
        if self.config.shuffle_rows:
            rng = random.Random(file_seed)
            rng.shuffle(row_indices)

        for row_index in row_indices:
            row = frame.row(row_index, named=True)
            if self.config.use_t:
                sample = _sample_from_row(
                    row,
                    cameras=self.config.cameras,
                    embedding_shape=self.config.embedding_shape,
                    suffix="t",
                )
                if sample is not None:
                    yield sample
            if self.config.use_t_5:
                sample = _sample_from_row(
                    row,
                    cameras=self.config.cameras,
                    embedding_shape=self.config.embedding_shape,
                    suffix="t_5",
                )
                if sample is not None:
                    yield sample


def estimate_num_embedding_samples(
    *,
    data_root: Path,
    include_datasets: tuple[str, ...] = (),
    use_t: bool = True,
    use_t_5: bool = True,
    split: str = "train",
    val_fraction: float = 0.0,
    seed: int = 42,
) -> int:
    files = _split_files(
        _find_parquet_files(data_root, include_datasets),
        split=split,
        val_fraction=val_fraction,
        seed=seed,
    )
    multiplier = int(use_t) + int(use_t_5)
    total_rows = 0
    for path in files:
        total_rows += pl.scan_parquet(path).select(pl.len()).collect().item()
    return total_rows * multiplier


def _find_parquet_files(data_root: Path, include_datasets: tuple[str, ...]) -> list[Path]:
    if include_datasets:
        roots = [data_root / dataset_name for dataset_name in include_datasets]
    else:
        roots = [data_root]
    files: list[Path] = []
    for root in roots:
        files.extend(sorted(root.rglob("*.parquet")))
    return sorted(files)


def _split_files(files: list[Path], *, split: str, val_fraction: float, seed: int) -> list[Path]:
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


def _required_columns(cameras: tuple[str, ...]) -> list[str]:
    columns = ["embedding_dtype"]
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


def _sample_from_row(
    row: dict[str, object],
    *,
    cameras: tuple[str, ...],
    embedding_shape: tuple[int, int],
    suffix: str,
) -> dict[str, torch.Tensor] | None:
    dtype = np.dtype(str(row["embedding_dtype"]))
    embeddings = np.zeros((len(cameras), *embedding_shape), dtype=np.float16)
    valid = np.zeros((len(cameras),), dtype=np.bool_)

    for camera_index, camera in enumerate(cameras):
        valid_key = f"{camera}_embedding_valid" if suffix == "t" else f"{camera}_embedding_t_5_valid"
        bytes_key = f"{camera}_embedding_{suffix}"
        is_valid = bool(row[valid_key])
        valid[camera_index] = is_valid
        if not is_valid:
            continue
        value = row[bytes_key]
        if value is None:
            valid[camera_index] = False
            continue
        array = np.frombuffer(value, dtype=dtype).reshape(embedding_shape)
        embeddings[camera_index] = array.astype(np.float16, copy=False)

    if not valid.any():
        return None

    return {
        "embeddings": torch.from_numpy(embeddings),
        "valid": torch.from_numpy(valid),
    }
