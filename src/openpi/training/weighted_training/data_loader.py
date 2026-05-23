from __future__ import annotations

import logging

import jax
import numpy as np
import torch
import tqdm_loggable.auto as tqdm

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader


def _coerce_weight(value) -> float:
    if value is None:
        return 1.0
    arr = np.asarray(value, dtype=np.float64)
    if arr.size == 0:
        return 1.0
    weight = float(arr.reshape(-1)[0])
    if not np.isfinite(weight) or weight <= 0.0:
        return 1.0
    return weight


def _unwrap_dataset(dataset):
    cur = dataset
    seen = set()
    while True:
        next_dataset = getattr(cur, "_dataset", None)
        if next_dataset is None or id(next_dataset) in seen:
            return cur
        seen.add(id(cur))
        cur = next_dataset


def _extract_sample_weights_fast(dataset) -> torch.Tensor | None:
    raw_dataset = _unwrap_dataset(dataset)
    for attr in ("hf_dataset", "_hf_dataset", "dataset", "_dataset"):
        candidate = getattr(raw_dataset, attr, None)
        if candidate is None:
            continue
        column_names = getattr(candidate, "column_names", None)
        if column_names is None or "sample_weight" not in column_names:
            continue
        try:
            weights = candidate["sample_weight"]
            weights_tensor = torch.as_tensor([_coerce_weight(v) for v in weights], dtype=torch.double)
            if weights_tensor.numel() == 0:
                raise ValueError("Cannot build weighted sampler for an empty dataset.")
            logging.info("Loaded sample weights from dataset column without per-sample decoding.")
            return weights_tensor
        except Exception as exc:
            logging.warning("Fast sample_weight extraction failed, falling back to row scan: %s", exc)
            return None
    return None


def _extract_sample_weights(dataset: _data_loader.Dataset) -> torch.Tensor:
    fast_weights = _extract_sample_weights_fast(dataset)
    if fast_weights is not None:
        return fast_weights

    dataset_len = len(dataset)
    logging.info("Extracting sample weights by scanning %d dataset rows.", dataset_len)
    weights = []
    for i in tqdm.tqdm(range(dataset_len), desc="Extracting sample weights", unit="sample", dynamic_ncols=True):
        sample = dataset[i]
        weights.append(_coerce_weight(sample.get("sample_weight")))
    weights_tensor = torch.as_tensor(weights, dtype=torch.double)
    if weights_tensor.numel() == 0:
        raise ValueError("Cannot build weighted sampler for an empty dataset.")
    return weights_tensor


def create_weighted_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: str = "jax",
) -> _data_loader.DataLoader[tuple[_model.Observation, _model.Actions]]:
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info("weighted data_config: %s", data_config)

    if data_config.rlds_data_dir is not None:
        raise NotImplementedError("Weighted sampling is only implemented for torch/LeRobot datasets.")

    return create_weighted_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
    )


def create_weighted_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
) -> _data_loader.DataLoader[tuple[_model.Observation, _model.Actions]]:
    if data_config.multi_dataset_specs:
        import openpi.training.multi_dataset as _multi_dataset

        dataset = _multi_dataset.create_multi_dataset(
            data_config,
            action_horizon=action_horizon,
            model_config=model_config,
            skip_norm_stats=skip_norm_stats,
        )
        sample_weights = torch.as_tensor(dataset.sample_weights, dtype=torch.double)
    else:
        base_dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
        sample_weights = _extract_sample_weights(base_dataset)
        dataset = _data_loader.transform_dataset(base_dataset, data_config, skip_norm_stats=skip_norm_stats)

    if framework == "pytorch" and torch.distributed.is_initialized():
        raise NotImplementedError("Weighted sampling is not implemented for PyTorch distributed training.")

    if framework == "pytorch":
        local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    if local_batch_size <= 0:
        raise ValueError(f"Invalid local batch size computed from batch_size={batch_size}.")

    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )

    logging.info(
        "weighted sampler stats: num_samples=%d, min_weight=%.6f, max_weight=%.6f, mean_weight=%.6f",
        len(sample_weights),
        float(sample_weights.min().item()),
        float(sample_weights.max().item()),
        float(sample_weights.mean().item()),
    )
    logging.info("weighted local_batch_size: %d", local_batch_size)

    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=False,
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )
    return _data_loader.DataLoaderImpl(data_config, data_loader)
