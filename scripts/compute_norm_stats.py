"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import numpy as np
import torch
import tqdm
from typing_extensions import Annotated
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.multi_dataset as _multi_dataset
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def _collate_norm_stats_fn(items: list[dict]) -> dict:
    keys = ("state", "state_mask", "actions", "action_mask", "sample_weight")
    return {
        key: np.stack([np.asarray(item[key]) for item in items], axis=0)
        for key in keys
        if all(key in item for item in items)
    }


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *_data_loader._make_sidecar_transforms(data_config),
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *_data_loader._make_sidecar_transforms(data_config),
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(
    config_name: str,
    max_frames: int | None = None,
    avoid_image_dims: Annotated[
        bool,
        tyro.conf.arg(
            aliases=("--avoid_image_dims",),
            help="Skip image column selection and image parsing for supported norm-stats datasets.",
        ),
    ] = False,
):
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    keys = {"state": "state_mask", "actions": "action_mask"}
    stats = {key: normalize.RunningStats() for key in keys}

    if data_config.multi_dataset_specs:
        dataset = _multi_dataset.create_multi_dataset(
            data_config,
            action_horizon=config.model.action_horizon,
            model_config=config.model,
            for_norm_stats=True,
            avoid_image_dims=avoid_image_dims,
        )
        dataset_len = len(dataset)
        if max_frames is not None:
            dataset_len = min(dataset_len, max_frames)
            dataset = torch.utils.data.Subset(dataset, range(dataset_len))
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            persistent_workers=config.num_workers > 0,
            collate_fn=_collate_norm_stats_fn,
            drop_last=False,
        )
        num_batches = (dataset_len + config.batch_size - 1) // config.batch_size
        for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing weighted stats"):
            weights = np.asarray(batch["sample_weight"], dtype=np.float64)
            for key, mask_key in keys.items():
                mask = None if mask_key not in batch else np.asarray(batch[mask_key])
                stats[key].update(np.asarray(batch[key]), mask=mask, weights=weights)

        norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}
        output_path = config.assets_dirs / (data_config.asset_id or data_config.repo_id)
        print(f"Writing stats to: {output_path}")
        normalize.save(output_path, norm_stats)
        return

    if avoid_image_dims:
        print(
            "--avoid-image-dims/--avoid_image_dims is only implemented for multi_dataset_specs configs; "
            "using current behavior."
        )

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
        )

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key, mask_key in keys.items():
            mask = None if mask_key not in batch else np.asarray(batch[mask_key])
            stats[key].update(np.asarray(batch[key]), mask=mask)

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    output_path = config.assets_dirs / (data_config.asset_id or data_config.repo_id)
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
