from __future__ import annotations

import bisect
from collections.abc import Sequence
import dataclasses
from typing import SupportsIndex

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np

import openpi.models.model as _model
import openpi.policies.lehome_camera_cv_policy as lehome_camera_cv_policy
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.future_latent_sidecar as _future_latent_sidecar
import openpi.transforms as _transforms


def _lehome_repack_transform(
    *,
    include_images: bool,
    include_wrist_images: bool,
    include_prompt: bool,
    include_future_latent: bool = False,
) -> _transforms.RepackTransform:
    structure = {
        "observation/state": "observation.state",
        "actions": "actions",
    }
    if include_images:
        structure["observation/top_rgb"] = "observation.images.top_rgb"
    if include_prompt:
        structure["prompt"] = "prompt"
    if include_images and include_wrist_images:
        structure.update(
            {
                "observation/left_rgb": "observation.images.left_rgb",
                "observation/right_rgb": "observation.images.right_rgb",
            }
        )
    if include_future_latent:
        structure.update(
            {
                "future_latent_pred": "future_latent/pred",
                "future_latent_true": "future_latent/true",
                "future_latent_valid_mask": "future_latent/valid_mask",
            }
        )
    return _transforms.RepackTransform(structure)


def _create_lerobot_dataset(
    spec: _config.LehomeCameraCVDatasetSpec,
    *,
    action_horizon: int,
    prompt_from_task: bool,
) -> _data_loader.Dataset:
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(spec.repo_id)
    dataset = lerobot_dataset.LeRobotDataset(
        spec.repo_id,
        delta_timestamps={"actions": [t / dataset_meta.fps for t in range(action_horizon)]},
    )
    if prompt_from_task:
        dataset = _data_loader.TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])
    return dataset


def _input_transform_for_spec(
    spec: _config.LehomeCameraCVDatasetSpec,
    *,
    model_type: _model.ModelType,
    state_unit: str,
    pose_quat_order: str,
    action_dim: int,
    include_images: bool = True,
) -> _transforms.DataTransformFn:
    if spec.apply_camera_cv_transform:
        return lehome_camera_cv_policy.LehomeCameraCVInputs(
            model_type=model_type,
            fk_json_path=spec.fk_json_path,
            camera_config_json_path=spec.camera_config_json_path,
            state_unit=state_unit,
            pose_quat_order=pose_quat_order,
            dataset_joint_order_csv=spec.dataset_joint_order_csv,
            valid_image_names_csv=spec.valid_image_names_csv,
            masked_state_indices_csv=spec.masked_state_indices_csv,
            masked_action_indices_csv=spec.masked_action_indices_csv,
            target_image_height=spec.target_image_height,
            target_image_width=spec.target_image_width,
            include_images=include_images,
        )
    return lehome_camera_cv_policy.LehomePrecomputed16DInputs(
        model_type=model_type,
        state_dim=action_dim,
        gripper_dim_indices_csv="",
        valid_image_names_csv=spec.valid_image_names_csv,
        masked_state_indices_csv=spec.masked_state_indices_csv,
        masked_action_indices_csv=spec.masked_action_indices_csv,
        target_image_height=spec.target_image_height,
        target_image_width=spec.target_image_width,
        include_images=include_images,
    )


@dataclasses.dataclass(frozen=True)
class _WeightedDatasetRecord:
    dataset: _data_loader.Dataset
    sample_weight: float
    repo_id: str


class WeightedConcatDataset(_data_loader.Dataset):
    def __init__(self, records: Sequence[_WeightedDatasetRecord]):
        if not records:
            raise ValueError("WeightedConcatDataset requires at least one dataset.")
        self._records = tuple(records)
        self._cumulative_sizes = []
        total = 0
        for record in self._records:
            dataset_len = len(record.dataset)
            if dataset_len <= 0:
                raise ValueError(f"Dataset {record.repo_id} is empty.")
            total += dataset_len
            self._cumulative_sizes.append(total)

    def __len__(self) -> int:
        return self._cumulative_sizes[-1]

    def _locate(self, index: int) -> tuple[_WeightedDatasetRecord, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        dataset_idx = bisect.bisect_right(self._cumulative_sizes, index)
        prev = 0 if dataset_idx == 0 else self._cumulative_sizes[dataset_idx - 1]
        return self._records[dataset_idx], index - prev

    def __getitem__(self, index: SupportsIndex) -> dict:
        record, local_index = self._locate(index.__index__())
        sample = dict(record.dataset[local_index])
        # `state_joint` is only needed by policy output post-processing. Training
        # batches mix dataset specs, and not every spec can provide this key.
        sample.pop("state_joint", None)
        sample["sample_weight"] = np.asarray(record.sample_weight, dtype=np.float32)
        return sample

    @property
    def sample_weights(self) -> np.ndarray:
        chunks = [
            np.full(len(record.dataset), float(record.sample_weight), dtype=np.float64)
            for record in self._records
        ]
        return np.concatenate(chunks, axis=0)


def create_multi_dataset(
    data_config: _config.DataConfig,
    *,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
    skip_norm_stats: bool = False,
    for_norm_stats: bool = False,
    avoid_image_dims: bool = False,
) -> WeightedConcatDataset:
    norm_stats = {}
    if not for_norm_stats and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    include_images = not (for_norm_stats and avoid_image_dims)
    records = []
    for spec in data_config.multi_dataset_specs:
        if not isinstance(spec, _config.LehomeCameraCVDatasetSpec):
            raise TypeError(f"Expected LehomeCameraCVDatasetSpec, got {type(spec)}")
        if data_config.future_latent_filter_included_datasets and not spec.include_in_future_latent_dataset:
            continue

        include_future_latent = data_config.future_latent_sidecar_root is not None
        transforms: list[_transforms.DataTransformFn] = []
        if include_future_latent:
            transforms.append(
                _future_latent_sidecar.FutureLatentSidecarTransform(
                    sidecar_root=data_config.future_latent_sidecar_root or "",
                    source_repo_id=spec.repo_id,
                    required=data_config.future_latent_sidecar_required,
                )
            )
        transforms.extend(
            [
                _lehome_repack_transform(
                    include_images=include_images,
                    include_wrist_images=spec.apply_camera_cv_transform,
                    include_prompt=data_config.prompt_from_task,
                    include_future_latent=include_future_latent,
                ),
                _input_transform_for_spec(
                    spec,
                    model_type=model_config.model_type,
                    state_unit=data_config.multi_state_unit,
                    pose_quat_order=data_config.multi_pose_quat_order,
                    action_dim=data_config.multi_action_dim,
                    include_images=include_images,
                ),
            ]
        )
        if not for_norm_stats:
            transforms.extend(
                [
                    _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
                    *data_config.model_transforms.inputs,
                ]
            )

        dataset = _create_lerobot_dataset(
            spec,
            action_horizon=action_horizon,
            prompt_from_task=data_config.prompt_from_task,
        )
        records.append(
            _WeightedDatasetRecord(
                dataset=_data_loader.TransformedDataset(dataset, transforms),
                sample_weight=float(spec.sample_weight),
                repo_id=spec.repo_id,
            )
        )

    if not records:
        raise ValueError("No multi-dataset records remain after filtering.")

    return WeightedConcatDataset(records)
