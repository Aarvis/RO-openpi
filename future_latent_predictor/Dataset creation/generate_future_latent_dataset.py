from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path
from typing import Any

import flax.nnx as nnx
import jax
import numpy as np
import polars as pl
from tqdm.auto import tqdm

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.training.multi_dataset as _multi_dataset
import openpi.training.weight_loaders as _weight_loaders
import openpi.transforms as _transforms


DEFAULT_CONFIG_NAME = "pi05_lehome_camera_cv_multi_cotrain_robot_finetune_future_latent"
CAMERA_TO_COLUMN = {
    "base_0_rgb": "top",
    "right_wrist_0_rgb": "right_wrist",
    "left_wrist_0_rgb": "left_wrist",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create future-latent training parquets by running LeHome frames through the pi0.5 image encoder. "
            "Original LeRobot datasets are not modified."
        )
    )
    parser.add_argument("--config-name", default=DEFAULT_CONFIG_NAME)
    parser.add_argument(
        "--params-path",
        default=None,
        help=(
            "Optional params checkpoint path. If omitted, the config weight_loader is used. "
            "Use this for a trained cotrain checkpoint, e.g. ./checkpoints/<config>/<exp>/<step>/params."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("openpi/future_latent_predictor/Dataset creation/output"),
    )
    parser.add_argument("--future-offset", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--shard-size", type=int, default=512)
    parser.add_argument("--max-rows-per-dataset", type=int, default=None)
    parser.add_argument("--embedding-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a dataset if its output directory already contains parquet shards.",
    )
    return parser.parse_args()


def _load_model(config: _config.TrainConfig, *, params_path: str | None) -> _model.BaseModel:
    rng = jax.random.PRNGKey(config.seed)
    model = config.model.create(rng)
    params = nnx.state(model)
    loader = _weight_loaders.CheckpointWeightLoader(params_path) if params_path else config.weight_loader
    loaded_params = loader.load(params.to_pure_dict())
    graphdef, state = nnx.split(model)
    state.replace_by_pure_dict(loaded_params)
    model = nnx.merge(graphdef, state)
    model.eval()
    return model


def _enabled_specs(data_config: _config.DataConfig) -> list[_config.LehomeCameraCVDatasetSpec]:
    specs = []
    for spec in data_config.multi_dataset_specs:
        if not isinstance(spec, _config.LehomeCameraCVDatasetSpec):
            raise TypeError(f"Expected LehomeCameraCVDatasetSpec, got {type(spec)}")
        if spec.include_in_future_latent_dataset:
            specs.append(spec)
    return specs


def _transforms_for_spec(
    *,
    spec: _config.LehomeCameraCVDatasetSpec,
    train_config: _config.TrainConfig,
    data_config: _config.DataConfig,
) -> list[_transforms.DataTransformFn]:
    transforms: list[_transforms.DataTransformFn] = [
        _multi_dataset._lehome_repack_transform(  # noqa: SLF001
            include_images=True,
            include_wrist_images=spec.apply_camera_cv_transform,
            include_prompt=data_config.prompt_from_task,
        ),
        _multi_dataset._input_transform_for_spec(  # noqa: SLF001
            spec,
            model_type=train_config.model.model_type,
            state_unit=data_config.multi_state_unit,
            pose_quat_order=data_config.multi_pose_quat_order,
            action_dim=data_config.multi_action_dim,
            include_images=True,
        ),
    ]
    if data_config.norm_stats is not None:
        transforms.append(_transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm))
    transforms.extend(data_config.model_transforms.inputs)
    return transforms


def _apply_transforms(sample: dict[str, Any], transforms: list[_transforms.DataTransformFn]) -> dict[str, Any]:
    data = sample
    for transform in transforms:
        data = transform(data)
    return data


def _batch_items(items: list[Any]) -> Any:
    first = items[0]
    if isinstance(first, dict):
        return {key: _batch_items([item[key] for item in items]) for key in first}
    return np.stack(items, axis=0)


def _observation_from_items(items: list[dict[str, Any]]) -> _model.Observation:
    observation_items = []
    for item in items:
        observation_item = {
            "image": item["image"],
            "image_mask": item["image_mask"],
            "state": item["state"],
        }
        if item.get("tokenized_prompt") is not None:
            observation_item["tokenized_prompt"] = item["tokenized_prompt"]
            observation_item["tokenized_prompt_mask"] = item["tokenized_prompt_mask"]
        observation_items.append(observation_item)
    batched = _batch_items(observation_items)
    return _model.Observation.from_dict(batched)


def _scalar(value: Any) -> int | float | str | None:
    if value is None:
        return None
    array = np.asarray(value)
    if array.shape == ():
        return array.item()
    if array.size == 1:
        return array.reshape(()).item()
    return None


def _same_episode(current: dict[str, Any], future: dict[str, Any]) -> bool:
    current_episode = _scalar(current.get("episode_index"))
    future_episode = _scalar(future.get("episode_index"))
    if current_episode is None or future_episode is None:
        return True
    return int(current_episode) == int(future_episode)


def _metadata_from_sample(sample: dict[str, Any], *, source_repo_id: str, source_index: int) -> dict[str, Any]:
    metadata = {
        "source_repo_id": source_repo_id,
        "source_index": source_index,
    }
    for key in ("episode_index", "frame_index", "timestamp", "task_index", "index"):
        value = _scalar(sample.get(key))
        if value is not None:
            metadata[key] = value
    return metadata


def _array_to_bytes(array: np.ndarray, *, dtype: np.dtype) -> bytes:
    return np.asarray(array, dtype=dtype).tobytes(order="C")


@dataclasses.dataclass
class _FrameEmbedding:
    raw: dict[str, Any]
    item: dict[str, Any]
    source_index: int
    embeddings: dict[str, np.ndarray]


@nnx.jit
def _extract_image_embeddings(model: _model.BaseModel, observation: _model.Observation) -> dict[str, jax.Array]:
    observation = _model.preprocess_observation(None, observation, train=False)
    outputs = {}
    for name in _model.IMAGE_KEYS:
        image_tokens, _ = model.PaliGemma.img(observation.images[name], train=False)
        outputs[name] = image_tokens
    return outputs


def _extract_frame_embeddings(
    *,
    model: _model.BaseModel,
    frames: list[tuple[dict[str, Any], dict[str, Any], int]],
) -> list[_FrameEmbedding]:
    obs = _observation_from_items([item for _, item, _ in frames])
    batch_embeddings = jax.device_get(_extract_image_embeddings(model, obs))
    entries = []
    for batch_index, (raw, item, source_index) in enumerate(frames):
        entries.append(
            _FrameEmbedding(
                raw=raw,
                item=item,
                source_index=source_index,
                embeddings={
                    camera_name: np.asarray(batch_embeddings[camera_name][batch_index])
                    for camera_name in CAMERA_TO_COLUMN
                },
            )
        )
    return entries


def _row_from_embedding_pair(
    *,
    current: _FrameEmbedding,
    future: _FrameEmbedding,
    source_repo_id: str,
    embedding_dtype: np.dtype,
    future_offset: int,
) -> dict[str, Any] | None:
    if future.source_index != current.source_index + future_offset:
        return None
    if not _same_episode(current.raw, future.raw):
        return None

    row = {
        **_metadata_from_sample(current.raw, source_repo_id=source_repo_id, source_index=current.source_index),
        "state": np.asarray(current.item["state"], dtype=np.float32).reshape(-1).tolist(),
        "action_chunk": np.asarray(current.item.get("actions", []), dtype=np.float32).reshape(-1).tolist(),
        "embedding_dtype": np.dtype(embedding_dtype).name,
        "embedding_shape": [256, 2048],
    }
    for camera_name, column_name in CAMERA_TO_COLUMN.items():
        current_valid = bool(np.asarray(current.item["image_mask"][camera_name]).item())
        future_valid = bool(np.asarray(future.item["image_mask"][camera_name]).item())
        row[f"{column_name}_embedding_valid"] = current_valid
        row[f"{column_name}_embedding_t_5_valid"] = future_valid
        row[f"{column_name}_embedding_t"] = _array_to_bytes(
            current.embeddings[camera_name],
            dtype=embedding_dtype,
        )
        row[f"{column_name}_embedding_t_5"] = _array_to_bytes(
            future.embeddings[camera_name],
            dtype=embedding_dtype,
        )
    return row


def _write_shard(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(output_path)


def _safe_dataset_name(repo_id: str) -> str:
    return repo_id.replace("/", "__").replace("\\", "__")


def _write_dataset_metadata(
    *,
    output_dir: Path,
    train_config: _config.TrainConfig,
    spec: _config.LehomeCameraCVDatasetSpec,
    args: argparse.Namespace,
    num_rows: int,
    num_skipped: int,
) -> None:
    payload = {
        "config_name": train_config.name,
        "source_repo_id": spec.repo_id,
        "future_offset": args.future_offset,
        "num_rows": num_rows,
        "num_skipped": num_skipped,
        "embedding_dtype": args.embedding_dtype,
        "embedding_shape_per_camera": [256, 2048],
        "columns": [
            "top_embedding_t",
            "right_wrist_embedding_t",
            "left_wrist_embedding_t",
            "top_embedding_t_5",
            "right_wrist_embedding_t_5",
            "left_wrist_embedding_t_5",
        ],
        "dataset_spec": dataclasses.asdict(spec),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "future_latent_metadata.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _process_spec(
    *,
    train_config: _config.TrainConfig,
    data_config: _config.DataConfig,
    spec: _config.LehomeCameraCVDatasetSpec,
    model: _model.BaseModel,
    args: argparse.Namespace,
) -> None:
    dataset_name = _safe_dataset_name(spec.repo_id)
    output_dir = args.output_dir / dataset_name
    if args.skip_existing and list(output_dir.glob("*.parquet")):
        logging.info("Skipping %s because parquet shards already exist in %s", spec.repo_id, output_dir)
        return

    raw_dataset = _multi_dataset._create_lerobot_dataset(  # noqa: SLF001
        spec,
        action_horizon=train_config.model.action_horizon,
        prompt_from_task=data_config.prompt_from_task,
    )
    transforms = _transforms_for_spec(spec=spec, train_config=train_config, data_config=data_config)

    embedding_dtype = np.float16 if args.embedding_dtype == "float16" else np.float32
    limit = len(raw_dataset)
    if args.max_rows_per_dataset is not None:
        limit = min(limit, args.max_rows_per_dataset + args.future_offset)

    shard_rows: list[dict[str, Any]] = []
    frame_batch: list[tuple[dict[str, Any], dict[str, Any], int]] = []
    window: list[_FrameEmbedding] = []
    num_rows = 0
    num_skipped = 0
    shard_index = 0

    def flush_frame_batch() -> bool:
        nonlocal frame_batch, num_rows, num_skipped, shard_index, shard_rows
        if not frame_batch:
            return False

        entries = _extract_frame_embeddings(model=model, frames=frame_batch)
        frame_batch = []
        for entry in entries:
            window.append(entry)
            if len(window) <= args.future_offset:
                continue

            current = window.pop(0)
            row = _row_from_embedding_pair(
                current=current,
                future=window[-1],
                source_repo_id=spec.repo_id,
                embedding_dtype=embedding_dtype,
                future_offset=args.future_offset,
            )
            if row is None:
                num_skipped += 1
                continue

            shard_rows.append(row)
            num_rows += 1
            if len(shard_rows) >= args.shard_size:
                _write_shard(shard_rows, output_dir / f"part-{shard_index:06d}.parquet")
                shard_rows = []
                shard_index += 1
            if args.max_rows_per_dataset is not None and num_rows >= args.max_rows_per_dataset:
                return True
        return False

    progress = tqdm(range(limit), desc=f"Embedding {spec.repo_id}", unit="frame")
    stop = False
    for source_index in progress:
        try:
            raw = dict(raw_dataset[source_index])
            item = _apply_transforms(raw, transforms)
        except Exception as exc:  # noqa: BLE001
            logging.warning("Skipping %s index %s: %s", spec.repo_id, source_index, exc)
            num_skipped += 1
            continue

        frame_batch.append((raw, item, source_index))
        if len(frame_batch) >= args.batch_size:
            stop = flush_frame_batch()
            progress.set_postfix(rows=num_rows, skipped=num_skipped)
        if stop:
            break

    if not stop and frame_batch:
        flush_frame_batch()

    if shard_rows:
        _write_shard(shard_rows, output_dir / f"part-{shard_index:06d}.parquet")

    _write_dataset_metadata(
        output_dir=output_dir,
        train_config=train_config,
        spec=spec,
        args=args,
        num_rows=num_rows,
        num_skipped=num_skipped,
    )
    logging.info("Wrote %s rows for %s to %s; skipped=%s", num_rows, spec.repo_id, output_dir, num_skipped)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    train_config = _config.get_config(args.config_name)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    specs = _enabled_specs(data_config)
    if not specs:
        raise ValueError(
            f"No dataset specs are enabled for future latent generation in config {args.config_name!r}."
        )

    model = _load_model(train_config, params_path=args.params_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        _process_spec(
            train_config=train_config,
            data_config=data_config,
            spec=spec,
            model=model,
            args=args,
        )


if __name__ == "__main__":
    main()
