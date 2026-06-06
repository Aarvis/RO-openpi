from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
from tqdm.auto import tqdm

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.multi_dataset as _multi_dataset
import openpi.training.weight_loaders as _weight_loaders
import openpi.transforms as _transforms


DEFAULT_CONFIG_NAME = "pi05_lehome_camera_cv_multi_cotrain_robot_finetune_future_latent"
CAMERA_TO_COLUMN = {
    "base_0_rgb": "top",
    "right_wrist_0_rgb": "right_wrist",
    "left_wrist_0_rgb": "left_wrist",
}
EMBEDDING_SHAPE = (256, 2048)
FUTURE_LATENT_REPACK_KEYS = {
    "future_latent_pred",
    "future_latent_true",
    "future_latent_valid_mask",
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
    parser.add_argument(
        "--read-workers",
        type=int,
        default=2,
        help=(
            "Number of background threads per GPU worker used to read and transform dataset rows. "
            "Set to 0 to read synchronously in the main thread."
        ),
    )
    parser.add_argument(
        "--read-prefetch-batches",
        type=int,
        default=2,
        help=(
            "Number of batches to keep prefetched per GPU worker. Higher values can improve GPU feeding but use more RAM."
        ),
    )
    parser.add_argument(
        "--max-pending-shard-writes",
        type=int,
        default=2,
        help=(
            "Maximum number of parquet shard writes allowed to run in the background per worker. "
            "Higher values overlap writing with GPU encoding more, but use more host RAM."
        ),
    )
    parser.add_argument("--max-rows-per-dataset", type=int, default=None)
    parser.add_argument("--embedding-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument(
        "--num-gpu-workers",
        default="auto",
        help=(
            "Number of GPU worker subprocesses to launch. Use 'auto' to use all IDs in CUDA_VISIBLE_DEVICES. "
            "Set to 1 to run in the current process."
        ),
    )
    parser.add_argument(
        "--worker-launch-stagger-seconds",
        type=float,
        default=8.0,
        help=(
            "Seconds to wait between launching GPU worker subprocesses. This reduces simultaneous CUDA/XLA "
            "initialization pressure on large multi-GPU nodes. Set to 0 to disable."
        ),
    )
    parser.add_argument("--worker-index", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--num-workers", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a dataset if its output directory already contains parquet shards.",
    )
    parser.add_argument(
        "--debug-start-image-count",
        type=int,
        default=5,
        help=(
            "Save transformed camera images for the first N frames on worker 0 for quick visual inspection. "
            "Set to 0 to disable."
        ),
    )
    return parser.parse_args()


def _visible_gpu_ids() -> list[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible or visible.strip() in {"", "-1"}:
        return []
    return [item.strip() for item in visible.split(",") if item.strip()]


def _resolve_num_gpu_workers(value: str) -> int:
    if value == "auto":
        return max(1, len(_visible_gpu_ids()))
    try:
        workers = int(value)
    except ValueError as exc:
        raise ValueError(f"--num-gpu-workers must be 'auto' or an integer, got {value!r}") from exc
    if workers < 1:
        raise ValueError(f"--num-gpu-workers must be >= 1, got {workers}")
    return workers


def _child_command(args: argparse.Namespace, *, worker_index: int, num_workers: int) -> list[str]:
    command = [
        sys.executable,
        __file__,
        "--config-name",
        args.config_name,
        "--output-dir",
        str(args.output_dir),
        "--future-offset",
        str(args.future_offset),
        "--batch-size",
        str(args.batch_size),
        "--shard-size",
        str(args.shard_size),
        "--read-workers",
        str(args.read_workers),
        "--read-prefetch-batches",
        str(args.read_prefetch_batches),
        "--max-pending-shard-writes",
        str(args.max_pending_shard_writes),
        "--embedding-dtype",
        args.embedding_dtype,
        "--num-gpu-workers",
        "1",
        "--worker-launch-stagger-seconds",
        "0",
        "--worker-index",
        str(worker_index),
        "--num-workers",
        str(num_workers),
    ]
    if args.params_path is not None:
        command.extend(["--params-path", args.params_path])
    if args.max_rows_per_dataset is not None:
        command.extend(["--max-rows-per-dataset", str(args.max_rows_per_dataset)])
    if args.skip_existing:
        command.append("--skip-existing")
    command.extend(["--debug-start-image-count", str(args.debug_start_image_count)])
    return command


def _maybe_launch_gpu_workers(args: argparse.Namespace) -> bool:
    if args.num_workers != 1 or args.worker_index != 0:
        return False

    num_workers = _resolve_num_gpu_workers(str(args.num_gpu_workers))
    if num_workers <= 1:
        return False

    visible_gpu_ids = _visible_gpu_ids()
    if len(visible_gpu_ids) < num_workers:
        raise ValueError(
            f"Requested {num_workers} GPU workers but CUDA_VISIBLE_DEVICES exposes {len(visible_gpu_ids)} GPUs: "
            f"{visible_gpu_ids}"
        )

    logging.info("Launching %s GPU workers across CUDA_VISIBLE_DEVICES=%s", num_workers, visible_gpu_ids)
    processes = []
    for worker_index, gpu_id in enumerate(visible_gpu_ids[:num_workers]):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        command = _child_command(args, worker_index=worker_index, num_workers=num_workers)
        logging.info("Starting worker %s/%s on GPU %s", worker_index, num_workers, gpu_id)
        processes.append(subprocess.Popen(command, env=env))  # noqa: S603
        launch_delay = float(args.worker_launch_stagger_seconds)
        if launch_delay > 0 and worker_index + 1 < num_workers:
            logging.info("Waiting %.1fs before launching next worker", launch_delay)
            time.sleep(launch_delay)

    failed = []
    for worker_index, process in enumerate(processes):
        return_code = process.wait()
        if return_code != 0:
            failed.append((worker_index, return_code))
    if failed:
        raise RuntimeError(f"Future latent GPU workers failed: {failed}")
    return True


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


@dataclasses.dataclass(frozen=True)
class _DatasetSource:
    repo_id: str
    raw_dataset: _data_loader.Dataset
    transforms: list[_transforms.DataTransformFn]
    dataset_spec: _config.LehomeCameraCVDatasetSpec | None = None


def _strip_future_latent_repack_fields(transform: _transforms.DataTransformFn) -> _transforms.DataTransformFn:
    if not isinstance(transform, _transforms.RepackTransform):
        return transform

    structure = {
        key: value
        for key, value in transform.structure.items()
        if key not in FUTURE_LATENT_REPACK_KEYS
    }
    return _transforms.RepackTransform(structure)


def _transforms_for_data_config(data_config: _config.DataConfig) -> list[_transforms.DataTransformFn]:
    transforms: list[_transforms.DataTransformFn] = [
        _strip_future_latent_repack_fields(transform)
        for transform in data_config.repack_transforms.inputs
    ]
    transforms.extend(data_config.data_transforms.inputs)
    if data_config.norm_stats is not None:
        transforms.append(_transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm))
    transforms.extend(data_config.model_transforms.inputs)
    return transforms


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


def _dataset_sources(
    train_config: _config.TrainConfig,
    data_config: _config.DataConfig,
) -> list[_DatasetSource]:
    if data_config.multi_dataset_specs:
        sources = []
        for spec in _enabled_specs(data_config):
            sources.append(
                _DatasetSource(
                    repo_id=spec.repo_id,
                    raw_dataset=_multi_dataset._create_lerobot_dataset(  # noqa: SLF001
                        spec,
                        action_horizon=train_config.model.action_horizon,
                        prompt_from_task=data_config.prompt_from_task,
                    ),
                    transforms=_transforms_for_spec(
                        spec=spec,
                        train_config=train_config,
                        data_config=data_config,
                    ),
                    dataset_spec=spec,
                )
            )
        return sources

    if data_config.repo_id is None:
        raise ValueError(
            f"Config {train_config.name!r} has neither multi_dataset_specs nor a single repo_id."
        )

    return [
        _DatasetSource(
            repo_id=data_config.repo_id,
            raw_dataset=_data_loader.create_torch_dataset(
                data_config,
                action_horizon=train_config.model.action_horizon,
                model_config=train_config.model,
            ),
            transforms=_transforms_for_data_config(data_config),
        )
    ]


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
def _extract_one_image_embedding(model: _model.BaseModel, image: jax.Array) -> jax.Array:
    image_tokens, _ = model.PaliGemma.img(image, train=False)
    return image_tokens


def _extract_frame_embeddings(
    *,
    model: _model.BaseModel,
    frames: list[tuple[dict[str, Any], dict[str, Any], int]],
) -> list[_FrameEmbedding]:
    obs = _observation_from_items([item for _, item, _ in frames])
    obs = _model.preprocess_observation(None, obs, train=False)
    batch_embeddings: dict[str, np.ndarray] = {}
    valid_camera_names = []
    valid_camera_images = []
    for camera_name in CAMERA_TO_COLUMN:
        camera_has_valid_image = any(
            bool(np.asarray(item["image_mask"][camera_name]).item())
            for _, item, _ in frames
        )
        if camera_has_valid_image:
            valid_camera_names.append(camera_name)
            valid_camera_images.append(obs.images[camera_name])
        else:
            batch_embeddings[camera_name] = np.zeros(
                (len(frames), *EMBEDDING_SHAPE),
                dtype=np.float32,
            )

    if valid_camera_images:
        stacked_images = jnp.concatenate(valid_camera_images, axis=0)
        stacked_embeddings = np.asarray(
            jax.device_get(_extract_one_image_embedding(model, stacked_images))
        )
        for camera_index, camera_name in enumerate(valid_camera_names):
            start = camera_index * len(frames)
            end = start + len(frames)
            batch_embeddings[camera_name] = stacked_embeddings[start:end]

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
    import polars as pl

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Embeddings are MiB-scale binary blobs. Parquet statistics can duplicate
    # large binary page values, roughly doubling shard size.
    pl.DataFrame(rows).write_parquet(output_path, statistics=False)


def _safe_dataset_name(repo_id: str) -> str:
    return repo_id.replace("/", "__").replace("\\", "__")


def _write_dataset_metadata(
    *,
    output_dir: Path,
    train_config: _config.TrainConfig,
    source: _DatasetSource,
    args: argparse.Namespace,
    num_rows: int,
    num_skipped: int,
) -> None:
    payload = {
        "config_name": train_config.name,
        "source_repo_id": source.repo_id,
        "future_offset": args.future_offset,
        "num_rows": num_rows,
        "num_skipped": num_skipped,
        "worker_index": args.worker_index,
        "num_workers": args.num_workers,
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
        "dataset_spec": dataclasses.asdict(source.dataset_spec) if source.dataset_spec is not None else None,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_name = (
        "future_latent_metadata.json"
        if args.num_workers == 1
        else f"future_latent_metadata_worker-{args.worker_index:03d}.json"
    )
    (output_dir / metadata_name).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _row_range_for_worker(*, total_rows: int, worker_index: int, num_workers: int) -> tuple[int, int]:
    if worker_index < 0 or worker_index >= num_workers:
        raise ValueError(f"worker_index must be in [0, {num_workers}), got {worker_index}")
    rows_per_worker = total_rows // num_workers
    remainder = total_rows % num_workers
    start = worker_index * rows_per_worker + min(worker_index, remainder)
    end = start + rows_per_worker + (1 if worker_index < remainder else 0)
    return start, end


def _shard_path(output_dir: Path, *, shard_index: int, args: argparse.Namespace) -> Path:
    if args.num_workers == 1:
        return output_dir / f"part-{shard_index:06d}.parquet"
    return output_dir / f"part-worker-{args.worker_index:03d}-{shard_index:06d}.parquet"


def _format_seconds(seconds: float | None) -> str:
    if seconds is None or not np.isfinite(seconds):
        return "?"
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _image_to_uint8(image: Any) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 3 and array.shape[0] in {1, 3, 4} and array.shape[-1] not in {1, 3, 4}:
        array = np.moveaxis(array, 0, -1)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.ndim != 3:
        raise ValueError(f"Expected image with 2 or 3 dims, got shape {array.shape}")
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] > 3:
        array = array[..., :3]

    if np.issubdtype(array.dtype, np.floating):
        finite = array[np.isfinite(array)]
        if finite.size == 0:
            array = np.zeros_like(array, dtype=np.float32)
        min_value = float(finite.min()) if finite.size else 0.0
        max_value = float(finite.max()) if finite.size else 0.0
        if min_value >= -1.1 and max_value <= 1.1 and min_value < 0.0:
            array = (array + 1.0) * 127.5
        elif min_value >= 0.0 and max_value <= 1.1:
            array = array * 255.0
        array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
        array = np.clip(array, 0.0, 255.0).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def _save_debug_start_images(
    *,
    item: dict[str, Any],
    source_index: int,
    source_repo_id: str,
    args: argparse.Namespace,
) -> None:
    if args.worker_index != 0 or args.debug_start_image_count <= 0:
        return
    saved_count = getattr(args, "_debug_saved_start_image_count", 0)
    if saved_count >= args.debug_start_image_count:
        return

    from PIL import Image

    dataset_name = _safe_dataset_name(source_repo_id)
    debug_dir = args.output_dir / "debug_start_images" / dataset_name
    debug_dir.mkdir(parents=True, exist_ok=True)
    frame_index = saved_count
    for camera_name, column_name in CAMERA_TO_COLUMN.items():
        image = _image_to_uint8(item["image"][camera_name])
        valid = bool(np.asarray(item["image_mask"][camera_name]).item())
        output_path = debug_dir / (
            f"frame_{frame_index:03d}_source_{source_index:08d}_{column_name}_valid{int(valid)}.png"
        )
        Image.fromarray(image).save(output_path)
    setattr(args, "_debug_saved_start_image_count", saved_count + 1)


def _process_source(
    *,
    train_config: _config.TrainConfig,
    source: _DatasetSource,
    model: _model.BaseModel,
    args: argparse.Namespace,
) -> None:
    dataset_name = _safe_dataset_name(source.repo_id)
    output_dir = args.output_dir / dataset_name
    if args.skip_existing and list(output_dir.glob("*.parquet")):
        logging.info("Skipping %s because parquet shards already exist in %s", source.repo_id, output_dir)
        return

    raw_dataset = source.raw_dataset
    transforms = source.transforms

    embedding_dtype = np.float16 if args.embedding_dtype == "float16" else np.float32
    total_rows = max(0, len(raw_dataset) - args.future_offset)
    if args.max_rows_per_dataset is not None:
        total_rows = min(total_rows, args.max_rows_per_dataset)
    row_start, row_end = _row_range_for_worker(
        total_rows=total_rows,
        worker_index=args.worker_index,
        num_workers=args.num_workers,
    )
    frame_start = row_start
    frame_end = row_end + args.future_offset

    shard_rows: list[dict[str, Any]] = []
    frame_batch: list[tuple[dict[str, Any], dict[str, Any], int]] = []
    window: list[_FrameEmbedding] = []
    num_rows = 0
    num_skipped = 0
    shard_index = 0
    progress_total = frame_end - frame_start
    progress_started_at = time.monotonic()
    progress_phase = "starting"
    debug_image_lock = threading.Lock()
    max_pending_writes = max(1, int(args.max_pending_shard_writes))
    write_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    pending_writes: list[concurrent.futures.Future] = []

    def reap_completed_writes() -> None:
        nonlocal pending_writes
        remaining = []
        for future in pending_writes:
            if future.done():
                future.result()
            else:
                remaining.append(future)
        pending_writes = remaining

    def wait_for_write_slot() -> None:
        nonlocal progress_phase
        reap_completed_writes()
        while len(pending_writes) >= max_pending_writes:
            progress_phase = "waiting_write_slot"
            done, _ = concurrent.futures.wait(
                pending_writes,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                future.result()
            reap_completed_writes()

    def schedule_shard_write(rows: list[dict[str, Any]], output_path: Path) -> None:
        nonlocal progress_phase
        wait_for_write_slot()
        progress_phase = "scheduling_write"
        pending_writes.append(write_executor.submit(_write_shard, rows, output_path))
        progress_phase = "pairing"

    def finish_pending_writes() -> None:
        nonlocal progress_phase
        progress_phase = "waiting_writes"
        for future in pending_writes:
            future.result()
        pending_writes.clear()
        write_executor.shutdown(wait=True)

    def flush_frame_batch() -> bool:
        nonlocal frame_batch, num_rows, num_skipped, shard_index, shard_rows, progress_phase
        if not frame_batch:
            return False

        batch_len = len(frame_batch)
        progress_phase = "encoding"
        entries = _extract_frame_embeddings(model=model, frames=frame_batch)
        frame_batch = []
        progress.update(batch_len)
        progress_phase = "pairing"
        for entry in entries:
            window.append(entry)
            if len(window) <= args.future_offset:
                continue

            current = window.pop(0)
            row = _row_from_embedding_pair(
                current=current,
                future=window[-1],
                source_repo_id=source.repo_id,
                embedding_dtype=embedding_dtype,
                future_offset=args.future_offset,
            )
            if row is None:
                num_skipped += 1
                continue

            shard_rows.append(row)
            num_rows += 1
            if len(shard_rows) >= args.shard_size:
                progress_phase = "scheduling_write"
                schedule_shard_write(shard_rows, _shard_path(output_dir, shard_index=shard_index, args=args))
                shard_rows = []
                shard_index += 1
                progress_phase = "pairing"
            if num_rows >= row_end - row_start:
                _refresh_progress_postfix(refresh=True)
                return True
        _refresh_progress_postfix(refresh=False)
        progress_phase = "reading"
        return False

    if frame_start >= frame_end:
        write_executor.shutdown(wait=True)
        _write_dataset_metadata(
            output_dir=output_dir,
            train_config=train_config,
            source=source,
            args=args,
            num_rows=0,
            num_skipped=0,
        )
        logging.info(
            "Worker %s/%s has no rows for %s",
            args.worker_index,
            args.num_workers,
            source.repo_id,
        )
        return

    def _progress_postfix() -> str:
        elapsed = time.monotonic() - progress_started_at
        fps = progress.n / elapsed if elapsed > 0 else 0.0
        remaining = progress_total - progress.n
        eta = remaining / fps if fps > 0 else None
        return (
            f"phase={progress_phase} rows={num_rows} skipped={num_skipped} "
            f"elapsed={_format_seconds(elapsed)} fps={fps:.2f} eta={_format_seconds(eta)}"
        )

    def _refresh_progress_postfix(*, refresh: bool) -> None:
        progress.set_postfix_str(_progress_postfix(), refresh=refresh)

    progress = tqdm(
        total=progress_total,
        desc=f"Embedding {source.repo_id} worker {args.worker_index}/{args.num_workers}",
        unit="frame",
        position=args.worker_index,
        leave=True,
        dynamic_ncols=True,
    )
    heartbeat_stop = threading.Event()

    def heartbeat() -> None:
        while not heartbeat_stop.wait(2.0):
            _refresh_progress_postfix(refresh=True)

    heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
    heartbeat_thread.start()

    try:
        _refresh_progress_postfix(refresh=False)
        stop = False

        def read_one(source_index: int) -> tuple[dict[str, Any], dict[str, Any], int] | None:
            try:
                raw = dict(raw_dataset[source_index])
                item = _apply_transforms(raw, transforms)
                with debug_image_lock:
                    _save_debug_start_images(
                        item=item,
                        source_index=source_index,
                        source_repo_id=source.repo_id,
                        args=args,
                    )
                return raw, item, source_index
            except Exception as exc:  # noqa: BLE001
                logging.warning("Skipping %s index %s: %s", source.repo_id, source_index, exc)
                return None

        def add_read_result(result: tuple[dict[str, Any], dict[str, Any], int] | None) -> bool:
            nonlocal num_skipped
            if result is None:
                num_skipped += 1
                progress.update(1)
                _refresh_progress_postfix(refresh=False)
                return False

            frame_batch.append(result)
            if len(frame_batch) >= args.batch_size:
                return flush_frame_batch()
            return False

        progress_phase = "reading"
        if args.read_workers <= 0:
            for source_index in range(frame_start, frame_end):
                stop = add_read_result(read_one(source_index))
                if stop:
                    break
        else:
            max_pending_reads = max(1, int(args.read_prefetch_batches)) * int(args.batch_size)
            with concurrent.futures.ThreadPoolExecutor(max_workers=int(args.read_workers)) as read_executor:
                pending_reads: dict[int, concurrent.futures.Future] = {}
                next_source_index = frame_start
                next_consume_index = frame_start

                def submit_reads() -> None:
                    nonlocal next_source_index
                    while next_source_index < frame_end and len(pending_reads) < max_pending_reads:
                        pending_reads[next_source_index] = read_executor.submit(read_one, next_source_index)
                        next_source_index += 1

                submit_reads()
                while pending_reads and next_consume_index < frame_end:
                    submit_reads()
                    future = pending_reads.pop(next_consume_index)
                    progress_phase = "reading"
                    stop = add_read_result(future.result())
                    if stop:
                        for future in pending_reads.values():
                            future.cancel()
                        break
                    next_consume_index += 1
                    submit_reads()

        if not stop and frame_batch:
            flush_frame_batch()
    finally:
        had_error = sys.exc_info()[0] is not None
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=2.0)
        _refresh_progress_postfix(refresh=True)
        progress.close()
        if had_error:
            write_executor.shutdown(wait=False, cancel_futures=True)

    try:
        if shard_rows:
            schedule_shard_write(shard_rows, _shard_path(output_dir, shard_index=shard_index, args=args))
            shard_rows = []
    finally:
        finish_pending_writes()

    _write_dataset_metadata(
        output_dir=output_dir,
        train_config=train_config,
        source=source,
        args=args,
        num_rows=num_rows,
        num_skipped=num_skipped,
    )
    logging.info(
        "Worker %s/%s wrote %s rows for %s to %s; skipped=%s; row_range=[%s,%s)",
        args.worker_index,
        args.num_workers,
        num_rows,
        source.repo_id,
        output_dir,
        num_skipped,
        row_start,
        row_end,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    if args.future_offset <= 0:
        raise ValueError(f"--future-offset must be > 0, got {args.future_offset}")
    if args.batch_size <= 0:
        raise ValueError(f"--batch-size must be > 0, got {args.batch_size}")
    if args.shard_size <= 0:
        raise ValueError(f"--shard-size must be > 0, got {args.shard_size}")
    if args.debug_start_image_count < 0:
        raise ValueError(f"--debug-start-image-count must be >= 0, got {args.debug_start_image_count}")
    setattr(args, "_debug_saved_start_image_count", 0)
    if _maybe_launch_gpu_workers(args):
        return

    train_config = _config.get_config(args.config_name)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    sources = _dataset_sources(train_config, data_config)
    if not sources:
        raise ValueError(
            f"No datasets are enabled for future latent generation in config {args.config_name!r}."
        )

    model = _load_model(train_config, params_path=args.params_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for source in sources:
        _process_source(
            train_config=train_config,
            source=source,
            model=model,
            args=args,
        )


if __name__ == "__main__":
    main()
