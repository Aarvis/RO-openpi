from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import polars as pl
import torch
from torch import nn
from tqdm.auto import tqdm

import openpi.training.config as _config
from future_latent_predictor.future_predictor.model import FutureLatentPredictor
from future_latent_predictor.resampler_autoencoder.model import CAMERAS
from future_latent_predictor.resampler_autoencoder.model import ResamplerAutoencoder


DEFAULT_CONFIG_PATH = Path(
    "openpi/future_latent_predictor/future_latent_dataset_generatore/configs/example_vla_sidecar_generation.json"
)
CAMERA_ORDER = ("top", "right_wrist", "left_wrist")
EMBEDDING_SHAPE = (256, 2048)
LATENT_SHAPE = (24, 512)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate compact future-latent sidecar parquets for VLA fine-tuning. "
            "This consumes full image-embedding parquets produced by Dataset creation/generate_future_latent_dataset.py."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--config-name", default=None)
    parser.add_argument("--embedding-data-root", action="append", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resampler-checkpoint", type=Path, default=None)
    parser.add_argument("--future-predictor-checkpoint", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--shard-size", type=int, default=None)
    parser.add_argument("--num-gpu-workers", default=None)
    parser.add_argument("--max-rows-per-dataset", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true", default=None)
    parser.add_argument(
        "--check-inputs-only",
        action="store_true",
        help="Print and save the embedding-root dataset summary, then exit before loading models.",
    )
    parser.add_argument("--worker-index", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--num-workers", type=int, default=1, help=argparse.SUPPRESS)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def resolve_config(args: argparse.Namespace) -> dict[str, Any]:
    config = load_json(args.config)
    if args.config_name is not None:
        config["config_name"] = args.config_name
    if args.embedding_data_root is not None:
        config["embedding_dataset_roots"] = [str(path) for path in args.embedding_data_root]
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir)
    if args.resampler_checkpoint is not None:
        config["resampler_checkpoint"] = str(args.resampler_checkpoint)
    if args.future_predictor_checkpoint is not None:
        config["future_predictor_checkpoint"] = str(args.future_predictor_checkpoint)
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.shard_size is not None:
        config["shard_size"] = args.shard_size
    if args.num_gpu_workers is not None:
        config["num_gpu_workers"] = args.num_gpu_workers
    if args.max_rows_per_dataset is not None:
        config["max_rows_per_dataset"] = args.max_rows_per_dataset
    if args.skip_existing is not None:
        config["skip_existing"] = bool(args.skip_existing)
    return config


def visible_gpu_ids() -> list[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible or visible.strip() in {"", "-1"}:
        return []
    return [item.strip() for item in visible.split(",") if item.strip()]


def resolve_num_gpu_workers(value: str | int) -> int:
    if str(value) == "auto":
        return max(1, len(visible_gpu_ids()))
    workers = int(value)
    if workers < 1:
        raise ValueError(f"num_gpu_workers must be >= 1, got {workers}")
    return workers


def child_command(args: argparse.Namespace, *, worker_index: int, num_workers: int) -> list[str]:
    command = [
        sys.executable,
        __file__,
        "--config",
        str(args.config),
        "--worker-index",
        str(worker_index),
        "--num-workers",
        str(num_workers),
    ]
    return command


def maybe_launch_gpu_workers(args: argparse.Namespace, config: dict[str, Any]) -> bool:
    if args.num_workers != 1 or args.worker_index != 0:
        return False
    num_workers = resolve_num_gpu_workers(config.get("num_gpu_workers", "auto"))
    if num_workers <= 1:
        return False

    gpu_ids = visible_gpu_ids()
    if len(gpu_ids) < num_workers:
        raise ValueError(
            f"Requested {num_workers} GPU workers but CUDA_VISIBLE_DEVICES exposes {len(gpu_ids)} GPUs: {gpu_ids}"
        )

    processes = []
    for worker_index, gpu_id in enumerate(gpu_ids[:num_workers]):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        command = child_command(args, worker_index=worker_index, num_workers=num_workers)
        logging.info("Starting sidecar worker %s/%s on GPU %s", worker_index, num_workers, gpu_id)
        processes.append(subprocess.Popen(command, env=env))  # noqa: S603

    failed = []
    for worker_index, process in enumerate(processes):
        return_code = process.wait()
        if return_code != 0:
            failed.append((worker_index, return_code))
    if failed:
        raise RuntimeError(f"VLA sidecar GPU workers failed: {failed}")
    return True


def safe_dataset_name(repo_id: str) -> str:
    return repo_id.replace("/", "__").replace("\\", "__")


def enabled_specs(config_name: str) -> list[_config.LehomeCameraCVDatasetSpec]:
    train_config = _config.get_config(config_name)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    specs = []
    for spec in data_config.multi_dataset_specs:
        if not isinstance(spec, _config.LehomeCameraCVDatasetSpec):
            raise TypeError(f"Expected LehomeCameraCVDatasetSpec, got {type(spec)}")
        if spec.include_in_future_latent_dataset:
            specs.append(spec)
    if specs:
        return specs

    repo_id = getattr(data_config, "repo_id", None) or getattr(train_config.data, "repo_id", None)
    if repo_id:
        return [_config.LehomeCameraCVDatasetSpec(repo_id=str(repo_id), include_in_future_latent_dataset=True)]

    raise ValueError(f"No datasets have include_in_future_latent_dataset=True in config {config_name!r}")


@dataclasses.dataclass(frozen=True)
class DatasetInput:
    source_repo_id: str
    safe_name: str
    root: Path
    files: tuple[Path, ...]


def embedding_input_summary(config: dict[str, Any]) -> dict[str, Any]:
    roots = [Path(path) for path in config.get("embedding_dataset_roots", [])]
    specs = enabled_specs(str(config["config_name"]))
    datasets = []
    for spec in specs:
        source = spec.repo_id
        safe_name = safe_dataset_name(source)
        root_summaries = []
        total_files = 0
        for embedding_root in roots:
            dataset_root = embedding_root / safe_name
            files = tuple(sorted(dataset_root.glob("*.parquet")))
            total_files += len(files)
            root_summaries.append(
                {
                    "embedding_root": str(embedding_root),
                    "dataset_root": str(dataset_root),
                    "exists": dataset_root.exists(),
                    "is_symlink": dataset_root.is_symlink(),
                    "resolved_path": str(dataset_root.resolve())
                    if dataset_root.exists() or dataset_root.is_symlink()
                    else None,
                    "parquet_files": len(files),
                }
            )
        datasets.append(
            {
                "source_repo_id": source,
                "safe_dataset_name": safe_name,
                "total_parquet_files": total_files,
                "roots": root_summaries,
            }
        )

    return {
        "config_name": str(config["config_name"]),
        "embedding_dataset_roots": [str(root) for root in roots],
        "output_dir": str(config["output_dir"]),
        "enabled_future_latent_datasets": len(datasets),
        "total_parquet_files": sum(int(dataset["total_parquet_files"]) for dataset in datasets),
        "missing_or_empty_datasets": [
            dataset["source_repo_id"] for dataset in datasets if int(dataset["total_parquet_files"]) == 0
        ],
        "datasets": datasets,
    }


def print_embedding_input_summary(summary: dict[str, Any]) -> None:
    print("VLA sidecar embedding input summary:")
    print(f"  config_name: {summary['config_name']}")
    print(f"  output_dir: {summary['output_dir']}")
    print(f"  embedding_dataset_roots: {summary['embedding_dataset_roots']}")
    print(f"  enabled future-latent datasets: {summary['enabled_future_latent_datasets']}")
    print(f"  total parquet files found: {summary['total_parquet_files']}")
    for dataset in summary["datasets"]:
        status = "found" if dataset["total_parquet_files"] else "missing/empty"
        print(
            f"  - {dataset['source_repo_id']} ({dataset['safe_dataset_name']}): "
            f"{status}, files={dataset['total_parquet_files']}"
        )
        for root_summary in dataset["roots"]:
            symlink = " symlink" if root_summary["is_symlink"] else ""
            print(
                f"    root={root_summary['embedding_root']} "
                f"files={root_summary['parquet_files']} exists={root_summary['exists']}{symlink}"
            )
            print(f"    dataset_path={root_summary['dataset_root']}")
            if root_summary["resolved_path"] and root_summary["resolved_path"] != root_summary["dataset_root"]:
                print(f"    resolves_to={root_summary['resolved_path']}")
    if summary["missing_or_empty_datasets"]:
        print(f"  WARNING missing/empty datasets: {summary['missing_or_empty_datasets']}")


def discover_inputs(config: dict[str, Any]) -> list[DatasetInput]:
    roots = [Path(path) for path in config.get("embedding_dataset_roots", [])]
    if not roots:
        raise ValueError("Config must provide at least one embedding_dataset_roots entry.")

    duplicate_policy = str(config.get("duplicate_dataset_policy", "error"))
    if duplicate_policy not in {"error", "skip", "merge"}:
        raise ValueError("duplicate_dataset_policy must be one of: error, skip, merge")

    by_source: dict[str, DatasetInput] = {}
    merge_files: dict[str, list[Path]] = {}
    merge_root: dict[str, Path] = {}
    for spec in enabled_specs(str(config["config_name"])):
        source = spec.repo_id
        safe_name = safe_dataset_name(source)
        for embedding_root in roots:
            dataset_root = embedding_root / safe_name
            files = tuple(sorted(dataset_root.glob("*.parquet")))
            if not files:
                continue
            if source in by_source or source in merge_files:
                if duplicate_policy == "error":
                    raise ValueError(
                        f"Dataset {source!r} appears in multiple embedding roots. "
                        "Set duplicate_dataset_policy to 'merge' or 'skip' if intended."
                    )
                if duplicate_policy == "skip":
                    continue
            if duplicate_policy == "merge":
                merge_files.setdefault(source, []).extend(files)
                merge_root.setdefault(source, dataset_root)
            else:
                by_source[source] = DatasetInput(source, safe_name, dataset_root, files)

    if duplicate_policy == "merge":
        by_source = {
            source: DatasetInput(source, safe_dataset_name(source), merge_root[source], tuple(sorted(files)))
            for source, files in merge_files.items()
        }

    inputs = list(by_source.values())
    if not inputs:
        searched = ", ".join(str(root) for root in roots)
        raise FileNotFoundError(f"No generated embedding parquet datasets found under: {searched}")
    return sorted(inputs, key=lambda item: item.source_repo_id)


def build_resampler_from_checkpoint(path: Path, *, device: torch.device) -> ResamplerAutoencoder:
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint.get("config", {}).get("model", {})
    resampler = ResamplerAutoencoder(
        cameras=tuple(config.get("cameras", CAMERAS)),
        input_tokens=int(config.get("input_tokens", 256)),
        input_dim=int(config.get("input_dim", 2048)),
        latent_tokens=int(config.get("latent_tokens", 24)),
        latent_dim=int(config.get("latent_dim", 512)),
        encoder_layers=int(config.get("encoder_layers", 2)),
        decoder_layers=int(config.get("decoder_layers", 2)),
        num_heads=int(config.get("num_heads", 16)),
        mlp_ratio=float(config.get("mlp_ratio", 4.0)),
        dropout=float(config.get("dropout", 0.0)),
    ).to(device)

    if "encoder" in checkpoint:
        missing, unexpected = resampler.load_state_dict(checkpoint["encoder"], strict=False)
        unexpected = [key for key in unexpected if not key.startswith("camera_decoders.")]
        missing = [key for key in missing if key.startswith("camera_encoders.")]
        if missing or unexpected:
            raise RuntimeError(f"Could not load resampler encoder from {path}: missing={missing[:5]} unexpected={unexpected[:5]}")
    elif "model" in checkpoint:
        resampler.load_state_dict(checkpoint["model"], strict=True)
    else:
        resampler.load_state_dict(checkpoint, strict=True)

    resampler.camera_decoders = nn.ModuleDict()
    resampler.eval()
    for parameter in resampler.parameters():
        parameter.requires_grad_(False)
    return resampler


def build_future_predictor_from_checkpoint(path: Path, *, device: torch.device) -> FutureLatentPredictor:
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint.get("config", {}).get("model", {})
    if config.get("state_dim") is None:
        raise ValueError(
            f"Future predictor checkpoint {path} does not contain config.model.state_dim. "
            "Use a checkpoint saved after training initialized state_dim."
        )
    model = FutureLatentPredictor(
        state_dim=int(config["state_dim"]),
        num_cameras=len(config.get("cameras", CAMERA_ORDER)),
        latent_tokens=int(config.get("latent_tokens", 24)),
        latent_dim=int(config.get("latent_dim", 512)),
        num_state_tokens=int(config.get("num_state_tokens", 4)),
        state_hidden_dim=int(config.get("state_hidden_dim", 1024)),
        transformer_layers=int(config.get("transformer_layers", 6)),
        num_heads=int(config.get("num_heads", 16)),
        mlp_ratio=float(config.get("mlp_ratio", 4.0)),
        dropout=float(config.get("dropout", 0.0)),
        predict_residual=bool(config.get("predict_residual", True)),
    ).to(device)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def required_columns(cameras: tuple[str, ...]) -> list[str]:
    columns = [
        "source_repo_id",
        "source_index",
        "state",
        "embedding_dtype",
        "episode_index",
        "frame_index",
        "index",
        "task_index",
    ]
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


def available_required_columns(path: Path, cameras: tuple[str, ...]) -> list[str]:
    schema = pl.read_parquet_schema(path)
    return [column for column in required_columns(cameras) if column in schema]


def scalar_or_none(value: Any) -> int | float | str | None:
    if value is None:
        return None
    array = np.asarray(value)
    if array.shape == ():
        return array.item()
    if array.size == 1:
        return array.reshape(()).item()
    return None


def decode_embedding(value: Any, *, dtype: np.dtype, shape: tuple[int, int]) -> np.ndarray:
    if value is None:
        return np.zeros(shape, dtype=np.float16)
    return np.frombuffer(value, dtype=dtype).reshape(shape).astype(np.float16, copy=False)


def batch_from_rows(rows: list[dict[str, Any]], *, cameras: tuple[str, ...]) -> dict[str, Any]:
    current = np.zeros((len(rows), len(cameras), *EMBEDDING_SHAPE), dtype=np.float16)
    future = np.zeros((len(rows), len(cameras), *EMBEDDING_SHAPE), dtype=np.float16)
    valid = np.zeros((len(rows), len(cameras)), dtype=np.bool_)
    states = []
    metadata = []

    for row_index, row in enumerate(rows):
        dtype = np.dtype(str(row["embedding_dtype"]))
        states.append(np.asarray(row["state"], dtype=np.float32).reshape(-1))
        metadata.append(row)
        for camera_index, camera in enumerate(cameras):
            current_valid = bool(row.get(f"{camera}_embedding_valid", False))
            future_valid = bool(row.get(f"{camera}_embedding_t_5_valid", False))
            is_valid = current_valid and future_valid
            valid[row_index, camera_index] = is_valid
            if not is_valid:
                continue
            current[row_index, camera_index] = decode_embedding(
                row[f"{camera}_embedding_t"],
                dtype=dtype,
                shape=EMBEDDING_SHAPE,
            )
            future[row_index, camera_index] = decode_embedding(
                row[f"{camera}_embedding_t_5"],
                dtype=dtype,
                shape=EMBEDDING_SHAPE,
            )

    return {
        "current": torch.from_numpy(current),
        "future": torch.from_numpy(future),
        "valid": valid,
        "state": torch.from_numpy(np.stack(states, axis=0).astype(np.float32)),
        "metadata": metadata,
    }


def latent_to_bytes(array: np.ndarray, *, dtype: np.dtype) -> bytes:
    return np.asarray(array, dtype=dtype).tobytes(order="C")


def sidecar_rows_from_batch(
    *,
    batch: dict[str, Any],
    current_compact: torch.Tensor,
    future_true: torch.Tensor,
    future_pred: torch.Tensor,
    source_repo_id: str,
    future_offset: int,
    output_dtype: np.dtype,
    cameras: tuple[str, ...],
) -> list[dict[str, Any]]:
    current_compact_np = current_compact.detach().cpu().numpy()
    future_true_np = future_true.detach().cpu().numpy()
    future_pred_np = future_pred.detach().cpu().numpy()
    rows = []
    for row_index, metadata in enumerate(batch["metadata"]):
        valid = batch["valid"][row_index]
        row: dict[str, Any] = {
            "source_dataset": source_repo_id,
            "source_repo_id": source_repo_id,
            "future_offset": int(future_offset),
            "latent_dtype": np.dtype(output_dtype).name,
            "latent_shape": list(LATENT_SHAPE),
        }
        for key in ("episode_index", "frame_index", "index", "task_index", "source_index"):
            value = scalar_or_none(metadata.get(key))
            if value is not None:
                row[key] = value
        for camera_index, camera in enumerate(cameras):
            camera_valid = bool(valid[camera_index])
            row[f"{camera}_future_latent_valid"] = camera_valid
            if camera_valid:
                pred_value = future_pred_np[row_index, camera_index]
                true_value = future_true_np[row_index, camera_index]
                current_value = current_compact_np[row_index, camera_index]
            else:
                pred_value = np.zeros(LATENT_SHAPE, dtype=output_dtype)
                true_value = np.zeros(LATENT_SHAPE, dtype=output_dtype)
                current_value = np.zeros(LATENT_SHAPE, dtype=output_dtype)
            row[f"{camera}_future_latent_pred"] = latent_to_bytes(pred_value, dtype=output_dtype)
            row[f"{camera}_future_latent_true"] = latent_to_bytes(true_value, dtype=output_dtype)
            if bool(metadata.get("_write_current_latents", False)):
                row[f"{camera}_current_latent_compact"] = latent_to_bytes(current_value, dtype=output_dtype)
        rows.append(row)
    return rows


def write_shard(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(output_path)


def output_shard_path(output_root: Path, safe_name: str, *, worker_index: int, num_workers: int, shard_index: int) -> Path:
    if num_workers == 1:
        return output_root / safe_name / f"part-{shard_index:06d}.parquet"
    return output_root / safe_name / f"part-worker-{worker_index:03d}-{shard_index:06d}.parquet"


def write_metadata(
    *,
    output_root: Path,
    dataset_input: DatasetInput,
    config: dict[str, Any],
    rows_written: int,
    rows_skipped: int,
    worker_index: int,
    num_workers: int,
) -> None:
    metadata = {
        "source_repo_id": dataset_input.source_repo_id,
        "safe_dataset_name": dataset_input.safe_name,
        "embedding_dataset_root": str(dataset_input.root),
        "rows_written": rows_written,
        "rows_skipped": rows_skipped,
        "worker_index": worker_index,
        "num_workers": num_workers,
        "future_offset": int(config["future_offset"]),
        "latent_dtype": str(config.get("dtype", "float16")),
        "latent_shape_per_camera": list(LATENT_SHAPE),
        "camera_order": list(config.get("cameras", CAMERA_ORDER)),
        "resampler_checkpoint": str(config["resampler_checkpoint"]),
        "future_predictor_checkpoint": str(config["future_predictor_checkpoint"]),
        "columns": [
            "top_future_latent_pred",
            "right_wrist_future_latent_pred",
            "left_wrist_future_latent_pred",
            "top_future_latent_true",
            "right_wrist_future_latent_true",
            "left_wrist_future_latent_true",
            "top_future_latent_valid",
            "right_wrist_future_latent_valid",
            "left_wrist_future_latent_valid",
            "future_offset",
            "episode_index",
            "frame_index",
            "index",
            "task_index",
            "source_dataset",
        ],
    }
    output_dir = output_root / dataset_input.safe_name
    output_dir.mkdir(parents=True, exist_ok=True)
    name = "vla_sidecar_metadata.json" if num_workers == 1 else f"vla_sidecar_metadata_worker-{worker_index:03d}.json"
    save_json(output_dir / name, metadata)


def row_batches_from_file(path: Path, *, batch_size: int, cameras: tuple[str, ...]) -> list[list[dict[str, Any]]]:
    frame = pl.read_parquet(path, columns=available_required_columns(path, cameras))
    rows = frame.to_dicts()
    return [rows[index : index + batch_size] for index in range(0, len(rows), batch_size)]


def parquet_row_count(path: Path) -> int:
    return int(pl.scan_parquet(path).select(pl.len()).collect().item())


def progress_total_for_files(files: list[Path], *, max_rows: int | None, num_workers: int) -> int:
    if max_rows is not None:
        return max(0, (max_rows + num_workers - 1) // num_workers)
    return sum(parquet_row_count(path) for path in files)


def process_dataset(
    *,
    dataset_input: DatasetInput,
    config: dict[str, Any],
    resampler: ResamplerAutoencoder,
    predictor: FutureLatentPredictor,
    device: torch.device,
    worker_index: int,
    num_workers: int,
) -> None:
    output_root = Path(config["output_dir"])
    output_dir = output_root / dataset_input.safe_name
    if bool(config.get("skip_existing", False)) and list(output_dir.glob("*.parquet")):
        logging.info("Skipping %s because %s already contains parquet shards.", dataset_input.source_repo_id, output_dir)
        return

    cameras = tuple(config.get("cameras", CAMERA_ORDER))
    output_dtype = np.dtype(str(config.get("dtype", "float16")))
    batch_size = int(config.get("batch_size", 64))
    shard_size = int(config.get("shard_size", 1024))
    future_offset = int(config["future_offset"])
    max_rows = config.get("max_rows_per_dataset")
    max_rows = None if max_rows is None else int(max_rows)
    write_current_latents = bool(config.get("write_current_latents", False))
    precision = str(config.get("precision", "fp16"))
    use_autocast = device.type == "cuda" and precision in {"fp16", "float16", "bf16", "bfloat16"}
    autocast_dtype = torch.bfloat16 if precision in {"bf16", "bfloat16"} else torch.float16

    files = [
        path
        for index, path in enumerate(dataset_input.files)
        if index % num_workers == worker_index
    ]
    rows_written = 0
    rows_skipped = 0
    shard_rows: list[dict[str, Any]] = []
    shard_index = 0
    progress_total = progress_total_for_files(files, max_rows=max_rows, num_workers=num_workers)
    progress = tqdm(
        total=progress_total,
        desc=f"Sidecar {dataset_input.safe_name} worker {worker_index}/{num_workers}",
        unit="row",
        dynamic_ncols=True,
        position=worker_index,
        leave=True,
    )
    progress.set_postfix(
        {
            "written": rows_written,
            "skipped": rows_skipped,
            "shard": shard_index,
            "files": len(files),
        },
        refresh=False,
    )

    try:
        with torch.inference_mode():
            for path in files:
                for rows in row_batches_from_file(path, batch_size=batch_size, cameras=cameras):
                    if max_rows is not None and rows_written + rows_skipped >= max_rows:
                        break
                    for row in rows:
                        row["_write_current_latents"] = write_current_latents
                    batch = batch_from_rows(rows, cameras=cameras)
                    if not batch["valid"].any():
                        rows_skipped += len(rows)
                        progress.update(len(rows))
                        progress.set_postfix(
                            {
                                "written": rows_written,
                                "skipped": rows_skipped,
                                "shard": shard_index,
                                "file": path.name,
                            },
                            refresh=False,
                        )
                        continue
                    current = batch["current"].to(device, non_blocking=True)
                    future = batch["future"].to(device, non_blocking=True)
                    state = batch["state"].to(device, non_blocking=True)
                    with torch.amp.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
                        current_compact = resampler.encode(current)
                        future_true = resampler.encode(future)
                        future_pred = predictor(current_compact, state)["prediction"]
                    new_rows = sidecar_rows_from_batch(
                        batch=batch,
                        current_compact=current_compact,
                        future_true=future_true,
                        future_pred=future_pred,
                        source_repo_id=dataset_input.source_repo_id,
                        future_offset=future_offset,
                        output_dtype=output_dtype,
                        cameras=cameras,
                    )
                    shard_rows.extend(new_rows)
                    rows_written += len(new_rows)
                    progress.update(len(rows))
                    progress.set_postfix(
                        {
                            "written": rows_written,
                            "skipped": rows_skipped,
                            "shard": shard_index,
                            "file": path.name,
                        },
                        refresh=False,
                    )
                    if len(shard_rows) >= shard_size:
                        write_shard(
                            shard_rows,
                            output_shard_path(
                                output_root,
                                dataset_input.safe_name,
                                worker_index=worker_index,
                                num_workers=num_workers,
                                shard_index=shard_index,
                            ),
                        )
                        shard_rows = []
                        shard_index += 1
                        progress.set_postfix(
                            {
                                "written": rows_written,
                                "skipped": rows_skipped,
                                "shard": shard_index,
                                "file": path.name,
                            },
                            refresh=False,
                        )
                if max_rows is not None and rows_written + rows_skipped >= max_rows:
                    break
        if shard_rows:
            write_shard(
                shard_rows,
                output_shard_path(
                    output_root,
                    dataset_input.safe_name,
                    worker_index=worker_index,
                    num_workers=num_workers,
                    shard_index=shard_index,
                ),
            )
        write_metadata(
            output_root=output_root,
            dataset_input=dataset_input,
            config=config,
            rows_written=rows_written,
            rows_skipped=rows_skipped,
            worker_index=worker_index,
            num_workers=num_workers,
        )
    finally:
        progress.close()


def validate_config(config: dict[str, Any]) -> None:
    required = [
        "config_name",
        "embedding_dataset_roots",
        "output_dir",
        "resampler_checkpoint",
        "future_predictor_checkpoint",
        "future_offset",
    ]
    missing = [
        key
        for key in required
        if key not in config
        or config[key] is None
        or (isinstance(config[key], str) and config[key] == "")
        or (isinstance(config[key], list) and not config[key])
    ]
    if missing:
        raise ValueError(f"Generator config is missing required keys: {missing}")
    if int(config.get("future_offset", 0)) <= 0:
        raise ValueError("future_offset must be > 0")
    if int(config.get("batch_size", 0)) <= 0:
        raise ValueError("batch_size must be > 0")
    if int(config.get("shard_size", 0)) <= 0:
        raise ValueError("shard_size must be > 0")
    if str(config.get("dtype", "float16")) not in {"float16", "float32"}:
        raise ValueError("dtype must be 'float16' or 'float32'")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    config = resolve_config(args)
    validate_config(config)

    if args.num_workers == 1 and args.worker_index == 0:
        output_dir = Path(config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        input_summary = embedding_input_summary(config)
        print_embedding_input_summary(input_summary)
        save_json(output_dir / "vla_sidecar_embedding_input_summary.json", input_summary)
        if args.check_inputs_only:
            if input_summary["missing_or_empty_datasets"]:
                raise FileNotFoundError(
                    f"Missing/empty embedding datasets: {input_summary['missing_or_empty_datasets']}"
                )
            return

    if maybe_launch_gpu_workers(args, config):
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    inputs = discover_inputs(config)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.worker_index == 0:
        save_json(output_dir / "resolved_vla_sidecar_generation_config.json", config)

    resampler = build_resampler_from_checkpoint(Path(config["resampler_checkpoint"]), device=device)
    predictor = build_future_predictor_from_checkpoint(Path(config["future_predictor_checkpoint"]), device=device)

    for dataset_input in inputs:
        process_dataset(
            dataset_input=dataset_input,
            config=config,
            resampler=resampler,
            predictor=predictor,
            device=device,
            worker_index=args.worker_index,
            num_workers=args.num_workers,
        )


if __name__ == "__main__":
    main()
