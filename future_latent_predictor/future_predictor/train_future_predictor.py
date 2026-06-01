from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from future_latent_predictor.future_predictor.dataset import FuturePredictionDataConfig
from future_latent_predictor.future_predictor.dataset import FuturePredictionDataset
from future_latent_predictor.future_predictor.dataset import estimate_num_prediction_samples
from future_latent_predictor.future_predictor.dataset import infer_state_dim
from future_latent_predictor.future_predictor.dataset import parquet_file_summary
from future_latent_predictor.future_predictor.model import FutureLatentPredictor
from future_latent_predictor.future_predictor.model import count_parameters
from future_latent_predictor.resampler_autoencoder.model import CAMERAS
from future_latent_predictor.resampler_autoencoder.model import ResamplerAutoencoder
from future_latent_predictor.resampler_autoencoder.model import count_parameters as count_resampler_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the compact future-latent predictor.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None, help="Override data.data_root in the JSON config.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override training.output_dir in the JSON config.")
    parser.add_argument(
        "--resampler-checkpoint",
        type=Path,
        default=None,
        help="Override resampler.checkpoint_path in the JSON config.",
    )
    parser.add_argument("--resume", type=Path, default=None, help="Resume from a future predictor checkpoint.")
    return parser.parse_args()


def setup_distributed() -> tuple[int, int, int, bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_distributed = world_size > 1
    if is_distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    return rank, local_rank, world_size, is_distributed


def cleanup_distributed(is_distributed: bool) -> None:
    if is_distributed:
        dist.destroy_process_group()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def seed_everything(seed: int, rank: int) -> None:
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataset_config(config: dict[str, Any], *, split: str) -> FuturePredictionDataConfig:
    data = config["data"]
    return FuturePredictionDataConfig(
        data_root=Path(data["data_root"]),
        include_datasets=tuple(data.get("include_datasets", [])),
        cameras=tuple(data.get("cameras", CAMERAS)),
        embedding_shape=tuple(data.get("embedding_shape", [256, 2048])),
        shuffle_files=bool(data.get("shuffle_files", True)),
        shuffle_rows=bool(data.get("shuffle_rows", True)),
        seed=int(config["training"].get("seed", 42)),
        split=split,
        val_fraction=float(data.get("val_fraction", 0.0)),
        dataset_weights=dict(data.get("dataset_weights", {})),
    )


def print_dataset_file_summary(summary: dict[str, Any]) -> None:
    print("Future predictor dataset file summary:")
    print(f"  data_root: {summary['data_root']}")
    include_datasets = summary.get("include_datasets") or ["<all datasets under data_root>"]
    print(f"  include_datasets: {include_datasets}")
    print(f"  total parquet files found: {summary['total_parquet_files']}")
    for root_summary in summary["configured_roots"]:
        status = "found" if root_summary["parquet_files"] else "missing/empty"
        symlink = " symlink" if root_summary["is_symlink"] else ""
        print(
            f"  - {root_summary['dataset']}: {status}, "
            f"files={root_summary['parquet_files']}, exists={root_summary['exists']}{symlink}"
        )
        print(f"    path: {root_summary['path']}")
        if root_summary["resolved_path"] and root_summary["resolved_path"] != root_summary["path"]:
            print(f"    resolves_to: {root_summary['resolved_path']}")
    for split_name in ("train", "val"):
        split_summary = summary[split_name]
        print(f"  {split_name} parquet files: {split_summary['total_parquet_files']}")
        for dataset_name, count in split_summary["datasets"].items():
            print(f"    {dataset_name}: {count}")
    if summary["missing_or_empty_datasets"]:
        print(f"  WARNING missing/empty datasets: {summary['missing_or_empty_datasets']}")


def build_future_predictor(config: dict[str, Any], *, state_dim: int) -> FutureLatentPredictor:
    model_config = config["model"]
    return FutureLatentPredictor(
        state_dim=state_dim,
        num_cameras=len(model_config.get("cameras", CAMERAS)),
        latent_tokens=int(model_config.get("latent_tokens", 24)),
        latent_dim=int(model_config.get("latent_dim", 512)),
        num_state_tokens=int(model_config.get("num_state_tokens", 4)),
        state_hidden_dim=int(model_config.get("state_hidden_dim", 1024)),
        transformer_layers=int(model_config.get("transformer_layers", 6)),
        num_heads=int(model_config.get("num_heads", 16)),
        mlp_ratio=float(model_config.get("mlp_ratio", 4.0)),
        dropout=float(model_config.get("dropout", 0.0)),
        predict_residual=bool(model_config.get("predict_residual", True)),
    )


def build_resampler_from_checkpoint(config: dict[str, Any], *, device: torch.device) -> ResamplerAutoencoder:
    checkpoint_path = Path(config["resampler"]["checkpoint_path"])
    checkpoint = torch.load(checkpoint_path, map_location=device)
    resampler_config = checkpoint.get("config", {}).get("model", config["resampler"].get("model", {}))
    resampler = ResamplerAutoencoder(
        cameras=tuple(resampler_config.get("cameras", CAMERAS)),
        input_tokens=int(resampler_config.get("input_tokens", 256)),
        input_dim=int(resampler_config.get("input_dim", 2048)),
        latent_tokens=int(resampler_config.get("latent_tokens", 24)),
        latent_dim=int(resampler_config.get("latent_dim", 512)),
        encoder_layers=int(resampler_config.get("encoder_layers", 2)),
        decoder_layers=int(resampler_config.get("decoder_layers", 2)),
        num_heads=int(resampler_config.get("num_heads", 16)),
        mlp_ratio=float(resampler_config.get("mlp_ratio", 4.0)),
        dropout=float(resampler_config.get("dropout", 0.0)),
    ).to(device)

    if "encoder" in checkpoint:
        state_dict = checkpoint["encoder"]
        missing, unexpected = resampler.load_state_dict(state_dict, strict=False)
        unexpected = [key for key in unexpected if not key.startswith("camera_decoders.")]
        if unexpected:
            raise RuntimeError(f"Unexpected resampler encoder keys in {checkpoint_path}: {unexpected}")
        missing = [key for key in missing if key.startswith("camera_encoders.")]
        if missing:
            raise RuntimeError(f"Missing resampler encoder keys in {checkpoint_path}: {missing[:10]}")
    elif "model" in checkpoint:
        resampler.load_state_dict(checkpoint["model"], strict=True)
    else:
        resampler.load_state_dict(checkpoint, strict=True)

    resampler.camera_decoders = nn.ModuleDict()
    resampler.eval()
    for parameter in resampler.parameters():
        parameter.requires_grad_(False)
    return resampler


def compact_prediction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    current: torch.Tensor,
    valid: torch.Tensor,
    *,
    mse_weight: float,
    cosine_weight: float,
    delta_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    valid_camera_mask = valid[:, :, None, None].to(torch.float32)
    valid_token_mask = valid[:, :, None].to(torch.float32)

    prediction_float = prediction.float()
    target_float = target.float()
    current_float = current.float()

    mse_denominator = (valid_camera_mask.sum() * target.shape[2] * target.shape[3]).clamp_min(1.0)
    cosine_denominator = (valid_token_mask.sum() * target.shape[2]).clamp_min(1.0)

    mse = ((prediction_float - target_float).square() * valid_camera_mask).sum() / mse_denominator
    cosine = F.cosine_similarity(prediction_float, target_float, dim=-1)
    cosine_loss = ((1.0 - cosine) * valid_token_mask).sum() / cosine_denominator

    target_delta = target_float - current_float
    predicted_delta = prediction_float - current_float
    delta_mse = ((predicted_delta - target_delta).square() * valid_camera_mask).sum() / mse_denominator
    delta_cosine = F.cosine_similarity(predicted_delta, target_delta, dim=-1)
    delta_cosine_score = (delta_cosine * valid_token_mask).sum() / cosine_denominator

    predicted_delta_norm = predicted_delta.norm(dim=-1)
    target_delta_norm = target_delta.norm(dim=-1)
    delta_norm_ratio = (predicted_delta_norm / target_delta_norm.clamp_min(1e-8) * valid_token_mask).sum() / cosine_denominator

    delta_dot = (predicted_delta * target_delta).sum(dim=-1)
    target_delta_norm_sq = target_delta.square().sum(dim=-1)
    delta_projection_ratio = (
        delta_dot / target_delta_norm_sq.clamp_min(1e-8) * valid_token_mask
    ).sum() / cosine_denominator

    copy_mse = ((current_float - target_float).square() * valid_camera_mask).sum() / mse_denominator
    copy_cosine = F.cosine_similarity(current_float, target_float, dim=-1)
    copy_cosine_loss = ((1.0 - copy_cosine) * valid_token_mask).sum() / cosine_denominator

    loss = mse * mse_weight + cosine_loss * cosine_weight + delta_mse * delta_weight
    improvement = 1.0 - (mse / copy_mse.clamp_min(1e-8))
    return loss, {
        "loss": float(loss.detach().cpu()),
        "mse": float(mse.detach().cpu()),
        "cosine": float(cosine_loss.detach().cpu()),
        "delta_mse": float(delta_mse.detach().cpu()),
        "delta_cosine": float(delta_cosine_score.detach().cpu()),
        "delta_norm_ratio": float(delta_norm_ratio.detach().cpu()),
        "delta_projection_ratio": float(delta_projection_ratio.detach().cpu()),
        "copy_mse": float(copy_mse.detach().cpu()),
        "copy_cosine": float(copy_cosine_loss.detach().cpu()),
        "mse_improvement": float(improvement.detach().cpu()),
    }


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-8, float(step + 1) / float(warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def next_batch(iterator: Any, loader: DataLoader) -> tuple[Any, Any]:
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def init_wandb(config: dict[str, Any], *, rank: int) -> Any | None:
    wandb_config = config.get("wandb", {})
    if rank != 0 or not bool(wandb_config.get("enabled", False)):
        return None
    try:
        import wandb
    except ImportError:
        print("wandb.enabled=true but wandb is not installed; continuing without wandb logging.")
        return None

    token = str(wandb_config.get("token", "") or os.environ.get("WANDB_API_KEY", ""))
    if token:
        wandb.login(key=token)
    return wandb.init(
        project=wandb_config.get("project", "lehome-future-latent"),
        entity=wandb_config.get("entity") or None,
        name=wandb_config.get("name") or None,
        config=config,
    )


def save_checkpoint(
    *,
    output_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.amp.GradScaler,
    config: dict[str, Any],
    epoch: int,
    global_step: int,
    parameter_count: int,
    name: str,
) -> None:
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    state_dict = raw_model.state_dict()
    checkpoint = {
        "model": state_dict,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "config": config,
        "epoch": epoch,
        "global_step": global_step,
        "parameter_count": parameter_count,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_dir / f"{name}.pt")
    torch.save(
        {
            "model": state_dict,
            "config": config,
            "parameter_count": parameter_count,
            "epoch": epoch,
            "global_step": global_step,
        },
        output_dir / "future_predictor_latest.pt",
    )


def load_checkpoint(
    *,
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> tuple[int, int]:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
    return int(checkpoint["epoch"]) + 1, int(checkpoint["global_step"])


def write_log(output_dir: Path, payload: dict[str, Any]) -> None:
    with (output_dir / "train_log.jsonl").open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload) + "\n")


def print_metric_line(payload: dict[str, Any]) -> None:
    split = payload.get("split", "train")
    if split == "val":
        message = (
            f"val step={payload['global_step']} epoch={payload['epoch'] + 1} "
            f"loss={payload['val_loss']:.5f} mse={payload['val_mse']:.5f} "
            f"copy={payload['val_copy_mse']:.5f} imp={payload['val_mse_improvement']:.3f} "
            f"cos={payload['val_cosine']:.5f} copy_cos={payload['val_copy_cosine']:.5f} "
            f"dcos={payload['val_delta_cosine']:.3f} "
            f"dnorm={payload['val_delta_norm_ratio']:.3f} "
            f"dproj={payload['val_delta_projection_ratio']:.3f}"
        )
    else:
        message = (
            f"train step={payload['global_step']} epoch={payload['epoch'] + 1} "
            f"loss={payload['loss']:.5f} mse={payload['mse']:.5f} "
            f"copy={payload['copy_mse']:.5f} imp={payload['mse_improvement']:.3f} "
            f"cos={payload['cosine']:.5f} dcos={payload['delta_cosine']:.3f} "
            f"dnorm={payload['delta_norm_ratio']:.3f} "
            f"dproj={payload['delta_projection_ratio']:.3f} lr={payload['lr']:.2e}"
        )
    tqdm.write(message)


@torch.no_grad()
def validate(
    *,
    model: nn.Module,
    resampler: ResamplerAutoencoder,
    loader: DataLoader,
    device: torch.device,
    val_batches: int,
    use_fp16: bool,
    mse_weight: float,
    cosine_weight: float,
    delta_weight: float,
    show_progress: bool = False,
    desc: str = "validation",
) -> dict[str, float]:
    model.eval()
    resampler.eval()
    iterator = iter(loader)
    total_stats = {
        "loss": 0.0,
        "mse": 0.0,
        "cosine": 0.0,
        "delta_mse": 0.0,
        "delta_cosine": 0.0,
        "delta_norm_ratio": 0.0,
        "delta_projection_ratio": 0.0,
        "copy_mse": 0.0,
        "copy_cosine": 0.0,
        "mse_improvement": 0.0,
    }
    completed_batches = 0

    progress = tqdm(
        range(val_batches),
        desc=desc,
        disable=not show_progress,
        dynamic_ncols=True,
        leave=False,
        position=1,
        unit="batch",
    )
    for _ in progress:
        try:
            batch = next(iterator)
        except StopIteration:
            break
        current_embeddings = batch["current_embeddings"].to(device=device, non_blocking=True)
        future_embeddings = batch["future_embeddings"].to(device=device, non_blocking=True)
        valid = batch["valid"].to(device=device, non_blocking=True)
        state = batch["state"].to(device=device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_fp16):
            current_latents = resampler.encode(current_embeddings)
            future_latents = resampler.encode(future_embeddings)
            outputs = model(current_latents, state)
            _, stats = compact_prediction_loss(
                outputs["prediction"],
                future_latents,
                current_latents,
                valid,
                mse_weight=mse_weight,
                cosine_weight=cosine_weight,
                delta_weight=delta_weight,
            )
        for key in total_stats:
            total_stats[key] += stats[key]
        completed_batches += 1
        if show_progress:
            progress.set_postfix(
                loss=total_stats["loss"] / completed_batches,
                mse=total_stats["mse"] / completed_batches,
                copy=total_stats["copy_mse"] / completed_batches,
                imp=total_stats["mse_improvement"] / completed_batches,
                dcos=total_stats["delta_cosine"] / completed_batches,
                dnorm=total_stats["delta_norm_ratio"] / completed_batches,
                dproj=total_stats["delta_projection_ratio"] / completed_batches,
            )

    model.train()
    if completed_batches == 0:
        return {f"val_{key}": float("nan") for key in total_stats}
    averaged = {f"val_{key}": value / completed_batches for key, value in total_stats.items()}
    averaged["val_mse_improvement"] = 1.0 - (
        averaged["val_mse"] / max(averaged["val_copy_mse"], 1e-8)
    )
    return averaged


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.data_root is not None:
        config["data"]["data_root"] = str(args.data_root)
    if args.output_dir is not None:
        config["training"]["output_dir"] = str(args.output_dir)
    if args.resampler_checkpoint is not None:
        config["resampler"]["checkpoint_path"] = str(args.resampler_checkpoint)

    rank, local_rank, world_size, is_distributed = setup_distributed()
    training_config = config["training"]
    output_dir = Path(training_config["output_dir"])
    seed_everything(int(training_config.get("seed", 42)), rank)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    dataset_config = build_dataset_config(config, split="train")
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset_summary = parquet_file_summary(
            data_root=dataset_config.data_root,
            include_datasets=dataset_config.include_datasets,
            val_fraction=dataset_config.val_fraction,
            seed=dataset_config.seed,
        )
        print_dataset_file_summary(dataset_summary)
        save_json(output_dir / "dataset_file_summary.json", dataset_summary)

    if config["model"].get("state_dim") is None:
        state_dim = infer_state_dim(
            data_root=dataset_config.data_root,
            include_datasets=dataset_config.include_datasets,
        )
        config["model"]["state_dim"] = state_dim
    else:
        state_dim = int(config["model"]["state_dim"])

    dataset = FuturePredictionDataset(dataset_config, rank=rank, world_size=world_size)
    loader = DataLoader(
        dataset,
        batch_size=int(training_config["batch_size_per_gpu"]),
        num_workers=int(training_config.get("num_workers", 2)),
        pin_memory=bool(training_config.get("pin_memory", True)) and device.type == "cuda",
        persistent_workers=False,
        drop_last=True,
    )
    val_loader = None
    val_fraction = float(config["data"].get("val_fraction", 0.0))
    if val_fraction > 0.0 and rank == 0:
        val_config = build_dataset_config(config, split="val")
        val_config = dataclasses.replace(val_config, shuffle_files=False, shuffle_rows=False)
        val_dataset = FuturePredictionDataset(val_config, rank=0, world_size=1)
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(training_config["batch_size_per_gpu"]),
            num_workers=int(training_config.get("val_num_workers", 0)),
            pin_memory=bool(training_config.get("pin_memory", True)) and device.type == "cuda",
            persistent_workers=False,
            drop_last=False,
        )

    resampler = build_resampler_from_checkpoint(config, device=device)
    model = build_future_predictor(config, state_dim=state_dim).to(device)
    parameter_count = count_parameters(model)
    resampler_parameter_count = count_resampler_parameters(resampler)
    if is_distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config.get("weight_decay", 0.01)),
        betas=tuple(training_config.get("adam_betas", [0.9, 0.95])),
    )

    steps_per_epoch = training_config.get("steps_per_epoch")
    if steps_per_epoch is None:
        total_samples = estimate_num_prediction_samples(
            data_root=dataset_config.data_root,
            include_datasets=dataset_config.include_datasets,
            split=dataset_config.split,
            val_fraction=dataset_config.val_fraction,
            seed=dataset_config.seed,
        )
        global_batch = (
            world_size
            * int(training_config["batch_size_per_gpu"])
            * int(training_config.get("gradient_accumulation_steps", 1))
        )
        steps_per_epoch = max(1, math.ceil(total_samples / global_batch))
    steps_per_epoch = int(steps_per_epoch)

    total_optimizer_steps = steps_per_epoch * int(training_config["epochs"])
    scheduler = make_scheduler(
        optimizer,
        warmup_steps=int(training_config.get("warmup_steps", 500)),
        total_steps=total_optimizer_steps,
        min_lr_ratio=float(training_config.get("min_lr_ratio", 0.05)),
    )
    use_fp16 = str(training_config.get("precision", "fp16")).lower() == "fp16" and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    start_epoch = 0
    global_step = 0
    if args.resume is not None:
        start_epoch, global_step = load_checkpoint(
            path=args.resume,
            model=model.module if isinstance(model, DistributedDataParallel) else model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )

    wandb_run = init_wandb(config, rank=rank)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_json(output_dir / "resolved_config.json", config)
        print(f"Future predictor trainable parameters: {parameter_count:,}")
        print(f"Frozen resampler loaded parameters: {resampler_parameter_count:,}")
        print(f"World size: {world_size}; batch per GPU: {training_config['batch_size_per_gpu']}")
        print(f"Optimizer steps per epoch: {steps_per_epoch}")

    grad_accum_steps = int(training_config.get("gradient_accumulation_steps", 1))
    log_every_steps = int(training_config.get("log_every_steps", 20))
    val_every_steps = int(training_config.get("val_every_steps", 0))
    val_batches = int(training_config.get("val_batches", 100))
    save_every_epochs = int(training_config.get("save_every_epochs", 1))
    mse_weight = float(config["loss"].get("mse_weight", 1.0))
    cosine_weight = float(config["loss"].get("cosine_weight", 0.1))
    delta_weight = float(config["loss"].get("delta_weight", 0.0))
    clip_grad_norm = float(training_config.get("clip_grad_norm", 1.0))

    for epoch in range(start_epoch, int(training_config["epochs"])):
        dataset.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        data_iter = iter(loader)
        progress = tqdm(
            range(steps_per_epoch),
            desc=f"epoch {epoch + 1}/{training_config['epochs']}",
            disable=rank != 0,
            dynamic_ncols=True,
        )
        epoch_started_at = time.monotonic()
        running_loss = 0.0

        for step_in_epoch in progress:
            step_stats = {
                "loss": 0.0,
                "mse": 0.0,
                "cosine": 0.0,
                "delta_mse": 0.0,
                "delta_cosine": 0.0,
                "delta_norm_ratio": 0.0,
                "delta_projection_ratio": 0.0,
                "copy_mse": 0.0,
                "copy_cosine": 0.0,
                "mse_improvement": 0.0,
            }
            for micro_step in range(grad_accum_steps):
                batch, data_iter = next_batch(data_iter, loader)
                current_embeddings = batch["current_embeddings"].to(device=device, non_blocking=True)
                future_embeddings = batch["future_embeddings"].to(device=device, non_blocking=True)
                valid = batch["valid"].to(device=device, non_blocking=True)
                state = batch["state"].to(device=device, non_blocking=True)

                should_sync = micro_step == grad_accum_steps - 1
                sync_context = (
                    contextlib.nullcontext()
                    if should_sync or not isinstance(model, DistributedDataParallel)
                    else model.no_sync()
                )
                with torch.no_grad():
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_fp16):
                        current_latents = resampler.encode(current_embeddings)
                        future_latents = resampler.encode(future_embeddings)

                with sync_context:
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_fp16):
                        outputs = model(current_latents, state)
                        loss, stats = compact_prediction_loss(
                            outputs["prediction"],
                            future_latents,
                            current_latents,
                            valid,
                            mse_weight=mse_weight,
                            cosine_weight=cosine_weight,
                            delta_weight=delta_weight,
                        )
                        loss = loss / grad_accum_steps
                    scaler.scale(loss).backward()

                for key in step_stats:
                    step_stats[key] += stats[key] / grad_accum_steps

            if clip_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            running_loss += step_stats["loss"]
            if rank == 0 and (global_step % log_every_steps == 0 or step_in_epoch == 0):
                lr = scheduler.get_last_lr()[0]
                elapsed = time.monotonic() - epoch_started_at
                avg_loss = running_loss / max(1, step_in_epoch + 1)
                log_payload = {
                    "split": "train",
                    "epoch": epoch,
                    "step_in_epoch": step_in_epoch,
                    "global_step": global_step,
                    "elapsed_seconds": elapsed,
                    "lr": lr,
                    "avg_loss": avg_loss,
                    **step_stats,
                }
                write_log(output_dir, log_payload)
                print_metric_line(log_payload)
                if wandb_run is not None:
                    wandb_run.log(log_payload, step=global_step)
            if rank == 0 and val_loader is not None and val_every_steps > 0 and global_step % val_every_steps == 0:
                val_stats = validate(
                    model=model.module if isinstance(model, DistributedDataParallel) else model,
                    resampler=resampler,
                    loader=val_loader,
                    device=device,
                    val_batches=val_batches,
                    use_fp16=use_fp16,
                    mse_weight=mse_weight,
                    cosine_weight=cosine_weight,
                    delta_weight=delta_weight,
                    show_progress=True,
                    desc=f"val epoch {epoch + 1} step {global_step}",
                )
                val_payload = {
                    "split": "val",
                    "epoch": epoch,
                    "step_in_epoch": step_in_epoch,
                    "global_step": global_step,
                    "elapsed_seconds": time.monotonic() - epoch_started_at,
                    **val_stats,
                }
                write_log(output_dir, val_payload)
                print_metric_line(val_payload)
                if wandb_run is not None:
                    wandb_run.log(val_payload, step=global_step)

        if rank == 0 and ((epoch + 1) % save_every_epochs == 0 or epoch + 1 == int(training_config["epochs"])):
            save_checkpoint(
                output_dir=output_dir,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                config=config,
                epoch=epoch,
                global_step=global_step,
                parameter_count=parameter_count,
                name=f"checkpoint_epoch_{epoch + 1:04d}",
            )

    if wandb_run is not None:
        wandb_run.finish()
    cleanup_distributed(is_distributed)


if __name__ == "__main__":
    main()
