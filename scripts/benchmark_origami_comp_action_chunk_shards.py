from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
import sys
import time

import numpy as np
import torch
from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import openpi.training.config as _config
import openpi.training.origami_comp_action_chunk_shards as _shards


def _stack_tree(items):
    first = items[0]
    if isinstance(first, dict):
        return {key: _stack_tree([item[key] for item in items]) for key in first}
    if isinstance(first, tuple):
        return tuple(_stack_tree([item[index] for item in items]) for index in range(len(first)))
    if isinstance(first, list):
        return [_stack_tree([item[index] for item in items]) for index in range(len(first))]
    return np.stack([np.asarray(item) for item in items], axis=0)


def _collate_fn(items):
    return _stack_tree(items)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Origami comp action-chunk shard read throughput.")
    parser.add_argument("--config-name", type=str, default="pi05_origami_comp_action_chunk")
    parser.add_argument("--shard-root", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--manifest-root", type=Path, default=None)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-batches", type=int, default=200)
    parser.add_argument("--with-transforms", action="store_true")
    parser.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Randomize dataset indices before batching. Disabled by default for mmap shard throughput checks.",
    )
    parser.add_argument(
        "--debug-first-samples",
        type=int,
        default=0,
        help="Read this many sequential samples directly before constructing the DataLoader.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = _config.get_config(args.config_name)
    if not isinstance(config.data, _config.OrigamiCompActionChunkDataConfig):
        raise TypeError(
            f"Config {args.config_name!r} uses {type(config.data).__name__}; "
            "expected OrigamiCompActionChunkDataConfig."
        )
    settings = _shards.settings_from_data_factory(config.data, config.model)
    shard_root = args.shard_root or (Path(config.data.shard_root) if config.data.shard_root else None)
    if shard_root is None:
        raise ValueError("Set data.shard_root or pass --shard-root.")
    if args.dataset_root is not None:
        settings = dataclasses.replace(settings, dataset_root=str(args.dataset_root))
    if args.manifest_root is not None:
        settings = dataclasses.replace(settings, manifest_root=str(args.manifest_root))
    settings = dataclasses.replace(settings, dataset_backend="shard", shard_root=str(shard_root))
    dataset = _shards.OrigamiCompActionChunkShardDataset(settings, split=args.split)
    if args.with_transforms:
        import openpi.training.data_loader as _data_loader

        data_config = config.data.create(config.assets_dirs, config.model)
        data_config = dataclasses.replace(data_config, origami_vla=settings, dataset_split=args.split)
        dataset = _data_loader.transform_dataset(dataset, data_config)
    for index in tqdm(
        range(min(int(args.debug_first_samples), len(dataset))),
        desc="Debug direct samples",
        unit="sample",
        dynamic_ncols=True,
        disable=int(args.debug_first_samples) <= 0,
    ):
        _ = dataset[index]
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=bool(args.shuffle),
        num_workers=int(args.num_workers),
        multiprocessing_context="spawn" if int(args.num_workers) > 0 else None,
        persistent_workers=int(args.num_workers) > 0,
        collate_fn=_collate_fn,
        drop_last=True,
    )
    print("Origami comp action-chunk shard benchmark")
    print(f"  config_name    : {args.config_name}")
    print(f"  shard_root     : {shard_root}")
    print(f"  split          : {args.split}")
    print(f"  rows           : {len(dataset):,}")
    print(f"  batch_size     : {args.batch_size}")
    print(f"  num_workers    : {args.num_workers}")
    print(f"  shuffle        : {args.shuffle}")
    print(f"  with_transforms: {args.with_transforms}")

    start = time.perf_counter()
    sample_count = 0
    try:
        iterator = iter(loader)
        for _ in tqdm(range(int(args.num_batches)), desc="Benchmark batches", unit="batch", dynamic_ncols=True):
            batch = next(iterator)
            sample_count += int(next(iter(batch["image"].values())).shape[0])
    finally:
        close = getattr(dataset, "close", None)
        if callable(close):
            close()
    elapsed = max(time.perf_counter() - start, 1.0e-9)
    print(f"Elapsed seconds : {elapsed:.3f}")
    print(f"Batches/second  : {int(args.num_batches) / elapsed:.3f}")
    print(f"Samples/second  : {sample_count / elapsed:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
