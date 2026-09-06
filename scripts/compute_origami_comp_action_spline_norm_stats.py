from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
import sys

import numpy as np
from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.origami_comp_action_spline_shards as _spline_shards


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute train-split normalization stats for the Origami pi0.5 comp action-spline config. "
            "The shard target stores physical span widths; this script converts them to centered log-width "
            "logits before accumulating span stats."
        )
    )
    parser.add_argument("--config-name", type=str, default="pi05_origami_comp_action_spline")
    parser.add_argument("--shard-root", type=Path, default=None, help="Optional shard root override.")
    parser.add_argument("--dataset-root", type=Path, default=None, help="Optional dataset root override.")
    parser.add_argument("--split", type=str, default="train", choices=("train", "all"))
    parser.add_argument("--max-rows", type=int, default=None, help="Optional cap on rows used for stats.")
    parser.add_argument("--chunk-size", type=int, default=8192, help="Rows to process per shard chunk.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional output directory override.")
    return parser.parse_args()


def _resolve_settings(args: argparse.Namespace):
    config = _config.get_config(args.config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.origami_vla is None:
        raise RuntimeError(f"Config {args.config_name!r} did not produce Origami dataset settings.")
    settings = data_config.origami_vla
    if settings.action_source != "spline":
        raise ValueError(
            f"Config {args.config_name!r} uses action_source={settings.action_source!r}; expected 'spline'."
        )
    if settings.spline_span_representation != "logits":
        raise ValueError(
            "This comp action-spline norm-stats script expects "
            "model.origami_vla.spline_span_representation='logits'."
        )
    if args.shard_root is not None:
        settings = dataclasses.replace(settings, shard_root=str(args.shard_root))
    if args.dataset_root is not None:
        settings = dataclasses.replace(settings, dataset_root=str(args.dataset_root))
    if args.max_rows is not None:
        settings = dataclasses.replace(settings, max_rows=int(args.max_rows))
    return config, settings


def _span_widths_to_centered_logits(widths: np.ndarray, mask: np.ndarray, *, eps: float = 1.0e-6) -> np.ndarray:
    widths = np.asarray(widths, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    safe = np.where(mask, np.clip(widths, eps, None), 1.0)
    logits = np.log(safe)
    count = np.clip(np.sum(mask, axis=-1, keepdims=True), 1, None)
    mean = np.sum(np.where(mask, logits, 0.0), axis=-1, keepdims=True) / count
    return np.where(mask, logits - mean, 0.0).astype(np.float32)


def main() -> int:
    args = parse_args()
    config, settings = _resolve_settings(args)
    shard_root = _spline_shards.resolve_shard_root(settings)
    specs = _spline_shards.load_shard_specs(
        shard_root,
        split=args.split,
        manifest_filename=settings.shard_manifest_name,
        complete_marker_name=settings.shard_complete_marker_name,
        require_complete=settings.shard_require_complete,
    )
    if not specs:
        raise RuntimeError(f"No completed shards found for split {args.split!r} under {shard_root}.")

    row_limit = int(args.max_rows) if args.max_rows is not None else None
    total_rows = sum(int(spec.num_rows) for spec in specs)
    rows_for_stats = min(total_rows, row_limit) if row_limit is not None else total_rows
    if rows_for_stats <= 0:
        raise RuntimeError("No rows available for norm stats.")

    state_stats = _normalize.RunningStats()
    tactile_prompt_stats = _normalize.RunningStats()
    control_point_stats = _normalize.RunningStats()
    span_logit_stats = _normalize.RunningStats()

    chunk_size = max(1, int(args.chunk_size))
    max_control_points = int(settings.max_control_points)
    max_span_count = int(settings.max_span_count)

    asset_id = config.data.assets.asset_id or config.data.repo_id
    if asset_id is None and args.output_dir is None:
        raise RuntimeError("Could not resolve asset_id for writing norm stats.")
    output_dir = args.output_dir or (config.assets_dirs / str(asset_id))

    print("Origami comp action-spline normalization stats")
    print(f"  config_name          : {args.config_name}")
    print(f"  split                : {args.split}")
    print(f"  shard_root           : {shard_root}")
    print(f"  shards               : {len(specs)}")
    print(f"  rows available       : {total_rows:,}")
    print(f"  rows used            : {rows_for_stats:,}")
    print(f"  action target layout : actions=({settings.action_horizon}, {settings.action_dim})")
    print(f"  control points       : {max_control_points}")
    print(f"  span representation  : logits from physical widths")
    print("  sample_weight        : ignored for norm stats")
    print(f"  output_dir           : {output_dir}")

    rows_remaining = rows_for_stats
    progress = tqdm(total=rows_for_stats, desc="Spline norm stats", unit="row", dynamic_ncols=True)
    for spec in specs:
        if rows_remaining <= 0:
            break
        arrays_dir = spec.shard_dir / "arrays"
        state = np.load(arrays_dir / "state.npy", mmap_mode="r")
        tactile = np.load(arrays_dir / "tactile.npy", mmap_mode="r")
        actions = np.load(arrays_dir / "actions.npy", mmap_mode="r")
        action_mask = np.load(arrays_dir / "action_mask.npy", mmap_mode="r")
        shard_rows = min(int(spec.num_rows), rows_remaining)
        for start in range(0, shard_rows, chunk_size):
            end = min(start + chunk_size, shard_rows)
            state_chunk = np.asarray(state[start:end], dtype=np.float32)
            tactile_chunk = np.asarray(tactile[start:end], dtype=np.float32)
            action_chunk = np.asarray(actions[start:end], dtype=np.float32)
            mask_chunk = np.asarray(action_mask[start:end], dtype=bool)

            if state_chunk.shape[-1] != settings.state_dim:
                raise ValueError(f"Expected state dim {settings.state_dim}, got {state_chunk.shape}")
            if tactile_chunk.shape[-1] != settings.tactile_dim:
                raise ValueError(f"Expected tactile dim {settings.tactile_dim}, got {tactile_chunk.shape}")
            if action_chunk.shape[-2:] != (settings.action_horizon, settings.action_dim):
                raise ValueError(
                    f"Expected actions [N, {settings.action_horizon}, {settings.action_dim}], "
                    f"got {action_chunk.shape}"
                )

            control_points = action_chunk[:, :max_control_points, :]
            control_mask = mask_chunk[:, :max_control_points, :]
            widths = action_chunk[:, max_control_points, :max_span_count]
            span_mask = mask_chunk[:, max_control_points, :max_span_count]
            valid_widths = widths[span_mask]
            if valid_widths.size == 0:
                raise ValueError(f"No valid span widths in {spec.shard_name} rows {start}:{end}.")
            if not np.all(np.isfinite(valid_widths)):
                raise ValueError(f"Non-finite span widths in {spec.shard_name} rows {start}:{end}.")
            if np.any(valid_widths <= 0.0):
                raise ValueError(f"Non-positive span widths in {spec.shard_name} rows {start}:{end}.")
            width_sums = np.sum(np.where(span_mask, widths, 0.0), axis=-1)
            if not np.allclose(width_sums, 1.0, atol=1.0e-4, rtol=1.0e-4):
                bad_index = int(np.argmax(np.abs(width_sums - 1.0)))
                raise ValueError(
                    f"Span widths must sum to 1.0 before logit conversion; {spec.shard_name} "
                    f"row {start + bad_index} sums to {float(width_sums[bad_index]):.8f}."
                )
            span_logits = _span_widths_to_centered_logits(widths, span_mask)

            state_stats.update(state_chunk)
            tactile_prompt_stats.update(tactile_chunk)
            control_point_stats.update(control_points, mask=control_mask)
            span_logit_stats.update(span_logits, mask=span_mask)

            count = int(end - start)
            rows_remaining -= count
            progress.update(count)
            progress.set_postfix(shard=spec.shard_name)
            if rows_remaining <= 0:
                break

    progress.close()

    norm_stats = {
        "state": state_stats.get_statistics(),
        "tactile_prompt": tactile_prompt_stats.get_statistics(),
        "actions_control_points": control_point_stats.get_statistics(),
        "actions_span_logits": span_logit_stats.get_statistics(),
    }
    _normalize.save(output_dir, norm_stats)
    print(f"Writing stats to: {output_dir}")
    print("Saved keys: state, tactile_prompt, actions_control_points, actions_span_logits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
