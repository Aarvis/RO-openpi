"""Compute Phase-3 normalization assets directly from completed shard files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import openpi.shared.normalize as _normalize
import openpi.training.origami_comp_action_chunk_phase3_shards as _phase3


PROVENANCE_FILENAME = "phase3_norm_stats_metadata.json"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Phase-3 state/tactile/action-delta normalization stats.")
    parser.add_argument("--config-name", default="pi05_origami_comp_action_chunk_phase3")
    parser.add_argument("--shard-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--physical-chunk-rows", type=int, default=8192)
    parser.add_argument("--logical-chunk-rows", type=int, default=8192)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _array(directory: Path, name: str) -> np.ndarray:
    return np.load(directory / "arrays" / f"{name}.npy", mmap_mode="r")


def main() -> int:
    args = _args()
    config, data = _phase3.phase3_config(args.config_name)
    runtime = data.create(config.assets_dirs, config.model)
    if runtime.origami_vla is None or not runtime.origami_vla.phase3_mixed_speed_enabled:
        raise RuntimeError(f"{args.config_name!r} is not a runnable Phase-3 mixed-speed configuration.")
    settings = runtime.origami_vla
    root = args.shard_root or Path(settings.shard_root or "")
    asset_id = data.assets.asset_id or data.repo_id
    if asset_id is None and args.output_dir is None:
        raise RuntimeError("Could not resolve a Phase-3 normalization output directory.")
    output_dir = args.output_dir or (config.assets_dirs / str(asset_id))
    norm_path = output_dir / "norm_stats.json"
    if norm_path.exists() and not args.overwrite:
        raise FileExistsError(f"{norm_path} already exists; pass --overwrite to replace it.")
    physical_chunk = max(1, int(args.physical_chunk_rows))
    logical_chunk = max(1, int(args.logical_chunk_rows))
    shards = _phase3.load_phase3_specs(root, split="train", require_complete=True)
    manifest_path = root / _phase3.SHARD_MANIFEST_FILENAME
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    state_stats = _normalize.RunningStats()
    tactile_stats = _normalize.RunningStats()
    action_stats = _normalize.RunningStats()
    physical_rows = logical_rows = 0
    by_speed: dict[str, int] = {str(stride): 0 for stride in data.shard_build.mixed_speed.ordered_strides}
    for spec in tqdm(shards, desc="Phase-3 normalization", unit="shard"):
        metadata = json.loads((spec.shard_dir / _phase3.METADATA_FILENAME).read_text(encoding="utf-8"))
        if metadata.get("format") != settings.phase3_shard_format:
            raise ValueError(f"{spec.shard_dir}: incorrect Phase-3 shard format.")
        state = _array(spec.shard_dir, "state")
        tactile = _array(spec.shard_dir, "tactile")
        if len(state) != spec.num_rows or len(tactile) != spec.num_rows:
            raise ValueError(f"{spec.shard_dir}: physical array row count mismatch.")
        for start in range(0, spec.num_rows, physical_chunk):
            end = min(spec.num_rows, start + physical_chunk)
            state_stats.update(np.asarray(state[start:end], dtype=np.float32))
            tactile_stats.update(np.asarray(tactile[start:end], dtype=np.float32))
            physical_rows += end - start
        plan_file = spec.shard_dir / str(metadata["virtual_plan"]["filename"])
        plan = np.load(plan_file, mmap_mode="r")
        if plan.dtype != np.dtype(np.uint32) or plan.ndim != 2 or plan.shape[1] != 3:
            raise ValueError(f"{spec.shard_dir}: invalid virtual plan.")
        actions = {stride: _array(spec.shard_dir, f"actions_stride_{stride}") for stride in by_speed}
        for start in range(0, len(plan), logical_chunk):
            block = np.asarray(plan[start : start + logical_chunk])
            physical_ids = block[:, 0].astype(np.int64, copy=False)
            speeds = block[:, 1]
            anchors = np.asarray(state[physical_ids], dtype=np.float32)
            for stride_text, action_array in actions.items():
                stride = int(stride_text)
                chosen = np.flatnonzero(speeds == stride)
                if not len(chosen):
                    continue
                absolute = np.asarray(action_array[physical_ids[chosen]], dtype=np.float32)
                action_stats.update(absolute - anchors[chosen, None, :])
                by_speed[stride_text] += int(len(chosen))
            logical_rows += len(block)
    norm_stats = {
        "state": state_stats.get_statistics(),
        "tactile": tactile_stats.get_statistics(),
        # Tokenized tactile conditioning uses this duplicate of tactile.
        "tactile_prompt": tactile_stats.get_statistics(),
        "actions": action_stats.get_statistics(),
    }
    _normalize.save(output_dir, norm_stats)
    provenance = {
        "format": "origami_comp_action_chunk_phase3_norm_stats_v1",
        "config_name": args.config_name,
        "shard_format": settings.phase3_shard_format,
        "shard_manifest_sha256": manifest_sha256,
        "mixed_speed": data.shard_build.mixed_speed.stride_episode_coverage,
        "physical_train_rows": physical_rows,
        "logical_action_rows": logical_rows,
        "logical_action_rows_by_speed": by_speed,
        "state_dim": settings.state_dim,
        "tactile_dim": settings.tactile_dim,
        "action_dim": settings.action_dim,
        "action_horizon": settings.action_horizon,
        "action_semantics": "state_anchored_delta_from_selected_absolute_stride_chunk",
        "state_tactile_semantics": "physical_rows_once",
    }
    (output_dir / PROVENANCE_FILENAME).write_text(json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8")
    print(f"OK: wrote Phase-3 norm stats to {output_dir}")
    print(f"physical rows={physical_rows:,}; logical action rows={logical_rows:,}; by_speed={by_speed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
