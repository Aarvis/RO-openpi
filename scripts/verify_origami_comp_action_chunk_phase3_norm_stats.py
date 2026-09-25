"""Verify Phase-3 shard-derived normalization assets and provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import openpi.shared.normalize as _normalize
import openpi.training.origami_comp_action_chunk_phase3_shards as _phase3


PROVENANCE_FILENAME = "phase3_norm_stats_metadata.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Phase-3 normalization assets.")
    parser.add_argument("--config-name", default="pi05_origami_comp_action_chunk_phase3")
    parser.add_argument("--shard-root", type=Path, default=None)
    parser.add_argument("--norm-dir", type=Path, default=None)
    args = parser.parse_args()
    config, data = _phase3.phase3_config(args.config_name)
    runtime = data.create(config.assets_dirs, config.model)
    if runtime.origami_vla is None or not runtime.origami_vla.phase3_mixed_speed_enabled:
        raise RuntimeError(f"{args.config_name!r} is not a Phase-3 runtime config.")
    settings = runtime.origami_vla
    root = args.shard_root or Path(settings.shard_root or "")
    asset_id = data.assets.asset_id or data.repo_id
    norm_dir = args.norm_dir or (config.assets_dirs / str(asset_id))
    stats = _normalize.load(norm_dir)
    provenance = json.loads((norm_dir / PROVENANCE_FILENAME).read_text(encoding="utf-8"))
    failures: list[str] = []
    expected_shapes = {
        "state": (settings.state_dim,),
        "tactile": (settings.tactile_dim,),
        "tactile_prompt": (settings.tactile_dim,),
        "actions": (settings.action_dim,),
    }
    for name, shape in expected_shapes.items():
        item = stats.get(name)
        if item is None:
            failures.append(f"missing norm-stat key {name}")
            continue
        if tuple(item.mean.shape) != shape or tuple(item.std.shape) != shape:
            failures.append(f"{name}: unexpected mean/std shape")
        if not np.isfinite(item.mean).all() or not np.isfinite(item.std).all() or np.any(item.std < 0.0):
            failures.append(f"{name}: invalid mean/std values")
    if "tactile" in stats and "tactile_prompt" in stats:
        if not np.array_equal(stats["tactile"].mean, stats["tactile_prompt"].mean):
            failures.append("tactile and tactile_prompt means differ")
        if not np.array_equal(stats["tactile"].std, stats["tactile_prompt"].std):
            failures.append("tactile and tactile_prompt stds differ")
    manifest_sha256 = hashlib.sha256((root / _phase3.SHARD_MANIFEST_FILENAME).read_bytes()).hexdigest()
    if provenance.get("shard_manifest_sha256") != manifest_sha256:
        failures.append("normalization provenance does not match current shard manifest")
    if provenance.get("shard_format") != settings.phase3_shard_format:
        failures.append("normalization provenance has wrong shard format")
    if provenance.get("action_semantics") != "state_anchored_delta_from_selected_absolute_stride_chunk":
        failures.append("normalization provenance has wrong action semantics")
    if provenance.get("state_tactile_semantics") != "physical_rows_once":
        failures.append("normalization provenance has wrong state/tactile semantics")
    expected_physical = expected_logical = 0
    expected_by_speed = {str(stride): 0 for stride in data.shard_build.mixed_speed.ordered_strides}
    for spec in _phase3.load_phase3_specs(root, split="train", require_complete=True):
        expected_physical += int(spec.num_rows)
        metadata = json.loads((spec.shard_dir / _phase3.METADATA_FILENAME).read_text(encoding="utf-8"))
        plan = np.load(spec.shard_dir / str(metadata["virtual_plan"]["filename"]), mmap_mode="r")
        expected_logical += len(plan)
        for stride in expected_by_speed:
            expected_by_speed[stride] += int(np.count_nonzero(plan[:, 1] == int(stride)))
    if int(provenance.get("physical_train_rows", -1)) != expected_physical:
        failures.append("normalization provenance physical-row count differs from shards")
    if int(provenance.get("logical_action_rows", -1)) != expected_logical:
        failures.append("normalization provenance logical-row count differs from virtual plans")
    if provenance.get("logical_action_rows_by_speed") != expected_by_speed:
        failures.append("normalization provenance per-speed counts differ from virtual plans")
    if failures:
        print("FAILED")
        print("\n".join(f"- {failure}" for failure in failures))
        return 1
    print(f"OK: Phase-3 norm assets and provenance are valid: {norm_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
