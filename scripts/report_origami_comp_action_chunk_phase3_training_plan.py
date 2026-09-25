"""Report Phase-3 logical coverage and steps for a chosen global batch size."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import openpi.training.origami_comp_action_chunk_phase3_shards as _phase3


def main() -> int:
    parser = argparse.ArgumentParser(description="Report a completed Phase-3 logical training pass.")
    parser.add_argument("--config-name", default="pi05_origami_comp_action_chunk_phase3")
    parser.add_argument("--shard-root", type=Path, default=None)
    parser.add_argument("--global-batch-size", type=int, default=None)
    args = parser.parse_args()
    config, data = _phase3.phase3_config(args.config_name)
    runtime = data.create(config.assets_dirs, config.model)
    if runtime.origami_vla is None:
        raise RuntimeError("Missing Origami runtime settings.")
    root = args.shard_root or Path(runtime.origami_vla.shard_root or "")
    batch = int(args.global_batch_size or config.batch_size)
    if batch <= 0:
        raise ValueError("--global-batch-size must be positive.")
    physical = logical = 0
    counts = {str(stride): 0 for stride in data.shard_build.mixed_speed.ordered_strides}
    for spec in _phase3.load_phase3_specs(root, split="train", require_complete=True):
        physical += spec.num_rows
        metadata = json.loads((spec.shard_dir / _phase3.METADATA_FILENAME).read_text(encoding="utf-8"))
        plan = np.load(spec.shard_dir / str(metadata["virtual_plan"]["filename"]), mmap_mode="r")
        logical += len(plan)
        for stride in counts:
            counts[stride] += int(np.count_nonzero(plan[:, 1] == int(stride)))
    print("Phase-3 training plan")
    print(f"  physical train rows       : {physical:,}")
    print(f"  logical train rows        : {logical:,}")
    print(f"  logical rows by speed     : {counts}")
    print(f"  global batch size         : {batch:,}")
    print(f"  full batches / pass       : {logical // batch:,}")
    print(f"  final dropped remainder   : {logical % batch:,}")
    print(f"  configured training steps : {config.num_train_steps:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
