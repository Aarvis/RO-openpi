"""Verify the Phase-3 common-horizon, planner-free source manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import openpi.training.origami_comp_action_chunk_phase3_shards as _phase3


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Phase-3 source manifests.")
    parser.add_argument("--config-name", default="pi05_origami_comp_action_chunk_phase3_build")
    parser.add_argument("--manifest-root", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, default=None)
    args = parser.parse_args()
    config, data = _phase3.phase3_config(args.config_name)
    build = data.shard_build
    root = args.manifest_root or Path(data.manifest_root)
    dataset_root = args.dataset_root or Path(data.dataset_root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    common_stride = max(build.mixed_speed.ordered_strides)
    required_offset = (build.mixed_speed.action_horizon - 1) * common_stride
    train = pd.read_parquet(root / data.manifest_build.train_index_name)
    val = pd.read_parquet(root / data.manifest_build.val_index_name)
    def _episode_uids(frame: pd.DataFrame) -> set[str]:
        return set() if "episode_uid" not in frame else set(frame["episode_uid"].astype(str))

    train_uids = _episode_uids(train)
    val_uids = _episode_uids(val)
    failures: list[str] = []
    if train_uids & val_uids:
        failures.append(f"train/val episode overlap: {sorted(train_uids & val_uids)[:5]}")
    for split, frame in (("train", train), ("val", val)):
        # A Phase-3 training-only configuration is allowed to have zero
        # validation episodes. Pandas may serialize that empty parquet with no
        # planner column, which is valid because there are no rows to violate
        # the planner-free contract.
        if frame.empty:
            continue
        required_columns = {"episode_uid", "frame_position", "planner_enabled"}
        missing_columns = sorted(required_columns - set(frame.columns))
        if missing_columns:
            failures.append(f"{split}: missing required columns {missing_columns}")
            continue
        if frame.duplicated(subset=["episode_uid", "frame_position"]).any():
            failures.append(f"{split}: duplicate episode/frame rows")
        if frame["planner_enabled"].astype(bool).any():
            failures.append(f"{split}: planner_enabled must be false for every row")
        if "sample_weight" in frame and not np.allclose(frame["sample_weight"].to_numpy(dtype=np.float32), 1.0):
            failures.append(f"{split}: Phase-3 sample_weight must be exactly one")
        for uid, group in tqdm(frame.groupby("episode_uid", sort=False), desc=f"Verify {split} horizon", unit="episode"):
            actions = np.load(dataset_root / "episodes" / str(uid) / "arrays" / data.action_filename, mmap_mode="r")
            if (group["frame_position"].to_numpy(dtype=np.int64) + required_offset >= len(actions)).any():
                failures.append(f"{split}:{uid}: contains a start without common stride-{common_stride} horizon")
                break
    if int(manifest.get("common_horizon_max_stride", common_stride)) != common_stride:
        failures.append("manifest common_horizon_max_stride does not match Phase-3 config")
    if failures:
        print("FAILED")
        print("\n".join(f"- {failure}" for failure in failures))
        return 1
    print("OK: Phase-3 manifest has disjoint splits, unit weights, disabled planner rows, and complete horizons.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
