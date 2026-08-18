from __future__ import annotations

import argparse
from collections import defaultdict
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
import openpi.training.origami_vla_dataset as _origami_vla_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Origami VLA state/action normalization stats.")
    parser.add_argument("--config-name", type=str, default="pi05_origami_checkpoint_spline_vla")
    parser.add_argument("--manifest-root", type=Path, default=None, help="Optional manifest root override.")
    parser.add_argument("--dataset-root", type=Path, default=None, help="Optional dataset root override.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional cap on train rows.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = _config.get_config(args.config_name)
    if not isinstance(config.data, _config.OrigamiVlaDataConfig):
        raise TypeError(f"Config {args.config_name!r} is not an Origami VLA config.")
    if not isinstance(config.model, _config.pi0_config.Pi0Config) or not config.model.origami_vla.enabled:
        raise TypeError(f"Config {args.config_name!r} does not enable model.origami_vla.")

    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.origami_vla is None:
        raise RuntimeError("Origami settings were not populated in the data config.")
    settings = data_config.origami_vla
    if args.manifest_root is not None:
        settings = dataclasses.replace(settings, manifest_root=str(args.manifest_root))
    if args.dataset_root is not None:
        settings = dataclasses.replace(settings, dataset_root=str(args.dataset_root))
    if args.max_rows is not None:
        settings = dataclasses.replace(settings, max_rows=int(args.max_rows))

    train_rows = _origami_vla_dataset.load_manifest_rows(settings, "train")
    state_stats = _normalize.RunningStats()
    action_stats = _normalize.RunningStats()

    dataset_root = Path(settings.dataset_root)
    grouped_rows = defaultdict(list)
    for row in train_rows.to_dict(orient="records"):
        grouped_rows[str(row["episode_uid"])].append(row)

    for episode_uid in tqdm(sorted(grouped_rows), desc="Norm stats", unit="episode", dynamic_ncols=True):
        episode_root = dataset_root / "episodes" / episode_uid
        state = np.load(episode_root / "arrays" / "state_65d.npy", mmap_mode="r")
        archive = np.load(episode_root / "arrays" / settings.local_target_npz_name, allow_pickle=False)
        for row in grouped_rows[episode_uid]:
            frame_position = int(row["frame_position"])
            sample_index = int(row["local_target_npz_sample_index"])
            control_points, local_knots = _origami_vla_dataset.extract_target_sample(archive, sample_index)
            span_widths = _origami_vla_dataset.local_knots_to_span_widths(local_knots, settings.degree)
            actions, action_mask = _origami_vla_dataset.pack_spline_actions(
                control_points,
                span_widths,
                max_control_points=settings.max_control_points,
                max_span_count=settings.max_span_count,
                action_dim=settings.action_dim,
            )
            state_stats.update(np.asarray(state[frame_position], dtype=np.float32)[None, :])
            action_stats.update(actions[None, ...], mask=action_mask[None, ...])

    norm_stats = {
        "state": state_stats.get_statistics(),
        "actions": action_stats.get_statistics(),
    }
    asset_id = data_config.asset_id or data_config.repo_id
    if asset_id is None:
        raise RuntimeError("Could not resolve asset_id for writing norm stats.")
    output_dir = config.assets_dirs / str(asset_id)
    _normalize.save(output_dir, norm_stats)
    print("Origami VLA normalization stats")
    print(f"  config_name : {args.config_name}")
    print(f"  rows        : {len(train_rows)}")
    print(f"  output_dir  : {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
