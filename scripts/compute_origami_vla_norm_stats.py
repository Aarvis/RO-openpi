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
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2048,
        help="Number of manifest rows to aggregate before updating running statistics.",
    )
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
    control_point_stats = _normalize.RunningStats()
    span_width_stats = _normalize.RunningStats()

    dataset_root = Path(settings.dataset_root)
    grouped_rows = defaultdict(list)
    for row in train_rows.to_dict(orient="records"):
        grouped_rows[str(row["episode_uid"])].append(row)

    total_rows = len(train_rows)
    total_episodes = len(grouped_rows)
    print("Origami VLA normalization stats")
    print(f"  config_name : {args.config_name}")
    print(f"  rows        : {total_rows}")
    print(f"  episodes    : {total_episodes}")
    print(f"  chunk_size  : {args.chunk_size}")

    row_progress = tqdm(total=total_rows, desc="Norm stats", unit="row", dynamic_ncols=True)
    for episode_idx, episode_uid in enumerate(sorted(grouped_rows), start=1):
        episode_root = dataset_root / "episodes" / episode_uid
        state = np.load(episode_root / "arrays" / "state_65d.npy", mmap_mode="r")
        archive = np.load(episode_root / "arrays" / settings.local_target_npz_name, allow_pickle=False)
        sample_offsets = np.asarray(archive["control_point_offsets"], dtype=np.int64)
        knot_offsets = np.asarray(archive["local_knot_offsets"], dtype=np.int64)
        control_points_all = np.asarray(archive["local_delta_control_points"], dtype=np.float32)
        local_knots_all = np.asarray(archive["local_knots"], dtype=np.float32)
        episode_rows = grouped_rows[episode_uid]
        for chunk_start in range(0, len(episode_rows), args.chunk_size):
            chunk_rows = episode_rows[chunk_start : chunk_start + args.chunk_size]
            frame_positions = np.asarray([int(row["frame_position"]) for row in chunk_rows], dtype=np.int64)
            state_chunk = np.asarray(state[frame_positions], dtype=np.float32)
            state_stats.update(state_chunk)

            actions_chunk = np.zeros(
                (len(chunk_rows), settings.max_control_points + 1, settings.action_dim),
                dtype=np.float32,
            )
            action_mask_chunk = np.zeros_like(actions_chunk, dtype=bool)

            for sample_offset, row in enumerate(chunk_rows):
                sample_index = int(row["local_target_npz_sample_index"])
                cp_start = int(sample_offsets[sample_index])
                cp_end = int(sample_offsets[sample_index + 1])
                knot_start = int(knot_offsets[sample_index])
                knot_end = int(knot_offsets[sample_index + 1])
                if cp_end <= cp_start:
                    raise ValueError(f"Empty control-point slice for sample_index={sample_index} in {episode_uid}")
                if knot_end <= knot_start:
                    raise ValueError(f"Empty knot slice for sample_index={sample_index} in {episode_uid}")
                control_points = control_points_all[cp_start:cp_end]
                local_knots = local_knots_all[knot_start:knot_end]
                span_widths = _origami_vla_dataset.local_knots_to_span_widths(local_knots, settings.degree)
                actions, action_mask = _origami_vla_dataset.pack_spline_actions(
                    control_points,
                    span_widths,
                    max_control_points=settings.max_control_points,
                    max_span_count=settings.max_span_count,
                    action_dim=settings.action_dim,
                )
                actions_chunk[sample_offset] = actions
                action_mask_chunk[sample_offset] = action_mask

            control_point_stats.update(
                actions_chunk[:, : settings.max_control_points, :],
                mask=action_mask_chunk[:, : settings.max_control_points, :],
            )
            span_width_stats.update(
                actions_chunk[:, settings.max_control_points, : settings.max_span_count],
                mask=action_mask_chunk[:, settings.max_control_points, : settings.max_span_count],
            )
            row_progress.update(len(chunk_rows))
            row_progress.set_postfix(
                episode=f"{episode_idx}/{total_episodes}",
                episode_uid=episode_uid[-24:],
            )

    row_progress.close()

    norm_stats = {
        "state": state_stats.get_statistics(),
        "actions_control_points": control_point_stats.get_statistics(),
        "actions_span_widths": span_width_stats.get_statistics(),
    }
    asset_id = data_config.asset_id or data_config.repo_id
    if asset_id is None:
        raise RuntimeError("Could not resolve asset_id for writing norm stats.")
    output_dir = config.assets_dirs / str(asset_id)
    _normalize.save(output_dir, norm_stats)
    print(f"  output_dir  : {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
