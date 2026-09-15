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
    parser = argparse.ArgumentParser(
        description=(
            "Compute train-split normalization stats for the Origami pi0.5 B-spline point config. "
            "This writes state, tactile_prompt, physical B-spline point, and width-logit q01/q99 stats."
        )
    )
    parser.add_argument("--config-name", type=str, default="pi05_origami_comp_action_bspline_points")
    parser.add_argument("--manifest-root", type=Path, default=None, help="Optional manifest root override.")
    parser.add_argument("--dataset-root", type=Path, default=None, help="Optional dataset root override.")
    parser.add_argument("--split", type=str, default="train", choices=("train", "all"))
    parser.add_argument("--max-rows", type=int, default=None, help="Optional cap on manifest rows.")
    parser.add_argument("--chunk-size", type=int, default=8192, help="Rows to process per episode chunk.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional output directory override.")
    return parser.parse_args()


def _resolve_settings(args: argparse.Namespace) -> tuple[_config.TrainConfig, _origami_vla_dataset.OrigamiVlaSettings]:
    config = _config.get_config(args.config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.origami_vla is None:
        raise RuntimeError(f"Config {args.config_name!r} did not produce Origami dataset settings.")
    settings = data_config.origami_vla
    if settings.action_source != "bspline_points":
        raise ValueError(
            f"Config {args.config_name!r} uses action_source={settings.action_source!r}; "
            "expected 'bspline_points'."
        )
    if args.manifest_root is not None:
        settings = dataclasses.replace(settings, manifest_root=str(args.manifest_root))
    if args.dataset_root is not None:
        settings = dataclasses.replace(settings, dataset_root=str(args.dataset_root))
    if args.max_rows is not None:
        settings = dataclasses.replace(settings, max_rows=int(args.max_rows))
    return config, settings


def main() -> int:
    args = parse_args()
    config, settings = _resolve_settings(args)
    rows_df = _origami_vla_dataset.load_manifest_rows(settings, args.split)
    if rows_df.empty:
        raise RuntimeError(f"No rows found for split {args.split!r} under {settings.manifest_root}.")

    manifest_row_count = int(len(rows_df))
    rows_df = rows_df.drop_duplicates(
        subset=["episode_uid", "frame_position", "local_target_npz_sample_index"]
    ).copy()
    unique_row_count = int(len(rows_df))

    grouped_rows: dict[str, list[dict[str, int]]] = defaultdict(list)
    for row in rows_df.to_dict(orient="records"):
        grouped_rows[str(row["episode_uid"])].append(
            {
                "frame_position": int(row["frame_position"]),
                "target_sample_index": int(row["local_target_npz_sample_index"]),
            }
        )

    state_stats = _normalize.RunningStats()
    tactile_prompt_stats = _normalize.RunningStats()
    point_stats = _normalize.RunningStats()
    width_logit_stats = _normalize.RunningStats()

    dataset_root = Path(settings.dataset_root)
    point_count = int(settings.max_control_points)
    width_logit_count = int(settings.max_span_count)
    action_dim = int(settings.action_dim)
    chunk_size = max(1, int(args.chunk_size))

    asset_id = config.data.assets.asset_id or config.data.repo_id
    if asset_id is None and args.output_dir is None:
        raise RuntimeError("Could not resolve asset_id for writing norm stats.")
    output_dir = args.output_dir or (config.assets_dirs / str(asset_id))

    print("Origami B-spline point normalization stats")
    print(f"  config_name       : {args.config_name}")
    print(f"  split             : {args.split}")
    print(f"  manifest rows     : {manifest_row_count:,}")
    print(f"  unique targets    : {unique_row_count:,}")
    print(f"  episodes          : {len(grouped_rows):,}")
    print(f"  action_horizon    : {settings.action_horizon}")
    print(f"  point target shape: ({point_count}, {action_dim})")
    print(f"  width logits      : {width_logit_count}")
    print("  sample_weight     : ignored for norm stats")
    print(f"  output_dir        : {output_dir}")

    progress = tqdm(total=unique_row_count, desc="B-spline point norm stats", unit="frame", dynamic_ncols=True)
    for episode_index, episode_uid in enumerate(sorted(grouped_rows), start=1):
        episode_root = dataset_root / "episodes" / episode_uid
        arrays_root = episode_root / "arrays"
        state_array = np.load(arrays_root / "state_65d.npy", mmap_mode="r")
        tactile_array = np.load(arrays_root / settings.tactile_filename, mmap_mode="r")
        target_archive = np.load(arrays_root / settings.local_target_npz_name, allow_pickle=False)
        try:
            rows = grouped_rows[episode_uid]
            for chunk_start in range(0, len(rows), chunk_size):
                chunk = rows[chunk_start : chunk_start + chunk_size]
                frame_positions = np.asarray([row["frame_position"] for row in chunk], dtype=np.int64)
                sample_indices = np.asarray([row["target_sample_index"] for row in chunk], dtype=np.int64)
                state_chunk = np.asarray(state_array[frame_positions], dtype=np.float32)
                tactile_chunk = np.asarray(tactile_array[frame_positions], dtype=np.float32)
                points = np.asarray(target_archive["points"][sample_indices], dtype=np.float32)
                width_logits = np.asarray(target_archive["width_logits"][sample_indices], dtype=np.float32)

                if state_chunk.ndim != 2 or state_chunk.shape[-1] != settings.state_dim:
                    raise ValueError(f"Expected state shape [N, {settings.state_dim}], got {state_chunk.shape}")
                if tactile_chunk.ndim != 2 or tactile_chunk.shape[-1] != settings.tactile_dim:
                    raise ValueError(f"Expected tactile shape [N, {settings.tactile_dim}], got {tactile_chunk.shape}")
                if points.shape[-2:] != (point_count, action_dim):
                    raise ValueError(f"Expected point target shape [N, {point_count}, {action_dim}], got {points.shape}")
                if width_logits.shape[-1] != width_logit_count:
                    raise ValueError(f"Expected width logits [N, {width_logit_count}], got {width_logits.shape}")
                if not np.all(np.isfinite(points)):
                    raise ValueError(f"Non-finite B-spline point targets in {episode_uid}.")
                if not np.all(np.isfinite(width_logits)):
                    raise ValueError(f"Non-finite B-spline width logits in {episode_uid}.")

                state_stats.update(state_chunk)
                tactile_prompt_stats.update(tactile_chunk)
                point_stats.update(points)
                width_logit_stats.update(width_logits)

                progress.update(int(len(chunk)))
                progress.set_postfix(episode=f"{episode_index}/{len(grouped_rows)}", uid=episode_uid[-16:])
        finally:
            target_archive.close()

    progress.close()

    norm_stats = {
        "state": state_stats.get_statistics(),
        "tactile_prompt": tactile_prompt_stats.get_statistics(),
        "actions_bspline_points": point_stats.get_statistics(),
        "actions_bspline_width_logits": width_logit_stats.get_statistics(),
    }
    _normalize.save(output_dir, norm_stats)
    print(f"Writing stats to: {output_dir}")
    print("Saved keys: state, tactile_prompt, actions_bspline_points, actions_bspline_width_logits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
