from __future__ import annotations

import argparse
import dataclasses
from collections import defaultdict
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
            "Compute train-split normalization stats for the Origami pi0.5 action-chunk config. "
            "This writes state, delta action-chunk, and tactile_prompt q01/q99 stats."
        )
    )
    parser.add_argument("--config-name", type=str, default="pi05_origami_comp_action_chunk")
    parser.add_argument("--manifest-root", type=Path, default=None, help="Optional manifest root override.")
    parser.add_argument("--dataset-root", type=Path, default=None, help="Optional dataset root override.")
    parser.add_argument("--split", type=str, default="train", choices=("train", "all"))
    parser.add_argument("--max-rows", type=int, default=None, help="Optional cap on manifest rows.")
    parser.add_argument("--chunk-size", type=int, default=8192, help="Rows to process per episode chunk.")
    return parser.parse_args()


def _resolve_settings(args: argparse.Namespace) -> tuple[_config.TrainConfig, _origami_vla_dataset.OrigamiVlaSettings]:
    config = _config.get_config(args.config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.origami_vla is None:
        raise RuntimeError(f"Config {args.config_name!r} did not produce Origami dataset settings.")
    settings = data_config.origami_vla
    if settings.action_source != "action_chunk":
        raise ValueError(
            f"Config {args.config_name!r} uses action_source={settings.action_source!r}; expected 'action_chunk'."
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
    rows_df = rows_df.drop_duplicates(subset=["episode_uid", "frame_position"]).copy()
    unique_row_count = int(len(rows_df))

    grouped_rows: dict[str, list[int]] = defaultdict(list)
    for row in rows_df.to_dict(orient="records"):
        grouped_rows[str(row["episode_uid"])].append(int(row["frame_position"]))

    state_stats = _normalize.RunningStats()
    action_stats = _normalize.RunningStats()
    tactile_prompt_stats = _normalize.RunningStats()

    dataset_root = Path(settings.dataset_root)
    action_horizon = int(settings.action_horizon)
    action_dim = int(settings.action_dim)
    chunk_size = max(1, int(args.chunk_size))

    print("Origami action-chunk normalization stats")
    print(f"  config_name   : {args.config_name}")
    print(f"  split         : {args.split}")
    print(f"  manifest rows : {manifest_row_count}")
    print(f"  unique frames : {unique_row_count}")
    print(f"  episodes      : {len(grouped_rows)}")
    print(f"  action_horizon: {action_horizon}")
    print(f"  action_dim    : {action_dim}")
    print("  sample_weight : ignored for norm stats")
    print(f"  output_dir    : {config.assets_dirs / str(config.data.assets.asset_id or config.data.repo_id)}")

    progress = tqdm(total=unique_row_count, desc="Norm stats", unit="frame", dynamic_ncols=True)
    for episode_index, episode_uid in enumerate(sorted(grouped_rows), start=1):
        episode_root = dataset_root / "episodes" / episode_uid
        arrays_root = episode_root / "arrays"
        state_array = np.load(arrays_root / "state_65d.npy", mmap_mode="r")
        action_array = np.load(arrays_root / settings.action_filename, mmap_mode="r")
        tactile_array = np.load(arrays_root / settings.tactile_filename, mmap_mode="r")

        frame_positions = np.asarray(grouped_rows[episode_uid], dtype=np.int64)
        for chunk_start in range(0, frame_positions.shape[0], chunk_size):
            positions = frame_positions[chunk_start : chunk_start + chunk_size]
            state_chunk = np.asarray(state_array[positions], dtype=np.float32)
            tactile_chunk = np.asarray(tactile_array[positions], dtype=np.float32)
            if state_chunk.ndim != 2 or state_chunk.shape[-1] != settings.state_dim:
                raise ValueError(f"Expected state shape [N, {settings.state_dim}], got {state_chunk.shape}")
            if tactile_chunk.ndim != 2 or tactile_chunk.shape[-1] != settings.tactile_dim:
                raise ValueError(f"Expected tactile shape [N, {settings.tactile_dim}], got {tactile_chunk.shape}")

            action_chunk = np.zeros((positions.shape[0], action_horizon, action_dim), dtype=np.float32)
            action_mask = np.zeros_like(action_chunk, dtype=bool)
            offsets = np.arange(action_horizon, dtype=np.int64) * int(settings.action_chunk_stride)
            action_positions = positions[:, None] + offsets[None, :]
            valid = (action_positions >= 0) & (action_positions < int(action_array.shape[0]))
            for row_index in range(positions.shape[0]):
                valid_positions = action_positions[row_index, valid[row_index]]
                if valid_positions.size == 0:
                    continue
                valid_actions = np.asarray(action_array[valid_positions], dtype=np.float32)
                if valid_actions.ndim != 2 or valid_actions.shape[-1] != action_dim:
                    raise ValueError(f"Expected action shape [N, {action_dim}], got {valid_actions.shape}")
                action_chunk[row_index, valid[row_index], :] = valid_actions
                action_mask[row_index, valid[row_index], :] = True

            # Match the training data transform: targets are action[t + h] - state[t].
            action_chunk -= state_chunk[:, None, :action_dim]

            state_stats.update(state_chunk)
            tactile_prompt_stats.update(tactile_chunk)
            action_stats.update(action_chunk, mask=action_mask)

            progress.update(int(positions.shape[0]))
            progress.set_postfix(episode=f"{episode_index}/{len(grouped_rows)}", uid=episode_uid[-16:])

    progress.close()

    norm_stats = {
        "state": state_stats.get_statistics(),
        "actions": action_stats.get_statistics(),
        "tactile_prompt": tactile_prompt_stats.get_statistics(),
    }
    asset_id = config.data.assets.asset_id or config.data.repo_id
    if asset_id is None:
        raise RuntimeError("Could not resolve asset_id for writing norm stats.")
    output_dir = config.assets_dirs / str(asset_id)
    _normalize.save(output_dir, norm_stats)
    print(f"Writing stats to: {output_dir}")
    print("Saved keys: state, actions, tactile_prompt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
