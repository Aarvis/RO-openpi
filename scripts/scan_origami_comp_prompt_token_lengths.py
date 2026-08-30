from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import dataclasses
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import openpi.models.pi0_config as _pi0_config
import openpi.models.tokenizer as _tokenizer
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.origami_vla_dataset as _origami_vla_dataset

_TOKENIZER_CACHE: dict[int, _tokenizer.PaligemmaTokenizer] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure pi0.5 Origami action-chunk prompt token lengths for the fixed task text, "
            "65D state, and 60D tactile prompt fields."
        )
    )
    parser.add_argument("--config-name", type=str, default="pi05_origami_comp_action_chunk")
    parser.add_argument("--manifest-root", type=Path, default=None, help="Optional manifest root override.")
    parser.add_argument("--dataset-root", type=Path, default=None, help="Optional dataset root override.")
    parser.add_argument("--assets-dir", type=Path, default=None, help="Optional norm-stats directory override.")
    parser.add_argument("--split", type=str, default="train", choices=("train", "val", "all"))
    parser.add_argument("--max-rows", type=int, default=None, help="Optional cap on manifest rows.")
    parser.add_argument("--num-workers", type=int, default=None, help="Episode-parallel workers; defaults to min(8, CPU count).")
    parser.add_argument("--top-k", type=int, default=10, help="Number of longest samples to print.")
    return parser.parse_args()


def _get_tokenizer(max_len: int) -> _tokenizer.PaligemmaTokenizer:
    tokenizer = _TOKENIZER_CACHE.get(max_len)
    if tokenizer is None:
        tokenizer = _tokenizer.PaligemmaTokenizer(max_len=max_len)
        _TOKENIZER_CACHE[max_len] = tokenizer
    return tokenizer


def _normalize_quantile(values: np.ndarray, stats: _normalize.NormStats) -> np.ndarray:
    if stats.q01 is None or stats.q99 is None:
        raise ValueError("Quantile token scan requires q01/q99 stats.")
    q01 = np.asarray(stats.q01, dtype=np.float32)[..., : values.shape[-1]]
    q99 = np.asarray(stats.q99, dtype=np.float32)[..., : values.shape[-1]]
    return (values - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


def _build_full_prompt(
    tokenizer: _tokenizer.PaligemmaTokenizer,
    *,
    prompt: str,
    state: np.ndarray,
    tactile: np.ndarray,
    clip_discrete_inputs: bool,
) -> str:
    cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
    state_str = tokenizer._serialize_state(state, np.ones_like(state, dtype=bool), clip=clip_discrete_inputs)  # noqa: SLF001
    tactile_str = tokenizer._serialize_tactile(  # noqa: SLF001
        tactile,
        np.ones_like(tactile, dtype=bool),
        clip=clip_discrete_inputs,
    )
    return f"Task: {cleaned_text}, {state_str} {tactile_str}\nAction: "


def _summarize(values: np.ndarray) -> dict[str, float | int]:
    return {
        "count": int(values.size),
        "min": int(np.min(values)),
        "max": int(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }


def _episode_worker(
    *,
    episode_uid: str,
    frame_rows: list[tuple[int, int]],
    dataset_root: str,
    prompt: str,
    max_token_len: int,
    top_k: int,
    clip_discrete_inputs: bool,
    state_stats: _normalize.NormStats,
    tactile_stats: _normalize.NormStats,
) -> dict[str, Any]:
    tokenizer = _get_tokenizer(max(max_token_len, 4096))
    episode_root = Path(dataset_root) / "episodes" / episode_uid
    arrays_root = episode_root / "arrays"
    state_array = np.load(arrays_root / "state_65d.npy", mmap_mode="r")
    tactile_array = np.load(arrays_root / "tactile_60d.npy", mmap_mode="r")

    token_lengths: list[int] = []
    longest: list[dict[str, Any]] = []
    overflow_count = 0

    for frame_position, frame_index in frame_rows:
        state = np.asarray(state_array[frame_position], dtype=np.float32)
        tactile = np.asarray(tactile_array[frame_position], dtype=np.float32)
        state_norm = _normalize_quantile(state, state_stats)
        tactile_norm = _normalize_quantile(tactile, tactile_stats)
        full_prompt = _build_full_prompt(
            tokenizer,
            prompt=prompt,
            state=state_norm,
            tactile=tactile_norm,
            clip_discrete_inputs=clip_discrete_inputs,
        )
        token_len = int(len(tokenizer._tokenizer.encode(full_prompt, add_bos=True)))  # noqa: SLF001
        token_lengths.append(token_len)
        if token_len > max_token_len:
            overflow_count += 1

        sample_meta = {
            "token_len": token_len,
            "episode_uid": episode_uid,
            "frame_position": int(frame_position),
            "frame_index": int(frame_index),
        }
        if len(longest) < top_k:
            longest.append(sample_meta)
            longest.sort(key=lambda item: int(item["token_len"]), reverse=True)
        elif token_len > int(longest[-1]["token_len"]):
            longest[-1] = sample_meta
            longest.sort(key=lambda item: int(item["token_len"]), reverse=True)

    return {
        "episode_uid": episode_uid,
        "row_count": len(frame_rows),
        "overflow_count": overflow_count,
        "token_lengths": token_lengths,
        "longest": longest,
    }


def main() -> int:
    args = parse_args()
    config = _config.get_config(args.config_name)
    if not isinstance(config.model, _pi0_config.Pi0Config):
        raise TypeError(f"Config {args.config_name!r} is not a Pi0/Pi0.5 config.")
    if not config.model.pi05 or not config.model.discrete_state_input:
        raise ValueError(f"Config {args.config_name!r} must be a pi0.5 config with discrete_state_input=True.")
    if not config.model.origami_vla.tactile_prompt_input:
        raise ValueError(f"Config {args.config_name!r} does not enable tactile_prompt_input.")

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

    asset_id = config.data.assets.asset_id or config.data.repo_id
    stats_dir = args.assets_dir or (config.assets_dirs / str(asset_id))
    norm_stats = _normalize.load(stats_dir)
    missing_stats = [key for key in ("state", "tactile_prompt") if key not in norm_stats]
    if missing_stats:
        raise KeyError(f"Missing required norm-stats keys {missing_stats} in {stats_dir}")

    rows_df = _origami_vla_dataset.load_manifest_rows(settings, args.split)
    grouped_rows: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for row in rows_df.to_dict(orient="records"):
        grouped_rows[str(row["episode_uid"])].append((int(row["frame_position"]), int(row["frame_index"])))

    max_token_len = int(config.model.max_token_len)
    top_k = max(1, int(args.top_k))
    num_workers = max(1, int(args.num_workers or min(8, os.cpu_count() or 1)))
    clip_discrete_inputs = bool(config.model.origami_vla.prompt_discrete_clip)

    print("Origami action-chunk prompt token-length scan")
    print(f"  config_name   : {args.config_name}")
    print(f"  split         : {args.split}")
    print(f"  rows          : {len(rows_df)}")
    print(f"  episodes      : {len(grouped_rows)}")
    print(f"  max_token_len : {max_token_len}")
    print(f"  clip_discrete : {clip_discrete_inputs}")
    print(f"  stats_dir     : {stats_dir}")
    print(f"  prompt        : {settings.prompt}")

    worker_kwargs = {
        "dataset_root": str(settings.dataset_root),
        "prompt": settings.prompt,
        "max_token_len": max_token_len,
        "top_k": top_k,
        "clip_discrete_inputs": clip_discrete_inputs,
        "state_stats": norm_stats["state"],
        "tactile_stats": norm_stats["tactile_prompt"],
    }

    token_lengths: list[int] = []
    longest: list[dict[str, Any]] = []
    overflow_count = 0
    progress = tqdm(total=len(rows_df), desc="Token lengths", unit="row", dynamic_ncols=True)

    if num_workers == 1:
        results_iter = (
            _episode_worker(episode_uid=episode_uid, frame_rows=grouped_rows[episode_uid], **worker_kwargs)
            for episode_uid in sorted(grouped_rows)
        )
        for result in results_iter:
            overflow_count += int(result["overflow_count"])
            token_lengths.extend(result["token_lengths"])
            for item in result["longest"]:
                if len(longest) < top_k:
                    longest.append(item)
                    longest.sort(key=lambda entry: int(entry["token_len"]), reverse=True)
                elif int(item["token_len"]) > int(longest[-1]["token_len"]):
                    longest[-1] = item
                    longest.sort(key=lambda entry: int(entry["token_len"]), reverse=True)
            progress.update(int(result["row_count"]))
            progress.set_postfix(max_seen=max(token_lengths), overflow=overflow_count)
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    _episode_worker,
                    episode_uid=episode_uid,
                    frame_rows=grouped_rows[episode_uid],
                    **worker_kwargs,
                ): episode_uid
                for episode_uid in sorted(grouped_rows)
            }
            for future in as_completed(futures):
                result = future.result()
                overflow_count += int(result["overflow_count"])
                token_lengths.extend(result["token_lengths"])
                for item in result["longest"]:
                    if len(longest) < top_k:
                        longest.append(item)
                        longest.sort(key=lambda entry: int(entry["token_len"]), reverse=True)
                    elif int(item["token_len"]) > int(longest[-1]["token_len"]):
                        longest[-1] = item
                        longest.sort(key=lambda entry: int(entry["token_len"]), reverse=True)
                progress.update(int(result["row_count"]))
                progress.set_postfix(max_seen=max(token_lengths), overflow=overflow_count)
    progress.close()

    lengths = np.asarray(token_lengths, dtype=np.int32)
    summary = _summarize(lengths)
    overflow_fraction = float(overflow_count) / float(max(1, len(rows_df)))
    print("Token length summary")
    print(f"  min           : {summary['min']}")
    print(f"  p50           : {summary['p50']:.2f}")
    print(f"  p90           : {summary['p90']:.2f}")
    print(f"  p95           : {summary['p95']:.2f}")
    print(f"  p99           : {summary['p99']:.2f}")
    print(f"  max           : {summary['max']}")
    print(f"  mean          : {summary['mean']:.2f}")
    print(f"  std           : {summary['std']:.2f}")
    print(f"  overflow_count: {overflow_count}")
    print(f"  overflow_frac : {overflow_fraction:.6f}")
    print(f"  recommended_floor_max_token_len : {max(int(summary['max']), max_token_len)}")
    print("Longest samples")
    for index, item in enumerate(longest, start=1):
        print(
            f"  [{index:02d}] len={int(item['token_len'])} | "
            f"{item['episode_uid']} | frame_position={int(item['frame_position'])} | "
            f"frame_index={int(item['frame_index'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
