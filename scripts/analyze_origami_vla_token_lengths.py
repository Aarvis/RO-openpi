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

import openpi.models.tokenizer as _tokenizer
import openpi.training.config as _config
import openpi.training.origami_vla_dataset as _origami_vla_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure actual pi0.5 Origami prompt token lengths before truncation. "
            "This scans the manifest rows, serializes the 65D state exactly like the tokenizer, "
            "and reports the token-length distribution against model.max_token_len."
        )
    )
    parser.add_argument("--config-name", type=str, default="pi05_origami_checkpoint_spline_vla")
    parser.add_argument("--manifest-root", type=Path, default=None, help="Optional manifest root override.")
    parser.add_argument("--dataset-root", type=Path, default=None, help="Optional dataset root override.")
    parser.add_argument("--split", type=str, default="train", choices=("train", "val", "all"))
    parser.add_argument("--max-rows", type=int, default=None, help="Optional cap on manifest rows.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of longest samples to print.")
    return parser.parse_args()


def build_full_prompt(
    tokenizer: _tokenizer.PaligemmaTokenizer,
    *,
    prompt: str,
    state: np.ndarray,
    state_mask: np.ndarray | None = None,
) -> str:
    cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
    state_str = tokenizer._serialize_state(state, state_mask)  # noqa: SLF001
    return f"Task: {cleaned_text}, {state_str}\nAction: "


def summarize(values: np.ndarray) -> dict[str, float | int]:
    if values.size == 0:
        return {
            "count": 0,
            "min": 0,
            "max": 0,
            "mean": 0.0,
            "std": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }
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


def main() -> int:
    args = parse_args()
    config = _config.get_config(args.config_name)
    if not isinstance(config.data, _config.OrigamiVlaDataConfig):
        raise TypeError(f"Config {args.config_name!r} is not an Origami VLA config.")
    if not isinstance(config.model, _config.pi0_config.Pi0Config):
        raise TypeError(f"Config {args.config_name!r} is not a Pi0/Pi0.5 config.")
    if not config.model.pi05 or not config.model.discrete_state_input:
        raise ValueError(
            f"Config {args.config_name!r} does not use pi0.5 discrete-state tokenization, "
            "so this token-budget analysis is not applicable."
        )

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

    rows_df = _origami_vla_dataset.load_manifest_rows(settings, args.split)
    dataset_root = Path(settings.dataset_root)
    grouped_rows: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows_df.to_dict(orient="records"):
        grouped_rows[str(row["episode_uid"])].append(row)

    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=max(int(config.model.max_token_len), 4096))
    max_token_len = int(config.model.max_token_len)
    top_k = max(1, int(args.top_k))

    token_lengths: list[int] = []
    longest: list[dict[str, object]] = []
    overflow_count = 0

    print("Origami VLA token budget analysis")
    print(f"  config_name   : {args.config_name}")
    print(f"  split         : {args.split}")
    print(f"  manifest_rows : {len(rows_df)}")
    print(f"  episodes      : {len(grouped_rows)}")
    print(f"  max_token_len : {max_token_len}")
    print(f"  prompt        : {settings.prompt}")

    progress = tqdm(total=len(rows_df), desc="Token lengths", unit="row", dynamic_ncols=True)
    for episode_uid in sorted(grouped_rows):
        episode_root = dataset_root / "episodes" / episode_uid
        state_array = np.load(episode_root / "arrays" / "state_65d.npy", mmap_mode="r")
        for row in grouped_rows[episode_uid]:
            frame_position = int(row["frame_position"])
            frame_index = int(row["frame_index"])
            state = np.asarray(state_array[frame_position], dtype=np.float32)
            state_mask = np.ones_like(state, dtype=bool)
            full_prompt = build_full_prompt(
                tokenizer,
                prompt=settings.prompt,
                state=state,
                state_mask=state_mask,
            )
            tokens = tokenizer._tokenizer.encode(full_prompt, add_bos=True)  # noqa: SLF001
            token_len = int(len(tokens))
            token_lengths.append(token_len)
            if token_len > max_token_len:
                overflow_count += 1

            sample_meta = {
                "token_len": token_len,
                "episode_uid": episode_uid,
                "frame_position": frame_position,
                "frame_index": frame_index,
            }
            if len(longest) < top_k:
                longest.append(sample_meta)
                longest.sort(key=lambda item: int(item["token_len"]), reverse=True)
            elif token_len > int(longest[-1]["token_len"]):
                longest[-1] = sample_meta
                longest.sort(key=lambda item: int(item["token_len"]), reverse=True)

            progress.update(1)
            progress.set_postfix(max_seen=max(token_lengths), overflow=overflow_count)
    progress.close()

    token_lengths_np = np.asarray(token_lengths, dtype=np.int32)
    summary = summarize(token_lengths_np)
    overflow_fraction = (float(overflow_count) / float(len(token_lengths_np))) if token_lengths_np.size else 0.0
    recommended = int(max(summary["max"], max_token_len))

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
    print(f"  recommended_floor_max_token_len : {recommended}")
    print("Longest samples")
    for idx, item in enumerate(longest, start=1):
        print(
            f"  [{idx:02d}] len={int(item['token_len'])} | "
            f"{item['episode_uid']} | frame_position={int(item['frame_position'])} | "
            f"frame_index={int(item['frame_index'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
