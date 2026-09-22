#!/usr/bin/env python3
"""Audit Phase-2 planner-export coverage and exported history semantics.

This is intentionally read-only.  It checks the source lookup and the history
arrays saved by ``export_checkpoint_planner_vla_rollout_features.py``.

For a direct frame-stride mode such as ``frame_stride_25`` the saved history
must step backwards by exactly 25 frame positions at every valid adjacent pair.
For ``random_mix`` the script recreates the exact deterministic selection used
by the exporter and verifies that every saved history matches it.  It also
reports how often F25 and F30 were actually selected, rather than merely
checking that a random_mix export directory exists.
"""

from __future__ import annotations

import argparse
import json
import math
import zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


VIEW_MODE_TO_LOOKUP = {
    "fixed_3": "past_3span",
    "fixed_7": "past_7span",
    "fixed_15": "past_15span",
    "fixed_25": "past_25span",
    "fixed_40": "past_40span",
    "frame_stride_10": "past_10frame",
    "frame_stride_25": "past_25frame",
    "frame_stride_30": "past_30frame",
    "frame_stride_50": "past_50frame",
}


@dataclass(frozen=True)
class Lookup:
    positions: dict[str, np.ndarray]
    valid: dict[str, np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit planner-export coverage and saved F25/random-mix history positions."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["frame_stride_25", "random_mix"],
        help="Export modes to inspect. The default is the Phase-2 pair.",
    )
    parser.add_argument("--action-horizon", type=int, default=25)
    parser.add_argument("--action-chunk-stride", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument(
        "--lookup-npz-name",
        default="checkpoint_planner_past_span_lookup_F25_F30_F50.npz",
        help="Per-episode lookup file containing past_25frame and past_30frame arrays.",
    )
    parser.add_argument(
        "--random-mix-modes",
        nargs="+",
        default=["frame_stride_25", "frame_stride_30"],
        help="Candidate modes used when the exported view mode is random_mix.",
    )
    parser.add_argument("--random-mix-seed", type=int, default=1234)
    parser.add_argument("--index-filename", default="planner_vla_rollout_index.parquet")
    parser.add_argument("--arrays-filename", default="planner_vla_rollout_features.npz")
    parser.add_argument("--marker-filename", default="export_complete.marker")
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument(
        "--fail-on-history-error",
        action="store_true",
        help="Exit non-zero if any saved history fails the semantic checks.",
    )
    return parser.parse_args()


def require_positive(name: str, value: int) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be positive; got {value}.")
    return value


def list_episode_uids(dataset_root: Path) -> list[str]:
    episodes_root = dataset_root / "episodes"
    if not episodes_root.is_dir():
        raise FileNotFoundError(f"Episodes directory not found: {episodes_root}")
    return sorted(path.name for path in episodes_root.iterdir() if path.is_dir())


def load_lookup(path: Path) -> Lookup:
    positions: dict[str, np.ndarray] = {}
    valid: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as archive:
        for mode, prefix in VIEW_MODE_TO_LOOKUP.items():
            position_key = f"{prefix}_frame_position"
            valid_key = f"{prefix}_valid"
            if position_key in archive.files and valid_key in archive.files:
                positions[mode] = np.asarray(archive[position_key], dtype=np.int64)
                valid[mode] = np.asarray(archive[valid_key], dtype=bool)
    return Lookup(positions=positions, valid=valid)


def deterministic_random_mix_choice(
    *,
    episode_uid: str,
    anchor_position: int,
    current_position: int,
    history_depth: int,
    candidates: list[int],
    seed: int,
) -> int:
    """Exact copy of checkpoint-planner exporter selection logic."""
    if len(candidates) == 1:
        return int(candidates[0])
    episode_crc = zlib.crc32(episode_uid.encode("utf-8")) & 0xFFFFFFFF
    mixed_seed = (
        (int(seed) & 0xFFFFFFFF)
        ^ episode_crc
        ^ ((int(anchor_position) * 1_000_003) & 0xFFFFFFFF)
        ^ ((int(current_position) * 2_000_033) & 0xFFFFFFFF)
        ^ ((int(history_depth) * 5_000_099) & 0xFFFFFFFF)
    )
    rng = np.random.default_rng(mixed_seed)
    return int(candidates[int(rng.integers(0, len(candidates)))])


def expected_history(
    *,
    lookup: Lookup,
    episode_uid: str,
    anchor: int,
    sequence_length: int,
    view_mode: str,
    random_mix_modes: list[str],
    random_mix_seed: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Rebuild exactly the history that the planner exporter should have saved."""
    positions = [int(anchor)]
    selected_modes: list[str] = []
    current = int(anchor)
    while len(positions) < sequence_length:
        if view_mode in VIEW_MODE_TO_LOOKUP:
            if view_mode not in lookup.positions:
                raise KeyError(f"Lookup is missing {view_mode!r}")
            previous = int(lookup.positions[view_mode][current]) if lookup.valid[view_mode][current] else -1
            selected_mode = view_mode
        elif view_mode == "random_mix":
            candidates: list[tuple[str, int]] = []
            for candidate_mode in random_mix_modes:
                if candidate_mode not in lookup.positions:
                    continue
                if lookup.valid[candidate_mode][current]:
                    candidates.append((candidate_mode, int(lookup.positions[candidate_mode][current])))
            if not candidates:
                previous = -1
                selected_mode = ""
            else:
                choice_values = [position for _, position in candidates]
                previous = deterministic_random_mix_choice(
                    episode_uid=episode_uid,
                    anchor_position=anchor,
                    current_position=current,
                    history_depth=len(positions),
                    candidates=choice_values,
                    seed=random_mix_seed,
                )
                # Positions from different positive frame strides cannot coincide.
                selected_mode = next(mode for mode, position in candidates if position == previous)
        else:
            raise ValueError(f"Unsupported view mode {view_mode!r}")
        if previous < 0:
            break
        positions.append(previous)
        selected_modes.append(selected_mode)
        current = previous

    ordered = np.asarray(list(reversed(positions)), dtype=np.int64)
    valid = np.ones(len(ordered), dtype=bool)
    if len(ordered) < sequence_length:
        pad = sequence_length - len(ordered)
        ordered = np.concatenate([np.full(pad, -1, dtype=np.int64), ordered])
        valid = np.concatenate([np.zeros(pad, dtype=bool), valid])
    return ordered, valid, selected_modes


def load_valid_action_positions(episode_root: Path, horizon: int, stride: int) -> np.ndarray:
    """Return legal H/S action-chunk start positions for this episode."""
    arrays = episode_root / "arrays"
    action = np.load(arrays / "action_65d.npy", mmap_mode="r")
    state = np.load(arrays / "state_65d.npy", mmap_mode="r")
    tactile = np.load(arrays / "tactile_60d.npy", mmap_mode="r")
    usable = min(len(action), len(state), len(tactile))
    max_start = usable - (horizon - 1) * stride
    return np.arange(max(0, max_start), dtype=np.int64)


def validate_lookup_stride(lookup: Lookup, mode: str, expected_stride: int) -> tuple[int, int]:
    """Return (valid entries, invalid lookup entries) for a direct frame stride."""
    if mode not in lookup.positions:
        return 0, 1
    positions = lookup.positions[mode]
    valid = lookup.valid[mode]
    current = np.arange(len(positions), dtype=np.int64)
    errors = int(np.count_nonzero(valid & (positions != current - expected_stride)))
    errors += int(np.count_nonzero(~valid & (positions != -1)))
    errors += int(np.count_nonzero(valid & (positions < 0)))
    return int(valid.sum()), errors


def common_prefix(values: Iterable[np.ndarray]) -> np.ndarray:
    values = list(values)
    if not values:
        return np.empty((0,), dtype=np.int64)
    result = np.asarray(values[0], dtype=np.int64)
    for value in values[1:]:
        result = np.intersect1d(result, np.asarray(value, dtype=np.int64), assume_unique=False)
    return result


def main() -> int:
    args = parse_args()
    horizon = require_positive("--action-horizon", int(args.action_horizon))
    action_stride = require_positive("--action-chunk-stride", int(args.action_chunk_stride))
    sequence_length = require_positive("--sequence-length", int(args.sequence_length))
    modes = [str(mode) for mode in args.modes]
    random_mix_modes = [str(mode) for mode in args.random_mix_modes]
    unknown_mix = [mode for mode in random_mix_modes if mode not in VIEW_MODE_TO_LOOKUP]
    if unknown_mix:
        raise ValueError(f"Unsupported --random-mix-modes: {unknown_mix}")

    report_dir = args.report_dir.resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    episode_uids = list_episode_uids(args.dataset_root)

    totals = Counter()
    coverage_by_mode = Counter()
    history_by_mode = Counter()
    random_choice_counts = Counter()
    per_episode_rows: list[dict[str, object]] = []
    fully_covered = 0
    missing_export_files = 0

    progress = tqdm(episode_uids, desc="Audit planner-export coverage", unit="episode")
    for episode_uid in progress:
        episode_root = args.dataset_root / "episodes" / episode_uid
        valid_starts = load_valid_action_positions(episode_root, horizon, action_stride)
        totals["valid_action_starts"] += len(valid_starts)
        lookup_path = episode_root / "arrays" / args.lookup_npz_name
        if not lookup_path.is_file():
            raise FileNotFoundError(f"Missing lookup file: {lookup_path}")
        lookup = load_lookup(lookup_path)

        direct_lookup_checks: dict[str, tuple[int, int]] = {}
        for mode in {"frame_stride_25", *random_mix_modes}:
            if mode.startswith("frame_stride_"):
                direct_lookup_checks[mode] = validate_lookup_stride(
                    lookup, mode, int(mode.removeprefix("frame_stride_"))
                )

        exported_positions_by_mode: dict[str, np.ndarray] = {}
        episode_common: list[np.ndarray] = []
        episode_has_all_files = True
        for mode in modes:
            export_dir = args.export_root / episode_uid / mode
            index_path = export_dir / args.index_filename
            arrays_path = export_dir / args.arrays_filename
            marker_path = export_dir / args.marker_filename
            if not (index_path.is_file() and arrays_path.is_file() and marker_path.is_file()):
                episode_has_all_files = False
                missing_export_files += 1
                per_episode_rows.append(
                    {
                        "episode_uid": episode_uid,
                        "view_mode": mode,
                        "valid_action_starts": len(valid_starts),
                        "exported_rows": 0,
                        "covered_action_starts": 0,
                        "coverage_ratio": 0.0,
                        "status": "missing_export_files",
                    }
                )
                continue

            index_frame = pd.read_parquet(index_path, columns=["frame_position"])
            frame_positions = index_frame["frame_position"].to_numpy(dtype=np.int64, copy=True)
            with np.load(arrays_path, allow_pickle=False) as archive:
                if "history_positions" not in archive.files or "history_valid_mask" not in archive.files:
                    raise KeyError(f"{arrays_path} lacks history_positions/history_valid_mask")
                saved_history = np.asarray(archive["history_positions"], dtype=np.int64)
                saved_mask = np.asarray(archive["history_valid_mask"], dtype=bool)

            if saved_history.shape != saved_mask.shape:
                raise RuntimeError(f"{arrays_path}: history positions/mask shapes differ: {saved_history.shape} vs {saved_mask.shape}")
            if saved_history.ndim != 2 or saved_history.shape[0] != len(frame_positions):
                raise RuntimeError(
                    f"{arrays_path}: history rows {saved_history.shape} do not align with index rows {len(frame_positions)}"
                )
            if saved_history.shape[1] != sequence_length:
                raise RuntimeError(
                    f"{arrays_path}: sequence length {saved_history.shape[1]} != requested {sequence_length}. "
                    "Pass the exporter sequence length."
                )

            exported_unique = np.unique(frame_positions)
            covered = np.intersect1d(valid_starts, exported_unique, assume_unique=False)
            exported_positions_by_mode[mode] = covered
            episode_common.append(covered)
            coverage_by_mode[(mode, "covered")] += len(covered)
            coverage_by_mode[(mode, "exported_rows")] += len(frame_positions)

            semantic_errors = 0
            stored_history_errors = 0
            padding_errors = 0
            lookup_errors = 0
            mix_rows_both = 0
            mix_rows_only_25 = 0
            mix_rows_only_30 = 0
            mix_rows_no_step = 0
            mode_choices = Counter()

            if mode == "frame_stride_25":
                _, lookup_errors = direct_lookup_checks.get(mode, (0, 1))

            for row_index, anchor in enumerate(frame_positions.tolist()):
                expected_positions, expected_mask, selected_modes = expected_history(
                    lookup=lookup,
                    episode_uid=episode_uid,
                    anchor=int(anchor),
                    sequence_length=sequence_length,
                    view_mode=mode,
                    random_mix_modes=random_mix_modes,
                    random_mix_seed=int(args.random_mix_seed),
                )
                actual_positions = saved_history[row_index]
                actual_mask = saved_mask[row_index]
                if not np.array_equal(actual_positions, expected_positions) or not np.array_equal(actual_mask, expected_mask):
                    stored_history_errors += 1
                    continue

                # Generic format checks are retained independently of the exact
                # reconstruction, so corrupt values cannot pass by coincidence.
                if np.any(actual_positions[~actual_mask] != -1):
                    padding_errors += 1
                    continue
                selected = actual_positions[actual_mask]
                if len(selected) and selected[-1] != anchor:
                    semantic_errors += 1
                    continue
                if len(selected) > 1 and np.any(np.diff(selected) <= 0):
                    semantic_errors += 1
                    continue

                if mode == "frame_stride_25":
                    if len(selected) > 1 and not np.all(np.diff(selected) == 25):
                        semantic_errors += 1
                elif mode == "random_mix":
                    deltas = np.diff(selected)
                    if np.any(~np.isin(deltas, [25, 30])):
                        semantic_errors += 1
                    # selected_modes is newest-to-oldest. Count it by actual
                    # delta too, so the report remains interpretable.
                    mode_choices.update(selected_modes)
                    has_25 = bool(np.any(deltas == 25))
                    has_30 = bool(np.any(deltas == 30))
                    if not len(deltas):
                        mix_rows_no_step += 1
                    elif has_25 and has_30:
                        mix_rows_both += 1
                    elif has_25:
                        mix_rows_only_25 += 1
                    elif has_30:
                        mix_rows_only_30 += 1

            history_by_mode[(mode, "stored_history_errors")] += stored_history_errors
            history_by_mode[(mode, "semantic_errors")] += semantic_errors
            history_by_mode[(mode, "padding_errors")] += padding_errors
            history_by_mode[(mode, "lookup_errors")] += lookup_errors
            if mode == "random_mix":
                history_by_mode[(mode, "rows_both_f25_f30")] += mix_rows_both
                history_by_mode[(mode, "rows_only_f25")] += mix_rows_only_25
                history_by_mode[(mode, "rows_only_f30")] += mix_rows_only_30
                history_by_mode[(mode, "rows_no_step")] += mix_rows_no_step
                random_choice_counts.update(mode_choices)

            per_episode_rows.append(
                {
                    "episode_uid": episode_uid,
                    "view_mode": mode,
                    "valid_action_starts": len(valid_starts),
                    "exported_rows": len(frame_positions),
                    "covered_action_starts": len(covered),
                    "coverage_ratio": float(len(covered) / max(len(valid_starts), 1)),
                    "stored_history_errors": stored_history_errors,
                    "semantic_errors": semantic_errors,
                    "padding_errors": padding_errors,
                    "lookup_errors": lookup_errors,
                    "random_mix_rows_both_f25_f30": mix_rows_both if mode == "random_mix" else 0,
                    "random_mix_rows_only_f25": mix_rows_only_25 if mode == "random_mix" else 0,
                    "random_mix_rows_only_f30": mix_rows_only_30 if mode == "random_mix" else 0,
                    "status": "ok" if not (stored_history_errors or semantic_errors or padding_errors or lookup_errors) else "history_error",
                }
            )

        if episode_has_all_files and len(episode_common) == len(modes):
            common = common_prefix(episode_common)
            totals["common_covered"] += len(common)
            if len(common) == len(valid_starts):
                fully_covered += 1
        progress.set_postfix(
            common=f"{totals['common_covered']}/{totals['valid_action_starts']}",
            uid=episode_uid[-24:],
        )

    detail_path = report_dir / "coverage_by_episode_and_mode.csv"
    pd.DataFrame(per_episode_rows).to_csv(detail_path, index=False)

    per_mode_summary = {}
    for mode in modes:
        per_mode_summary[mode] = {
            "covered_action_starts": int(coverage_by_mode[(mode, "covered")]),
            "valid_action_starts": int(totals["valid_action_starts"]),
            "coverage_ratio": float(coverage_by_mode[(mode, "covered")] / max(totals["valid_action_starts"], 1)),
            "exported_rows": int(coverage_by_mode[(mode, "exported_rows")]),
            "history": {
                "stored_history_errors": int(history_by_mode[(mode, "stored_history_errors")]),
                "semantic_errors": int(history_by_mode[(mode, "semantic_errors")]),
                "padding_errors": int(history_by_mode[(mode, "padding_errors")]),
                "lookup_errors": int(history_by_mode[(mode, "lookup_errors")]),
            },
        }
    if "random_mix" in modes:
        per_mode_summary["random_mix"]["random_mix_selection"] = {
            "candidate_modes": random_mix_modes,
            "seed": int(args.random_mix_seed),
            "selected_steps": {mode: int(random_choice_counts[mode]) for mode in random_mix_modes},
            "rows_with_both_f25_and_f30": int(history_by_mode[("random_mix", "rows_both_f25_f30")]),
            "rows_only_f25": int(history_by_mode[("random_mix", "rows_only_f25")]),
            "rows_only_f30": int(history_by_mode[("random_mix", "rows_only_f30")]),
            "rows_without_a_history_step": int(history_by_mode[("random_mix", "rows_no_step")]),
        }

    total_history_errors = sum(
        values["history"][name]
        for values in per_mode_summary.values()
        for name in ("stored_history_errors", "semantic_errors", "padding_errors", "lookup_errors")
    )
    summary = {
        "dataset_root": str(args.dataset_root.resolve()),
        "export_root": str(args.export_root.resolve()),
        "action_horizon": horizon,
        "action_chunk_stride": action_stride,
        "sequence_length": sequence_length,
        "episodes": len(episode_uids),
        "valid_action_starts": int(totals["valid_action_starts"]),
        "modes": per_mode_summary,
        "common_coverage": {
            "covered_action_starts": int(totals["common_covered"]),
            "coverage_ratio": float(totals["common_covered"] / max(totals["valid_action_starts"], 1)),
            "fully_covered_episodes": fully_covered,
        },
        "missing_export_file_sets": missing_export_files,
        "total_history_errors": int(total_history_errors),
        "detail_csv": str(detail_path),
    }
    summary_path = report_dir / "coverage_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("Planner export coverage + history audit")
    print(f"episodes                    : {summary['episodes']}")
    print(f"valid H{horizon} starts       : {summary['valid_action_starts']:,}")
    for mode, values in per_mode_summary.items():
        coverage = values["coverage_ratio"] * 100.0
        print(f"{mode:28s}: {values['covered_action_starts']:,}/{summary['valid_action_starts']:,} ({coverage:.2f}%)")
        history = values["history"]
        print(
            f"  history errors              : stored={history['stored_history_errors']:,} "
            f"semantic={history['semantic_errors']:,} padding={history['padding_errors']:,} "
            f"lookup={history['lookup_errors']:,}"
        )
        if mode == "random_mix":
            selection = values["random_mix_selection"]
            counts = ", ".join(f"{key}={value:,}" for key, value in selection["selected_steps"].items())
            print(f"  selected history steps      : {counts}")
            print(
                "  random-mix rows              : "
                f"both={selection['rows_with_both_f25_and_f30']:,} "
                f"only_F25={selection['rows_only_f25']:,} only_F30={selection['rows_only_f30']:,}"
            )
    print(
        f"common coverage              : {summary['common_coverage']['covered_action_starts']:,}/"
        f"{summary['valid_action_starts']:,} ({summary['common_coverage']['coverage_ratio'] * 100.0:.2f}%)"
    )
    print(f"fully covered episodes       : {fully_covered}/{len(episode_uids)}")
    print(f"missing export file sets     : {missing_export_files}")
    print(f"detail CSV                   : {detail_path}")
    print(f"summary JSON                 : {summary_path}")
    if args.fail_on_history_error and total_history_errors:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
