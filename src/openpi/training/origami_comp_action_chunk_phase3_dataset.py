"""Portable logical-sample loader for completed Origami Phase-3 shards."""

from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import openpi.training.origami_comp_action_chunk_phase3_shards as _phase3
import openpi.training.origami_comp_action_chunk_shards as _chunk
import openpi.training.origami_vla_dataset as _dataset


def _close(array: np.ndarray) -> None:
    _phase3.close_memmap(array)


class _Bundle:
    def __init__(self, spec: _phase3.ShardSpec, rows: pd.DataFrame, plan: np.ndarray, arrays: dict[str, np.ndarray]):
        self.spec = spec
        self.rows = rows
        self.plan = plan
        self.arrays = arrays


class OrigamiCompActionChunkPhase3Dataset:
    """Map-style dataset whose rows are shuffled Phase-3 virtual occurrences.

    Concatenating per-shard plans produces one continuous logical stream. The
    standard Torch loader consequently makes a full boundary batch from the
    tail of one shard and head of the next shard, dropping only the final
    global remainder. With the default two-shard cache, the following shard is
    memory-mapped when a worker first enters the current one.
    """

    def __init__(self, settings: _dataset.OrigamiVlaSettings, *, split: str):
        if not settings.phase3_mixed_speed_enabled:
            raise ValueError("Phase-3 dataset requires phase3_mixed_speed_enabled=True.")
        if settings.action_source != "action_chunk":
            raise ValueError(f"Phase-3 requires action_chunk targets, got {settings.action_source!r}.")
        if not settings.shard_root:
            raise ValueError("Phase-3 dataset requires shard_root.")
        self._settings = settings
        self._split = split
        self._root = Path(settings.shard_root)
        self._shards = _phase3.load_phase3_specs(self._root, split=split, require_complete=True)
        self._logical_counts: list[int] = []
        self._cumulative_counts: list[int] = []
        total = 0
        for spec in self._shards:
            metadata = json.loads((spec.shard_dir / _phase3.METADATA_FILENAME).read_text(encoding="utf-8"))
            if metadata.get("format") != settings.phase3_shard_format:
                raise ValueError(f"{spec.shard_dir}: unexpected Phase-3 format {metadata.get('format')!r}.")
            plan_info = metadata.get("virtual_plan", {})
            if tuple(plan_info.get("columns", ())) != _phase3.VIRTUAL_PLAN_COLUMNS:
                raise ValueError(f"{spec.shard_dir}: virtual-plan columns differ from Phase-3 contract.")
            if Path(str(plan_info.get("filename", ""))).name != settings.phase3_virtual_plan_filename:
                raise ValueError(f"{spec.shard_dir}: unexpected virtual-plan filename.")
            count = int(plan_info.get("logical_samples", -1))
            if count < 0:
                raise ValueError(f"{spec.shard_dir}: missing virtual logical sample count.")
            total += count
            self._logical_counts.append(count)
            self._cumulative_counts.append(total)
        if total <= 0:
            raise RuntimeError(f"No logical Phase-3 samples found for split {split!r} in {self._root}.")
        self._cache: OrderedDict[int, _Bundle] = OrderedDict()

    def __len__(self) -> int:
        return self._cumulative_counts[-1]

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_cache"] = OrderedDict()
        return state

    def close(self) -> None:
        while self._cache:
            _index, bundle = self._cache.popitem(last=False)
            for array in (*bundle.arrays.values(), bundle.plan):
                _close(array)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _load_bundle(self, shard_index: int) -> _Bundle:
        cached = self._cache.get(shard_index)
        if cached is not None:
            self._cache.move_to_end(shard_index)
            return cached
        spec = self._shards[shard_index]
        metadata = json.loads((spec.shard_dir / _phase3.METADATA_FILENAME).read_text(encoding="utf-8"))
        arrays_dir = spec.shard_dir / "arrays"
        arrays: dict[str, np.ndarray] = {}
        for name, descriptor in metadata["arrays"].items():
            path = spec.shard_dir / str(descriptor["filename"])
            arrays[name] = np.load(path, mmap_mode="r")
        plan_path = spec.shard_dir / str(metadata["virtual_plan"]["filename"])
        plan = np.load(plan_path, mmap_mode="r")
        if plan.dtype != np.dtype(np.uint32) or plan.shape != (self._logical_counts[shard_index], 3):
            raise ValueError(f"{spec.shard_dir}: invalid Phase-3 virtual plan.")
        bundle = _Bundle(
            spec,
            pd.read_parquet(spec.shard_dir / _phase3.ROWS_FILENAME),
            plan,
            arrays,
        )
        self._cache[shard_index] = bundle
        while len(self._cache) > int(self._settings.shard_max_cached_shards):
            _old_index, old = self._cache.popitem(last=False)
            for array in (*old.arrays.values(), old.plan):
                _close(array)
        return bundle

    def _locate(self, index: int) -> tuple[_Bundle, int]:
        index = int(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_index = bisect_right(self._cumulative_counts, index)
        previous = 0 if shard_index == 0 else self._cumulative_counts[shard_index - 1]
        bundle = self._load_bundle(shard_index)
        # Keep the immediate successor open when cache capacity permits. This
        # is deliberately shard-local: it does not alter ordering or perform
        # cross-shard sampling, it only avoids a boundary-time mmap/open.
        if int(self._settings.shard_max_cached_shards) >= 2 and shard_index + 1 < len(self._shards):
            self._load_bundle(shard_index + 1)
        return bundle, index - previous

    def __getitem__(self, index: int) -> dict[str, Any]:
        bundle, plan_index = self._locate(index)
        physical_row, stride, occurrence = (int(value) for value in bundle.plan[plan_index])
        arrays = bundle.arrays
        if physical_row >= len(bundle.rows):
            raise RuntimeError(f"{bundle.spec.shard_dir}: virtual plan references an invalid physical row.")
        row = bundle.rows.iloc[physical_row]
        episode_uid = str(row["episode_uid"])
        frame_position = int(row["frame_position"])
        images = {
            key: np.asarray(arrays[f"{_chunk.IMAGE_ARRAY_PREFIX}{key}"][physical_row], dtype=np.uint8)
            for key in self._settings.image_modalities
        }
        tactile = np.asarray(arrays["tactile"][physical_row], dtype=np.float32)
        output: dict[str, Any] = {
            "image": images,
            "image_mask": {key: np.asarray(True) for key in images},
            "state": np.asarray(arrays["state"][physical_row], dtype=np.float32),
            "tactile": tactile,
            "tactile_prompt": np.array(tactile, copy=True),
            "tactile_prompt_mask": np.ones((self._settings.tactile_dim,), dtype=bool),
            "state_mask": np.ones((self._settings.state_dim,), dtype=bool),
            # Downstream DeltaActions mutates targets, so copy the read-only memmap slice.
            "actions": np.array(arrays[f"actions_stride_{stride}"][physical_row], dtype=np.float32, copy=True),
            "action_mask": np.ones((self._settings.action_horizon,), dtype=bool),
            "sample_weight": np.asarray(1.0, dtype=np.float32),
            "prompt": np.asarray(self._settings.phase3_prompt_template.format(speed=stride)),
            "frame_position": np.asarray(frame_position, dtype=np.int64),
            "frame_index": np.asarray(arrays["frame_index"][physical_row], dtype=np.int64),
            "timestamp": np.asarray(arrays["timestamp"][physical_row], dtype=np.float32),
            "source_row_index": np.asarray(int(row.get("source_row_index", -1)), dtype=np.int64),
            "phase3_speed_id": np.asarray(stride, dtype=np.uint32),
            "phase3_occurrence_id": np.asarray(occurrence, dtype=np.uint32),
        }
        if self._settings.include_planner_features:
            output.update(
                {
                    "planner_available": np.asarray(False, dtype=bool),
                    "planner_state_belief": np.asarray(arrays["planner_state_belief"][physical_row], dtype=np.float32),
                    "planner_progress_transition": np.asarray(
                        arrays["planner_progress_transition"][physical_row], dtype=np.float32
                    ),
                    "planner_uncertainty": np.asarray(arrays["planner_uncertainty"][physical_row], dtype=np.float32),
                    "planner_history_latent": np.asarray(arrays["planner_history_latent"][physical_row], dtype=np.float32),
                }
            )
        if self._settings.load_tactile_images:
            physical_raw_available = bool(arrays["tactile_raw_available"][physical_row])
            drop_raw = _phase3.drop_tactile_raw_for_virtual_sample(
                self._settings,
                episode_uid=episode_uid,
                frame_position=frame_position,
                stride=stride,
                occurrence_id=occurrence,
            )
            raw_available = physical_raw_available and not drop_raw
            raw_images = np.asarray(arrays["tactile_raw_images"][physical_row], dtype=np.uint8)
            if not raw_available:
                raw_images = np.zeros_like(raw_images)
            output.update(
                {
                    "tactile_deform_images": np.asarray(arrays["tactile_deform_images"][physical_row], dtype=np.uint8),
                    "tactile_raw_images": raw_images,
                    "tactile_deform_available": np.asarray(
                        arrays["tactile_deform_available"][physical_row], dtype=bool
                    ),
                    "tactile_raw_available": np.asarray(raw_available, dtype=bool),
                }
            )
        return output
