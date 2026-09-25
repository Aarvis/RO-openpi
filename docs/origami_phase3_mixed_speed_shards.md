# Phase 3 mixed-speed Origami shards

This document describes the Phase 3 dataset contract. It is intentionally
written before the Phase 3 builder and loader exist, so those later components
share one explicit source of truth.

## Scope and execution environment

Phase 3 separates the raw-data build environment from the portable training
environment.

- Build machines require the raw episode arrays, camera videos, tactile videos,
  and a source manifest.
- Training machines require only the completed Phase 3 shard package, Phase 3
  normalization assets, the Phase 2 checkpoint `params` directory, and the
  Phase 3 code/configuration.
- The Phase 3 trainer must not open raw episode videos, raw tactile videos,
  source action arrays, source manifests, or planner rollout exports.

All build, benchmark, verification, and training commands are intended to run
on the remote machine. Do not generate or validate large shard datasets on a
developer workstation.

## Configuration names

Two configurations will be registered in `openpi.training.config`:

```text
pi05_origami_comp_action_chunk_phase3_build
pi05_origami_comp_action_chunk_phase3
```

The `*_build` configuration is only for manifest/shard construction. The
second configuration is the portable training configuration and initializes
from the completed Phase 2 checkpoint.

## Mixed-speed contract

The default configuration is:

| Speed | Action stride | Episode-equivalent coverage |
| --- | ---: | ---: |
| 1x | 1 | 1.50 |
| 2x | 2 | 1.00 |
| 3x | 3 | 0.75 |
| 4x | 4 | 0.50 |
| 5x | 5 | 0.25 |

These values are configuration inputs, not hard-coded shard-builder behavior.
Changing a stride or coverage requires rebuilding the Phase 3 source manifest
and shards; it does not require copying image or tactile data once per speed.

For an action horizon of 25 and maximum stride 5, retained starts obey:

```text
frame_position + 24 * 5 < episode_length
```

Consequently every retained row has a full horizon for every configured speed.
Phase 3 stores no action-mask arrays.

Action targets in shards remain raw absolute actions. The training transform
later creates state-anchored deltas and applies Phase 3 normalization.

## Planner and speed weighting

Planner rollout data is not used in Phase 3. The model retains planner-shaped
inputs inherited from Phase 2, so shards will contain zero-valued planner arrays
and `planner_available=false` for every row. No planner NPZ export is needed at
build or training time.

Phase 3 uses no execution-speed, episode-duration, frame-level, or checkpoint
loss weighting. Every logical training sample has unit loss weight. Speed only
selects an action stride, a text prompt, and sampling frequency.

## Portable shard package

A completed Phase 3 shard package will include:

```text
shard_manifest.json
train/shard_*/complete.marker
train/shard_*/rows.parquet
train/shard_*/metadata.json
train/shard_*/arrays/*.npy
val/shard_*/...                 # when validation shards are built
```

The later Phase 3 shard schema will keep one physical copy of observations and
tactile inputs, five action arrays (`actions_stride_1.npy` through
`actions_stride_5.npy`), and a compact shard-local virtual sample plan containing
only `(physical_row_id, speed_id)` references.

The virtual plan is independent of training batch size. A future loader will
form speed-mixed batches inside the active shard using the remote machine's
configured batch size. It must never gather a batch across shards solely to
obtain a desired speed mixture.

## Resumability and verification

The future Phase 3 builder will retain incomplete shards and checkpoint at
completed-episode boundaries. It will never delete a compatible incomplete
shard when resuming. A final `complete.marker` is written only after every
episode, action array, metadata file, and virtual plan has been verified.

The future verification scripts will support portable structural/full checks and
optional raw-source reconstruction checks on the remote build machine. They will
validate all required arrays, row counts, finite numeric values, zero planner
inputs, action horizons, virtual-plan references, split isolation, and realized
speed coverage.
