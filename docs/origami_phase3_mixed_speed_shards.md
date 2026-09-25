# Phase 3 mixed-speed Origami shards

This document describes the implemented Phase 3 source-manifest, shard,
loader, and normalization contract.

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

The Phase 3 shard schema keeps one physical copy of observations and tactile
inputs, including the original raw tactile image when it exists, five action
arrays (`actions_stride_1.npy` through `actions_stride_5.npy`), and a compact
shard-local virtual sample plan containing
`(physical_row_id, speed_id, occurrence_id)` references.

Raw tactile dropout is deliberately not baked into physical shards. The loader
retains deform tactile and decides whether raw tactile is available
per logical sample with the deterministic key
`(raw_dropout_seed, episode_uid, frame_position, stride, occurrence_id)`. With
probability 0.5, this yields deform-only; otherwise it yields deform plus raw.
Thus repeated physical rows can receive independent, reproducible decisions.

The virtual plan is independent of training batch size. The loader consumes it
as one speed-mixed logical stream using the remote machine's configured batch
size. It never gathers a batch across shards solely to obtain a desired speed
mixture.

## Resumability and verification

The builder retains incomplete shards and checkpoints at completed-episode
boundaries. It never deletes a compatible incomplete shard when resuming. A
final `complete.marker` is written only after every episode, action array,
metadata file, and virtual plan has been written and flushed; the explicit
verification workflow below is the required final integrity check.

The verification scripts support portable structural/full checks and optional
raw-source reconstruction checks on the remote build machine. They validate
validate all required arrays, row counts, finite numeric values, zero planner
inputs, action horizons, virtual-plan references, split isolation, and realized
speed coverage.

## Implemented scripts

The Phase 3 build path is now separated from existing Phase 1/2 shard scripts:

```text
scripts/build_origami_comp_action_chunk_manifest.py
scripts/verify_origami_comp_action_chunk_phase3_manifest.py
scripts/build_origami_comp_action_chunk_phase3_shards.py
scripts/verify_origami_comp_action_chunk_phase3_shards.py
scripts/verify_origami_comp_action_chunk_phase3_source_reconstruction.py
scripts/compute_origami_comp_action_chunk_phase3_norm_stats.py
scripts/verify_origami_comp_action_chunk_phase3_norm_stats.py
scripts/verify_origami_comp_action_chunk_phase3_loader.py
scripts/report_origami_comp_action_chunk_phase3_training_plan.py
```

The generic manifest builder gained two Phase-3-safe controls:

```text
--common-horizon-max-stride 5
--force-planner-disabled
```

When invoked with `pi05_origami_comp_action_chunk_phase3_build`, the config
supplies both automatically.

## Remote workflow

All commands below must be run from the remote machine, inside the `openpi`
directory and its intended Python environment. Replace the Phase 2 checkpoint
placeholder in `src/openpi/training/config.py` before starting training. Do not
run these commands on a developer workstation.

### 1. Build and verify the source manifest

```powershell
python .\scripts\build_origami_comp_action_chunk_manifest.py `
  --config-name pi05_origami_comp_action_chunk_phase3_build

python .\scripts\verify_origami_comp_action_chunk_phase3_manifest.py `
  --config-name pi05_origami_comp_action_chunk_phase3_build
```

Expected result: every retained row has a complete 25-step horizon through
stride 5, every `planner_enabled` value is false, every sample weight is one,
and no episode appears in both splits.

### 2. Inspect shard allocation without decoding videos

```powershell
python .\scripts\build_origami_comp_action_chunk_phase3_shards.py `
  --config-name pi05_origami_comp_action_chunk_phase3_build `
  --plan-only
```

Use this report to confirm shard count, source season allocation, selected
splits, and estimated physical storage before the full build. Configure remote
worker count with `--num-workers`; it means one active shard writer per worker.

### 3. Build shards and safely resume

```powershell
python .\scripts\build_origami_comp_action_chunk_phase3_shards.py `
  --config-name pi05_origami_comp_action_chunk_phase3_build `
  --num-workers 8
```

Each worker writes one shard and processes its episodes sequentially. A stopped
run leaves `shard_*.incomplete` directories. Rerun the same command to resume:
completed episodes have `progress/<episode_uid>.done` markers and are skipped.
The builder refuses to resume if the source rows, stride/coverage mapping,
observation dimensions, or image layout changed.

Use `--overwrite` only when deliberately discarding a completed shard. It is
not a normal resume mechanism.

### 4. Run portable full verification

After every required shard has `complete.marker`, run:

```powershell
python .\scripts\verify_origami_comp_action_chunk_phase3_shards.py `
  --config-name pi05_origami_comp_action_chunk_phase3 `
  --split all `
  --mode full
```

`--mode structure` checks names, metadata, shapes, dtypes, completion state,
planner-zero policy, virtual references, and the configured per-speed coverage
without scanning every numeric value. `--mode full` additionally scans all
floating arrays for NaN/Inf and requires every virtual plan to exactly match its
deterministic episode-balanced reconstruction. Neither mode needs raw source
data.

### 5. Run source reconstruction verification

On the remote build machine, compare a random sample against raw source arrays:

```powershell
python .\scripts\verify_origami_comp_action_chunk_phase3_source_reconstruction.py `
  --config-name pi05_origami_comp_action_chunk_phase3_build `
  --split train `
  --num-samples 256 `
  --include-images
```

This verifies state, tactile vectors, all five action strides, and—when
`--include-images` is supplied—the decoded/resized camera and tactile images,
including tactile-raw availability. Run it again
with `--split val` when validation shards are present. Increase `--num-samples`
for stronger sampled evidence; use portable full verification for complete
array-level integrity.

### 6. Verify logical loader behavior and report the actual training pass

The loader consumes each shard's shuffled virtual plan as one continuous
logical stream. It does not enforce exact per-batch speed quotas. A batch may
cross only a shard boundary; therefore `drop_last=True` drops only the final
global remainder, not a tail from every shard.

```powershell
python .\scripts\report_origami_comp_action_chunk_phase3_training_plan.py `
  --config-name pi05_origami_comp_action_chunk_phase3 `
  --global-batch-size 160

python .\scripts\verify_origami_comp_action_chunk_phase3_loader.py `
  --config-name pi05_origami_comp_action_chunk_phase3 `
  --split train `
  --global-batch-size 160
```

The loader verifier reports physical raw tactile availability, logical raw
tactile retained/dropped/missing counts, the result for every configured speed,
the exact final batch remainder, planner-zero policy, action-mask policy,
speed-prompt pairing, and OpenPI image shape/layout checks. Raw tactile is
preserved physically and dropout is recomputed per logical occurrence from
`(seed, episode_uid, frame_position, stride, occurrence_id)`.

### 7. Compute and verify Phase 3 normalization assets

```powershell
python .\scripts\compute_origami_comp_action_chunk_phase3_norm_stats.py `
  --config-name pi05_origami_comp_action_chunk_phase3

python .\scripts\verify_origami_comp_action_chunk_phase3_norm_stats.py `
  --config-name pi05_origami_comp_action_chunk_phase3
```

This uses only completed train shards. State and tactile/tactile-prompt stats
scan each physical row exactly once. Action stats scan every virtual occurrence,
select its stride-specific absolute action chunk, and apply the existing
state-anchored delta rule before accumulating statistics. The saved provenance
binds assets to the exact shard-manifest hash and Phase 3 contract.

### 8. Remote smoke run, then full training

After the preceding checks pass, run a short remote smoke run with the actual
Phase 2 `params` path and a small number of steps. Then set the final global
batch size and calculate one full mixed pass as:

```text
floor(total logical train samples / global batch size)
```

Use the training-plan report rather than assuming 350,000 steps. The existing
pi0.5 training command needs no model-architecture change: it consumes the
Phase 3 shard loader, applies existing delta-action and normalization transforms,
and initializes from the Phase 2 checkpoint.

## Completion criteria

Treat the Phase 3 shard package as ready for download only when all of the
following have passed on the remote machine:

1. Source manifest verification exits zero.
2. Every requested train/validation shard has `complete.marker`.
3. Portable `--mode full` verification exits zero.
4. Source reconstruction verification exits zero for both available splits.
5. Loader verification exits zero at the intended remote global batch size.
6. Norm-stat computation and provenance verification both exit zero.
7. The top-level `shard_manifest.json` records the intended `1.5/1/0.75/0.5/0.25`
   episode-coverage mapping and the Phase 3 format version.

At that point copy the completed shard root, Phase 3 normalization assets once
created, and the Phase 2 `params` checkpoint to a training machine. The raw
dataset, original source manifests, and planner exports are not needed there.
