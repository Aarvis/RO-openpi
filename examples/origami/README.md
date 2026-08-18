# Origami VLA (Checkpoint-Planner Conditioned)

This path adds an Origami-specific `pi0.5` fine-tuning config on top of OpenPI:

- current images: `ooi`, `head_left`, `wrist_left`, `wrist_right`
- current `state_65d` through the stock pi0.5 discrete-state prompt path
- 4 checkpoint-planner prefix tokens:
  - belief
  - progress
  - uncertainty
  - temporal history
- local delta reached-state spline target packed as:
  - 18 control-point tokens of width 65
  - 1 span-width token using the first 15 dimensions

The registered config name is:

- `pi05_origami_checkpoint_spline_vla`

## Files Added

- `scripts/build_origami_vla_manifest.py`
- `scripts/compute_origami_vla_norm_stats.py`
- `src/openpi/training/origami_vla_dataset.py`
- `src/openpi/models/origami_planner_adapter.py`
- `src/openpi/models/origami_spline_losses.py`

## Sample Manifest Build

Edit the sample JSON if your dataset roots differ, then run:

```bash
python scripts/build_origami_vla_manifest.py \
  --config examples/origami/configs/build_manifest.sample.json
```

This writes:

- `train_index.parquet`
- `val_index.parquet`
- `manifest.json`

under:

- `D:/Sampled_Reprocessed_Dataset/metadata/openpi_origami_vla/no_hmm_v1`

## Normalization

Compute state/action norm stats from the train split only:

```bash
python scripts/compute_origami_vla_norm_stats.py \
  --config-name pi05_origami_checkpoint_spline_vla \
  --manifest-root D:/Sampled_Reprocessed_Dataset/metadata/openpi_origami_vla/no_hmm_v1 \
  --dataset-root D:/Sampled_Reprocessed_Dataset
```

This writes `norm_stats.json` under:

- `assets/pi05_origami_checkpoint_spline_vla/sampled_reprocessed_dataset_origami_vla`

## Training

Run JAX training with the registered config:

```bash
python scripts/train.py pi05_origami_checkpoint_spline_vla \
  --exp-name origami_vla_run_001 \
  --data.dataset_root D:/Sampled_Reprocessed_Dataset \
  --data.manifest_root D:/Sampled_Reprocessed_Dataset/metadata/openpi_origami_vla/no_hmm_v1
```

Useful overrides:

```bash
--batch-size 32
--num-workers 8
--num-train-steps 60000
--save-interval 5000
--val-frequency 2000
--checkpoint-base-dir ./checkpoints
--assets-base-dir ./assets
```

By default, this Origami config uses a two-group LR setup:

- `origami_planner_adapter`, `action_in_proj`, and `action_out_proj` train at the full scheduled LR
- the pretrained pi0.5 backbone trains at `0.5x` of that LR

## Notes

- This implementation currently targets the native JAX `scripts/train.py` path.
- The model loads the base `pi05` checkpoint where shapes still match.
- `action_in_proj`, `action_out_proj`, and the new planner adapter are initialized fresh.
- The checkpoint-planner belief token uses the post-prior `final_state_belief`.
- The spline auxiliary loss uses the saved `actions` normalization stats to denormalize packed spline tokens before curve loss evaluation.
- If you want `head_right` as a fifth image, add it in both:
  - `model.image_keys`
  - `data.image_modalities`
