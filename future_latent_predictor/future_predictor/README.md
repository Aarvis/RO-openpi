# Future Latent Predictor

This trains the next-stage compact latent predictor after the resampler autoencoder.

Input per generated dataset row:

```text
z_full[t]      [3, 256, 2048]
z_full[t+5]    [3, 256, 2048]
state[t]
```

The trainer loads the frozen resampler encoder from the previous step:

```text
z_full[t]      -> frozen resampler encoder -> z_compact[t]      [3, 24, 512]
z_full[t+5]    -> frozen resampler encoder -> z_compact[t+5]    [3, 24, 512]
```

Then it trains:

```text
state[t] + z_compact[t] -> future predictor -> z_hat_compact[t+5]
```

The model predicts a residual by default:

```text
z_hat[t+5] = z_compact[t] + predicted_delta
```

## 4x V100S Training

Run from the `openpi` repo root:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

uv run torchrun --standalone --nproc_per_node=4 \
  future_latent_predictor/future_predictor/train_future_predictor.py \
  --config future_latent_predictor/future_predictor/configs/v100s_4gpu_future_predictor.json \
  --data-root /scratch/real_robot_generated_dataset/output \
  --resampler-checkpoint /scratch/future_latent_runs/resampler_autoencoder_v1/resampler_encoder_latest.pt \
  --output-dir /scratch/future_latent_runs/future_predictor_v1
```

The default config uses:

```text
4 GPUs * batch_size_per_gpu 8 * gradient_accumulation_steps 4 = 128 effective batch
```

For `186K` rows, that is roughly:

```text
186K / 128 ~= 1453 optimizer steps per epoch
```

So the default `steps_per_epoch=1500` is close to one full pass.

## WandB

Set this in the config:

```json
"wandb": {
  "enabled": true,
  "project": "lehome-future-latent",
  "entity": "",
  "name": "future_predictor_v1",
  "token": ""
}
```

You can either put the token in `token`, or keep it empty and export:

```bash
export WANDB_API_KEY="..."
```

Only rank 0 logs to WandB.

## Validation

Validation is a deterministic parquet-shard split controlled by:

```json
"data": {
  "val_fraction": 0.02
},
"training": {
  "val_every_steps": 500,
  "val_batches": 100,
  "val_num_workers": 0
}
```

Validation rows are logged into `train_log.jsonl` and WandB, when enabled, with `"split": "val"` and metrics:

```text
val_mse
val_copy_mse
val_mse_improvement
val_cosine
val_copy_cosine
```

The important validation metric is `val_mse_improvement`; it should stay positive.

## Dataset Weights

Set per-dataset sampling weights in the config:

```json
"data": {
  "dataset_weights": {
    "local__lehome_pretrain_all_garment_round2_data": 0.5,
    "local__lehome_robot_sim_all_garment_round2_data": 2.0,
    "local__lehome_robot_real_all_garment_round2_data": 8.0
  }
}
```

Weights are applied at the dataset-folder level. The same ratio is used for train and validation sampling after each dataset folder is split into train/val shards. Set a dataset weight to `0.0` to exclude it.

## Outputs

The trainer writes:

```text
checkpoint_epoch_XXXX.pt
future_predictor_latest.pt
resolved_config.json
train_log.jsonl
```

Important logged metrics:

```text
mse                model prediction MSE to z_compact[t+5]
cosine             model cosine loss to z_compact[t+5]
copy_mse           baseline MSE for using z_compact[t] as prediction
copy_cosine        baseline cosine loss for using z_compact[t]
mse_improvement    1 - mse / copy_mse
```

The future predictor is useful only when `mse_improvement` is positive and stable.
