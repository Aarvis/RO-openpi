# Future Latent Dataset Creation

This folder contains standalone tooling for creating future-latent training parquets.
It does not modify the original LeRobot datasets or the normal OpenPI training/inference code.

The exporter uses the config:

```bash
pi05_lehome_camera_cv_multi_cotrain_robot_finetune_future_latent
```

Each `LehomeCameraCVDatasetSpec` has:

```python
include_in_future_latent_dataset=True
```

Set that field to `False` in the config for any source dataset that should be skipped.

Example:

```bash
cd E:/LeHome-Challenge/openpi

python "future_latent_predictor/Dataset creation/generate_future_latent_dataset.py" \
  --config-name pi05_lehome_camera_cv_multi_cotrain_robot_finetune_future_latent \
  --params-path "./checkpoints/pi05_lehome_camera_cv_multi_cotrain_robot_finetune/my_exp/1800/params" \
  --output-dir "./future_latent_predictor/Dataset creation/output" \
  --future-offset 5 \
  --batch-size 4 \
  --shard-size 128
```

If `--params-path` is omitted, the config's `weight_loader` is used.

If `CUDA_VISIBLE_DEVICES` exposes multiple GPUs, the default `--num-gpu-workers auto`
launches one subprocess per visible GPU and splits each source dataset into contiguous
row ranges. For example:

```bash
export CUDA_VISIBLE_DEVICES=0,1

python "future_latent_predictor/Dataset creation/generate_future_latent_dataset.py" \
  --config-name pi05_lehome_camera_cv_multi_cotrain_robot_finetune_future_latent \
  --params-path "./checkpoints/pi05_lehome_camera_cv_multi_cotrain_robot_finetune/my_exp/1800/params" \
  --output-dir "./future_latent_predictor/Dataset creation/output" \
  --future-offset 5 \
  --batch-size 4 \
  --shard-size 128
```

To force single-GPU/current-process execution:

```bash
--num-gpu-workers 1
```

In multi-worker mode, parquet shards are named with the worker id:

```text
part-worker-000-000000.parquet
part-worker-001-000000.parquet
```

Generated parquet rows include:

```text
top_embedding_t
right_wrist_embedding_t
left_wrist_embedding_t
top_embedding_t_5
right_wrist_embedding_t_5
left_wrist_embedding_t_5
```

Each embedding is stored as raw bytes with shape `[256, 2048]`.
The dtype is controlled by `--embedding-dtype` and defaults to `float16`.

Validity columns are also written, which matters for human-pretrain data where wrist images are masked:

```text
top_embedding_valid
right_wrist_embedding_valid
left_wrist_embedding_valid
top_embedding_t_5_valid
right_wrist_embedding_t_5_valid
left_wrist_embedding_t_5_valid
```

Rows whose `t + future_offset` frame crosses an episode boundary are skipped.

The exporter encodes each frame once and uses a sliding window to pair frame `t`
with frame `t + future_offset`. This avoids recomputing future-frame embeddings
for adjacent rows.

Progress bars advance only after an embedding batch completes. Their postfix is
refreshed by a heartbeat while JAX is compiling or a GPU batch is running, so
the elapsed time and ETA include time spent inside long model calls.

If a camera is masked invalid for an entire batch, the exporter skips that camera's
image encoder call and writes zero placeholder bytes with the corresponding
`*_embedding_valid=False` flag.
