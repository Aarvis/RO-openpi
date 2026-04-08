# Training on LeHome Episode JSON

This example shows how to convert a LeHome extracted episode JSON (for example `episode_200.json`) to a LeRobot dataset and fine-tune in `openpi`.

## 1. Convert JSON to LeRobot

```bash
uv run examples/lehome/convert_episode_json_to_lerobot.py \
  --json-path ../lehome-challenge/Datasets/episode_200.json \
  --repo-name your_hf_username/lehome_robot \
  --source-root ../lehome-challenge
```

This writes the dataset under `${HF_LEROBOT_HOME}/your_hf_username/lehome_robot`.

## 2. Compute normalization statistics

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_lehome_robot_finetune
```

## 3. Fine-tune

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
  pi05_lehome_robot_finetune \
  --exp-name=my_lehome_run \
  --overwrite
```

## Notes

- Update `repo_id` in `src/openpi/training/config.py` for `pi05_lehome_robot_finetune` to match your dataset repo name.
- If you later convert many episodes, keep the same feature schema and call `dataset.save_episode()` per episode.

## Pretrain 16D Top-Only Data

If your dataset already stores precomputed 16D camera-CV `observation.state` / `action` values and only has the
top camera, use:

```bash
uv run examples/lehome/convert_all_episode_json_precomputed_16d_to_lerobot.py \
  --json-root ../lehome-challenge/Datasets/your_pretrain_jsons \
  --json-glob "**/episode_*.json" \
  --repo-name local/lehome_precomputed_16d_pretrain \
  --source-root ../lehome-challenge
```

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_lehome_precomputed_16d_pretrain
```

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
  pi05_lehome_precomputed_16d_pretrain \
  --exp-name=my_lehome_pretrain16d_run \
  --overwrite
```

This config masks missing wrist cameras via `image_mask` and excludes the configured gripper dims from norm stats
and continuous action loss.
