
###  Extract as suitable Json format

Linux / WSL

```bash
python extract_all_episodes_with_images.py
```

## What You Need To Edit Before Running

If your source/output layout differs from the defaults, edit these constants inside the script:

- `EXAMPLE_ROOT`
- `OUTPUT_ROOT`
- `OUTPUT_FORMAT`
- `DATASET_MAX_WORKERS`

Typical edits:

- change `EXAMPLE_ROOT` if your input datasets live somewhere else
- change `OUTPUT_ROOT` if you want exports written to a different directory
- set `OUTPUT_FORMAT = "json"` if you only want JSON outputs
- reduce `DATASET_MAX_WORKERS` if CPU / disk load is too high


### Convert Dataset into HF Dataset compatible with open-pi

```bash
uv run examples/lehome/convert_all_episode_json_to_lerobot.py \
  --json-root /datadrive/LEHOME/lehome-challenge/Datasets/all_episode_exports \
  --json-glob "**/json/episode_*.json" \
  --repo-name local/lehome_all_episodes \
  --source-root .. \
  --overwrite \
  --workers 1
```

If you need weighted dataset sample weighted uniformly on garment type
```bash
uv run examples/lehome/convert_all_episode_json_to_lerobot_weighted.py \
  --json-root /datadrive/LEHOME/lehome-challenge/Datasets/all_garment_type_exports \
  --json-glob "**/json/episode_*.json" \
  --repo-name local/lehome_all_episodes_weighted \
  --source-root .. \
  --overwrite \
  --workers 1
```

### Set Environment Variables
```bash
mkdir -p /datadrive/cache/openpi /datadrive/cache/hf /datadrive/hf_cache/lerobot
export OPENPI_DATA_HOME=/datadrive/cache/openpi
export HF_HOME=/datadrive/cache/hf
export HUGGINGFACE_HUB_CACHE=/datadrive/cache/hf/hub
export HF_LEROBOT_HOME=/datadrive/hf_cache/lerobot
unset TRANSFORMERS_CACHE   # removes the deprecation warning path usage
```

run [text](../../setup_scratch_nvme.md)

```bash
mkdir -p /scratch/hf/datasets /scratch/tmp
export HF_DATASETS_CACHE=/scratch/hf/datasets
export TMPDIR=/scratch/tmp
```

##Compute Norm Stats and Train
```bash
uv run scripts/compute_norm_stats.py --config-name pi05_lehome_robot_finetune
uv run scripts/compute_norm_stats.py --config-name pi05_lehome_camera_cv_robot_finetune
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_lehome_robot_finetune --exp-name=my_lehome_run --overwrite
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_lehome_camera_cv_robot_finetune --exp-name=one_episode_run --overwrite

tmux show -g mouse
tmux set -g mouse on
tmux new -s train
tmux attach -t train
Ctrl-b d

XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train_weighted.py pi05_lehome_camera_cv_robot_finetune --exp-name lehome_cv_weighted_run1 --overwrite
```




