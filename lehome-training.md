Implemented in `openpi`:

1. New LeHome converter (JSON -> LeRobot dataset)
- `examples/lehome/convert_episode_json_to_lerobot.py:47`
- Reads your `episode_200.json`, loads saved image paths, creates a LeRobot dataset with:
  - `observation.images.top_rgb`
  - `observation.images.left_rgb`
  - `observation.images.right_rgb`
  - `observation.state`
  - `actions`
  - `task` (from prompt)
- Infers FPS from timestamps.

2. New LeHome input/output mapping for openpi
- `src/openpi/policies/lehome_policy.py:31` (`LehomeInputs`)
- `src/openpi/policies/lehome_policy.py:70` (`LehomeOutputs`)
- Maps LeHome observations/actions into openpi model format (PI0/PI05/PI0_FAST).

3. New LeHome data config + train config
- Import wired: `src/openpi/training/config.py:22`
- `LeRobotLehomeDataConfig`: `src/openpi/training/config.py:360`
- New train config: `src/openpi/training/config.py:828` (`pi05_lehome_robot_finetune`)

4. Added usage doc
- `examples/lehome/README_train.md:1`

How to run:
```bash
uv run examples/lehome/convert_episode_json_to_lerobot.py \
  --json-path ../lehome-challenge/Datasets/episode_200.json \
  --repo-name your_hf_username/lehome_robot \
  --source-root ../lehome-challenge
```

Set cache properly
mkdir -p /datadrive/cache/openpi /datadrive/cache/hf /datadrive/hf_cache/lerobot
export OPENPI_DATA_HOME=/datadrive/cache/openpi
export HF_HOME=/datadrive/cache/hf
export HUGGINGFACE_HUB_CACHE=/datadrive/cache/hf/hub
export HF_LEROBOT_HOME=/datadrive/hf_cache/lerobot
unset TRANSFORMERS_CACHE   # removes the deprecation warning path usage

Then:
```bash
uv run scripts/compute_norm_stats.py --config-name pi05_lehome_robot_finetune
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_lehome_robot_finetune --exp-name=my_lehome_run --overwrite
```

Important:
- Update `repo_id` in `src/openpi/training/config.py:832` to your actual LeRobot repo name.
- I could not fully execute conversion in this shell because `lerobot` is not installed in the current Python env (`ModuleNotFoundError: lerobot`).  
- Syntax compile check for modified files passed.


uv run examples/lehome/convert_episode_json_to_lerobot.py \
  --json-path examples/lehome/sample_data_episode/episode_200.json \
  --repo-name huggingaccounttest/lehome-openpi-episode \
  --source-root examples/lehome/sample_data_episode \
  --push-to-hub


-----
Run Eval Lehome
make sure assets are present
# This creates the Assets/ directory with all required simulation resources
hf download lehome/asset_challenge --repo-type dataset --local-dir Assets

hf download lehome/dataset_challenge_merged --repo-type dataset --local-dir Datasets/example

hf download lehome/dataset_challenge --repo-type dataset --local-dir Datasets/example


1) Upload Checkpoint to HF
cd /datadrive/lehome-openpi


hf auth login
hf repo create huggingaccounttest/pi05-lehome-my_lehome_run-599 --private

# one-command upload of the step folder root (contains params/ and assets/)
hf upload huggingaccounttest/pi05-lehome-my_lehome_run-599 \
  /datadrive/lehome-openpi/checkpoints/pi05_lehome_robot_finetune/my_lehome_run/599 \
  . \
  --repo-type model


2) Run Eval

export OPENPI_REPO=/datadrive/lehome-openpi
export OPENPI_CONFIG_NAME=pi05_lehome_robot_finetune
export HF_TOKEN=...   # if repo is private

python -m scripts.eval \
  --policy_type openpi \
  --policy_path "hf://huggingaccounttest/pi05-lehome-my_lehome_run-599" \
  --garment_type top_long \
  --num_episodes 2 \
  --task_description "fold the garment on the table" \
  --step_hz 30 \
  --enable_cameras \
  --device cpu
