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

```bash
uv run ./Datasets/examples/lehome/convert_episode_json_to_lerobot.py \
  --json-root /datadrive/LEHOME/lehome-challenge/Datasets/all_episode_exports \
  --json-glob "**/json/episode_*.json" \
  --repo-name local/lehome_all_episodes \
  --source-root .. \
  --overwrite
```

Set cache properly
mkdir -p /datadrive/cache/openpi /datadrive/cache/hf /datadrive/hf_cache/lerobot
export OPENPI_DATA_HOME=/datadrive/cache/openpi
export HF_HOME=/datadrive/cache/hf
export HUGGINGFACE_HUB_CACHE=/datadrive/cache/hf/hub
export HF_LEROBOT_HOME=/datadrive/hf_cache/lerobot
unset TRANSFORMERS_CACHE   # removes the deprecation warning path usage

mkdir -p /datadrive/cache/hf/datasets /datadrive/tmp
export HF_DATASETS_CACHE=/datadrive/cache/hf/datasets
export TMPDIR=/datadrive/tmp

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
hf upload huggingaccounttest/pi05-all-lehome-data-20000 \
  /datadrive/LEHOME/lehome-openpi/checkpoints/pi05_lehome_robot_finetune/all_data_3_epoch/20000 \
  . \
  --repo-type model

  


2) Run Eval

Good catch. This is an `openpi` downloader edge case with `hf://...` directory moves (`.partial` missing), not your checkpoint.

Use this reliable workaround: download from HF first, then serve from local path.

1. Clean broken cache entry:
```bash
rm -rf ~/.cache/openpi/huggingaccounttest/pi05-lehome-my_lehome_run-599*
```

2. Download model repo to local folder:
```bash
hf download huggingaccounttest/pi05-lehome-my_lehome_run-599 \
  --repo-type model \
  --local-dir /home/user/LEHOME/hf_ckpts/pi05-lehome-my_lehome_run-599
```

3. Verify it has `params/` and `assets/`:
```bash
ls /home/user/LEHOME/hf_ckpts/pi05-lehome-my_lehome_run-599
```

4. Start server from local path:
```bash
cd ~/LEHOME/lehome-openpi
uv run scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config pi05_lehome_robot_finetune \
  --policy.dir /home/user/LEHOME/hf_ckpts/pi05-lehome-my_lehome_run-599
```

Then keep using LeHome with `--policy_type openpi_ws` pointing to `ws://127.0.0.1:8000`.



cd ~/LEHOME/lehome-challenge

uv pip install --python "$(which python)" opencv-python-headless==4.10.0.84
python -c "import cv2; print(cv2.__version__, cv2.__file__)"

cd ~/LEHOME/lehome-challenge

uv pip install --python "$(which python)" opencv-python-headless==4.10.0.84
python -c "import cv2; print(cv2.__version__, cv2.__file__)"



export OPENPI_REPO=/home/user/LEHOME/lehome-openpi
export OPENPI_CONFIG_NAME=pi05_lehome_robot_finetune
export HF_TOKEN=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   # if repo is private

uv run scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config pi05_lehome_robot_finetune \
  --policy.dir /datadrive/LEHOME/lehome-openpi/checkpoints/pi05_lehome_robot_finetune/all_data_3_epoch/20000


export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

uv pip install -e ~/LEHOME/lehome-openpi/packages/openpi-client

uv run scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config pi05_lehome_robot_finetune \
  --policy.dir /datadrive/LEHOME/lehome-openpi/checkpoints/pi05_lehome_robot_finetune/all_data_3_epoch/20000

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

pip install -e ~/LEHOME/lehome-openpi/packages/openpi-client

python -m scripts.eval \
  --policy_type openpi_ws \
  --policy_path ws://20.244.4.116:8000 \
  --garment_type custom \
  --num_episodes 2 \
  --task_description "fold the garment on the table" \
  --step_hz 30 \
  --enable_cameras \
  --device cpu \
  --headless


python dik_solver_workflow/ik/augment_action_from_ik_and_obs_fk_to_camera_cv.py `
  --input-dir D:/LeHome-Challenge/lehome-challenge/Datasets/all_episode_exports/four_types_merged/chunk-000__file-000/json_with_action_from_ik `
  --output-dir D:/LeHome-Challenge/lehome-challenge/Datasets/all_episode_exports/four_types_merged/chunk-000__file-000/json_with_action_from_ik_camera_cv `
  --glob "episode_*.json" `
  --fk-json dik_solver_workflow/output/fk_from_usd_common.json `
  --state-unit rad `
  --pose-quat-order-world wxyz `
  --pose-quat-order-out wxyz `
  --overwrite


python lehome-camera-cv-policy-tests\evaluate_lehome_camera_cv_policy_roundtrip.py `
  --episodes-dir d:\LeHome-Challenge\lehome-challenge\Datasets\all_episode_exports\four_types_merged\chunk-000__file-000\json `
  --glob "episode_*.json" `
  --out-json d:\LeHome-Challenge\openpi\lehome-camera-cv-policy-tests\roundtrip_stats_policy_classes.json



python -m scripts.dataset_sim replay_json \
--episode_json "/home/user/LEHOME/lehome-challenge/Datasets/episode_000450_with_fk_ee_with_action_from_ik.json" \
--action_key "action.from_ik" \
--garment_name "Top_Long_Seen_8" \
--num_episodes 2 \
--step_hz 30 \
--device cpu


python -m scripts.dataset_sim replay_json \
  --episode_json "/home/user/LEHOME/lehome-challenge/Datasets/episode_000450_with_fk_ee_with_action_from_ik.json" \
  --action_key "action.from_ik" \
  --garment_name "Top_Long_Seen_8" \
  --num_episodes 2 \
  --step_hz 30 \
  --device cpu

python -m scripts.dataset_sim replay_json \
  --episodes_dir "/home/user/LEHOME/lehome-challenge/Datasets/all_episode_exports/four_types_merged/chunk-000__file-000/json_with_action_from_ik" \
  --glob "episode_*_with_fk_ee_with_action_from_ik.json" \
  --num_episodes 10 \
  --action_key "action.from_ik" \
  --garment_name "Top_Long_Seen_8" \
  --step_hz 30 \
  --device cpu

