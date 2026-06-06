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
  --json-path /datadrive/LEHOME/lehome-challenge/Datasets/all_episode_exports/four_types_merged/chunk-000__file-000/json/episode_000450.json \
  --repo-namelocal/lehome_all_e \
  --overwrite \
  --workers 1
```

```bash
uv run examples/lehome/convert_all_episode_json_to_lerobot.py \
  --json-root /datadrive/LEHOME/lehome-challenge/Datasets/all_episode_exports \
  --json-glob "**/json/episode_*.json" \
  --repo-name local/lehome_all_episodes \
  --source-root .. \
  --overwrite \
  --workers 1
```


uv run examples/lehome/convert_all_episode_json_precomputed_16d_to_lerobot.py \
  --json-root /workspace/lehome-challenge/Datasets/pretrain_all_garment_data_extracted \
  --json-glob "**/episode_*.json" \
  --repo-name local/lehome_all_garment_data_16d_pretrain \
  --source-root .. \
  --overwrite \
  --workers 1

uv run examples/lehome/convert_all_episode_json_to_lerobot.py \
  --json-root /datadrive/LEHOME/lehome-challenge/Datasets/train_all_garment_individual_extracted \
  --json-glob "**/json/episode_*.json" \
  --repo-name local/lehome_all_episodes_train \
  --source-root .. \
  --overwrite \
  --workers 1

hf upload huggingaccounttest/lehome_train_episodes \
  "/datadrive/hf_cache/lerobot/local/lehome_train_episodes" \
  . \
  --repo-type dataset && \
hf repos settings huggingaccounttest/lehome_train_episodes \
  --repo-type dataset \
  --gated manual


```bash
uv run examples/lehome/convert_all_episode_json_to_lerobot.py \
  --json-root /workspace/LEHOME/lehome-challenge/Datasets/all_garment_data_images_extracted \
  --json-glob "**/json/episode_*.json" \
  --repo-name local/lehome_all_garment_data \
  --source-root .. \
  --overwrite \
  --workers 1
```

```bash
uv run examples/lehome/convert_all_episode_json_to_lerobot_weighted.py \
  --json-root /datadrive/LEHOME/lehome-challenge/Datasets/all_garment_type_exports \
  --json-glob "**/json/episode_*.json" \
  --repo-name local/lehome_train_episodes \
  --source-root .. \
  --overwrite \
  --workers 1
```

```bash
uv run examples/lehome/parallel_convert_all_episode_json_to_lerobot_weighted.py \
  --json-root /datadrive/LEHOME/lehome-challenge/Datasets/all_garment_type_exports \
  --json-glob "**/json/episode_*.json" \
  --repo-name local/lehome_one_episode_weighted_parallel \
  --source-root .. \
  --overwrite \
  --workers 8 \
  --temp-root /scratch/tmp
```


Set cache properly
```bash
mkdir -p /datadrive/cache/openpi /datadrive/cache/hf /datadrive/hf_cache/lerobot
export OPENPI_DATA_HOME=/datadrive/cache/openpi
export HF_HOME=/datadrive/cache/hf
export HUGGINGFACE_HUB_CACHE=/datadrive/cache/hf/hub
export HF_LEROBOT_HOME=/datadrive/hf_cache/lerobot

unset TRANSFORMERS_CACHE   # removes the deprecation warning path usage
```

```bash
mkdir -p /workspace/cache/openpi
export OPENPI_DATA_HOME=/workspace/cache/openpi
export HF_HOME=/workspace/.hf_home
export HUGGINGFACE_HUB_CACHE=/workspace/.hf_home/hub
export HF_LEROBOT_HOME=/workspace/.hf_home/lerobot
unset TRANSFORMERS_CACHE   # removes the deprecation warning path usage

mkdir -p /workspace/tmp
export HF_DATASETS_CACHE=/workspace/.hf_home/datasets
export TMPDIR=/workspace/tmp

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

uv run scripts/train.py \
  pi05_lehome_camera_cv_robot_finetune\
  --exp-name lehome_train_eval \
  --overwrite \
  --fsdp-devices 1
```


export HUGGINGFACE_HUB_CACHE=/scratch/hf/lerobot

Then:
```bash
mkdir -p /scratch/hf/datasets /scratch/tmp
export HF_DATASETS_CACHE=/scratch/hf/datasets
export TMPDIR=/scratch/tmp

uv run scripts/compute_norm_stats.py --config-name pi05_lehome_robot_finetune
uv run scripts/compute_norm_stats.py --config-name pi05_lehome_camera_cv_robot_finetune
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_lehome_robot_finetune --exp-name=my_lehome_run --overwrite
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_lehome_camera_cv_robot_finetune --exp-name=one_episode_run --overwrite



tmux show -g mouse
tmux set -g mouse on
tmux new -s train
tmux attach -t train
Ctrl-b d

tmux new -s server

XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train_weighted.py pi05_lehome_camera_cv_robot_finetune --exp-name lehome_cv_weighted_run1 --overwrite
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
hf upload huggingaccounttest/pi05-all-lehome-dik-20K \
  /datadrive/LEHOME/lehome-openpi/checkpoints/pi05_lehome_camera_cv_robot_finetune/lehome_cv_weighted_run1/20000 \
  . \
  --repo-type model




hf upload huggingaccounttest/mid-16epoch-OF-run \
  /workspace/LEHOME/lehome-openpi/checkpoints/pi05_lehome_camera_cv_robot_finetune/lehome_train_eval/latest_val/9000 \
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
```bash
uv run scripts/multi_serve_policy.py \
  --num-servers 4 \
  --start-port 8000 \
  --gpu-id 0 \
  --total-gpu-fraction 0.96 \
  --xla-preallocate \
  --stagger-seconds 2.0 \
  --log-dir logs/multi_serve_policy \
  -- \
  policy:checkpoint \
  --policy.config pi05_lehome_camera_cv_robot_finetune \
  --policy.dir /datadrive/LEHOME/lehome-openpi/checkpoints/pi05_lehome_camera_cv_robot_finetune/lehome_cv_weighted_run1/63000
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

find /usr /etc /opt /lib /lib64 -type f -name 'nvidia_icd.json' 2>/dev/null


export CUDA_DEVICE_ORDER=PCI_BUS_ID
unset CUDA_VISIBLE_DEVICES
# or explicitly: export CUDA_VISIBLE_DEVICES=0,1,2,3
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

find /usr /etc /opt /lib /lib64 -type f -name 'nvidia_icd.json' 2>/dev/null

export CUDA_DEVICE_ORDER=PCI_BUS_ID
unset CUDA_VISIBLE_DEVICES
# or explicitly: export CUDA_VISIBLE_DEVICES=0,1,2,3
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
--allow_duplicate_garments

python -m scripts.eval \
    --policy_type docker \
    --docker_url http://174.88.252.119:15019 \
    --garment_type custom \
    --num_episodes 10 \
    --headless \
    --device cpu \
    --enable_cameras


uv pip install -e /workspace/LEHOME/lehome-openpi/packages/openpi-client

uv run scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config pi05_lehome_camera_cv_robot_finetune \
  --policy.dir /datadrive/LEHOME/lehome-openpi/checkpoints/pi05_lehome_camera_cv_robot_finetune/all_episode_2_epoch/8399

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

uv pip install -e ~/LEHOME/lehome-openpi/packages/openpi-client

python -m scripts.eval \
  --policy_type openpi_ws \
  --policy_path ws://20.244.4.116:8000 \
  --garment_type custom \
  --num_episodes 1 \
  --task_description "fold the garment on the table" \
  --step_hz 30 \
  --enable_cameras \
  --device cpu \
  --headless

python -m parallel_eval \
  --headless \
  --enable_cameras \
  --step_hz 30 \
  --garment_type custom \
  --num_episodes 5 \
  --max_workers 10 \
  --policy_type openpi_ws \
  --policy_base_ws_url ws://20.244.4.116 \
  --policy_start_port 8000 \
  --policy_port_step 1 \
  --policy_server_count 3 \
  --device cpu


python -m parallel_eval \
  --headless \
  --garment_type custom \
  --num_episodes 15 \
  --max_workers 5 \
  --policy_type openpi_ws \
  --policy_base_ws_url ws://20.244.4.116 \
  --policy_start_port 8000 \
  --policy_port_step 1 \
  --policy_server_count 5 \
  --device cpu


python -m round2_sim_inference_eval_test \
  --headless \
  --enable_cameras \
  --policy_server_addr localhost:8080 \
  --actions_per_chunk 5 \
  --garment_type custom \
  --num_episodes 5 \
  --max_steps 600 \
  --step_hz 30 \
  --sim_device cpu \
  --device cpu

python -m round2_sim_inference_eval_test \
  --headless \
  --enable_cameras \
  --policy_server_addr http://146.115.17.138:62328 \
  --actions_per_chunk 5 \
  --garment_type top_long \
  --num_episodes 5 \
  --max_steps 600 \
  --step_hz 30 \
  --sim_device cpu \
  --device cpu



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

uv run lehome-camera-cv-policy-tests/evaluate_lehome_camera_cv_policy_roundtrip.py \
--episodes-dir /datadrive/LEHOME/lehome-challenge/Datasets/all_episode_exports/four_types_merged/chunk-000__file-000/json \
--glob "episode_*.json" \
--workers 16 \
--out-json lehome-camera-cv-policy-tests\roundtrip_stats_policy_classes.json


python -m parallel_eval.tail_log --run_dir outputs/parallel_eval/run_20260311_132357_custom --worker_id 1 --follow

python -m parallel_eval \
  --headless \
  --enable_cameras \
  --garment_type custom \
  --num_episodes 2 \
  --max_workers 8 \
  --gpu_ids 0,1,2,3,4,5,6,7 \
  --ramp_up_episode_gate 0 \
  --worker_timeout_sec 7200 \
  --policy_type openpi_ws \
  --policy_base_ws_url ws://20.244.4.116 \
  --policy_start_port 8000 \
  --policy_port_step 1 \
  --step_hz 30 \
  --sim_device cpu \
  --device cpu \
  --time-analytics \


python -m parallel_eval \
  --headless \
  --enable_cameras \
  --garment_type custom \
  --num_episodes 2 \
  --max_workers 8 \
  --gpu_ids 0,1,2,3,4,5,6,7 \
  --ramp_up_episode_gate 0 \
  --worker_timeout_sec 7200 \
  --policy_type openpi_ws \
  --policy_base_ws_url ws://20.244.4.116 \
  --policy_start_port 8000 \
  --policy_port_step 1 \
  --step_hz 30 \
  --sim_device cpu \
  --device cpu \
  --time-analytics \


python -m parallel_eval \
  --headless \
  --enable_cameras \
  --garment_type custom \
  --num_episodes 10 \
  --max_workers 8 \
  --gpu_ids 0,1,2,3,4,5,6,7 \
  --ramp_up_episode_gate 1 \
  --worker_timeout_sec 10800 \
  --policy_type openpi_ws \
  --policy_paths ws://154.57.34.105:11072,ws://154.57.34.105:13499,ws://154.57.34.105:17594,ws://154.57.34.105:17589,ws://154.57.34.105:10419,ws://154.57.34.105:18465 \
  --step_hz 30 \
  --sim_device cpu \
  --device cpu \
  --time-analytics



python -m parallel_eval \
  --headless \
  --enable_cameras \
  --garment_type custom \
  --num_episodes 2 \
  --max_workers 4 \
  --gpu_ids 0,1,2,3 \
  --worker_timeout_sec 86400 \
  --policy_type openpi_ws \
  --policy_base_ws_url ws://20.244.4.116 \
  --policy_start_port 8000 \
  --policy_port_step 1 \
  --step_hz 30 \
  --sim_device cpu \
  --device cpu \
  --record_episodes \
  --record_all_episodes \
  --record_inbuilt_step_rewards \
  --record_policy_latent \
  --time-analytics \
  --use_random_seed

python -m parallel_eval \
  --headless \
  --enable_cameras \
  --garment_type custom \
  --num_episodes 50 \
  --max_workers 4 \
  --gpu_ids 0,1,2,3 \
  --ramp_up_episode_gate 1 \
  --worker_timeout_sec 86400 \
  --policy_type openpi_ws \
  --policy_base_ws_url ws://20.244.4.116 \
  --policy_start_port 8004 \
  --policy_port_step 1 \
  --step_hz 30 \
  --sim_device cpu \
  --device cpu \
  --record_episodes \
  --record_video \
  --record_all_episodes \
  --no-record_keep_frame_images \
  --record_inbuilt_step_rewards \
  --record_policy_latent \
  --time-analytics \
  --use_random_seed
  --allow


uv run lehome-camera-cv-policy-tests/plot_test_continuity/plot_episode_continuity.py --episode-json /datadrive/LEHOME/lehome-challenge/Datasets/all_garment_type_exports/top_long_merged/chunk-000__file-000/json/episode_000150.json

uv run lehome-camera-cv-policy-tests/plot_test_continuity/plot_episode_continuity.py --episode-json /datadrive/LEHOME/lehome-challenge/Datasets/all_garment_type_exports/top_long_merged/chunk-000__file-000/json/episode_000240.json



uv run lehome-camera-cv-policy-tests/plot_test_continuity/plot_episode_continuity.py \
  --episode-json /datadrive/LEHOME/lehome-challenge/Datasets/all_garment_type_exports/top_long_merged/chunk-000__file-000/json/episode_000100.json \
  --out-dir test_policy_continuity

uv run lehome-camera-cv-policy-tests/evaluate_lehome_camera_cv_policy_roundtrip.py \
  --episodes-dir /datadrive/LEHOME/lehome-challenge/Datasets/all_garment_type_exports/top_long_merged/chunk-000__file-000/json \
  --glob "episode_*.json" \
  --model-type pi05 \
  --workers 8 \
  --out-json output/lehome_camera_cv_policy_roundtrip_stats.json

python dik_solver_workflow/utils/plot_obs_ee_xyz_camera_frame.py --episode-json D:\LeHome-Challenge\lehome-challenge\test_outputs_camera_cv\episode_000450_with_fk_ee_with_camera_cv.json

python dik_solver_workflow/utils/plot_obs_ee_xyz_camera_frame.py --episode-json D:\LeHome-Challenge\lehome-challenge\test_outputs_camera_cv\episode_000450_with_fk_ee_with_camera_cv.json



python dik_solver_workflow/augment_all_episode_jsons_with_fk_ee.py --input-dir D:\LeHome-Challenge\lehome-challenge\Datasets\all_garment_type_exports\top_long_merged\chunk-000__file-000\json --glob "episode_000200.json" --output-dir test_outputs --overwrite


python dik_solver_workflow/augment_all_episode_jsons_with_fk_ee.py --input-dir D:\LeHome-Challenge\lehome-challenge\Datasets\all_episode_exports\four_types_merged\chunk-000__file-000\json --glob "episode_000450.json" --output-dir test_outputs --overwrite


python dik_solver_workflow/ik/write_action_from_ik_batch_parallel.py `
  --episodes-dir test_outputs `
  --glob "episode_000450_with_fk_ee.json" `
  --fk-json dik_solver_workflow/output/fk_from_usd_common.json `
  --state-unit rad `
  --pose-quat-order wxyz `
  --line-search-alphas 2,1,0.5,0.25,0.05 `
  --damping 0.1 `
  --alpha 1.0 `
  --max-iters 200 `
  --rot-weight 2 `
  --workers 1 `
  --out-dir D:\LeHome-Challenge\lehome-challenge\Datasets\all_episode_exports\four_types_merged\chunk-000__file-000



python dik_solver_workflow/ik/augment_action_from_ik_and_obs_fk_to_camera_cv.py `
  --input-dir D:\LeHome-Challenge\lehome-challenge\Datasets\all_episode_exports\four_types_merged\chunk-000__file-000 `
  --output-dir test_outputs_camera_cv `
  --glob "episode_000450_with_fk_ee_with_action_from_ik.json" `
  --overwrite


python dik_solver_workflow/ik/augment_action_from_ik_and_obs_fk_to_camera_cv.py `
  --input-dir test_outputs `
  --output-dir test_outputs_camera_cv `
  --glob "episode_000450_with_fk_ee.json" `
  --overwrite

python dik_solver_workflow/utils/stitch_episode_images_to_video.py `
  --episode-json D:\LeHome-Challenge\lehome-challenge\Datasets\all_garment_type_exports\top_long_merged\chunk-000__file-000\json\episode_000200.json `
  --output-dir D:\LeHome-Challenge\lehome-challenge\test_outputs\videos `
  --fps 30


python dik_solver_workflow/utils/plot_obs_ee_xyz_camera_frame.py \
  --episode-json /datadrive/LEHOME/lehome-challenge/Datasets/all_garment_type_exports/top_long_merged/chunk-000__file-000/json/episode_000100.json 




python -m parallel_eval   --headless   --enable_cameras   --garment_type custom   --num_episodes 20  --max_workers 8   --gpu_ids 0,1,2,3,4,5,6,7   --ramp_up_episode_gate 1   --worker_timeout_sec 86400   --policy_type openpi_ws   --policy_paths ws://57.128.84.121:8000,ws://57.128.84.121:8001,ws://57.128.84.121:8002,ws://57.128.84.121:8003 --step_hz 30  --sim_device cpu  --device cpu --time-analytics


python -m parallel_eval   --headless   --enable_cameras   --garment_type custom   --num_episodes 20  --max_workers 4   --gpu_ids 0,1,2,3 --ramp_up_episode_gate 1   --worker_timeout_sec 86400   --policy_type openpi_ws   --policy_paths ws://69.19.136.171:8000,ws://69.19.136.171:8001,ws://69.19.136.171:8002,ws://69.19.136.171:8003  --step_hz 30  --sim_device cpu  --device cpu --time-analytics 

--use_random_seed --allow_duplicate_garments 


--record_episodes --record_all_episodes --record_inbuilt_step_rewards --record_policy_latent 


python -m parallel_eval   --headless   --enable_cameras   --garment_type custom   --num_episodes 100   --max_workers 8   --gpu_ids 0,1,2,3,4,5,6,7   --ramp_up_episode_gate 1   --worker_timeout_sec 40800   --policy_type openpi_ws   --policy_paths ws://192.222.55.139:12385,ws://192.222.55.139:52609,ws://192.222.55.139:58557,ws://192.222.55.139:44628,ws://192.222.55.139:52200,ws://192.222.55.139:6396,ws://192.222.55.139:45045 --step_hz 30   --sim_device cpu   --device cpu   --time-analytics


python -m parallel_eval --headless --enable_cameras   --garment_type custom   --num_episodes 10   --max_workers 1  --gpu_ids 0   --worker_timeout_sec 10800   --policy_type openpi_ws   --policy_paths ws://20.38.175.29:1603 --step_hz 30   --sim_device cpu   --device cpu   --time-analytics   --record_episodes --record_all_episodes


python -m parallel_eval   --headless   --enable_cameras   --garment_type custom   --num_episodes 50   --max_workers 14   --gpu_ids 0,1,2,3,4,5,6,7,8,9,10,11,12,13   --ramp_up_episode_gate 1   --worker_timeout_sec 86400   --policy_type openpi_ws   --policy_paths ws://20.150.146.205:8045,ws://20.150.146.205:8779,ws://20.150.146.205:6517,ws://20.150.146.205:6494,ws://20.150.146.205:7981,ws://20.150.146.205:9801 --step_hz 30  --sim_device cpu   --device cpu --time-analytics


python -m parallel_eval   --headless   --enable_cameras   --garment_type custom   --num_episodes 5  --max_workers 1   --gpu_ids 0 --ramp_up_episode_gate 1   --worker_timeout_sec 86400   --policy_type openpi_ws   --policy_paths ws://81.183.231.113:54949 --step_hz 30  --sim_device cpu   --device cpu --time-analytics

python -m parallel_eval   --headless   --enable_cameras   --garment_type custom   --num_episodes 50   --max_workers 8   --gpu_ids 0,1,2,3,4,5,6,7  --ramp_up_episode_gate 1   --worker_timeout_sec 86400   --policy_type openpi_ws   --policy_paths ws://20.150.146.205:6517,ws://20.150.146.205:6494,ws://20.150.146.205:7981,ws://20.150.146.205:9801,ws://20.150.146.205:8045,ws://20.150.146.205:8779 --step_hz 30  --sim_device cpu   --device cpu --time-analytics




python -m parallel_eval   --headless   --enable_cameras   --garment_type custom   --num_episodes 20   --max_workers 4   --gpu_ids 0,1,2,3  --ramp_up_episode_gate 1   --worker_timeout_sec 86400   --policy_type openpi_ws   --policy_paths ws://38.79.155.163:61163,ws://38.79.155.163:61669,ws://38.79.155.163:61650,ws://38.79.155.163:61289 --step_hz 30  --sim_device cpu   --device cpu --time-analytics

python -m parallel_eval   --headless   --enable_cameras   --garment_type custom   --num_episodes 50   --max_workers 8   --gpu_ids 0,1,2,3,4,5,6,7   --ramp_up_episode_gate 1   --worker_timeout_sec 86400   --policy_type openpi_ws   --policy_paths ws://20.150.146.205:8045,ws://20.150.146.205:8779,ws://20.150.146.205:6517,ws://20.150.146.205:6494,ws://20.150.146.205:7981,ws://20.150.146.205:9801,ws://20.150.146.205:5275,ws://20.150.146.205:8601 --step_hz 30  --sim_device cpu   --device cpu --time-analytics


python -m round2_sim_inference_eval_test \
  --headless \
  --enable_cameras \
  --policy_server_addr 81.183.231.113:54949 \
  --actions_per_chunk 5 \
  --garment_type custom \
  --num_episodes 5 \
  --max_steps 600 \
  --step_hz 30 \
  --sim_device cpu \
  --device cpu




python .\Datasets\OnlineRL\1_frame_final_reward_per_chunk_Step.py `
  "E:\LeHome-Challenge\lehome-challenge\Datasets\OnlineRL\PPO_soft_launch_roll_out_2b" `
  --output-dir "E:\LeHome-Challenge\lehome-challenge\Datasets\OnlineRL\PPO_soft_launch_roll_out_2b_rewarded" `
  --config "E:\LeHome-Challenge\lehome-challenge\Datasets\OnlineRL\config.json" `
  --workers 8 `
  --overwrite

python .\Datasets\OnlineRL\2_create_monte_carlo_all_rewarded_episodes.py `
  "E:\LeHome-Challenge\lehome-challenge\Datasets\OnlineRL\PPO_soft_launch_roll_out_2b_rewarded" `
  --output-dir "E:\LeHome-Challenge\lehome-challenge\Datasets\OnlineRL\PPO_soft_launch_roll_out_2b_rewarded_mc" `
  --config "E:\LeHome-Challenge\lehome-challenge\Datasets\OnlineRL\config.json" `
  --workers 8 `
  --overwrite





