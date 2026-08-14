wand token: 86db3e27e1bed5224ebac6150d2b59c668eb8f76

git clone --recurse-submodules https://github.com/Aarvis/lehome-openpi.git

cd lehome-openpi

git fetch --all

git switch with_val

git submodule update --init --recursive

curl -LsSf https://astral.sh/uv/install.sh | sh

GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .


pip install huggingface_hub




mkdir -p /ephemeral/cache/openpi
export OPENPI_DATA_HOME=/ephemeral/cache/openpi
export HF_HOME=/ephemeral/.hf_home
export HUGGINGFACE_HUB_CACHE=/ephemeral/.hf_home/hub
export HF_LEROBOT_HOME=/ephemeral/.hf_home/lerobot
unset TRANSFORMERS_CACHE   # removes the deprecation warning path usage

mkdir -p /ephemeral/tmp
export HF_DATASETS_CACHE=/ephemeral/.hf_home/datasets
export TMPDIR=/ephemeral/tmp

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.98

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

hf auth login
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


mkdir -p "${HF_LEROBOT_HOME}/local/lehome_all_garment_data" && \
hf download huggingaccounttest/lehome_all_garment_data \
  --repo-type dataset \
  --local-dir "${HF_LEROBOT_HOME}/local/lehome_all_garment_data"


hf download huggingaccounttest/cotrain_base_future_latent_ah10_robot_only_polish8 \
  --repo-type model \
  --local-dir "/workspace/lehome-openpi/cotrain_base_future_latent_ah10_robot_only_polish8"



hf download huggingaccounttest/sim_only_trained_future_latents_sim_round_config \
  --repo-type dataset \
  --local-dir "/ephemeral/lehome-openpi/sim_only_trained_future_latents_sim_round_config"




export DISPLAY=:99

xdpyinfo >/dev/null && echo "DISPLAY OK" && \
python -m parallel_eval \
  --headless \
  --enable_cameras \
  --garment_type custom \
  --num_episodes 20 \
  --max_workers 8 \
  --gpu_ids 0,1,2,3,4,5,6,7 \
  --ramp_up_episode_gate 1 \
  --worker_timeout_sec 86400 \
  --policy_type openpi_ws \
  --policy_paths ws://38.65.239.41:36925,ws://38.65.239.41:15223,ws://38.65.239.41:37449,ws://38.65.239.41:26268 \
  --step_hz 30 \
  --sim_device cpu \
  --device cpu \
  --time-analytics


XLA_PYTHON_CLIENT_MEM_FRACTION=0.96 uv run scripts/train.py \
  pi05_lehome_camera_cv_robot_finetune_future_latent\
  --exp-name sim_future_latent_finetune_furthur_from_10epoch\
  --overwrite \
  --fsdp-devices 1


uv run scripts/multi_serve_policy.py \
  --num-servers 4 \
  --start-port 8000 \
  --gpu-id 0 \
  --total-gpu-fraction 0.95 \
  --xla-preallocate \
  --log-dir logs/multi_serve_policy_future_latent \
  -- \
  policy:checkpoint \
  --policy.config pi05_lehome_camera_cv_multi_cotrain_robot_finetune_future_latent \
  --policy.dir /workspace/LEHOME/lehome-openpi/cotrain_base_future_latent_sim_only_trial_1500