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


hf download huggingaccounttest/pretrain_base_all_garment_4_epoch\
  --local-dir ./cotrain_base_ah10_robot_only_polish3 \
  --repo-type model


mkdir -p "${HF_LEROBOT_HOME}/local/lehome_all_garment_data" && \
hf download huggingaccounttest/lehome_all_garment_data \
  --repo-type dataset \
  --local-dir "${HF_LEROBOT_HOME}/local/lehome_all_garment_data"

hf download huggingaccounttest/sim_future_latents_dependency_multi_cotrain_base3 \
  --repo-type model \
  --local-dir "/workspace/LEHOME/lehome-openpi/robot_future_latents_dependency_multi_cotrain_base3"

hf download huggingaccounttest/sim_future_latents_dependency_multi_cotrain_base3 \
  --repo-type model \
  --local-dir "/ephemeral2/sim_future_latents_dependency_multi_cotrain_base3"


hf download huggingaccounttest/pretrain_base_all_garment_4_epoch \
  --repo-type model \
  --local-dir "/ephemeral2/pretrain_base_all_garment_4_epoch"


  

Update Training Config for Policy

compute norm_stats for training data

uv run scripts/compute_norm_stats.py --config-name pi05_lehome_camera_cv_robot_finetune


XLA_PYTHON_CLIENT_MEM_FRACTION=0.96 uv run scripts/train.py \
  pi05_lehome_precomputed_16d_pretrain\
  --exp-name lehome_pretrain_with_state_all_garments\
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